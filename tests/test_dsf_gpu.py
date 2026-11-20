"""Real CUDA parity for every DSF channel and bounded streaming modes."""

from contextlib import closing

import numpy as np
import pytest

from mdtrace.dsf import gpu
from mdtrace.dsf.dsf import compute_dsf
from mdtrace.dsf.results import load_spectra, save_dsf
from mdtrace.dsf.spectra import block_spectra
from tests.test_dsf_modes import _params, _write


@pytest.fixture
def cuda():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device required")
    except cp.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")
    return cp


def test_gpu_stream_plan_bounds_memory_without_changing_time_block():
    batch, q_batch, resident = gpu._stream_plan(100, 1000, 100, True, 64, 256 * 1024**2)
    assert resident and 1 <= batch <= 64 and 1 <= q_batch <= 100
    batch, q_batch, resident = gpu._stream_plan(100, 10000, 1000, True, 64, 64 * 1024**2)
    assert not resident and 1 <= batch <= 64 and 1 <= q_batch < 1000
    with pytest.raises(MemoryError, match="GPU memory"):
        gpu._stream_plan(100, 10000, 1000, True, 64, 1)


def test_gpu_stream_plan_balances_training_q_points_within_available_memory():
    budget = int(.90 * 22.74 * 1024**3)
    assert gpu._stream_plan(73728, 3500, 500, True, 64, budget) == (64, 125, True)
    assert gpu._stream_plan(73728, 3500, 501, True, 64, budget) == (64, 126, True)
    batch, q_batch, _ = gpu._stream_plan(73728, 3500, 500, True, 64, budget // 2)
    assert batch == 64 and 1 <= q_batch < 125


@pytest.mark.parametrize(("n_q", "q_batch", "summary"), [
    (500, 93, "500 Q vectors in 6 batches (5 × 93 + 35)"),
    (500, 100, "500 Q vectors in 5 batches (5 × 100)"),
    (500, 500, "500 Q vectors in 1 batch"),
    (1, 1, "1 Q vector in 1 batch"),
])
def test_gpu_q_batch_summary(n_q, q_batch, summary):
    assert gpu._q_batch_summary(n_q, q_batch) == summary


@pytest.mark.parametrize("mode", ["density", "currents", "self", "neutron", "partials"])
@pytest.mark.parametrize("normalization", ["density", "legacy"])
@pytest.mark.parametrize("subtract_mean", [False, True])
def test_gpu_all_channels_match_cpu(cuda, mode, normalization, subtract_mean):
    rng = np.random.default_rng(43)
    positions, velocities = rng.normal(size=(2, 33, 8, 3))
    q = np.array([[0., 0, 0], [1., 0, 0], [-1., 0, 0], [0, 1., 0]])
    labels = np.array(["A", "B"] * 4)
    coherent = np.where(labels == "A", 2 + 3j, -1 + .7j) if mode in ("neutron", "partials") else np.ones(8)
    incoherent = np.full(8, .7) if mode in ("neutron", "partials") else np.zeros(8)
    options = dict(dt_ps=.005, normalization=normalization, subtract_mean=subtract_mean,
                   velocities=velocities if mode != "density" else None,
                   species_labels=labels if mode == "partials" else None,
                   compute_self=mode in ("self", "neutron"))
    expected = block_spectra(positions, q, coherent, incoherent, **options)
    progress = []
    with gpu.GPUBackend() as backend:
        actual = backend.compute(positions, q, coherent, incoherent, progress=progress.append, **options)
    assert progress[0] == 0 and progress[-1] == 1
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=2e-11, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("currents", [False, True])
@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("prefetch", [False, True])
def test_gpu_stream_matches_cpu_with_q_and_frame_batches(cuda, tmp_path, monkeypatch, currents, resident, prefetch):
    rng = np.random.default_rng(15)
    positions, velocities = rng.normal(size=(2, 34, 8, 3))
    path = tmp_path / "trajectory.nc"
    _write(path, positions, velocities=velocities)
    q = np.array([[0., 0, 0], [1., 0, 0], [-1., 0, 0], [0, 1., 0]])
    coherent = np.array([1 + 2j, -.5 + .3j] * 4)
    monkeypatch.setattr(gpu, "_batch_plan", lambda *args: (7, 1, prefetch))
    monkeypatch.setattr(gpu, "_stream_plan", lambda *args: (7, 2, resident))
    options = dict(dt_ps=.01, subtract_mean=True)
    progress = []
    stages = []
    with gpu.GPUBackend() as backend:
        with closing(backend.stream_blocks(path, 17, 2, 8, q, coherent,
                                          currents=currents, prefetch=prefetch,
                                          progress=progress.append,
                                          batch_status=lambda _, stage, done, total: stages.append((" ".join(stage.split()), done, total)),
                                          **options)) as results:
            for block, actual in enumerate(results):
                frames = slice(block * 17, (block + 1) * 17)
                expected = block_spectra(positions[frames], q, coherent, np.zeros(8),
                                         velocities=velocities[frames] if currents else None,
                                         compute_self=False, **options)
                assert actual.keys() == expected.keys()
                for name in expected:
                    np.testing.assert_allclose(actual[name], expected[name], rtol=2e-11, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(progress, [0, 7/17, 14/17, .99, 1, 1 + 7/17, 1 + 14/17, 1.99])
    labels = ["frames 1-7 / 34", "frames 8-14 / 34", "frames 15-17 / 34", "FFT · frames 1-17 / 34",
              "frames 18-24 / 34", "frames 25-31 / 34", "frames 32-34 / 34", "FFT · frames 18-34 / 34"]
    assert stages == [(label, done, 4) for label in labels for done in (0, 2, 4)]


@pytest.mark.parametrize("failure_stage", ["phase", "fft"])
def test_gpu_retries_unfinished_q_vectors_under_pool_limit(cuda, tmp_path, monkeypatch, failure_stage):
    rng = np.random.default_rng(23)
    positions, velocities = rng.normal(size=(2, 14, 8, 3))
    q = rng.normal(size=(8, 3))
    q[0] = 0
    path = tmp_path / "trajectory.nc"
    _write(path, positions, velocities=velocities)
    monkeypatch.setattr(gpu, "_stream_plan", lambda *args: (7, 4, True))
    outer = cuda.cuda.MemoryPool()
    outer.set_limit(size=64 * 1024**2)
    attempts = []

    def check_allocation(width):
        attempts.append(width)
        pool = cuda.cuda.get_allocator().__self__
        assert 0 < pool.get_limit() <= int(.90 * (outer.get_limit() - 1024**2))
        if len(attempts) == 2:
            cuda.empty(pool.get_limit() + 1, dtype=cuda.uint8)

    if failure_stage == "phase":
        original_exp = cuda.exp

        def exp(phase, **kwargs):
            check_allocation(phase.shape[1])
            return original_exp(phase, **kwargs)

        monkeypatch.setattr(cuda, "exp", exp)
    else:
        original_fft = gpu._fft_spectra

        def fft(cp, signals, *args, **kwargs):
            check_allocation(len(signals))
            return original_fft(cp, signals, *args, **kwargs)

        monkeypatch.setattr(gpu, "_fft_spectra", fft)

    stages = {}
    with cuda.cuda.using_allocator(outer.malloc), gpu.GPUBackend() as backend:
        held = cuda.empty(1024**2, dtype=cuda.uint8)
        with closing(backend.stream_blocks(
            path, 7, 2, 8, q, np.ones(8), currents=True, dt_ps=.01,
            batch_status=lambda _, label, done, total: stages.setdefault(" ".join(label.split()), []).append(done),
        )) as blocks:
            for block, actual in enumerate(blocks):
                assert cuda.cuda.get_allocator().__self__ is outer
                frames = slice(block * 7, (block + 1) * 7)
                expected = block_spectra(positions[frames], q, np.ones(8), np.zeros(8),
                                         velocities=velocities[frames], compute_self=False, dt_ps=.01)
                for name in expected:
                    np.testing.assert_allclose(actual[name], expected[name], rtol=2e-11, atol=1e-12, equal_nan=True)
        assert outer.get_limit() == 64 * 1024**2 and held.nbytes == 1024**2
    label = "frames 1-7 / 14" if failure_stage == "phase" else "FFT · frames 1-7 / 14"
    assert stages[label] == [0, 4, 6, 8]
    assert attempts == [4, 4, 2, 2, 2, 2, 2, 2]


@pytest.mark.parametrize("components", [False, True])
def test_gpu_pipeline_preserves_shells_blocks_and_text(cuda, tmp_path, components):
    rng = np.random.default_rng(17)
    positions, velocities = rng.normal(size=(2, 34, 8, 3))
    path = tmp_path / "trajectory.nc"
    _write(path, positions, velocities=velocities)
    options = dict(dsf_q_mode="spherical", dsf_qpoints=None, dsf_q_min=.5, dsf_q_max=1.5,
                   dsf_q_bin_width=.5, dsf_max_points=0, dsf_currents=True,
                   dsf_output_components=components, max_cores=1)
    expected = compute_dsf(_params(path, **options))
    params = _params(path, device="gpu", **options)
    actual = compute_dsf(params)
    for name, values in expected.items():
        if isinstance(values, np.ndarray) and values.dtype.kind in "fc":
            np.testing.assert_allclose(actual[name], values, rtol=2e-11, atol=1e-12, equal_nan=True)
    save_dsf(actual, params)
    loaded = load_spectra(params.out_files_name)
    np.testing.assert_array_equal(loaded["sqw"], actual["sqw"])
