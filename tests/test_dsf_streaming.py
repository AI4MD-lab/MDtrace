"""Full-block FFT parity, bounded reads, prefetch ownership and failure cleanup."""

from contextlib import closing, contextmanager
from multiprocessing import shared_memory
import threading

import numpy as np
import pytest

from mdtrace.dsf import streaming
from mdtrace.dsf.parallel import DSFExecutor
from mdtrace.dsf.spectra import block_spectra
from mdtrace.dsf.dsf import compute_dsf
from tests.test_dsf_modes import _write, _params


@pytest.fixture
def sample(tmp_path):
    rng = np.random.default_rng(5)
    positions = rng.normal(size=(34, 8, 3))
    velocities = rng.normal(size=positions.shape)
    q = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0]])
    path = tmp_path / 'trajectory.nc'
    _write(path, positions, velocities=velocities)
    return path, positions, velocities, q


@pytest.mark.parametrize('workers,prefetch', [(1, False), (1, True), (2, False), (2, True)])
@pytest.mark.parametrize('currents,normalization', [(False, 'density'),
                                                 (True, 'density'),
                                                 (True, 'legacy')])
def test_stream_matches_complete_block(sample, monkeypatch, capsys, workers, prefetch, currents, normalization):
    path, positions, velocities, q = sample
    monkeypatch.setattr(streaming, '_batch_plan', lambda *args: (7, workers, prefetch))
    weights = np.array([1 + 2j, -.3 + .5j] * 4)
    options = dict(dt_ps=.01, normalization=normalization, subtract_mean=True)
    progress = []
    stages = []
    with DSFExecutor(workers) as executor:
        with closing(streaming.stream_blocks(executor, path, 17, 2, 8, q, weights,
                                             currents=currents, prefetch=prefetch,
                                             progress=progress.append,
                                             batch_status=lambda _, label, done, total: stages.append((" ".join(label.split()), done, total)),
                                             **options)) as results:
            for block, actual in enumerate(results):
                frames = slice(block * 17, (block + 1) * 17)
                expected = block_spectra(positions[frames], q, weights, np.zeros(8),
                                        velocities=velocities[frames] if currents else None,
                                        compute_self=False, **options)
                assert actual.keys() == expected.keys()
                for key in expected:
                    np.testing.assert_allclose(actual[key], expected[key], rtol=2e-12, atol=1e-13, equal_nan=True)
    output = capsys.readouterr().out
    np.testing.assert_allclose(progress, [0, 7/17, 14/17, .99, 1, 1 + 7/17, 1 + 14/17, 1.99])
    labels = ["frames 1-7 / 34", "frames 8-14 / 34", "frames 15-17 / 34", "FFT · frames 1-17 / 34",
              "frames 18-24 / 34", "frames 25-31 / 34", "frames 32-34 / 34", "FFT · frames 18-34 / 34"]
    counts = (0, 1, 2, 3, 4) if workers == 1 else (0, 2, 4)
    assert stages == [(label, count, 4) for label in labels for count in counts]
    assert 'Time signals' not in output and 'FFT' not in output and 'ETA' not in output


def test_bounded_reader_owns_all_netcdf_calls(sample, monkeypatch):
    path, _, _, q = sample
    real_reader = streaming.NetCDFReader
    calls = []
    threads = set()

    class Reader(real_reader):
        def __init__(self, path):
            threads.add(threading.get_ident())
            super().__init__(path)

        def read(self, field, frames):
            threads.add(threading.get_ident())
            calls.append((field, frames.start, frames.stop))
            return super().read(field, frames)

        def close(self):
            threads.add(threading.get_ident())
            super().close()

    monkeypatch.setattr(streaming, 'NetCDFReader', Reader)
    monkeypatch.setattr(streaming, '_batch_plan', lambda *args: (7, 2, True))
    with DSFExecutor(2) as executor:
        list(streaming.stream_blocks(executor, path, 17, 2, 8, q, np.ones(8), currents=True, dt_ps=.1))
    assert len(threads) == 1 and threading.get_ident() not in threads
    assert [(a, b) for name, a, b in calls if name == 'positions'] == [(0, 7), (7, 14), (14, 17), (17, 24), (24, 31), (31, 34)]


@pytest.mark.parametrize('failure', ['early_exit', 'read', 'worker', 'worker_phase'])
def test_stream_cleans_reader_and_shared_buffers(sample, monkeypatch, failure):
    path, _, _, q = sample
    monkeypatch.setattr(streaming, '_batch_plan', lambda *args: (7, 2, True))
    original_buffer = streaming._buffer
    original_source = streaming._trajectory_batches
    names, closed = [], []

    @contextmanager
    def track_buffer(*args):
        with original_buffer(*args) as (descriptor, data):
            if descriptor is not None:
                names.append(descriptor[0])
            yield descriptor, data

    def source(*args):
        try:
            with closing(original_source(*args)) as batches:
                for i, arrays in enumerate(batches):
                    if failure == 'read' and i == 2:
                        raise ValueError('synthetic read failure')
                    if failure == 'worker_phase' and i == 2:
                        arrays[0][0, 0, 0] = np.nan
                    yield arrays
        finally:
            closed.append(True)

    monkeypatch.setattr(streaming, '_buffer', track_buffer)
    monkeypatch.setattr(streaming, '_trajectory_batches', source)
    options = dict(dt_ps=.1, normalization='bad' if failure == 'worker' else 'density')
    progress = []
    with DSFExecutor(2) as executor:
        with closing(streaming.stream_blocks(executor, path, 17, 2, 8, q, np.ones(8),
                                             progress=progress.append, **options)) as results:
            if failure == 'early_exit':
                next(results)
            else:
                with pytest.raises(ValueError):
                    list(results)
    assert closed == [True] and len(names) == 2
    assert progress and max(progress) < 1
    for name in names:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)


def test_full_pipeline_output_and_atomic_paths(sample, monkeypatch):
    from mdtrace.dsf.results import save_dsf, load_spectra
    from mdtrace.dsf.plot import plot_dsf
    import mdtrace.dsf.dsf as driver

    path, _, _, q = sample
    monkeypatch.setattr(streaming, '_batch_plan', lambda *args: (7, 2, True))
    params = _params(path, max_cores=2, dsf_currents=True, dsf_qpoints=q.ravel().tolist())
    result = compute_dsf(params)
    save_dsf(result, params)
    loaded = load_spectra(params.out_files_name)
    np.testing.assert_allclose(loaded['longitudinal_current'], result['longitudinal_current'], equal_nan=True)
    plot_dsf(params)
    monkeypatch.setattr(driver, 'stream_blocks', lambda *a, **k: pytest.fail('atom-wise request must keep full-block path'))
    for option in ('dsf_output_components', 'dsf_save_q_resolved', 'dsf_partials'):
        p = _params(path, max_cores=1, dsf_currents=True, dsf_qpoints=q.ravel().tolist(), **{option: True})
        if option == 'dsf_partials':
            # Species metadata is required by this existing output mode.
            _write(path, sample[1], velocities=sample[2], types=np.ones(8, dtype=int))
        assert 'self_sqw' in compute_dsf(p)
    neutron = _params(path, max_cores=1, experiment='neutron', atom_types=['H'])
    assert np.any(compute_dsf(neutron)['incoherent_sqw'] > 0)


def test_stream_respects_atom_alignment(sample, monkeypatch):
    path, positions, velocities, q = sample
    orders = np.tile(np.arange(8), (34, 1))
    orders[7::2] = orders[7::2, ::-1]
    _write(path, np.take_along_axis(positions, orders[:, :, None], axis=1),
           velocities=np.take_along_axis(velocities, orders[:, :, None], axis=1), ids=orders + 1)
    monkeypatch.setattr(streaming, '_batch_plan', lambda *args: (7, 2, True))
    p = _params(path, max_cores=2, dsf_currents=True, dsf_qpoints=q.ravel().tolist())
    actual = compute_dsf(p)
    refs = [block_spectra(positions[i:i + 17], q, np.ones(8), np.zeros(8), dt_ps=.01,
                         velocities=velocities[i:i + 17], compute_self=False) for i in (0, 17)]
    for name in refs[0]:
        np.testing.assert_allclose(actual[name], (refs[0][name] + refs[1][name]) / 2,
                                   rtol=2e-12, atol=1e-13, equal_nan=True)


def test_batch_plan_memory_controls(monkeypatch):
    monkeypatch.setattr(streaming, 'available_cpu_memory', lambda: None)
    assert streaming._batch_plan(8, 1000, 5, True, 4, True)[::2] == (32, False)
    monkeypatch.setattr(streaming, 'available_cpu_memory', lambda: 200 * 1024**2)
    batch, _, _ = streaming._batch_plan(10000, 1000, 5, True, 4, True)
    assert batch < 128
    monkeypatch.setattr(streaming, 'available_cpu_memory', lambda: 1)
    with pytest.raises(MemoryError, match='signals'):
        streaming._batch_plan(8, 1000, 5, True, 4, True)


def test_prefetch_reads_ahead_by_only_one_batch(sample, monkeypatch):
    path, _, _, q = sample
    monkeypatch.setattr(streaming, '_batch_plan', lambda *args: (7, 1, True))
    original_source = streaming._trajectory_batches
    original_compute = streaming.density_current_signals
    next_ready = threading.Event()
    read = []
    computed = []

    def source(*args):
        with closing(original_source(*args)) as batches:
            for i, arrays in enumerate(batches):
                read.append(i)
                if i == 1:
                    next_ready.set()
                yield arrays

    def compute(*args, **kwargs):
        if not computed:
            assert next_ready.wait(5), 'next batch must be read during current computation'
            assert read == [0, 1], 'must not prefetch a third batch'
        computed.append(True)
        return original_compute(*args, **kwargs)

    monkeypatch.setattr(streaming, '_trajectory_batches', source)
    monkeypatch.setattr(streaming, 'density_current_signals', compute)
    with DSFExecutor(1) as executor:
        list(streaming.stream_blocks(executor, path, 17, 2, 8, q, np.ones(8), dt_ps=.1))
    assert len(read) == len(computed) == 6


def test_unavailable_shared_storage_keeps_streamed_serial_path(sample, monkeypatch):
    path, _, _, q = sample
    monkeypatch.setattr(streaming, '_batch_plan', lambda *args: (7, 2, True))
    original = streaming._buffer

    @contextmanager
    def buffer(shape, dtype, shared):
        if shared:
            raise OSError('synthetic shared storage failure')
        with original(shape, dtype, shared) as values:
            yield values

    monkeypatch.setattr(streaming, '_buffer', buffer)
    with DSFExecutor(2) as executor, pytest.warns(RuntimeWarning, match='streamed serial'):
        result = list(streaming.stream_blocks(executor, path, 17, 2, 8, q, np.ones(8), dt_ps=.1))
        assert executor.workers == 1 and executor.pool is None
    assert all(np.isfinite(block['sqw']).all() for block in result)
