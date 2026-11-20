"""Spawn workers must preserve every DSF channel and release shared storage."""

from multiprocessing import shared_memory

import numpy as np
import pytest

from mdtrace.dsf import parallel
from mdtrace.dsf.spectra import block_spectra
from tests.test_dsf_modes import _params, _write
from mdtrace.dsf.dsf import compute_dsf


@pytest.fixture
def inputs():
    rng = np.random.default_rng(73)
    positions = rng.normal(size=(32, 8, 3))
    velocities = rng.normal(size=positions.shape)
    q = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return positions, velocities, q


def test_parallel_all_channels_and_pool_reuse(inputs, monkeypatch, capsys):
    positions, velocities, q = inputs
    # Exercise real spawn workers even on a constrained CI machine (tiny data).
    monkeypatch.setattr(parallel, '_worker_count', lambda *args: 2)
    labels = np.array(['A', 'B'] * 4)
    with parallel.DSFExecutor(2) as executor:
        for mode in ('density', 'currents', 'self', 'neutron', 'partials'):
            coherent = np.where(labels == 'A', 1 + 2j, -3 + .5j) if mode in ('neutron', 'partials') else np.ones(8)
            incoherent = np.ones(8) * .7 if mode in ('neutron', 'partials') else np.zeros(8)
            options = dict(dt_ps=.01, subtract_mean=True,
                           compute_self=mode in ('self', 'neutron'),
                           velocities=None if mode == 'density' else velocities,
                           species_labels=labels if mode == 'partials' else None)
            reference = block_spectra(positions, q, coherent, incoherent, **options)
            progress = []
            result = executor.compute(positions, q, coherent, incoherent, progress=progress.append, **options)
            assert progress[0] == 0 and progress[-1] == 1
            assert progress == sorted(progress)
            assert result.keys() == reference.keys()
            for name in reference:
                np.testing.assert_allclose(result[name], reference[name], rtol=1e-12, atol=1e-13, equal_nan=True)
            if mode == 'density':
                pool = executor.pool
            assert executor.pool is pool
    assert executor.pool is None
    output = capsys.readouterr().out
    assert 'Q vectors' not in output and 'ETA' not in output


def test_parallel_compute_keeps_shell_average_sem_and_text(tmp_path, inputs, monkeypatch):
    from mdtrace.dsf.results import save_dsf, load_spectra

    positions, velocities, _ = inputs
    path = tmp_path / 'input.nc'
    _write(path, positions, velocities=velocities, types=np.array([1, 2] * 4))
    options = dict(dsf_q_mode='spherical', dsf_qpoints=None, dsf_q_min=.5, dsf_q_max=1.5,
                   dsf_q_bin_width=.5, dsf_max_points_per_shell=0, dsf_currents=True,
                   experiment='neutron', atom_types=['H', 'O'], dsf_partials=True,
                   dsf_output_components=True, dsf_save_q_resolved=True)
    reference = compute_dsf(_params(path, max_cores=1, **options))
    monkeypatch.setattr(parallel, '_worker_count', lambda *args: 2)
    params = _params(path, max_cores=2, **options)
    result = compute_dsf(params)
    for name, values in reference.items():
        if isinstance(values, np.ndarray) and values.dtype.kind in 'fc':
            np.testing.assert_allclose(result[name], values, rtol=1e-12, atol=1e-13, equal_nan=True)
    save_dsf(result, params)
    loaded = load_spectra(params.out_files_name)
    np.testing.assert_allclose(loaded['sqw'], reference['sqw'], rtol=1e-12, atol=1e-13)


def test_low_memory_uses_serial_without_allocating_shared(inputs, monkeypatch, capsys):
    positions, _, q = inputs
    monkeypatch.setattr(parallel, 'available_cpu_memory', lambda: 1)
    monkeypatch.setattr(parallel, '_share', lambda _: pytest.fail('must not allocate shared memory'))
    with parallel.DSFExecutor(4) as executor:
        result = executor.compute(positions, q, np.ones(8), np.zeros(8), dt_ps=.1)
        assert executor.workers == 1 and executor.pool is None
        assert np.isfinite(result['sqw']).all()
    assert 'Q vectors' not in capsys.readouterr().out


def test_shared_memory_cleaned_after_worker_error(inputs, monkeypatch):
    positions, _, q = inputs
    monkeypatch.setattr(parallel, '_worker_count', lambda *args: 2)
    original = parallel._share
    names = []
    from contextlib import contextmanager

    @contextmanager
    def record(values):
        with original(values) as descriptor:
            names.append(descriptor[0])
            yield descriptor

    monkeypatch.setattr(parallel, '_share', record)
    with pytest.raises(ValueError, match='normalization'):
        with parallel.DSFExecutor(2) as executor:
            executor.compute(positions, q, np.ones(8), np.zeros(8), dt_ps=.1, normalization='invalid')
    assert names and executor.pool is None
    for name in names:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)


def test_memory_budget_and_mode_estimates(inputs, monkeypatch):
    positions, velocities, q = inputs
    monkeypatch.setattr(parallel, 'available_cpu_memory', lambda: 10000)
    monkeypatch.setattr(parallel.os, 'cpu_count', lambda: 8)
    monkeypatch.setattr(parallel.os, 'sched_getaffinity', lambda _: set(range(8)), raising=False)
    assert parallel._worker_count(4, 5, 1000, 1000, 3000) == 2
    assert parallel._worker_count(4, 1, 1000, 1000, 3000) == 1
    density = parallel._memory_estimate(positions, velocities, len(q), np.zeros(8), dict(compute_self=False))
    incoherent = parallel._memory_estimate(positions, velocities, len(q), np.ones(8), dict(compute_self=False))
    partials = parallel._memory_estimate(positions, velocities, len(q), np.zeros(8), dict(species_labels=['A', 'B'] * 4))
    assert density[2] < incoherent[2] < partials[2]
