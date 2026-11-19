"""Analytic, periodicity, and workflow checks for crystal/amorphous DSF."""

import json
import os
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest

from mdtrace.dsf.dsf import compute_dsf, save_dsf
from mdtrace.dsf.plot import plot_dsf
from mdtrace.dsf.qpoints import build_qpoints, average_q_groups
from mdtrace.dsf.spectra import block_spectra
from mdtrace.parser import read_input
from mdtrace.pipeline import _dsf_ready, run


def _params(path, **options):
    values = dict(
        action="compute", experiment="number", trajectory_path=str(path),
        dsf_qpoints=[1, 0, 0], dsf_q_coordinates="cartesian", num_blocks=2,
        time_step=10.0, output_data_stride=1, out_files_name=str(path.with_suffix("")),
    )
    values.update(options)
    return SimpleNamespace(**values)


def _write(path, positions, *, velocities=None, cell=2 * np.pi * np.eye(3),
           types=None, ids=None, times=None):
    netcdf4 = pytest.importorskip("netCDF4")
    with netcdf4.Dataset(str(path), "w") as dataset:
        for name, size in (("frame", len(positions)), ("atom", positions.shape[1]), ("spatial", 3)):
            dataset.createDimension(name, size)
        fields = [("coordinates", positions, "angstrom", ("frame", "atom", "spatial"))]
        if velocities is not None:
            fields.append(("velocities", velocities, "angstrom/picosecond", ("frame", "atom", "spatial")))
        if cell is not None:
            dims = ("spatial", "spatial") if np.ndim(cell) == 2 else ("frame", "spatial", "spatial")
            fields.append(("cell_vectors", cell, "angstrom", dims))
        if types is not None:
            fields.append(("type", types, None, ("atom",) if np.ndim(types) == 1 else ("frame", "atom")))
        if ids is not None:
            fields.append(("atom_id", ids, None, ("atom",) if np.ndim(ids) == 1 else ("frame", "atom")))
        if times is not None:
            fields.append(("time", times, "picosecond", ("frame",)))
        for name, values, units, dims in fields:
            variable = dataset.createVariable(name, "i8" if name in {"type", "atom_id"} else "f8", dims)
            variable[:] = values
            if units:
                variable.units = units


@pytest.mark.parametrize("window", ["none", "hann"])
@pytest.mark.parametrize("subtract_mean", [False, True])
def test_density_and_current_integrals_obey_parseval(window, subtract_mean):
    rng = np.random.default_rng(50)
    positions = rng.normal(size=(32, 4, 3))
    velocities = rng.normal(size=positions.shape)
    qs = np.array([[0.7, -0.2, 1.1], [-1.0, 0.3, 0.4]])
    b = np.array([-3.0, 1 + 2j, 0.3j, 2.0])
    inc = np.array([3.0, 2.0, 0.0, 1.0])
    dt = 0.02
    spectra = block_spectra(positions, qs, b, inc, dt_ps=dt, velocities=velocities,
                            window=window, subtract_mean=subtract_mean)
    taper = np.ones(32) if window == "none" else np.sin(np.pi * np.arange(32) / 32)**2

    def time_integral(series):
        if subtract_mean:
            series = series - series.mean(axis=0)
        return np.sum(taper.reshape((-1,) + (1,) * (series.ndim - 1))**2 * abs(series)**2,
                      axis=0) / (4 * (taper @ taper))

    for index, q in enumerate(qs):
        phase = np.exp(1j * (positions @ q))
        current = np.sum(velocities * phase[:, :, None], axis=1)
        direction = q / np.linalg.norm(q)
        longitudinal = current @ direction
        transverse = current - longitudinal[:, None] * direction
        expected = {
            "coherent_sqw": time_integral(phase @ b),
            "incoherent_sqw": time_integral(phase) @ inc,
            "self_sqw": time_integral(phase).sum(),
            "longitudinal_current": time_integral(longitudinal),
            "transverse_current": time_integral(transverse).sum(),
        }
        for name, reference in expected.items():
            # Frequency-bin integration of the independently calculated time
            # averages, including window energy and complex neutron amplitudes.
            np.testing.assert_allclose(spectra[name][index].sum() / (32 * dt), reference, rtol=1e-13)
        np.testing.assert_allclose(
            (spectra["longitudinal_current"][index] + spectra["transverse_current"][index]).sum() / (32 * dt),
            time_integral(current).sum(), rtol=1e-13,
        )


def test_traveling_particle_continuity_and_transverse_sidebands():
    frames, dt = 128, 0.02
    time = np.arange(frames) * dt
    fundamental = 1 / (frames * dt)
    velocity = 2 * np.pi * 4 * fundamental
    omega = 2 * np.pi * 2 * fundamental
    positions = np.zeros((frames, 1, 3))
    velocities = np.zeros_like(positions)
    positions[:, 0, 0], velocities[:, 0, 0] = velocity * time, velocity
    positions[:, 0, 1] = 0.1 * np.sin(omega * time)
    velocities[:, 0, 1] = 0.1 * omega * np.cos(omega * time)
    result = block_spectra(positions, np.array([[1., 0, 0]]), np.ones(1), np.zeros(1),
                           dt_ps=dt, velocities=velocities)
    frequency = np.fft.fftshift(np.fft.fftfreq(frames, dt))
    assert frequency[result["sqw"][0].argmax()] == pytest.approx(4 * fundamental)
    peaks = np.argsort(result["transverse_current"][0])[-2:]
    np.testing.assert_allclose(np.sort(frequency[peaks]), [2 * fundamental, 6 * fundamental])
    # Exact periodic trajectory: omega^2 S(Q,omega) = Q^2 C_L(Q,omega).
    np.testing.assert_allclose((2 * np.pi * frequency)**2 * result["sqw"][0],
                               result["longitudinal_current"][0], atol=1e-24, rtol=1e-12)


def test_transverse_oscillation_is_visible_in_current_but_not_density():
    frames, dt = 64, 0.01
    omega = 2 * np.pi * 5 / (frames * dt)
    time = np.arange(frames) * dt
    positions = np.zeros((frames, 1, 3))
    velocities = np.zeros_like(positions)
    positions[:, 0, 1] = np.sin(omega * time)
    velocities[:, 0, 1] = omega * np.cos(omega * time)
    result = block_spectra(positions, np.array([[1., 0, 0], [0, 0, 0]]),
                           np.ones(1), np.zeros(1), dt_ps=dt, velocities=velocities)
    assert np.count_nonzero(result["sqw"][0] > 1e-10) == 1
    np.testing.assert_array_equal(result["longitudinal_current"][0], 0)
    peaks = np.argsort(result["transverse_current"][0])[-2:]
    np.testing.assert_array_equal(np.sort(peaks), [frames // 2 - 5, frames // 2 + 5])
    assert np.isnan(result["transverse_current"][1]).all()
    assert np.isnan(result["longitudinal_current"][1]).all()


@pytest.mark.parametrize("cell", [2 * np.pi * np.eye(3),
                                 2 * np.pi * np.array([[1, 0, 0], [0.4, 1.3, 0], [-0.2, 0.1, 0.9]])])
def test_spherical_sampling_matches_brute_force_reciprocal_lattice(cell):
    params = SimpleNamespace(dsf_q_mode="spherical", dsf_q_min=0.3, dsf_q_max=2.2,
                             dsf_q_bin_width=0.4, dsf_max_points_per_shell=0)
    sample = build_qpoints(params, cell)
    grid = np.array(list(product(range(-4, 5), repeat=3)))
    q = 2 * np.pi * grid @ np.linalg.inv(cell).T
    norms = np.linalg.norm(q, axis=1)
    expected = grid[(norms >= 0.3 - 1e-10) & (norms <= 2.2 + 1e-10)]
    assert {tuple(row) for row in np.rint(sample["q_box"]).astype(int)} == {tuple(row) for row in expected}
    np.testing.assert_array_equal(sample["q_counts"], sample["q_available_counts"])
    # Selecting different periodic images of each atom leaves every phase intact.
    positions = np.random.default_rng(6).normal(size=(3, 4, 3))
    shifts = np.random.default_rng(7).integers(-3, 4, size=positions.shape) @ cell
    np.testing.assert_allclose(np.exp(1j * (positions @ sample["q_cart"].T)),
                               np.exp(1j * ((positions + shifts) @ sample["q_cart"].T)), atol=1e-13)


def test_shell_subsampling_is_reproducible_and_keeps_opposite_directions():
    params = SimpleNamespace(dsf_q_mode="spherical", dsf_q_max=4.2, dsf_q_bin_width=1.,
                             dsf_max_points_per_shell=6, dsf_seed=40)
    first = build_qpoints(params, 2 * np.pi * np.eye(3))
    second = build_qpoints(params, 2 * np.pi * np.eye(3))
    np.testing.assert_array_equal(first["q_cart"], second["q_cart"])
    assert np.all(first["q_counts"] <= 6)
    assert np.any(first["q_available_counts"] > 6)
    vectors = {tuple(row) for row in first["q_cart"]}
    assert vectors == {tuple(-row) for row in first["q_cart"]}
    params.dsf_seed = 41
    assert not np.array_equal(first["q_cart"], build_qpoints(params, 2 * np.pi * np.eye(3))["q_cart"])


def test_crystal_path_preserves_extended_zone_and_nonorthogonal_cell():
    primitive = np.array([[1, 0.2, 0], [0.1, 1.2, 0], [0, 0.1, 1.5]])
    cell = np.array([[4, 2, 0], [0, 4, 0], [0, 0, 2]]) @ primitive
    params = SimpleNamespace(dsf_q_mode="path", prim_unitcell=primitive,
                             dsf_q_path=[0, 0, 0, 1, 0, 0, 1.5, 0, 0],
                             dsf_q_path_labels=["G", "G+1", "X+1"])
    sample = build_qpoints(params, cell)
    np.testing.assert_allclose(sample["q_frac"][:, 0], np.arange(7) / 4, atol=1e-14)
    np.testing.assert_allclose(sample["q_box"], np.rint(sample["q_box"]), atol=1e-13)
    assert np.all(np.diff(sample["q_axis"]) > 0)
    assert sample["q_path_labels"] == params.dsf_q_path_labels


def test_shell_average_is_of_spectra_not_amplitudes():
    params = SimpleNamespace(dsf_q_mode="spherical", dsf_q_max=1.1, dsf_q_bin_width=0.2)
    sample = build_qpoints(params, np.diag([2 * np.pi, np.pi, np.pi]))
    assert len(sample["q_cart"]) == 2
    positions = np.tile([[[np.pi / 2, 0, 0]]], (8, 1, 1))
    spectra = block_spectra(positions, sample["q_cart"], np.ones(1), np.zeros(1), dt_ps=0.1)
    averaged = average_q_groups(spectra["sqw"], sample)
    assert averaged.shape == (1, 8)
    assert averaged[0, 4] == pytest.approx(0.8)
    # The amplitudes at +/-Q cancel, but their intensities must not cancel.
    assert abs(np.exp(1j * (sample["q_cart"] @ positions[0, 0])).mean()) < 1e-14


def test_current_spectra_follow_atom_ids_and_do_not_use_neutron_weights(tmp_path):
    rng = np.random.default_rng(12)
    positions, velocities = rng.normal(size=(2, 12, 3, 3))
    order = np.array([np.arange(3)] + [rng.permutation(3) for _ in range(11)])
    ids, types = np.array([71, 9, 23]), np.array([4, 7, 4])
    reference, shuffled = tmp_path / "reference.nc", tmp_path / "shuffled.nc"
    _write(reference, positions, velocities=velocities, types=types, ids=ids)
    _write(shuffled, np.take_along_axis(positions, order[:, :, None], axis=1),
           velocities=np.take_along_axis(velocities, order[:, :, None], axis=1),
           types=types[order], ids=ids[order])
    normal = compute_dsf(_params(reference, dsf_currents=True))
    neutron = compute_dsf(_params(shuffled, dsf_currents=True, experiment="neutron", atom_types=["3He", "H"]))
    for name in ("longitudinal_current", "transverse_current", "self_sqw"):
        np.testing.assert_allclose(neutron[name], normal[name], rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(neutron[name + "_sem"], normal[name + "_sem"], rtol=1e-13, atol=1e-14)
    assert not np.allclose(neutron["sqw"], normal["sqw"])
    # Reader m/s -> A/ps conversion must occur exactly once.
    expected = block_spectra(positions[:6], np.array([[1., 0, 0]]), np.ones(3), np.zeros(3),
                             velocities=velocities[:6], dt_ps=.01)
    expected2 = block_spectra(positions[6:], np.array([[1., 0, 0]]), np.ones(3), np.zeros(3),
                              velocities=velocities[6:], dt_ps=.01)
    np.testing.assert_allclose(normal["longitudinal_current"],
                               (expected["longitudinal_current"] + expected2["longitudinal_current"]) / 2)


@pytest.mark.parametrize("problem,match", [
    ("no_cell", "simulation cell"), ("changing_cell", "fixed simulation cell"),
    ("wrong_cell", "does not match"), ("bad_q", "not commensurate"),
    ("no_velocities", "velocities"), ("bad_velocity", "velocities must be finite"),
    ("bad_time", "uniformly spaced"),
])
def test_physical_input_requirements_are_enforced(tmp_path, problem, match):
    path = tmp_path / "invalid.nc"
    positions = np.zeros((8, 2, 3))
    cell = 2 * np.pi * np.eye(3)
    options, data = {}, {}
    if problem == "no_cell":
        cell = None
    elif problem == "changing_cell":
        cell = np.tile(cell, (8, 1, 1))
        cell[4, 0, 0] += 0.01
    elif problem == "wrong_cell":
        options["dsf_cell"] = np.eye(3)
    elif problem == "bad_q":
        options["dsf_qpoints"] = [0.5, 0, 0]
    elif problem in {"no_velocities", "bad_velocity"}:
        options["dsf_currents"] = True
        if problem == "bad_velocity":
            data["velocities"] = np.full_like(positions, np.nan)
    elif problem == "bad_time":
        data["times"] = np.arange(8) * .02
    _write(path, positions, cell=cell, **data)
    with pytest.raises(ValueError, match=match):
        compute_dsf(_params(path, **options))


def test_number_mode_needs_neither_primitive_nor_scattering_table_and_saves_units(tmp_path):
    path = tmp_path / "amorphous.nc"
    positions = np.random.default_rng(9).normal(size=(16, 7, 3))
    _write(path, positions, cell=None)
    params = _params(path, dsf_qpoints=None, dsf_q_mode="spherical", dsf_q_max=1.5,
                     dsf_cell=2 * np.pi * np.eye(3))
    result = compute_dsf(params)
    np.testing.assert_array_equal(result["incoherent_sqw"], 0)
    assert result["q_frac"].shape == (0, 3)
    assert np.any(result["self_sqw"] > 0)
    assert result["metadata"]["intensity_unit"] == "1/THz"
    save_dsf(result, params)
    assert _dsf_ready(params)
    with np.load(params.out_files_name + ".dsf.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["sqw"], result["sqw"])
        assert json.loads(str(saved["metadata_json"]))["current_weighting"] == "number"
    assert "per THz" in Path(params.out_files_name + ".dsf_meV").read_text()
    assert plot_dsf(params).stat().st_size > 10000


def test_gpumd_thinking_pipeline_computes_current_spectra_and_plot(tmp_path):
    source = tmp_path / "dump.xyz"
    lines = []
    cell = " ".join(f"{x:.17g}" for x in (2 * np.pi * np.eye(3)).ravel())
    for frame in range(16):
        lines.extend(["2", f'Lattice="{cell}" Properties=species:S:1:pos:R:3:vel:R:3'])
        # GPUMD text velocities are A/fs: these become 1 and 2 A/ps.
        lines.extend([f"Si {frame * .01:.17g} 0 0 0.001 0 0", "Ge 0 1 0 0 0.002 0"])
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    control = tmp_path / "input.in"
    prefix = tmp_path / "result"
    control.write_text(
        f'method = dsf\naction = thinking\nexperiment = number\ntrajectory_file = dump.xyz\n'
        f'out_files_name = "{prefix}"\ntime_step = 1\noutput_data_stride = 10\nnum_blocks = 2\n'
        'dsf_q_mode = spherical\ndsf_q_max = 1.1\ndsf_currents = 1\n', encoding="utf-8",
    )
    params = read_input(control)
    run(params)
    assert _dsf_ready(params, check_settings=True)
    assert Path(str(prefix) + ".dsf.png").stat().st_size > 10000
    old_time = Path(str(prefix) + ".dsf.npz").stat().st_mtime_ns
    run(params)
    assert Path(str(prefix) + ".dsf.npz").stat().st_mtime_ns == old_time
    # A changed numerical setting must not silently reuse a previous spectrum.
    params.dsf_subtract_mean = True
    assert not _dsf_ready(params, check_settings=True)
    with np.load(str(prefix) + ".dsf.npz", allow_pickle=False) as saved:
        assert np.isfinite(saved["longitudinal_current"]).all()
        assert np.isfinite(saved["transverse_current"]).all()
        # Average total number-current squared: only self terms survive after
        # averaging +/-x,+/-y,+/-z for the orthogonal velocity directions.
        integral = (saved["longitudinal_current"] + saved["transverse_current"]).sum(axis=1) / .08
        np.testing.assert_allclose(integral, (1**2 + 2**2) / 2, rtol=2e-6)


def test_crystal_cli_resolves_known_longitudinal_and_transverse_modes(tmp_path):
    lattice = np.array(list(product(range(4), repeat=3)), dtype=float)
    frames, dt = 128, .01
    time = np.arange(2 * frames) * dt
    f_long, f_trans = 3 / (frames * dt), 1 / (frames * dt)
    spatial = np.cos((np.pi / 2) * lattice[:, 0])
    positions = np.tile(lattice, (2 * frames, 1, 1))
    velocities = np.zeros_like(positions)
    for axis, frequency in ((0, f_long), (1, f_trans)):
        positions[:, :, axis] += .01 * np.sin(2 * np.pi * frequency * time)[:, None] * spatial
        velocities[:, :, axis] = .01 * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * time)[:, None] * spatial
    path = tmp_path / "crystal.nc"
    _write(path, positions, velocities=velocities, cell=4 * np.eye(3), times=time)
    control, prefix = tmp_path / "crystal.in", tmp_path / "crystal"
    control.write_text(
        f'method = dsf\naction = thinking\nexperiment = number\ntrajectory_file = crystal.nc\n'
        f'out_files_name = "{prefix}"\ntime_step = 1\noutput_data_stride = 10\nnum_blocks = 2\n'
        'prim_unitcell = 1 0 0 0 1 0 0 0 1\ndsf_q_mode = path\n'
        'dsf_q_path = 0 0 0 1/2 0 0\ndsf_q_path_labels = G X\ndsf_currents = 1\n'
        'dsf_subtract_mean = 1\ndsf_plot_frequency_max = 5\n', encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "mdtrace.main", str(control)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        encoding="utf-8", env={**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    with np.load(str(prefix) + ".dsf.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(saved["q_frac"][:, 0], [0, .25, .5])
        frequencies = saved["freq_thz"]
        positive = frequencies > 0
        for name, expected in (("sqw", f_long), ("longitudinal_current", f_long),
                               ("transverse_current", f_trans)):
            peak = frequencies[positive][saved[name][1, positive].argmax()]
            assert peak == pytest.approx(expected)
            np.testing.assert_allclose(saved[name + "_sem"][1], 0, atol=1e-14)
        assert np.isnan(saved["longitudinal_current"][0]).all()
        assert np.isnan(saved["transverse_current_sem"][0]).all()
    assert Path(str(prefix) + ".dsf.png").stat().st_size > 10000
    # Plot-only use must not require the production trajectory or compute input.
    path.unlink()
    plot_input = tmp_path / "plot.in"
    plot_input.write_text(f'method = dsf\naction = plot\nout_files_name = "{prefix}"\n'
                          'dsf_plot_frequency_max = 5\n', encoding="utf-8")
    run(read_input(plot_input))


@pytest.mark.parametrize("filename", ["crystal.in", "amorphous.in"])
def test_distributed_templates_parse_without_external_structure_files(filename):
    path = Path(__file__).resolve().parents[1] / "examples" / "dsf" / filename
    params = read_input(path)
    assert params.method == "dsf" and params.dsf_currents
    assert params.experiment == "number"


def test_compute_api_validates_options_even_with_plot_action(tmp_path):
    params = _params(tmp_path / "unused.nc", action="plot", backend="cupy")
    with pytest.raises(ValueError, match="backend = numpy only"):
        compute_dsf(params)


def test_direct_use_of_outdated_text_cache_requires_regeneration(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    path = tmp_path / "old.xyz.mdtrace.nc"
    _write(path, np.zeros((4, 1, 3)))
    with netcdf4.Dataset(str(path), "a") as dataset:
        dataset.mdtrace_schema_version = 2
    with pytest.raises(ValueError, match="original text trajectory"):
        compute_dsf(_params(path))


@pytest.mark.parametrize("settings,match", [
    ("dsf_q_mode = spherical\ndsf_q_max = 0", "dsf_q_min"),
    ("dsf_q_mode = spherical\ndsf_q_max = 2\ndsf_max_points_per_shell = 3", "positive even"),
    ("dsf_q_mode = path\ndsf_q_path = 0 0 0 0.5 0 0", "prim_unitcell"),
    ("dsf_q_coordinates = cartesian\ndsf_qpoints = 1 0", "groups of 3"),
    ("dsf_q_coordinates = cartesian\ndsf_qpoints = 1 0 0\ndsf_normalization = wrong", "dsf_normalization"),
])
def test_new_parameters_reject_ambiguous_or_invalid_input(tmp_path, settings, match):
    control = tmp_path / "bad.in"
    control.write_text("method = dsf\naction = compute\nexperiment = number\ntime_step = 1\n"
                       "output_data_stride = 1\n" + settings + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        read_input(control)
