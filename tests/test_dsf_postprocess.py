"""Conservation, covariance and workflow checks for reusable DSF spectra."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mdtrace.dsf import compute_dsf, save_dsf
from mdtrace.dsf.spectra import mean_and_sem as _mean_and_sem, PLANCK_MEV_PS
from mdtrace.dsf.partials import combine_partials, species_layout
from mdtrace.dsf.plot import plot_dsf
from mdtrace.dsf.postprocess import (
    average_shells, apply_resolution, load_spectra, main, reweight_scattering,
    to_energy_density,
)
from mdtrace.dsf.qpoints import sampling_diagnostics
from mdtrace.dsf.spectra import block_spectra
from tests.test_dsf_modes import _params, _write


def _sample(tmp_path, **options):
    rng = np.random.default_rng(26)
    positions = rng.normal(size=(64, 4, 3))
    velocities = rng.normal(size=positions.shape)
    path = tmp_path / "mixed.nc"
    _write(path, positions, velocities=velocities, types=np.array([4, 7, 4, 7]))
    settings = dict(dsf_qpoints=[1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0],
                    dsf_partials=True, dsf_currents=True, dsf_output_components=True, dsf_output_sem=True, atom_types=["3He", "H"])
    settings.update(options)
    params = _params(path, **settings)
    return compute_dsf(params), params


@pytest.mark.parametrize("per_mev", [False, True])
@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("experiment", ["number", "neutron"])
def test_text_columns_follow_q_mapping_and_rows_follow_frequency(tmp_path, per_mev, version, experiment):
    from mdtrace.dsf.results import spectra_ready

    result, params = _sample(tmp_path, experiment=experiment)
    if per_mev:
        result = to_energy_density(result)
    save_dsf(result, params)
    prefix = params.out_files_name
    qfile = Path(prefix + ".dsf_Qaxis")
    lines = qfile.read_text(encoding="utf-8").splitlines()
    header = lines[0]
    assert header.startswith("# " + result["q_axis_label"])
    used_end = header.index("Used") + len("Used")
    rows = [line for line in lines if not line.startswith("#")]
    for row, q, used, available in zip(rows, result["q_axis"], result["q_counts"],
                                       result["q_available_counts"], strict=True):
        assert row.startswith(f"  {q:.18e} ")
        assert row[:used_end].endswith(f" {used}")
        assert len(row) == len(header) and row.endswith(f" {available}")
    assert not lines[1].startswith("#")  # Q data immediately follows the short header.
    assert all("q_cart" not in line for line in lines)
    total_file = Path(prefix + ".dsf_Ctotal")
    expected_total = result["longitudinal_current"] + result["transverse_current"]
    assert total_file.read_text(encoding="utf-8").startswith(
        f"# Ctotal(Q,{'E' if per_mev else 'f'}) [{result['metadata']['current_unit']}]\n# Ctotal = CL + CT\n")
    np.testing.assert_array_equal(np.loadtxt(total_file, ndmin=2).T, expected_total)
    if version < 3:
        total_file.unlink()  # Both historical layouts predate the total-current file.
        # Rebuild both historical layouts, including their original blocks file.
        blockfile = Path(prefix + ".dsf_blocks")
        detail, blocks = blockfile.read_text(encoding="utf-8").split("\n", 1)
        blockfile.write_text(blocks, encoding="utf-8")
        record = json.loads(detail.removeprefix("# "))
        record.update(text_format_version=version, blocks=True, spectra={
            name + extra: suffix + extra for name, suffix in (
                ("sqw", ".dsf"), ("coherent_sqw", ".dsf_coherent"), ("incoherent_sqw", ".dsf_incoherent"),
                ("self_sqw", ".dsf_self"), ("longitudinal_current", ".dsf_CL"), ("transverse_current", ".dsf_CT"))
            for extra in ("", "_sem")})
        record["metadata"]["current_unit"] = record["metadata"]["current_unit"].replace("Å", "Angstrom")
        legacy = [header, *rows]
        legacy.insert(0 if version == 1 else len(legacy), "# " + json.dumps(record))
        qfile.write_text("\n".join(legacy) + "\n", encoding="utf-8")
    assert spectra_ready(params, check_settings=True) == (not per_mev)
    frequencies = np.loadtxt(prefix + ".dsf_THz")
    energies = load_spectra(prefix)["energy_mev"]
    qmap = list(enumerate(result["q_group"]))
    saved_text = load_spectra(prefix)
    np.testing.assert_array_equal(saved_text["total_current"], expected_total)
    assert saved_text["metadata"] == result["metadata"]
    for name, values in result.items():
        if name.startswith("q_") and isinstance(values, np.ndarray):
            np.testing.assert_array_equal(saved_text[name], values)
    np.testing.assert_allclose(energies, PLANCK_MEV_PS * frequencies)
    for suffix, name in ((".dsf", "sqw"), (".dsf_coherent", "coherent_sqw"),
                         (".dsf_incoherent", "incoherent_sqw"), (".dsf_self", "self_sqw"),
                         (".dsf_CL", "longitudinal_current"), (".dsf_CT", "transverse_current")):
        for extra in ("", "_sem"):
            table = np.loadtxt(prefix + suffix + extra, ndmin=2)
            assert table.shape == (len(frequencies), len(result["q_axis"]))
            for _, column in qmap:
                np.testing.assert_allclose(table[:, column], result[name + extra][column])
            np.testing.assert_array_equal(saved_text[name + extra], result[name + extra])
    assert not Path(prefix + ".dsf.npz").exists()
    for name in ("qblocks__pair_density", "qblocks__species_self"):
        np.testing.assert_array_equal(saved_text[name], result[name])
    Path(params.trajectory_path).unlink()  # This test's small synthetic trajectory only.
    assert plot_dsf(SimpleNamespace(out_files_name=prefix)).stat().st_size > 10000


def test_one_block_has_undefined_sem_and_two_blocks_keep_covariance():
    _, sem = _mean_and_sem([np.array([1., 5.])])
    assert np.isnan(sem).all()
    _, sem = _mean_and_sem([np.array([1., 5.]), np.array([3., 1.])])
    np.testing.assert_allclose(sem, [1., 2.])


def test_direction_diagnostics_detect_elongated_box_and_are_rotation_invariant():
    from mdtrace.dsf.qpoints import build_qpoints

    sample = build_qpoints(SimpleNamespace(dsf_q_mode="spherical", dsf_q_min=.07,
                           dsf_q_max=.3, dsf_q_bin_width=.1), np.diag([80., 20., 20.]))
    np.testing.assert_array_equal(sample["q_direction_counts"], 1)
    np.testing.assert_allclose(sample["q_direction_eigenvalues"], [[0, 0, 1]] * 3)
    assert sample["q_coverage_poor"].all()
    rotation, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
    rotated = sampling_diagnostics(sample["q_cart"] @ rotation, sample["q_group"], 3)
    for name in ("q_magnitude_std", "q_direction_eigenvalues", "q_direction_counts"):
        np.testing.assert_allclose(sample[name], rotated[name], atol=1e-14)


def test_complex_partials_reconstruct_direct_totals_and_preserve_phase():
    rng = np.random.default_rng(123)
    positions, velocities = rng.normal(size=(2, 48, 5, 3))
    labels = np.array(["A", "B", "A", "B", "B"])
    weights = np.where(labels == "A", 2+3j, -1+.7j)
    incoherent = np.where(labels == "A", 4., 2.)
    q = np.array([[.8, -.3, .4], [-.8, .3, -.4], [0, 0, 0]])
    args = dict(dt_ps=.01, velocities=velocities, subtract_mean=True)
    direct = block_spectra(positions, q, weights, incoherent, **args)
    partial = block_spectra(positions, q, weights, incoherent, species_labels=labels, **args)
    for name in direct:
        np.testing.assert_allclose(partial[name], direct[name], atol=1e-13)
    for name in ("coherent_sqw", "incoherent_sqw", "longitudinal_current", "transverse_current"):
        np.testing.assert_allclose(partial["partial_" + name].sum(axis=1), direct[name], atol=1e-13)
    assert np.max(abs(partial["pair_density"].imag)) > 1e-5
    assert np.any(partial["partial_coherent_sqw"] < 0)
    assert not np.allclose(partial["coherent_sqw"][0], partial["coherent_sqw"][1, ::-1])
    _, _, pairs = species_layout(labels)
    lost_phase = {**partial, "pair_density": partial["pair_density"].real}
    wrong = combine_partials(lost_phase, pairs, [2+3j, -1+.7j], [4, 2])
    assert not np.allclose(wrong["coherent_sqw"], direct["coherent_sqw"])


@pytest.mark.parametrize("experiment", ["number", "neutron"])
def test_plot_reads_only_text_spectra_and_matches_in_memory_result(tmp_path, monkeypatch, experiment):
    from mdtrace.dsf import plot

    result, params = _sample(tmp_path, experiment=experiment)
    save_dsf(result, params)
    # Reference plot uses the original in-memory arrays, independently of I/O.
    with monkeypatch.context() as patch:
        patch.setattr(plot, "load_spectra", lambda *a, **k: result)
        expected_image = plot_dsf(params).read_bytes()
    # Plot must not need the potentially large optional block file.
    Path(params.out_files_name + ".dsf_blocks").unlink()
    assert plot_dsf(params).read_bytes() == expected_image
    from mdtrace.pipeline import step_plot_dsf
    step_plot_dsf(params)


@pytest.mark.parametrize("density", [False, True])
@pytest.mark.parametrize("currents", [False, True])
def test_plot_switches_only_require_selected_spectra(tmp_path, density, currents):
    from mdtrace.pipeline import step_plot_dsf

    result, params = _sample(tmp_path, dsf_partials=False)
    save_dsf(result, params)
    params.dsf_plot_density = density
    params.dsf_plot_currents = currents
    if not density:
        Path(params.out_files_name + ".dsf").unlink()
    if not currents:
        Path(params.out_files_name + ".dsf_CL").unlink()
    step_plot_dsf(params)
    expected = {".dsf.png"} if density else set()
    if currents:
        expected.update((".dsf_Ctotal.png", ".dsf_CL.png", ".dsf_CT.png"))
    prefix = Path(params.out_files_name).name
    assert {p.name for p in tmp_path.glob("*.png")} == {prefix + suffix for suffix in expected}


@pytest.mark.parametrize("scale", ["log", "linear"])
def test_spherical_plot_keeps_empty_shells_blank_and_uses_automatic_colorbar_ticks(tmp_path, monkeypatch, scale):
    from matplotlib.figure import Figure
    from matplotlib.ticker import AutoLocator
    from mdtrace.dsf import plot

    params = SimpleNamespace(out_files_name=str(tmp_path / "sample"), dsf_plot_scale=scale,
                             dsf_plot_frequency_max=2.)
    values = np.array([[1e-18, 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    data = dict(sqw=np.column_stack((values, [1e9, 0., np.nan])), freq_thz=np.arange(4.),
                q_shell_intervals=np.array([[1, 2], [2, 3], [4, 5]]),
                metadata=dict(q_mode="spherical", shell_grid=[1, 5, 1], experiment="number", currents=True,
                              intensity_unit="1/THz", current_unit="(Å/ps)^2/THz"))
    data["longitudinal_current"] = 2 * data["sqw"]
    data["transverse_current"] = 3 * data["sqw"]
    factors = {"Density": 1, "Longitudinal current": 2, "Transverse current": 3, "Total current": 5}
    monkeypatch.setattr(plot, "load_spectra", lambda *args, **kwargs: data)
    def check_figure(fig, *args, **kwargs):
        ax, bar = fig.axes
        factor = factors[ax.get_title(loc="center")]
        assert ax.get_xlim() == (0, 5)
        assert ax.get_xticks()[0] == 0
        assert [tuple(image.get_extent()[:2]) for image in ax.images] == [(1, 3), (4, 5)]
        image = ax.images[0]
        assert image.get_interpolation() == "hanning"
        own_values = factor * values
        np.testing.assert_allclose(image.get_array(), (np.log(own_values) if scale == "log" else own_values)[:2].T)
        expected = [own_values.min(), own_values.max()]
        if scale == "log" and factor != 1:
            expected = [2 * values.min(), 5 * values.max()]
        np.testing.assert_allclose(image.get_clim(), np.log(expected) if scale == "log" else expected, atol=0)
        assert isinstance(bar.yaxis.get_major_locator(), AutoLocator)
        assert (r"\ln" in bar.get_ylabel()) == (scale == "log")

    monkeypatch.setattr(Figure, "savefig", check_figure)
    plot_dsf(params)


def test_reweight_rebin_roundtrip_and_sem_after_averaging(tmp_path):
    result, params = _sample(tmp_path)
    save_dsf(result, params)
    loaded = load_spectra(params.out_files_name)
    for name, value in result.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(loaded[name], value, err_msg=name)
    assert loaded["qblocks__pair_density"].shape == (2, 4, 3, 32)
    weighted = reweight_scattering(loaded)
    params.experiment = "neutron"
    direct = compute_dsf(params)
    for name in direct["metadata"]["spectrum_names"]:
        np.testing.assert_allclose(weighted[name], direct[name], atol=1e-12)
        np.testing.assert_allclose(weighted[name + "_sem"], direct[name + "_sem"], atol=1e-12)
    weights = [1., 2., 3., 4.]
    with pytest.warns(RuntimeWarning, match="coverage"):
        averaged = average_shells(weighted, [0, .8, 1.1, 2.], weights=weights)
    block_average = np.average(weighted["qblocks__sqw"], axis=1, weights=weights)
    np.testing.assert_allclose(averaged["sqw"][0], block_average.mean(axis=0))
    np.testing.assert_allclose(averaged["sqw_sem"][0], abs(np.diff(block_average, axis=0)[0]) / 2)
    assert not np.allclose(averaged["sqw_sem"][0], np.average(weighted["sqw_sem"], axis=0, weights=weights))
    np.testing.assert_array_equal(averaged["q_shell_intervals"], [[.8, 1.1]])
    assert averaged["metadata"]["q_population_scope"].startswith("retained")
    assert averaged["metadata"]["q_available_total"] == len(averaged["q_cart"])
    assert averaged["metadata"]["q_available_shells"] == len(averaged["q_counts"])
    with pytest.warns(RuntimeWarning):
        number_first = average_shells(loaded, [0, 1.1], weights=weights)
    reordered = reweight_scattering(number_first)
    np.testing.assert_allclose(reordered["sqw"], averaged["sqw"])
    np.testing.assert_allclose(reordered["sqw_sem"], averaged["sqw_sem"])
    np.testing.assert_allclose(reweight_scattering(weighted, experiment="number")["sqw"], loaded["sqw"])
    equal = reweight_scattering(weighted, density_weighting="equal")
    np.testing.assert_allclose(equal["sqw"], loaded["sqw"])
    assert equal["metadata"]["density_weighting"] == "equal"
    with pytest.raises(ValueError, match="not both"):
        reweight_scattering(weighted, density_weighting="equal", experiment="number")
    assert loaded["metadata"]["experiment"] == "number"  # caller's saved_text remains intact


def test_custom_weights_require_complete_mapping_and_keep_number_currents(tmp_path):
    result, _ = _sample(tmp_path)
    with pytest.raises(ValueError, match="exactly"):
        reweight_scattering(result, coherent={"3He": 1}, incoherent={"3He": 0})
    weighted = reweight_scattering(result, coherent={"3He": 1+2j, "H": 2-1j},
                                   incoherent={"3He": .3j, "H": .7})
    assert not np.allclose(weighted["sqw"], result["sqw"])
    np.testing.assert_allclose(weighted["longitudinal_current"], result["longitudinal_current"])
    with pytest.raises(ValueError, match="finite"):
        reweight_scattering(result, coherent={"3He": np.nan, "H": 1},
                            incoherent={"3He": 0, "H": 0})


def test_resolution_preserves_central_area_and_propagates_block_covariance(tmp_path):
    result, _ = _sample(tmp_path, dsf_qpoints=[0, 0, 0, 1, 0, 0], time_step=100., num_blocks=2)
    # Replace one spectrum with independently specified central impulses.
    blocks = np.zeros_like(result["qblocks__sqw"])
    blocks[0, :, 15] = 1
    blocks[1, :, 17] = 2
    result["qblocks__sqw"] = blocks
    fwhm = 2 * np.diff(result["energy_mev"])[0]
    blurred = apply_resolution(result, fwhm)
    np.testing.assert_allclose(blurred["qblocks__sqw"].sum(axis=-1), blocks.sum(axis=-1), atol=1e-12)
    np.testing.assert_allclose(blurred["sqw_sem"], abs(np.diff(blurred["qblocks__sqw"], axis=0)[0]) / 2)
    assert np.isnan(blurred["longitudinal_current"][0]).all()
    assert np.isfinite(blurred["longitudinal_current"][1]).all()
    energy = to_energy_density(blurred)
    df = np.diff(result["freq_thz"])[0]
    de = np.diff(result["energy_mev"])[0]
    np.testing.assert_allclose(energy["sqw"].sum(-1) * de, blurred["sqw"].sum(-1) * df)
    assert energy["metadata"]["intensity_unit"] == "1/meV"
    assert to_energy_density(energy) is energy
    np.testing.assert_allclose(energy["sqw_sem"], blurred["sqw_sem"] / PLANCK_MEV_PS)
    edge = deepcopy(result)
    edge["qblocks__sqw"][:] = 0
    edge["qblocks__sqw"][..., 0] = 1
    out = apply_resolution(edge, fwhm)
    assert np.max(abs(out["qblocks__sqw"][..., -1])) < 1e-12
    assert np.all(out["qblocks__sqw"].sum(-1) < 1)  # truncated tail, not circular wrapping


@pytest.mark.parametrize("weight_option", ["--density-weighting", "--experiment"])
def test_postprocessing_cli_needs_no_trajectory_and_plots_qens_energy_window(tmp_path, weight_option):
    result, params = _sample(tmp_path, time_step=100.)
    save_dsf(result, params)
    Path(params.trajectory_path).unlink()
    prefix = str(tmp_path / "neutron_qens")
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "mdtrace.dsf.postprocess", params.out_files_name,
         "--output", prefix, weight_option, "neutron", "--q-edges", ".8", "1.1",
         "--resolution-fwhm-mev", "2", "--per-mev"], capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "coverage" in completed.stderr
    final = load_spectra(prefix)
    assert final["metadata"]["intensity_unit"] == "fm^2/meV"
    assert "S(Q,E) [fm^2/meV]; neutron weights" in Path(prefix + ".dsf").read_text()
    assert final["metadata"]["spectral_density_axis"] == "energy_mev"
    figure = plot_dsf(SimpleNamespace(out_files_name=prefix, dsf_plot_energy_min=-2., dsf_plot_energy_max=2.))
    assert figure.stat().st_size > 10000
    assert "q_direction_counts" in final


def test_postprocessing_preserves_input_and_invalidates_compute_readiness(tmp_path):
    from mdtrace.pipeline import _dsf_ready

    result, params = _sample(tmp_path)
    save_dsf(result, params)
    saved_text = Path(params.out_files_name + ".dsf_Qaxis")
    before = saved_text.read_bytes()
    with pytest.raises(SystemExit):
        main([params.out_files_name, "--output", params.out_files_name, "--per-mev"])
    assert saved_text.read_bytes() == before
    assert _dsf_ready(params, check_settings=True)
    changed = reweight_scattering(result)
    save_dsf(changed, params)
    assert _dsf_ready(params, check_settings=False)
    assert not _dsf_ready(params, check_settings=True)


def test_resolution_lorentzian_and_too_narrow_resolution_warning(tmp_path):
    result, _ = _sample(tmp_path, time_step=100.)
    de = np.diff(result["energy_mev"])[0]
    broadened = apply_resolution(result, 2*de, kind="lorentzian")
    assert np.isfinite(broadened["sqw"]).all()
    np.testing.assert_allclose(broadened["sqw"],
                               broadened["coherent_sqw"] + broadened["incoherent_sqw"], atol=1e-13)
    with pytest.warns(RuntimeWarning, match="smaller than one energy bin"):
        apply_resolution(result, de/2)
    with pytest.raises(ValueError, match="positive finite"):
        apply_resolution(result, 0)


def test_saved_q_only_supports_rebin_but_reweight_requires_partials(tmp_path):
    result, _ = _sample(tmp_path, dsf_partials=False, dsf_save_q_resolved=True)
    with pytest.warns(RuntimeWarning):
        assert average_shells(result, [.5, 1.1])["sqw"].shape[0] == 1
    with pytest.raises(ValueError, match="dsf_partials"):
        reweight_scattering(result)
    with pytest.raises(ValueError, match="strictly increasing"):
        average_shells(result, [1, 0])
    with pytest.raises(ValueError, match="positive"):
        average_shells(result, [0, 2], weights=[1, -1, 1, 1])


def test_number_partials_without_element_names_use_type_labels(tmp_path):
    result, _ = _sample(tmp_path, atom_types=None)
    assert result["metadata"]["species_labels"] == ["type_4", "type_7"]
    with pytest.raises(ValueError, match="no coherent neutron"):
        reweight_scattering(result)


@pytest.mark.parametrize("currents", [False, True])
def test_default_output_is_minimal_text_and_ignores_old_optional_files(tmp_path, currents):
    from mdtrace.pipeline import _dsf_ready, step_plot_dsf

    result, params = _sample(tmp_path, dsf_partials=False, dsf_currents=currents,
                             dsf_output_components=False, dsf_output_sem=False)
    save_dsf(result, params)
    prefix = Path(params.out_files_name)
    expected = {prefix.name + suffix for suffix in (".dsf", ".dsf_THz", ".dsf_Qaxis")}
    if currents:
        expected.update(prefix.name + suffix for suffix in (".dsf_CL", ".dsf_CT", ".dsf_Ctotal"))
    assert {p.name for p in tmp_path.glob(prefix.name + ".dsf*")} == expected
    qtable = Path(str(prefix) + ".dsf_Qaxis").read_text(encoding="utf-8")
    assert "q_cart" not in qtable and "q_arrays" not in qtable and "Per-Q data" not in qtable
    assert "q_cart" not in load_spectra(prefix)
    # Historical optional outputs are not deleted, loaded or mistaken for new data.
    historical = Path(str(prefix) + ".dsf_self")
    historical.write_text("historical result, preserve me")
    assert "self_sqw" not in load_spectra(prefix)
    assert _dsf_ready(params, check_settings=True)
    params.dsf_plot_scale = "linear"
    assert _dsf_ready(params, check_settings=True)
    Path(params.trajectory_path).unlink()
    assert not _dsf_ready(params, check_settings=True)
    step_plot_dsf(params)
    assert historical.read_text() == "historical result, preserve me"
    Path(str(prefix) + ".dsf_THz").unlink()
    assert not _dsf_ready(params)
    with pytest.raises(ValueError, match="incomplete DSF text"):
        step_plot_dsf(params)


def test_text_reader_rejects_transposed_or_truncated_spectrum(tmp_path):
    result, params = _sample(tmp_path, dsf_partials=False)
    save_dsf(result, params)
    np.savetxt(params.out_files_name + ".dsf", result["sqw"])
    with pytest.raises(ValueError, match="frequency/Q shape"):
        load_spectra(params.out_files_name)


def test_incomplete_q_table_without_footer_is_not_ready(tmp_path):
    from mdtrace.dsf.results import spectra_ready

    result, params = _sample(tmp_path, dsf_partials=False)
    save_dsf(result, params)
    qfile = Path(params.out_files_name + ".dsf_Qaxis")
    lines = qfile.read_text(encoding="utf-8").splitlines()
    qfile.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    assert not spectra_ready(params)
    with pytest.raises(ValueError, match="missing DSF text metadata"):
        load_spectra(params.out_files_name)


@pytest.mark.parametrize("experiment", ["number", "neutron"])
def test_plot_needs_only_displayed_spectra_without_source_or_optional_outputs(tmp_path, experiment):
    from mdtrace.pipeline import _dsf_ready, step_plot_dsf
    from mdtrace.dsf.results import result_files

    result, params = _sample(tmp_path, experiment=experiment)
    save_dsf(result, params)
    expected = plot_dsf(params).read_bytes()
    Path(params.trajectory_path).unlink()
    required = set(result_files(params.out_files_name))
    removed = set(result_files(params.out_files_name, include_blocks=True)) - required
    assert removed
    for path in removed:
        path.unlink()  # synthetic test data only
    assert _dsf_ready(params)
    assert not _dsf_ready(params, check_settings=True)
    selected = load_spectra(params.out_files_name, for_plot=True)
    assert "q_cart" not in selected
    assert "self_sqw" not in selected
    assert not any(name.endswith("_sem") or name.startswith("qblocks__") for name in selected)
    assert "coherent_sqw" not in selected
    step_plot_dsf(params)
    assert Path(params.out_files_name + ".dsf.png").read_bytes() == expected
