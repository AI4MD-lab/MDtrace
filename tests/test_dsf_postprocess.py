"""Conservation, covariance and workflow checks for reusable DSF spectra."""

from copy import deepcopy
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
                    dsf_partials=True, dsf_currents=True, atom_types=["3He", "H"])
    settings.update(options)
    params = _params(path, **settings)
    return compute_dsf(params), params


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


@pytest.mark.parametrize("window", ["none", "hann"])
def test_complex_partials_reconstruct_direct_totals_and_preserve_phase(window):
    rng = np.random.default_rng(123)
    positions, velocities = rng.normal(size=(2, 48, 5, 3))
    labels = np.array(["A", "B", "A", "B", "B"])
    weights = np.where(labels == "A", 2+3j, -1+.7j)
    incoherent = np.where(labels == "A", 4., 2.)
    q = np.array([[.8, -.3, .4], [-.8, .3, -.4], [0, 0, 0]])
    args = dict(dt_ps=.01, velocities=velocities, window=window, subtract_mean=True)
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
def test_plot_skips_block_and_partial_arrays_without_changing_image(tmp_path, monkeypatch, experiment):
    from mdtrace.dsf import plot

    result, params = _sample(tmp_path, experiment=experiment)
    save_dsf(result, params)
    archive_path = params.out_files_name + ".dsf.npz"
    with np.load(archive_path) as archive:
        archive_type = type(archive)
        bulk_fields = {name for name in archive.files
                       if name.startswith(("qblocks__", "qresolved__", "partial_"))
                       or name in {"pair_density", "species_self"}}
    assert bulk_fields

    # Reference rendering loads the complete archive, as before this change.
    with monkeypatch.context() as patch:
        patch.setattr(plot, "load_spectra", lambda path, **kwargs: load_spectra(path))
        expected_image = plot_dsf(params).read_bytes()

    original_getitem = archive_type.__getitem__

    def read_without_bulk(self, name):
        assert name not in bulk_fields, f"plot unnecessarily decompresses {name}"
        return original_getitem(self, name)

    with monkeypatch.context() as patch:
        patch.setattr(archive_type, "__getitem__", read_without_bulk)
        assert plot_dsf(params).read_bytes() == expected_image

    # Full numerical post-processing still has all retained block arrays.
    assert bulk_fields.issubset(load_spectra(archive_path))


def test_reweight_rebin_roundtrip_and_sem_after_averaging(tmp_path):
    result, params = _sample(tmp_path)
    save_dsf(result, params)
    loaded = load_spectra(params.out_files_name + ".dsf.npz")
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
    with pytest.warns(RuntimeWarning):
        number_first = average_shells(loaded, [0, 1.1], weights=weights)
    reordered = reweight_scattering(number_first)
    np.testing.assert_allclose(reordered["sqw"], averaged["sqw"])
    np.testing.assert_allclose(reordered["sqw_sem"], averaged["sqw_sem"])
    np.testing.assert_allclose(reweight_scattering(weighted, experiment="number")["sqw"], loaded["sqw"])
    assert loaded["metadata"]["experiment"] == "number"  # caller's archive remains intact


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


def test_postprocessing_cli_needs_no_trajectory_and_plots_qens_energy_window(tmp_path):
    result, params = _sample(tmp_path, time_step=100.)
    save_dsf(result, params)
    Path(params.trajectory_path).unlink()
    prefix = str(tmp_path / "neutron_qens")
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "mdtrace.dsf.postprocess", params.out_files_name + ".dsf.npz",
         "--output", prefix, "--experiment", "neutron", "--q-edges", ".8", "1.1",
         "--resolution-fwhm-mev", "2", "--per-mev"], capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "coverage" in completed.stderr
    final = load_spectra(prefix + ".dsf.npz")
    assert final["metadata"]["intensity_unit"] == "fm^2/meV"
    assert "per meV" in Path(prefix + ".dsf").read_text()
    assert "per THz" not in Path(prefix + ".dsf_meV").read_text()
    figure = plot_dsf(SimpleNamespace(out_files_name=prefix, dsf_plot_energy_min=-2., dsf_plot_energy_max=2.))
    assert figure.stat().st_size > 10000
    assert Path(prefix + ".dsf_Qdiagnostics").is_file()


def test_postprocessing_preserves_input_and_invalidates_compute_readiness(tmp_path):
    from mdtrace.pipeline import _dsf_ready

    result, params = _sample(tmp_path)
    save_dsf(result, params)
    archive = Path(params.out_files_name + ".dsf.npz")
    before = archive.read_bytes()
    with pytest.raises(SystemExit):
        main([str(archive), "--output", params.out_files_name, "--per-mev"])
    assert archive.read_bytes() == before
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
