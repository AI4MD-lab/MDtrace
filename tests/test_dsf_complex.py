"""Complex neutron weights checked against analytic and pair-correlation results."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import numpy as np
import pytest

from mdtrace.dsf.scattering import neutron_weights as _neutron_weights
from mdtrace.dsf.dsf import (
    _compute_block_spectra,
    compute_dsf,
    save_dsf,
)
from mdtrace.dsf.neutron_scattering_lengths import (
    coherent_scattering_lengths,
    incoherent_scattering_lengths,
)
from mdtrace.io.prepare import prepare_trajectory


ALIASES = {"D": "2H", "T": "3H"}
SUPPORTED_LABELS = sorted(
    coherent_scattering_lengths.keys() & incoherent_scattering_lengths.keys()
) + list(ALIASES)
MIXTURES = [
    ("H", "O", "Si"),
    ("3He", "6Li", "10B"),
    ("149Sm", "151Eu", "157Gd"),
    ("H", "O", "Cd", "Gd"),
    ("Si", "28Si", "29Si", "30Si"),
    ("D", "T", "6Li", "none"),
]


def _source_lengths(labels):
    keys = [ALIASES.get(label, label) for label in labels]
    return (
        np.array([coherent_scattering_lengths[key] for key in keys]),
        np.array([incoherent_scattering_lengths[key] for key in keys]),
    )


def _pair_correlation_spectra(positions, qpoints, b_coherent, b_incoherent):
    """Expand every time/atom pair, independently of the density FFT code.

    Gamma[j,k] = conj(b[j])*b[k] for the coherent term; only j=k
    contributes to the incoherent term, with weight Re(b[j])**2+Im(b[j])**2.
    The double time sum has T**2 terms; the legacy FFT normalization adds
    another N*T divisor. Frequencies are returned in fftshift order.
    """
    n_frames, n_atoms, _ = positions.shape
    frequencies = np.arange(-(n_frames // 2), n_frames - n_frames // 2)
    gamma = b_coherent.conj()[:, None] * b_coherent[None, :]
    self_weights = b_incoherent.real**2 + b_incoherent.imag**2
    result = np.zeros((2, len(qpoints), n_frames), dtype=complex)
    for iq, q in enumerate(qpoints):
        for t0 in range(n_frames):
            for t1 in range(n_frames):
                separation = positions[t1][None, :, :] - positions[t0][:, None, :]
                phase = np.exp(1j * (separation @ q))
                time_phase = np.exp(-2j * np.pi * frequencies * (t1 - t0) / n_frames)
                result[0, iq] += np.sum(gamma * phase) * time_phase
                result[1, iq] += np.sum(self_weights * np.diag(phase)) * time_phase
    result /= n_atoms * n_frames**3
    np.testing.assert_allclose(result.imag, 0.0, atol=1e-10)
    return result.real


@pytest.mark.parametrize("label", SUPPORTED_LABELS)
def test_every_supported_label_has_complete_weights_and_single_atom_spectrum(label):
    b_coherent, b_incoherent = _source_lengths([label])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        coherent_weights, incoherent_weights = _neutron_weights(np.array([label]))

    assert np.iscomplexobj(coherent_weights)
    assert not np.iscomplexobj(incoherent_weights)
    np.testing.assert_array_equal(coherent_weights, b_coherent)
    np.testing.assert_allclose(
        incoherent_weights, b_incoherent.real**2 + b_incoherent.imag**2,
        rtol=1e-14, atol=0.0,
    )

    # A stationary single atom has only an elastic peak, for any Q.
    positions = np.tile([[[0.12, -0.4, 0.31]]], (5, 1, 1))
    qpoints = np.array([[0.0, 0.0, 0.0], [0.7, -1.3, 2.1]])
    actual = _compute_block_spectra(
        positions, qpoints, coherent_weights, incoherent_weights,
    )
    for spectrum, length in zip(actual, (b_coherent[0], b_incoherent[0])):
        expected = np.zeros((2, 5))
        expected[:, 2] = (length.real**2 + length.imag**2) / 5
        np.testing.assert_allclose(spectrum, expected, rtol=1e-13, atol=1e-12)
        assert np.isrealobj(spectrum)
        assert np.all(np.isfinite(spectrum))
        assert np.all(spectrum >= 0)


@pytest.mark.parametrize("labels", MIXTURES)
@pytest.mark.parametrize("n_frames", [4, 5])
def test_mixed_element_spectra_match_full_pair_correlation(labels, n_frames):
    positions = np.random.default_rng(106).normal(size=(n_frames, len(labels), 3))
    qpoints = np.array([[0, 0, 0], [1.3, -0.7, 0.2], [-1.3, 0.7, -0.2]])
    b_coherent, b_incoherent = _source_lengths(labels)
    expected = _pair_correlation_spectra(positions, qpoints, b_coherent, b_incoherent)
    weights = _neutron_weights(np.array(labels))
    actual = _compute_block_spectra(positions, qpoints, *weights)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-11)
    assert np.isrealobj(actual)
    assert np.all(np.isfinite(actual))
    assert np.all(np.asarray(actual) >= 0)

    # Permuting atom order must not change elemental weights or interference.
    order = np.arange(len(labels))[::-1]
    reordered = _compute_block_spectra(
        positions[:, order], qpoints, *[weight[order] for weight in weights],
    )
    np.testing.assert_allclose(reordered, actual, rtol=1e-13, atol=1e-11)


def test_relative_complex_phase_controls_interference_at_opposite_q():
    # At +Q the two amplitudes are i+i; at -Q they are i-i.
    # Taking |b|, Re(b), or conj(b) before summation gives different answers.
    with (
        patch.dict(coherent_scattering_lengths, {"H": 1j, "He": 1 + 0j}),
        patch.dict(incoherent_scattering_lengths, {"H": -2 + 3j, "He": 4 - 5j}),
    ):
        weights = _neutron_weights(np.array(["H", "He"]))
    positions = np.zeros((4, 2, 3))
    positions[:, 1, 0] = 0.25
    qpoints = np.array([[2 * np.pi, 0, 0], [-2 * np.pi, 0, 0]])
    coherent, incoherent = _compute_block_spectra(positions, qpoints, *weights)
    expected_coherent = np.zeros((2, 4))
    expected_coherent[0, 2] = 4 / 8
    expected_incoherent = np.zeros((2, 4))
    expected_incoherent[:, 2] = (13 + 41) / 8
    np.testing.assert_allclose(coherent, expected_coherent, atol=1e-15)
    np.testing.assert_allclose(incoherent, expected_incoherent, atol=1e-15)


@pytest.mark.parametrize("component", ["coherent", "incoherent"])
@pytest.mark.parametrize("value", [complex(np.nan, 0), complex(0, np.nan),
                                   complex(np.inf, 0), complex(0, -np.inf)])
def test_nonfinite_scattering_lengths_are_rejected(component, value):
    table = (coherent_scattering_lengths if component == "coherent"
             else incoherent_scattering_lengths)
    with patch.dict(table, {"H": value}):
        with pytest.raises(ValueError, match=f"non-finite {component}.*H"):
            _neutron_weights(np.array(["H"]))


def _write_netcdf_trajectory(path, positions, type_ids, atom_ids=None):
    netcdf4 = pytest.importorskip("netCDF4")
    with netcdf4.Dataset(str(path), "w", format="NETCDF4") as dataset:
        for name, count in (("frame", len(positions)), ("atom", positions.shape[1]),
                            ("spatial", 3)):
            dataset.createDimension(name, count)
        coordinates = dataset.createVariable("coordinates", "f8", ("frame", "atom", "spatial"))
        coordinates.units = "angstrom"
        coordinates[:] = positions
        cell = dataset.createVariable("cell_vectors", "f8", ("spatial", "spatial"))
        cell.units = "angstrom"
        cell[:] = 10 * np.eye(3)
        for name, values in (("type", type_ids), ("atom_id", atom_ids)):
            if values is not None:
                values = np.asarray(values)
                dimensions = ("atom",) if values.ndim == 1 else ("frame", "atom")
                variable = dataset.createVariable(name, values.dtype, dimensions)
                variable[:] = values


@pytest.mark.parametrize("reorder_frames", [False, True])
@pytest.mark.parametrize("single_element", [False, True])
@pytest.mark.parametrize("static_types", [False, True])
def test_complex_mixture_through_netcdf_blocks_and_text_output(
    tmp_path, reorder_frames, single_element, static_types,
):
    # Repeated, noncontiguous type IDs exercise the real trajectory mapping.
    type_ids = np.array([42, 7, 105, 42, 7])
    labels = ["157Gd", "H", "3He", "157Gd", "H"]
    if single_element:
        type_ids[:] = 42
        labels = ["3He"] * 5
    positions = np.random.default_rng(401).normal(size=(10, 5, 3))
    trajectory = tmp_path / "mixed.nc"
    order = np.tile(np.arange(5), (10, 1))
    if reorder_frames:
        rng = np.random.default_rng(902)
        order[1:] = [rng.permutation(5) for _ in range(9)]
    # Noncontiguous, unsorted IDs; the first-frame ordering remains the reference.
    atom_ids = np.array([93, 10, 206, 5, 71])
    _write_netcdf_trajectory(
        trajectory, np.take_along_axis(positions, order[:, :, None], axis=1),
        type_ids if static_types else type_ids[order], atom_ids[order],
    )
    q_frac = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    params = SimpleNamespace(
        device="cpu", experiment="neutron", trajectory_path=str(trajectory),
        dsf_normalization="legacy",
        dsf_qpoints=q_frac.ravel().tolist(), prim_unitcell=np.eye(3),
        atom_types=["3He"] if single_element else ["H", "157Gd", "3He"], num_blocks=2,
        time_step=2.0, output_data_stride=5, out_files_name=str(tmp_path / "mixed"),
    )
    result = compute_dsf(params)
    assert result["type_mapping"] == (
        {42: "3He"} if single_element else {7: "H", 42: "157Gd", 105: "3He"}
    )
    b_coherent, b_incoherent = _source_lengths(labels)
    reference_blocks = np.array([
        _pair_correlation_spectra(block, 2 * np.pi * q_frac, b_coherent, b_incoherent)
        for block in (positions[:5], positions[5:])
    ])
    references = {
        "coherent_sqw": reference_blocks[:, 0],
        "incoherent_sqw": reference_blocks[:, 1],
        "sqw": reference_blocks.sum(axis=1),
    }
    for name, blocks in references.items():
        np.testing.assert_allclose(result[name], blocks.mean(axis=0), rtol=1e-12, atol=1e-11)
        # With two blocks SEM = |block1-block2|/2, including their covariance
        # when the total coherent+incoherent signal is formed before averaging.
        np.testing.assert_allclose(result[name + "_sem"], np.abs(blocks[0] - blocks[1]) / 2,
                                   rtol=1e-12, atol=1e-11)
        assert np.isrealobj(result[name])
        assert np.all(np.isfinite(result[name]))
        assert np.all(result[name] >= 0)
    assert np.max(result["sqw_sem"]) > 0

    params.dsf_output_sem = True
    save_dsf(result, params)
    for suffix, name in ((".dsf", "sqw"), (".dsf_coherent", "coherent_sqw"),
                         (".dsf_incoherent", "incoherent_sqw"),
                         (".dsf_sem", "sqw_sem"), (".dsf_coherent_sem", "coherent_sqw_sem"),
                         (".dsf_incoherent_sem", "incoherent_sqw_sem")):
        output = Path(params.out_files_name + suffix)
        np.testing.assert_allclose(np.loadtxt(output), result[name].T, rtol=1e-14, atol=1e-12)
        assert "neutron weights" in output.read_text().splitlines()[0]


def _trajectory_params(path):
    return SimpleNamespace(
        density_weighting="neutron",
        trajectory_path=str(path), dsf_qpoints=[0.4, 0.2, -0.3],
        prim_unitcell=np.eye(3), atom_types=["H"], num_blocks=1,
        dsf_normalization="legacy",
        time_step=1.0, output_data_stride=10,
    )


@pytest.mark.parametrize("atom_ids", [
    [[10, 10], [10, 10]],  # Duplicate identities even when ordering stays fixed.
    [[10, 20], [10, 30]],  # Atom replacement must not be treated as a self trajectory.
    [[10, 20], [10, 10]],
    [[10, 20], [10, np.nan]],
    [[10, 20], [10, 20.5]],
])
def test_invalid_atom_identity_is_rejected(tmp_path, atom_ids):
    trajectory = tmp_path / "invalid_ids.nc"
    _write_netcdf_trajectory(trajectory, np.zeros((2, 2, 3)), [1, 1], atom_ids)
    with pytest.raises(ValueError, match="atom IDs"):
        compute_dsf(_trajectory_params(trajectory))


def test_type_changes_are_rejected_after_atom_alignment(tmp_path):
    trajectory = tmp_path / "changing_types.nc"
    # Atom 10 really changes from type 1 to 2; this is not merely row reordering.
    _write_netcdf_trajectory(
        trajectory, np.zeros((2, 2, 3)), [[1, 2], [2, 2]], [[10, 20], [20, 10]],
    )
    params = _trajectory_params(trajectory)
    params.atom_types = ["H", "O"]
    with pytest.raises(ValueError, match="type ID.*constant"):
        compute_dsf(params)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_positions_are_rejected(tmp_path, value):
    trajectory = tmp_path / "invalid_positions.nc"
    positions = np.zeros((2, 2, 3))
    positions[1, 0, 2] = value
    _write_netcdf_trajectory(trajectory, positions, [1, 1])
    with pytest.raises(ValueError, match="positions.*finite"):
        compute_dsf(_trajectory_params(trajectory))


@pytest.mark.parametrize("time_step,stride", [
    (np.nan, 1), (np.inf, 1), (-1, -1), (1, 0), (1, 1.5),
])
def test_invalid_sampling_interval_is_rejected(tmp_path, time_step, stride):
    trajectory = tmp_path / "invalid_time.nc"
    _write_netcdf_trajectory(trajectory, np.zeros((2, 2, 3)), [1, 1])
    params = _trajectory_params(trajectory)
    params.time_step, params.output_data_stride = time_step, stride
    with pytest.raises(ValueError, match="time_step|output_data_stride"):
        compute_dsf(params)


def test_gpumd_mixed_species_cache_matches_pair_correlation(tmp_path):
    # GPUMD text conversion assigns type IDs in first-appearance order, not
    # alphabetically or by atomic number. Repeated species need the same weight.
    labels = ["O", "H", "O", "3He", "H"]
    positions = np.random.default_rng(17).normal(size=(6, 5, 3))
    source = tmp_path / "mixed.xyz"
    lines = []
    for frame in positions:
        lines.extend(["5", 'Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3'])
        lines.extend(f"{label} {x:.17g} {y:.17g} {z:.17g}"
                     for label, (x, y, z) in zip(labels, frame))
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cache, _ = prepare_trajectory(source, batch_size=2, storage_dtype="float64")
    params = _trajectory_params(cache)
    params.atom_types = ["O", "H", "3He"]
    result = compute_dsf(params)
    assert result["type_mapping"] == {0: "O", 1: "H", 2: "3He"}
    expected = _pair_correlation_spectra(
        positions, result["q_cart"], *_source_lengths(labels),
    )
    np.testing.assert_allclose(result["coherent_sqw"], expected[0], atol=1e-11)
    np.testing.assert_allclose(result["incoherent_sqw"], expected[1], atol=1e-11)
