"""Optional, real dynasor 2.5 comparisons (CPU, deterministic tiny trajectories).

dynasor returns even, unbiased linear time correlations. MDtrace returns a
finite-record periodogram. To compare estimators exactly we restore (T-lag)/T
weights and fold linear lags into a circular correlation, then transform it.
This is the finite-record Wiener-Khinchin identity, not fitted rescaling.
dynasor 2.5 averages transverse polarizations; MDtrace sums them (factor two).
"""

from types import SimpleNamespace

import numpy as np
import pytest

dynasor = pytest.importorskip("dynasor", minversion="2.5")

from ase import Atoms, units
from ase.io.trajectory import Trajectory as ASETrajectory
from numba import set_num_threads

from mdtrace.dsf.dsf import _neutron_weights
from mdtrace.dsf.partials import species_layout
from mdtrace.dsf.qpoints import build_qpoints
from mdtrace.dsf.spectra import block_spectra


def _aligned_psd(acf, dt_ps):
    n = acf.shape[-1]
    lag = np.arange(n)
    circular = acf * ((n - lag) / n)
    circular[..., 1:] += acf[..., :0:-1] * (lag[1:] / n)
    return np.fft.fftshift(np.fft.fft(circular, axis=-1).real * dt_ps, axes=-1)


def _even_frequency(spectrum):
    unshifted = np.fft.ifftshift(spectrum, axes=-1)
    opposite = (-np.arange(unshifted.shape[-1])) % unshifted.shape[-1]
    return np.fft.fftshift((unshifted + unshifted[..., opposite]) / 2, axes=-1)


@pytest.mark.parametrize("cell", [2*np.pi*np.eye(3),
                                 2*np.pi*np.array([[1, .2, 0], [.3, 1.2, .1], [0, 0, .9]])])
def test_complete_reciprocal_spheres_match_dynasor(cell):
    from dynasor.qpoints import get_spherical_qpoints

    ours = build_qpoints(SimpleNamespace(dsf_q_mode="spherical", dsf_q_min=.2,
                         dsf_q_max=2.21, dsf_q_bin_width=.3, dsf_max_points_per_shell=0), cell)
    theirs = get_spherical_qpoints(cell, q_min=.2, q_max=2.21)
    assert {tuple(x) for x in np.round(ours["q_cart"], 10)} == {tuple(x) for x in np.round(theirs, 10)}


@pytest.mark.parametrize("disordered", [False, True])
def test_full_dynasor_pipeline_density_self_currents_pairs_and_neutron_weights(tmp_path, disordered):
    set_num_threads(1)
    if hasattr(dynasor, "set_compute_config"):
        dynasor.set_compute_config(backend="numba")
    rng = np.random.default_rng(19)
    frames, atoms, dt_ps = 33, 8, .01
    cell = 2*np.pi*np.eye(3)
    base = rng.uniform(0, 2*np.pi, size=(atoms, 3)) if disordered else np.array(
        [(x, y, z) for x in (0, np.pi) for y in (0, np.pi) for z in (0, np.pi)])
    t = np.arange(frames) * dt_ps
    phases = rng.uniform(0, 2*np.pi, size=(atoms, 3))
    omega = 2*np.pi*np.array([1, 3, 2]) / (frames * dt_ps)
    displacement = .07*np.sin(t[:, None, None]*omega + phases)
    positions = base[None] + displacement
    velocities = .07*omega*np.cos(t[:, None, None]*omega + phases)
    labels = np.array(["3He", "H"] * 4)
    q = np.array([[1., 0, 0], [-1., 0, 0], [0, 1., 0], [0, -1., 0]])
    path = tmp_path / "reference.traj"
    with ASETrajectory(str(path), "w") as writer:
        for p, v in zip(positions, velocities):
            structure = Atoms("HeH" * 4, positions=p, cell=cell, pbc=True)
            structure.set_velocities(v / 1000 / units.fs)  # A/ps -> ASE time units
            writer.write(structure)
    trajectory = dynasor.Trajectory(str(path), trajectory_format="ase",
                  atomic_indices={name: np.flatnonzero(labels == name).tolist() for name in ("3He", "H")},
                  length_unit="Angstrom", time_unit="fs")
    theirs = dynasor.compute_dynamic_structure_factors(trajectory, q, dt=dt_ps*1000,
                  window_size=frames-1, window_step=1, calculate_currents=True,
                  calculate_incoherent=True, logging_interval=0)
    ours = block_spectra(positions, q, np.ones(atoms), np.zeros(atoms),
                        dt_ps=dt_ps, velocities=velocities, species_labels=labels)
    references = {"coherent_sqw": theirs.Fqt_coh, "self_sqw": theirs.Fqt_incoh,
                  "longitudinal_current": theirs.Clqt*1e6,
                  "transverse_current": theirs.Ctqt*2e6}
    for name, acf in references.items():
        np.testing.assert_allclose(_even_frequency(ours[name]), _aligned_psd(acf, dt_ps), atol=2e-12, rtol=2e-11)
        # Absolute integrated strength, independent of the spectral grid.
        np.testing.assert_allclose(ours[name].sum(axis=-1)/(frames*dt_ps), acf[:, 0], atol=2e-12)
    species, _, pairs = species_layout(labels)
    for k, (a, b) in enumerate(pairs):
        pair_name = f"{species[a]}_{species[b]}"
        for local, remote, conversion in (("partial_coherent_sqw", "Fqt_coh_", 1.),
                                           ("partial_longitudinal_current", "Clqt_", 1e6),
                                           ("partial_transverse_current", "Ctqt_", 2e6)):
            np.testing.assert_allclose(_even_frequency(ours[local][:, k]),
                                       _aligned_psd(theirs[remote + pair_name]*conversion, dt_ps),
                                       atol=2e-12, rtol=2e-11)
    # Complex neutron contrast: dynasor uses real symmetrized pair functions.
    # Compare only after both +/-Q and +/-frequency averaging; the full complex
    # directional cross terms are validated separately by explicit pair sums.
    b, inc = _neutron_weights(labels)
    weighted = block_spectra(positions, q, b, inc, dt_ps=dt_ps)
    bs, incs = _neutron_weights(species)
    acf = np.zeros_like(theirs.Fqt_coh)
    for a, b in pairs:
        acf += (bs[a]*bs[b].conjugate()).real * theirs[f"Fqt_coh_{species[a]}_{species[b]}"]
    for i, name in enumerate(species):
        acf += incs[i] * theirs[f"Fqt_incoh_{name}"]
    np.testing.assert_allclose(_even_frequency(weighted["sqw"]).mean(axis=0),
                               _aligned_psd(acf, dt_ps).mean(axis=0), atol=2e-11, rtol=2e-11)
