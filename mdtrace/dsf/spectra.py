# Copyright 2026 Ting Liang and MDTRACE development team
# SPDX-License-Identifier: GPL-3.0-or-later
"""Density and particle-current periodograms using MDtrace's FFT approach.

The standard two-sided spectrum is per ordinary frequency f (THz), not per
angular frequency or energy. With a window w and N atoms it is
dt * |FFT(w*x)|**2 / (N * sum(w**2)). Its discrete frequency integral is the
window-weighted mean squared amplitude per atom (Parseval's identity).
The legacy pynamic convention is available for reproducibility.

Particle-current definitions: https://doi.org/10.1002/adts.202000240
They are number weighted, irrespective of the neutron weights of the DSF.
"""

import numpy as np

from mdtrace.dsf.partials import combine_partials, species_layout


PLANCK_MEV_PS = 4.135667696


def block_spectra(positions, q_cart, coherent_weights, incoherent_weights_sq,
                  *, dt_ps, velocities=None, normalization="density",
                  window="none", subtract_mean=False, species_labels=None):
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (frame, atom, 3)")
    n_frames, n_atoms, _ = positions.shape
    if n_frames == 0 or n_atoms == 0:
        raise ValueError("DSF requires at least one frame and one atom")
    if not np.isfinite(positions).all():
        raise ValueError("trajectory positions must be finite")
    if coherent_weights.shape != (n_atoms,) or incoherent_weights_sq.shape != (n_atoms,):
        raise ValueError("scattering weights do not match the number of atoms")
    if velocities is not None:
        velocities = np.asarray(velocities, dtype=float)
        if velocities.shape != positions.shape or not np.isfinite(velocities).all():
            raise ValueError("velocities must be finite and match positions (frame, atom, 3)")
    if window == "none":
        taper = np.ones(n_frames)
    elif window == "hann":
        taper = np.sin(np.pi * np.arange(n_frames) / n_frames)**2
    else:
        raise ValueError("dsf_window must be none or hann")
    window_energy = float(taper @ taper)
    if window_energy == 0:
        raise ValueError("the selected DSF window has zero weight; increase block length")
    if normalization == "density":
        scale = dt_ps / (n_atoms * window_energy)
    elif normalization == "legacy":
        scale = 1.0 / (n_atoms * n_frames**2 * window_energy)
    else:
        raise ValueError("dsf_normalization must be density or legacy")

    def transform(series):
        if subtract_mean:
            series = series - series.mean(axis=0, keepdims=True)
        weights = taper.reshape((n_frames,) + (1,) * (series.ndim - 1))
        return np.fft.fftshift(np.fft.fft(series * weights, axis=0), axes=0)

    names = ["coherent_sqw", "incoherent_sqw", "self_sqw"]
    if velocities is not None:
        names += ["longitudinal_current", "transverse_current"]
    result = {name: np.empty((len(q_cart), n_frames)) for name in names}
    if species_labels is not None:
        if np.shape(species_labels) != (n_atoms,):
            raise ValueError("species_labels must have one label per atom")
        species, species_index, pairs = species_layout(species_labels)
        masks = [species_index == i for i in range(len(species))]
        species_coh = np.array([coherent_weights[m][0] for m in masks])
        species_inc = np.array([incoherent_weights_sq[m][0] for m in masks])
        for i, mask in enumerate(masks):
            if (not np.all(coherent_weights[mask] == species_coh[i])
                    or not np.all(incoherent_weights_sq[mask] == species_inc[i])):
                raise ValueError("atoms with the same species label must have the same scattering weights")
        result["pair_density"] = np.empty((len(q_cart), len(pairs), n_frames), complex)
        result["species_self"] = np.empty((len(q_cart), len(species), n_frames))
        if velocities is not None:
            for name in ("longitudinal_current", "transverse_current"):
                result["pair_" + name] = np.empty((len(q_cart), len(pairs), n_frames), complex)
    for index, q_vector in enumerate(q_cart):
        phase = np.exp(1j * (positions @ q_vector))
        density = phase @ coherent_weights
        result["coherent_sqw"][index] = abs(transform(density))**2 * scale
        atom_power = abs(transform(phase))**2
        result["self_sqw"][index] = atom_power.sum(axis=1) * scale
        result["incoherent_sqw"][index] = atom_power @ incoherent_weights_sq * scale
        if species_labels is not None:
            amplitudes = np.stack([transform(phase[:, m].sum(axis=1)) for m in masks])
            a, b = pairs.T
            result["pair_density"][index] = amplitudes[a] * amplitudes[b].conj() * scale
            result["species_self"][index] = np.stack([atom_power[:, m].sum(axis=1) for m in masks]) * scale
        if velocities is not None:
            magnitude = np.linalg.norm(q_vector)
            if magnitude == 0:
                # There is no longitudinal direction at Q=0.
                result["longitudinal_current"][index] = np.nan
                result["transverse_current"][index] = np.nan
                if species_labels is not None:
                    result["pair_longitudinal_current"][index] = np.nan
                    result["pair_transverse_current"][index] = np.nan
                continue
            direction = q_vector / magnitude
            current = np.einsum("tna,tn->ta", velocities, phase)
            current_fft = transform(current)
            longitudinal = current_fft @ direction
            transverse = current_fft - longitudinal[:, None] * direction
            result["longitudinal_current"][index] = abs(longitudinal)**2 * scale
            # Sum of the two transverse polarizations in 3D, not their average.
            result["transverse_current"][index] = (abs(transverse)**2).sum(axis=1) * scale
            if species_labels is not None:
                vectors = np.stack([transform(np.einsum("tna,tn->ta", velocities[:, m], phase[:, m]))
                                    for m in masks])
                along = vectors @ direction
                across = vectors - along[:, :, None] * direction
                result["pair_longitudinal_current"][index] = along[a] * along[b].conj() * scale
                result["pair_transverse_current"][index] = (across[a] * across[b].conj()).sum(axis=-1) * scale
    result["sqw"] = result["coherent_sqw"] + result["incoherent_sqw"]
    if species_labels is not None:
        result.update(combine_partials(result, pairs, species_coh, species_inc))
    return result


def mean_and_sem(blocks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    values = np.stack(blocks, axis=0)
    mean = np.mean(values, axis=0)
    if len(values) == 1:
        return mean, np.full(mean.shape, np.nan)
    sem = np.std(values, axis=0, ddof=1) / np.sqrt(len(values))
    return mean, sem
