# Copyright 2026 Ting Liang and MDTRACE development team
# SPDX-License-Identifier: GPL-3.0-or-later
"""Density and particle-current periodograms using MDtrace's FFT approach.

The standard two-sided spectrum is per ordinary frequency f (THz), not per
angular frequency or energy. With T frames and N atoms it is
dt * |FFT(x)|**2 / (N * T). Its discrete frequency integral is the
mean squared amplitude per atom (Parseval's identity).
The legacy pynamic convention is available for reproducibility.

Particle-current definitions: https://doi.org/10.1002/adts.202000240
They are number weighted, irrespective of the neutron weights of the DSF.
"""

import numpy as np

from mdtrace.dsf.partials import combine_partials, species_layout


PLANCK_MEV_PS = 4.135667696


def _spectral_scale(n_frames, n_atoms, dt_ps, normalization):
    if normalization == "density":
        return dt_ps / (n_atoms * n_frames)
    if normalization == "legacy":
        return 1.0 / (n_atoms * n_frames**3)
    raise ValueError("dsf_normalization must be density or legacy")


def _time_transform(series, subtract_mean, xp=np):
    if subtract_mean:
        series = series - series.mean(axis=0, keepdims=True)
    return xp.fft.fftshift(xp.fft.fft(series, axis=0), axes=0)


def _current_power(current_fft, direction, scale):
    longitudinal = current_fft @ direction
    transverse = current_fft - longitudinal[:, None] * direction
    return abs(longitudinal)**2 * scale, (abs(transverse)**2).sum(axis=1) * scale


def _phase(positions, q_vector, xp=np):
    # Evaluate in place to avoid a second full complex phase array.
    phase = xp.empty(positions.shape[:2], dtype=complex)
    phase.real = 0
    phase.imag = positions @ xp.asarray(q_vector)
    xp.exp(phase, out=phase)
    return phase


def density_current_signals(positions, q_cart, coherent_weights, output, velocities=None, *, q_completed=None):
    """Fill (Q, frame, 1 or 4) signals for a small batch, without a time FFT."""
    if not np.isfinite(positions).all():
        raise ValueError("trajectory positions must be finite")
    if velocities is not None and not np.isfinite(velocities).all():
        raise ValueError("velocities must be finite and match positions (frame, atom, 3)")
    for i, q_vector in enumerate(q_cart):
        phase = _phase(positions, q_vector)
        output[i, :, 0] = phase @ coherent_weights
        if velocities is not None:
            output[i, :, 1:] = np.einsum("tna,tn->ta", velocities, phase)
        del phase
        if q_completed is not None:
            q_completed(1)


def spectra_from_signals(signals, q_cart, n_atoms, *, dt_ps,
                         normalization="density", subtract_mean=False, q_completed=None):
    """Transform complete-block signals with the same estimator as block_spectra."""
    n_q, n_frames, channels = signals.shape
    scale = _spectral_scale(n_frames, n_atoms, dt_ps, normalization)
    result = {name: np.empty((n_q, n_frames)) for name in ("coherent_sqw", "incoherent_sqw")}
    result["incoherent_sqw"].fill(0)
    if channels == 4:
        for name in ("longitudinal_current", "transverse_current"):
            result[name] = np.empty((n_q, n_frames))
    for i, q_vector in enumerate(q_cart):
        density_fft = _time_transform(signals[i, :, 0], subtract_mean)
        result["coherent_sqw"][i] = abs(density_fft)**2 * scale
        if channels == 4:
            magnitude = np.linalg.norm(q_vector)
            if magnitude == 0:
                result["longitudinal_current"][i] = np.nan
                result["transverse_current"][i] = np.nan
            else:
                current_fft = _time_transform(signals[i, :, 1:], subtract_mean)
                result["longitudinal_current"][i], result["transverse_current"][i] = _current_power(
                    current_fft, q_vector / magnitude, scale,
                )
        if q_completed is not None:
            q_completed(1)
    result["sqw"] = result["coherent_sqw"] + result["incoherent_sqw"]
    return result


def block_spectra(positions, q_cart, coherent_weights, incoherent_weights_sq,
                  *, dt_ps, velocities=None, normalization="density",
                  subtract_mean=False, species_labels=None,
                  compute_self=True, progress=None, xp=np):
    """Compute requested spectra; partials also reconstruct their total self spectrum."""
    positions = xp.asarray(positions, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (frame, atom, 3)")
    n_frames, n_atoms, _ = positions.shape
    if n_frames == 0 or n_atoms == 0:
        raise ValueError("DSF requires at least one frame and one atom")
    if not xp.isfinite(positions).all():
        raise ValueError("trajectory positions must be finite")
    if coherent_weights.shape != (n_atoms,) or incoherent_weights_sq.shape != (n_atoms,):
        raise ValueError("scattering weights do not match the number of atoms")
    if velocities is not None:
        velocities = xp.asarray(velocities, dtype=float)
        if velocities.shape != positions.shape or not xp.isfinite(velocities).all():
            raise ValueError("velocities must be finite and match positions (frame, atom, 3)")
    scale = _spectral_scale(n_frames, n_atoms, dt_ps, normalization)

    def transform(series):
        return _time_transform(series, subtract_mean, xp)

    # Self/partial spectra and nonzero incoherent weights require atom-wise FFTs.
    # A density/current-only calculation needs just the summed time signals.
    atom_fft = compute_self or species_labels is not None or bool(xp.any(incoherent_weights_sq))
    names = ["coherent_sqw", "incoherent_sqw"]
    if compute_self:
        names.append("self_sqw")
    if velocities is not None:
        names += ["longitudinal_current", "transverse_current"]
    result = {name: xp.empty((len(q_cart), n_frames)) for name in names}
    if species_labels is not None:
        if np.shape(species_labels) != (n_atoms,):
            raise ValueError("species_labels must have one label per atom")
        species, species_index, pairs = species_layout(species_labels)
        masks = [xp.asarray(species_index == i) for i in range(len(species))]
        species_coh = xp.stack([coherent_weights[m][0] for m in masks])
        species_inc = xp.stack([incoherent_weights_sq[m][0] for m in masks])
        for i, mask in enumerate(masks):
            if (not xp.all(coherent_weights[mask] == species_coh[i])
                    or not xp.all(incoherent_weights_sq[mask] == species_inc[i])):
                raise ValueError("atoms with the same species label must have the same scattering weights")
        result["pair_density"] = xp.empty((len(q_cart), len(pairs), n_frames), complex)
        result["species_self"] = xp.empty((len(q_cart), len(species), n_frames))
        if velocities is not None:
            for name in ("longitudinal_current", "transverse_current"):
                result["pair_" + name] = xp.empty((len(q_cart), len(pairs), n_frames), complex)
    for index, q_vector in enumerate(q_cart):
        phase = _phase(positions, q_vector, xp)
        density = phase @ coherent_weights
        result["coherent_sqw"][index] = abs(transform(density))**2 * scale
        if atom_fft:
            atom_power = abs(transform(phase))**2
            if compute_self:
                result["self_sqw"][index] = atom_power.sum(axis=1) * scale
            result["incoherent_sqw"][index] = atom_power @ incoherent_weights_sq * scale
        else:
            result["incoherent_sqw"][index] = 0.0
        if species_labels is not None:
            amplitudes = xp.stack([transform(phase[:, m].sum(axis=1)) for m in masks])
            a, b = pairs.T
            result["pair_density"][index] = amplitudes[a] * amplitudes[b].conj() * scale
            result["species_self"][index] = xp.stack([atom_power[:, m].sum(axis=1) for m in masks]) * scale
        if velocities is not None:
            magnitude = np.linalg.norm(q_vector)
            if magnitude == 0:
                # There is no longitudinal direction at Q=0.
                result["longitudinal_current"][index] = np.nan
                result["transverse_current"][index] = np.nan
                if species_labels is not None:
                    result["pair_longitudinal_current"][index] = np.nan
                    result["pair_transverse_current"][index] = np.nan
                del phase
                if atom_fft:
                    del atom_power
                if progress is not None:
                    progress((index + 1) / len(q_cart))
                continue
            direction = xp.asarray(q_vector / magnitude)
            current = xp.einsum("tna,tn->ta", velocities, phase)
            current_fft = transform(current)
            # Sum of the two transverse polarizations in 3D, not their average.
            result["longitudinal_current"][index], result["transverse_current"][index] = _current_power(
                current_fft, direction, scale,
            )
            if species_labels is not None:
                vectors = xp.stack([transform(xp.einsum("tna,tn->ta", velocities[:, m], phase[:, m]))
                                    for m in masks])
                along = vectors @ direction
                across = vectors - along[:, :, None] * direction
                result["pair_longitudinal_current"][index] = along[a] * along[b].conj() * scale
                result["pair_transverse_current"][index] = (across[a] * across[b].conj()).sum(axis=-1) * scale
        del phase
        if atom_fft:
            del atom_power
        if progress is not None:
            progress((index + 1) / len(q_cart))
    result["sqw"] = result["coherent_sqw"] + result["incoherent_sqw"]
    if species_labels is not None:
        result.update(combine_partials(result, pairs, species_coh, species_inc, xp=xp))
    return result


def mean_and_sem(blocks) -> tuple[np.ndarray, np.ndarray]:
    """Stack a block list once, or reuse an already stacked array without copying."""
    values = np.asarray(blocks)
    mean = np.mean(values, axis=0)
    if len(values) == 1:
        return mean, np.full(mean.shape, np.nan)
    sem = np.std(values, axis=0, ddof=1) / np.sqrt(len(values))
    return mean, sem
