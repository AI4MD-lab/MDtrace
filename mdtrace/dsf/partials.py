# Copyright 2026 Ting Liang and MDTRACE development team
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reconstruct spectra from complex, unweighted element-pair cross spectra."""

from itertools import combinations_with_replacement

import numpy as np


def species_layout(labels):
    species, indices = np.unique(np.asarray(labels, dtype=str), return_inverse=True)
    pairs = np.array(list(combinations_with_replacement(range(len(species)), 2)), dtype=int)
    return species, indices, pairs


def combine_partials(spectra, pairs, coherent, incoherent_sq, *, xp=np):
    """Return weighted density contributions and number-current contributions.

    Raw pairs store A_a * conj(A_b), including their imaginary parts. The
    species/pair axis is penultimate; arbitrary leading block/Q axes are allowed.
    AA has factor 1; AB has factor 2 and contributes 2 Re[b_a b_b* P_ab].
    No concentration factors are needed: all raw spectra use total atom count.
    """
    pairs = np.asarray(pairs, dtype=int)
    coherent = xp.asarray(coherent, dtype=complex)
    incoherent_sq = xp.asarray(incoherent_sq, dtype=float)
    if (coherent.ndim != 1 or incoherent_sq.shape != coherent.shape
            or not xp.isfinite(coherent).all() or not xp.isfinite(incoherent_sq).all()
            or xp.any(incoherent_sq < 0)):
        raise ValueError("partial weights must be finite; incoherent squared weights must be nonnegative")
    expected = np.array(list(combinations_with_replacement(range(len(coherent)), 2)))
    if not np.array_equal(pairs, expected):
        raise ValueError("pair indices must contain each unordered species pair exactly once")
    if (spectra["pair_density"].shape[-2] != len(pairs)
            or spectra["species_self"].shape[-2] != len(coherent)):
        raise ValueError("partial spectrum dimensions do not match species and pair weights")
    a, b = pairs.T
    factors = xp.asarray(np.where(a == b, 1., 2.))
    pair_weights = factors * coherent[a] * coherent[b].conj()
    result = {
        "partial_coherent_sqw": (spectra["pair_density"] * pair_weights[:, None]).real,
        "partial_incoherent_sqw": spectra["species_self"] * incoherent_sq[:, None],
        "self_sqw": spectra["species_self"].sum(axis=-2),
    }
    result["coherent_sqw"] = result["partial_coherent_sqw"].sum(axis=-2)
    result["incoherent_sqw"] = result["partial_incoherent_sqw"].sum(axis=-2)
    result["sqw"] = result["coherent_sqw"] + result["incoherent_sqw"]
    for name in ("longitudinal_current", "transverse_current"):
        if "pair_" + name in spectra:
            partial = spectra["pair_" + name].real * factors[:, None]
            result["partial_" + name] = partial
            result[name] = partial.sum(axis=-2)
    return result
