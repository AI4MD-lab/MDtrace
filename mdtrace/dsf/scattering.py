# Copyright 2026 Ting Liang and MDTRACE development team
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bound neutron weights shared by DSF computation and reweighting."""

import numpy as np

from mdtrace.dsf.neutron_scattering_lengths import (
    coherent_scattering_lengths,
    incoherent_scattering_lengths,
)

_ISOTOPE_ALIASES = {"D": "2H", "T": "3H"}


def _scattering_length(table, label: str, component: str) -> complex:
    """Return a finite complex bound scattering length in fm."""

    lookup_label = _ISOTOPE_ALIASES.get(label, label)
    try:
        value = table[lookup_label]
    except KeyError as error:
        raise ValueError(
            f"no {component} neutron scattering length is available for "
            f"atom type '{label}'; use an element or isotope name from the "
            "bound neutron scattering table"
        ) from error

    value = complex(value)
    if not np.isfinite(value):
        raise ValueError(
            f"non-finite {component} neutron scattering length for "
            f"atom type '{label}'"
        )
    return value


def neutron_weights(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map labels to complex coherent b (fm) and real |b_inc|^2 (fm^2)."""

    coherent_by_label = {}
    incoherent_sq_by_label = {}
    for label in dict.fromkeys(labels.tolist()):
        coherent_by_label[label] = _scattering_length(
            coherent_scattering_lengths,
            label,
            "coherent",
        )
        incoherent = _scattering_length(
            incoherent_scattering_lengths,
            label,
            "incoherent",
        )
        incoherent_sq_by_label[label] = abs(incoherent)**2

    coherent = np.asarray(
        [coherent_by_label[label] for label in labels],
        dtype=complex,
    )
    incoherent_sq = np.asarray(
        [incoherent_sq_by_label[label] for label in labels],
        dtype=float,
    )
    return coherent, incoherent_sq
