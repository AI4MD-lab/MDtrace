# Copyright 2026 Ting Liang and MDTRACE development team
# SPDX-License-Identifier: GPL-3.0-or-later
"""Trajectory-independent averaging, scattering weights and resolution.

Every linear operation is applied to retained time blocks before estimating
SEM, preserving covariance across Q points, species and frequency bins.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
from scipy.signal import fftconvolve

from mdtrace.dsf.partials import combine_partials
from mdtrace.dsf.qpoints import average_q_groups, sampling_diagnostics


def load_spectra(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files if key != "metadata_json"}
        result["metadata"] = json.loads(str(archive["metadata_json"]))
    metadata = result["metadata"]
    for key in ("q_mode", "q_axis_label", "q_path_labels", "num_blocks", "block_size", "frames_cropped"):
        result[key] = metadata[key]
    return result


def _blocks(result):
    blocks = {key.removeprefix("qblocks__"): value for key, value in result.items()
              if key.startswith("qblocks__")}
    if not blocks:
        raise ValueError("post-processing requires dsf_save_q_resolved = 1 (or dsf_partials = 1) at computation")
    return blocks


def _summarize(result, blocks, operation):
    from mdtrace.dsf.dsf import _mean_and_sem

    result = dict(result)
    result["metadata"] = deepcopy(result["metadata"])
    for name, values in blocks.items():
        result["qblocks__" + name] = values
        result["qresolved__" + name], result["qresolved__" + name + "_sem"] = _mean_and_sem(values)
        grouped = [average_q_groups(block, result) for block in values]
        result[name], result[name + "_sem"] = _mean_and_sem(grouped)
    result["metadata"]["spectrum_names"] = list(blocks)
    result["metadata"].setdefault("postprocessing", []).append(operation)
    return result


def average_shells(result, edges, *, weights=None):
    """Re-bin stored vectors, without generating/interpolating new directions.

    Edges are in rad/Angstrom. Intervals are left-closed, right-open except
    for the last interval. Q=0 is excluded. Weights describe the desired
    average of the retained vectors; they cannot recover unsampled vectors.
    """
    blocks = _blocks(result)
    edges = np.asarray(edges, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or edges[0] < 0 or np.any(np.diff(edges) <= 0)):
        raise ValueError("shell edges must be finite, nonnegative and strictly increasing")
    q = result["q_cart"]
    weights = np.asarray(result.get("q_weights", np.ones(len(q))) if weights is None else weights, dtype=float)
    if weights.shape != (len(q),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Q weights must be finite and positive, one per stored Q vector")
    norms = np.linalg.norm(q, axis=1)
    selected = (norms > 0) & (norms >= edges[0]) & (norms <= edges[-1])
    if not np.any(selected):
        raise ValueError("no stored nonzero Q vectors fall inside the requested shells")
    norms, weights = norms[selected], weights[selected]
    bins = np.minimum(np.searchsorted(edges, norms, side="right") - 1, len(edges) - 2)
    occupied, groups, counts = np.unique(bins, return_inverse=True, return_counts=True)
    totals = np.bincount(groups, weights=weights)
    out = dict(result)
    out["metadata"] = deepcopy(result["metadata"])
    out.update(q_cart=q[selected], q_box=result["q_box"][selected],
               q_frac=result["q_frac"][selected] if result["q_frac"].size else result["q_frac"],
               q_weights=weights, q_group=groups, q_counts=counts, q_available_counts=counts.copy(),
               q_axis=np.bincount(groups, weights=weights * norms) / totals,
               q_shell_intervals=np.c_[edges[occupied], edges[occupied + 1]],
               q_mode="spherical", q_axis_label="Q magnitude (rad/Angstrom)",
               q_path_ticks=np.empty(0), q_path_labels=[])
    out.update(sampling_diagnostics(out["q_cart"], groups, len(counts), weights))
    if np.any(out["q_coverage_poor"]):
        warnings.warn("rebinned Q shells have sparse or anisotropic directional coverage; inspect Q diagnostics",
                      RuntimeWarning, stacklevel=2)
    out["metadata"].update(q_mode="spherical", q_axis_label=out["q_axis_label"],
                           q_path_labels=[], q_population_scope="retained vectors only, not all allowed box vectors")
    return _summarize(out, {name: values[:, selected] for name, values in blocks.items()},
                      {"operation": "average_shells", "edges_rad_per_A": edges.tolist(),
                       "weighting": "positive per-vector weights saved as q_weights",
                       "population": "retained vectors; changing shells does not undo original subsampling"})


def reweight_scattering(result, *, experiment="neutron", coherent=None, incoherent=None):
    """Reweight unmodified trajectories using species-resolved raw spectra.

    Custom mappings supply complex lengths in fm, not cross sections. Changing
    labels/weights changes scattering contrast only, not isotope masses in MD.
    Currents remain number weighted. For number spectra custom weights are invalid.
    """
    from mdtrace.dsf.dsf import _neutron_weights

    blocks = _blocks(result)
    labels = np.asarray(result["metadata"]["species_labels"])
    if "pair_density" not in blocks or not len(labels):
        raise ValueError("reweighting requires dsf_partials = 1 at computation")
    if experiment == "number":
        if coherent is not None or incoherent is not None:
            raise ValueError("custom scattering lengths require experiment='neutron'")
        b, inc_sq = np.ones(len(labels)), np.zeros(len(labels))
    elif experiment == "neutron":
        if coherent is None and incoherent is None:
            b, inc_sq = _neutron_weights(labels)
        elif coherent is None or incoherent is None:
            raise ValueError("provide both coherent and incoherent custom length mappings")
        else:
            if set(coherent) != set(labels) or set(incoherent) != set(labels):
                raise ValueError("custom mappings must cover exactly the stored species labels")
            b = np.array([coherent[label] for label in labels], dtype=complex)
            inc_sq = abs(np.array([incoherent[label] for label in labels], dtype=complex))**2
    else:
        raise ValueError("experiment must be number or neutron")
    blocks.update(combine_partials(blocks, result["metadata"]["pair_indices"], b, inc_sq))
    out = _summarize(result, blocks, {"operation": "reweight_scattering", "experiment": experiment,
                     "coherent_fm": [[float(x.real), float(x.imag)] for x in b],
                     "incoherent_squared_fm2": inc_sq.tolist()})
    meta = out["metadata"]
    meta["experiment"] = experiment
    if meta["normalization"] == "density":
        unit = "meV" if meta.get("spectral_density_axis") == "energy_mev" else "THz"
        meta["intensity_unit"] = ("fm^2/" if experiment == "neutron" else "1/") + unit
    return out


def apply_resolution(result, fwhm_mev, *, kind="gaussian"):
    """Convolve every block with a discrete unit-area energy-resolution kernel.

    Linear zero-padded convolution is used, never circular wraparound. Weight
    leaving the saved frequency range is lost, not silently renormalized.
    This is classical instrumental broadening, not a quantum correction.
    """
    from mdtrace.dsf.dsf import PLANCK_MEV_PS

    blocks = _blocks(result)
    frequencies = result["freq_thz"]
    if (not np.isfinite(fwhm_mev) or fwhm_mev <= 0 or len(frequencies) < 2
            or kind not in {"gaussian", "lorentzian"}):
        raise ValueError("resolution requires positive finite FWHM, >=2 bins, and gaussian or lorentzian kind")
    steps = np.diff(frequencies)
    if np.any(steps <= 0) or not np.allclose(steps, steps[0]):
        raise ValueError("resolution requires a uniformly spaced, ascending frequency axis")
    de = steps[0] * PLANCK_MEV_PS
    if fwhm_mev < de:
        warnings.warn("resolution FWHM is smaller than one energy bin; use a longer time block",
                      RuntimeWarning, stacklevel=2)
    offsets = np.arange(-(len(frequencies) - 1), len(frequencies)) * de
    if kind == "gaussian":
        kernel = np.exp(-4 * np.log(2) * (offsets / fwhm_mev)**2)
    else:
        kernel = 1 / (1 + (2 * offsets / fwhm_mev)**2)
    kernel /= kernel.sum()
    convolved = {}
    for name, values in blocks.items():
        # Q=0 currents are entirely undefined; preserve that mask explicitly.
        invalid = np.all(np.isnan(values), axis=-1)
        if np.any(np.isnan(values) & ~invalid[..., None]):
            raise ValueError("resolution cannot convolve spectra with partly missing frequency bins")
        clean = np.where(invalid[..., None], 0, values)
        shape = (1,) * (values.ndim - 1) + (len(kernel),)
        convolved[name] = fftconvolve(clean, kernel.reshape(shape), mode="same", axes=-1)
        convolved[name][invalid] = np.nan
    return _summarize(result, convolved, {"operation": "instrument_resolution", "kind": kind,
                      "fwhm_meV": float(fwhm_mev), "boundary": "zero-padded, truncated at saved range"})


def to_energy_density(result):
    """Convert per-THz spectra and uncertainties to per-meV with E=h*f."""
    from mdtrace.dsf.dsf import PLANCK_MEV_PS

    if result["metadata"]["normalization"] != "density":
        raise ValueError("energy-density conversion requires density normalization, not legacy arbitrary units")
    if result["metadata"].get("spectral_density_axis") == "energy_mev":
        return result
    blocks = {name: value / PLANCK_MEV_PS for name, value in _blocks(result).items()}
    out = _summarize(result, blocks, {"operation": "to_energy_density", "h_meV_ps": PLANCK_MEV_PS})
    out["metadata"]["spectral_density_axis"] = "energy_mev"
    out["metadata"]["energy_conversion"] = "spectra are per meV; E=h*f; frequency axis retained as an auxiliary coordinate"
    for key in ("intensity_unit", "self_unit", "current_unit"):
        out["metadata"][key] = out["metadata"][key].replace("/THz", "/meV")
    return out


def main(argv=None):
    from mdtrace.dsf.dsf import save_dsf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--output", required=True, help="new output prefix")
    parser.add_argument("--q-edges", type=float, nargs="+")
    parser.add_argument("--experiment", choices=("number", "neutron"))
    parser.add_argument("--resolution-fwhm-mev", type=float)
    parser.add_argument("--resolution-kind", choices=("gaussian", "lorentzian"), default="gaussian")
    parser.add_argument("--per-mev", action="store_true")
    args = parser.parse_args(argv)
    if Path(args.archive).resolve() == Path(args.output + ".dsf.npz").resolve():
        parser.error("use a different output prefix to preserve the input archive")
    result = load_spectra(args.archive)
    if args.experiment:
        result = reweight_scattering(result, experiment=args.experiment)
    if args.q_edges is not None:
        result = average_shells(result, args.q_edges)
    if args.resolution_fwhm_mev is not None:
        result = apply_resolution(result, args.resolution_fwhm_mev, kind=args.resolution_kind)
    if args.per_mev:
        result = to_energy_density(result)
    save_dsf(result, SimpleNamespace(out_files_name=args.output))


if __name__ == "__main__":
    main()
