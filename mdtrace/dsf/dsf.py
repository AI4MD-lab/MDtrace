#!/usr/bin/env python
# =============================================================================
#     Copyright 2025-2026 Ting Liang and MDTRACE development team
#     This file is part of MDTRACE.
#     MDTRACE is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#     MDTRACE is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#     You should have received a copy of the GNU General Public License
#     along with MDTRACE.  If not, see <http://www.gnu.org/licenses/>.
# =============================================================================

"""DSF and particle-current spectra for periodic crystals and disordered systems.

The original density-spectrum estimator and optional legacy normalization
follow ``pynamic-structure-factor``. The default is now a two-sided spectral
density per THz. Full complex neutron amplitudes and |b_inc|**2 are retained.
Longitudinal/transverse currents use number weights and velocities in A/ps.
These classical spectra are not absolute experimental cross sections.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import warnings

import numpy as np

from mdtrace.io.netcdf import NetCDFReader, is_netcdf
from mdtrace.io.schema import ATOM_IDS, CELLS, POSITIONS, TIMES, TYPES, VELOCITIES
from mdtrace.dsf.qpoints import average_q_groups, build_qpoints, validate_cell
from mdtrace.dsf.spectra import PLANCK_MEV_PS, block_spectra, mean_and_sem
from mdtrace.dsf.scattering import neutron_weights
from mdtrace.dsf.results import calculation_configuration, source_signature, save_dsf
from mdtrace.dsf.partials import species_layout


def _constant_type_ids(raw_types: np.ndarray, n_atoms: int) -> np.ndarray:
    """Validate that each atom keeps one trajectory type across all frames."""

    raw_types = np.asarray(raw_types)
    if raw_types.ndim == 1:
        type_ids = raw_types
    elif raw_types.ndim == 2:
        if raw_types.shape[0] == 0:
            raise ValueError("trajectory contains no atom type frames")
        type_ids = raw_types[0]
        if not np.all(raw_types == type_ids[None, :]):
            raise ValueError(
                "DSF requires each atom's trajectory type ID to remain "
                "constant across frames"
            )
    else:
        raise ValueError(
            "trajectory atom types must have shape (atom,) or (frame, atom)"
        )
    if type_ids.shape != (n_atoms,):
        raise ValueError(
            f"trajectory atom types have shape {type_ids.shape}, expected "
            f"({n_atoms},)"
        )
    return type_ids


def _atom_frame_order(raw_ids: np.ndarray, n_atoms: int, n_frames: int):
    """Align frame-dependent IDs to the first frame, keeping its atom order.

    A static ID array (or an unchanged order) needs no permutation. Without
    trajectory IDs the caller must assume a stable atom order across frames.
    """

    atom_ids = np.asarray(raw_ids)
    if atom_ids.shape not in {(n_atoms,), (n_frames, n_atoms)}:
        raise ValueError(
            "trajectory atom IDs must have shape (atom,) or (frame, atom) "
            "matching the positions"
        )
    if not np.all(np.isfinite(atom_ids)) or not np.all(atom_ids == np.floor(atom_ids)):
        raise ValueError("trajectory atom IDs must be finite integers")
    reference = atom_ids if atom_ids.ndim == 1 else atom_ids[0]
    reference_order = np.argsort(reference)
    sorted_reference = reference[reference_order]
    if np.any(sorted_reference[1:] == sorted_reference[:-1]):
        raise ValueError("trajectory atom IDs must be unique in each frame")
    if atom_ids.ndim == 1 or np.all(atom_ids == reference[None, :]):
        return None

    order = np.argsort(atom_ids, axis=1)
    if not np.all(np.take_along_axis(atom_ids, order, axis=1) == sorted_reference):
        raise ValueError("trajectory atom IDs must contain the same unique set in every frame")
    return order[:, np.argsort(reference_order)]


def _map_atom_labels(
    type_ids: np.ndarray,
    atom_types,
) -> tuple[np.ndarray, dict[object, str]]:
    """Map sorted trajectory type IDs to labels without count assumptions.

    GPUMD text caches number types by species' first appearance in the file.
    """

    if not atom_types:
        raise ValueError("atom_types is required for DSF computation")
    labels = [str(label) for label in atom_types]
    unique_ids = np.unique(type_ids)

    if len(labels) == len(unique_ids):
        mapping = {
            type_id.item() if hasattr(type_id, "item") else type_id: label
            for type_id, label in zip(unique_ids, labels)
        }
        per_atom = np.asarray(
            [mapping[value.item() if hasattr(value, "item") else value]
             for value in type_ids],
            dtype=str,
        )
        return per_atom, mapping

    # Retain the useful legacy form where one label is supplied per atom.
    if len(labels) == len(type_ids):
        per_atom = np.asarray(labels, dtype=str)
        mapping = {}
        for type_id in unique_ids:
            key = type_id.item() if hasattr(type_id, "item") else type_id
            assigned = np.unique(per_atom[type_ids == type_id])
            if len(assigned) != 1:
                raise ValueError(
                    f"trajectory type ID {key!r} maps to multiple atom_types: "
                    f"{assigned.tolist()}"
                )
            mapping[key] = str(assigned[0])
        return per_atom, mapping

    raise ValueError(
        f"atom_types contains {len(labels)} labels, but the trajectory has "
        f"{len(unique_ids)} unique type IDs and {len(type_ids)} atoms; "
        "provide one label per sorted unique type ID (recommended), or one "
        "label per atom"
    )


def _compute_block_spectra(
    positions: np.ndarray,
    q_cart: np.ndarray,
    coherent_weights: np.ndarray,
    incoherent_weights_sq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility entry point for the original pynamic-normalized estimator."""
    result = block_spectra(
        positions, q_cart, coherent_weights, incoherent_weights_sq,
        dt_ps=1.0, normalization="legacy",
    )
    return result["coherent_sqw"], result["incoherent_sqw"]


def _fixed_cell(trajectory, params, frames_used):
    supplied = getattr(params, "dsf_cell", None)
    if supplied is not None:
        supplied = validate_cell(supplied, "dsf_cell")
    if CELLS not in trajectory.info.fields:
        if supplied is None:
            raise ValueError("DSF requires a periodic simulation cell in the trajectory or dsf_cell")
        return supplied
    cells = trajectory.read_cells(slice(0, frames_used))
    reference = validate_cell(cells[0], "trajectory cell")
    if cells.shape != (frames_used, 3, 3) or not np.isfinite(cells).all():
        raise ValueError("trajectory cells must be finite and match the number of frames")
    if not np.allclose(cells, reference, rtol=0, atol=1e-5):
        raise ValueError("DSF currently requires a fixed simulation cell; variable-cell trajectories are unsupported")
    if supplied is not None and not np.allclose(supplied, reference, rtol=0, atol=1e-5):
        raise ValueError("dsf_cell does not match the simulation cell in the trajectory")
    return reference


def compute_dsf(params):
    """Compute classical density and optional number-current spectra.

    Atom IDs align positions, types, and velocities to the first-frame order.
    Without IDs the atom order must already be fixed. All Q points must belong
    to the reciprocal lattice of a fixed periodic simulation cell.
    """
    from mdtrace.parameters import dsf_params, validate_dsf

    validate_dsf(params, compute=True)
    option = lambda name: getattr(params, name, dsf_params[name].default)
    trajectory_path = Path(params.trajectory_path)
    if not is_netcdf(trajectory_path):
        raise ValueError("DSF requires a NetCDF trajectory/cache; set trajectory_read_mode = cache")
    num_blocks = int(params.num_blocks)
    if num_blocks < 1 or num_blocks != params.num_blocks:
        raise ValueError("num_blocks must be a positive integer")
    time_step = float(params.time_step)
    stride = float(params.output_data_stride)
    if not np.isfinite(time_step) or time_step <= 0:
        raise ValueError("time_step must be finite and positive")
    if not np.isfinite(stride) or stride < 1 or not stride.is_integer():
        raise ValueError("output_data_stride must be a positive integer")
    dt_ps = time_step * stride / 1000.0
    if not np.isfinite(dt_ps) or dt_ps <= 0:
        raise ValueError("time_step * output_data_stride must be finite and positive")

    blocks, q_blocks = {}, {}
    keep_q = option("dsf_save_q_resolved") or option("dsf_partials")
    species, pairs = np.empty(0, dtype=str), np.empty((0, 2), dtype=int)
    with NetCDFReader(trajectory_path) as trajectory:
        trajectory.require_current_cache()
        trajectory.require(POSITIONS)
        if option("dsf_currents"):
            if VELOCITIES not in trajectory.info.fields:
                raise ValueError("dsf_currents = 1 requires velocities saved alongside positions")
            trajectory.require(VELOCITIES)
        n_frames, n_atoms = trajectory.info.n_frames, trajectory.info.n_atoms
        if n_atoms == 0:
            raise ValueError("DSF requires at least one atom")
        if n_frames < num_blocks:
            raise ValueError(f"trajectory has {n_frames} frames, fewer than num_blocks ({num_blocks})")
        block_size = n_frames // num_blocks
        frames_used = block_size * num_blocks
        frames_cropped = n_frames - frames_used
        if frames_cropped:
            warnings.warn(f"cropping the final {frames_cropped} trajectory frame(s) to form equal DSF blocks",
                          RuntimeWarning, stacklevel=2)
        cell = _fixed_cell(trajectory, params, frames_used)
        sampling = build_qpoints(params, cell)
        if sampling["q_mode"] == "spherical" and np.any(sampling["q_coverage_poor"]):
            count = int(np.count_nonzero(sampling["q_coverage_poor"]))
            warnings.warn(
                f"{count} Q shell(s) have sparse or anisotropic directional coverage; "
                "inspect .dsf_Qdiagnostics and check cell size/shape and sampling convergence. "
                "A larger sampling cap cannot create missing reciprocal-box directions.",
                RuntimeWarning, stacklevel=2,
            )
        if TIMES in trajectory.info.fields:
            times = trajectory.read(TIMES, slice(0, frames_used))
            if (times.shape != (frames_used,) or not np.isfinite(times).all()
                    or not np.allclose(np.diff(times), dt_ps, rtol=1e-5, atol=1e-8)):
                raise ValueError("trajectory times must be uniformly spaced and match time_step * output_data_stride")

        atom_order = None
        if ATOM_IDS in trajectory.info.fields:
            atom_order = _atom_frame_order(
                trajectory.read(ATOM_IDS, slice(0, frames_used)), n_atoms, frames_used,
            )
        type_mapping, atom_counts = {}, {}
        if option("experiment") == "neutron" or option("atom_types") is not None or option("dsf_partials"):
            if TYPES not in trajectory.info.fields:
                raise ValueError("neutron weighting, partial spectra or atom_types mapping requires trajectory type IDs")
            trajectory.require(TYPES)
            raw_types = trajectory.read(TYPES, slice(0, frames_used))
            if raw_types.ndim == 2:
                if raw_types.shape != (frames_used, n_atoms):
                    raise ValueError("trajectory atom type frames must match the positions")
                if atom_order is not None:
                    raw_types = np.take_along_axis(raw_types, atom_order, axis=1)
            type_ids = _constant_type_ids(raw_types, n_atoms)
            type_labels = option("atom_types")
            if type_labels is None:
                type_labels = [f"type_{value}" for value in np.unique(type_ids)]
            labels, type_mapping = _map_atom_labels(type_ids, type_labels)
            atom_counts = dict(Counter(labels.tolist()))
            if option("dsf_partials"):
                species, _, pairs = species_layout(labels)
        if option("experiment") == "neutron":
            coherent_weights, incoherent_weights_sq = neutron_weights(labels)
        else:
            coherent_weights = np.ones(n_atoms, dtype=complex)
            incoherent_weights_sq = np.zeros(n_atoms)

        for block_index in range(num_blocks):
            start = block_index * block_size
            frames = slice(start, start + block_size)
            positions = trajectory.read(POSITIONS, frames)
            # Canonical trajectory velocities are m/s. Use A/ps with positions
            # in A and time in ps, so omega**2*S = Q**2*C_L has consistent units.
            velocities = trajectory.read(VELOCITIES, frames) / 100.0 if option("dsf_currents") else None
            if atom_order is not None:
                order = atom_order[frames, :, None]
                positions = np.take_along_axis(positions, order, axis=1)
                if velocities is not None:
                    velocities = np.take_along_axis(velocities, order, axis=1)
            spectra = block_spectra(
                positions, sampling["q_cart"], coherent_weights, incoherent_weights_sq,
                dt_ps=dt_ps, velocities=velocities,
                normalization=option("dsf_normalization"), window=option("dsf_window"),
                subtract_mean=option("dsf_subtract_mean"),
                species_labels=labels if option("dsf_partials") else None,
            )
            for name, values in spectra.items():
                blocks.setdefault(name, []).append(average_q_groups(values, sampling))
                if keep_q:
                    q_blocks.setdefault(name, []).append(values)

    result = dict(sampling)
    for name, values in blocks.items():
        result[name], result[name + "_sem"] = mean_and_sem(values)
    for name, values in q_blocks.items():
        result["qblocks__" + name] = np.stack(values)
        result["qresolved__" + name], result["qresolved__" + name + "_sem"] = mean_and_sem(values)
    freq_thz = np.fft.fftshift(np.fft.fftfreq(block_size, d=dt_ps))
    result.update(freq_thz=freq_thz, energy_mev=PLANCK_MEV_PS * freq_thz,
                  num_blocks=num_blocks, block_size=block_size,
                  frames_cropped=frames_cropped, type_mapping=type_mapping)
    density = option("dsf_normalization") == "density"
    intensity_unit = "fm^2/THz" if option("experiment") == "neutron" else "1/THz"
    result["metadata"] = {
        "format_version": 2, "configuration": calculation_configuration(params),
        "source_signature": source_signature(params),
        "trajectory": str(trajectory_path.resolve()), "n_atoms": n_atoms,
        "frames_used": frames_used, "frames_cropped": frames_cropped,
        "num_blocks": num_blocks, "block_size": block_size, "dt_ps": dt_ps,
        "frequency_resolution_THz": 1.0 / (block_size * dt_ps),
        "nyquist_THz": 0.5 / dt_ps, "cell_A": cell.tolist(),
        "experiment": option("experiment"), "normalization": option("dsf_normalization"),
        "window": option("dsf_window"), "subtract_mean": bool(option("dsf_subtract_mean")),
        "intensity_unit": intensity_unit if density else "legacy arbitrary units",
        "self_unit": "1/THz" if density else "legacy arbitrary units",
        "current_unit": "(Angstrom/ps)^2/THz" if density else "legacy current units",
        "currents": bool(option("dsf_currents")), "current_weighting": "number",
        "transverse_convention": "sum of two transverse polarizations",
        "zero_Q_current": "undefined (NaN)",
        "frequency_convention": "two-sided, ordinary frequency f in THz; no positive-frequency doubling",
        "spectral_density_axis": "frequency_thz",
        "energy_conversion": "spectra remain per THz; E=h*f; divide spectral densities by h=4.135667696 meV*ps for per-meV units",
        "q_mode": sampling["q_mode"], "q_axis_label": sampling["q_axis_label"],
        "q_path_labels": sampling["q_path_labels"],
        "q_fractional_basis": "prim_unitcell" if sampling["q_frac"].size else None,
        "q_rows": "q_group maps individual Q vectors to output rows; spherical rows average spectra",
        "sem_convention": "standard error across non-overlapping time blocks after direction averaging",
        "sem_estimated": num_blocks > 1,
        "sem_limits": "one block: NaN; does not include angular sampling uncertainty or block autocorrelation",
        "complex_sem": "RMS complex deviation across blocks; real/imaginary covariance is retained in qblocks",
        "q_resolved": bool(keep_q), "partials": bool(option("dsf_partials")),
        "spectrum_names": list(blocks),
        "species_labels": species.tolist(), "pair_indices": pairs.tolist(),
        "pair_convention": "raw A_a*conj(A_b)/N; AA once, AB contribution 2*Re(b_a*conj(b_b)*P_ab)",
        "partial_axes": "Q, pair_or_species, frequency; qblocks add leading time_block axis",
        "q_coverage_heuristic": "poor if fewer than 6 unoriented directions or second-moment eigenvalues outside [0.1,0.6]; not a convergence certificate",
        "q_population_scope": "all reciprocal-box vectors in each original shell",
        "atom_type_mapping": {str(key): value for key, value in type_mapping.items()},
        "atom_counts": atom_counts,
    }
    print(f"  Trajectory: {n_frames} frames x {n_atoms} atoms")
    print(f"  DSF blocks: {num_blocks} x {block_size} frames ({frames_cropped} cropped)")
    if type_mapping:
        print(f"  Atom type mapping: {type_mapping}; counts: {atom_counts}")
    print(f"  Q sampling: {sampling['q_mode']}, {len(sampling['q_cart'])} vectors -> {len(sampling['q_axis'])} output rows")
    print(f"  Spectra: {option('experiment')}, {option('dsf_normalization')} normalization, {len(freq_thz)} frequency bins")
    if option("dsf_currents") and np.any(np.linalg.norm(sampling["q_cart"], axis=1) == 0):
        print("  Longitudinal/transverse currents at Q=0 are undefined and saved as NaN.")
    if num_blocks == 1:
        print("  One time block: SEM cannot be estimated and is saved as NaN.")
    return result
