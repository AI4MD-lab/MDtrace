# Copyright 2026 Ting Liang and MDTRACE development team
# SPDX-License-Identifier: GPL-3.0-or-later
"""DSF result metadata, archive validation, loading and output files.

This module never computes spectra or imports a trajectory reader. Its writer
preserves the existing format and filenames; loading needs only the archive.
"""

import json
from pathlib import Path

import numpy as np


def calculation_configuration(params):
    """Serializable numerical settings for reproducibility and thinking mode."""
    from mdtrace.parameters import dsf_params

    values = {name: getattr(params, name, definition.default)
              for name, definition in dsf_params.items()
              if not name.startswith("dsf_plot_") and name != "if_show_figures"}
    for name in ("time_step", "output_data_stride", "num_blocks", "prim_unitcell"):
        values[name] = getattr(params, name, None)
    values["lammps_unit"] = getattr(params, "lammps_unit", "metal")
    return {name: value.tolist() if isinstance(value, np.ndarray) else value
            for name, value in values.items()}


def source_signature(params):
    source = Path(getattr(params, "trajectory_file", getattr(params, "trajectory_path", ""))).resolve()
    stat = source.stat()
    return {"path": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def spectra_ready(params, check_settings=False):
    """Check whether an archive is complete and optionally matches this run."""
    try:
        with np.load(params.out_files_name + ".dsf.npz", allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata_json"]))
            required = {"sqw", "coherent_sqw", "incoherent_sqw", "self_sqw", "freq_thz", "q_axis"}
            if metadata["currents"]:
                required.update(("longitudinal_current", "transverse_current"))
            if metadata.get("q_resolved"):
                required.update("qblocks__" + name for name in metadata["spectrum_names"])
            if metadata.get("partials"):
                required.update(("pair_density", "species_self"))
            if not required.issubset(saved.files):
                return False
            if check_settings and metadata["configuration"] != calculation_configuration(params):
                return False
            if check_settings:
                if metadata.get("postprocessing"):
                    return False
                if metadata["source_signature"] != source_signature(params):
                    return False
            if check_settings:
                return metadata["format_version"] == 2
            return metadata["format_version"] in {1, 2}
    except (OSError, ValueError, KeyError):
        return False


def load_spectra(path, *, fields=None):
    """Load all arrays, or only stored arrays named in fields, plus metadata.

    Restricting fields avoids decompressing block/partial data when plotting.
    The default retains the complete archive for numerical post-processing.
    """
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files
                  if key != "metadata_json" and (fields is None or key in fields)}
        result["metadata"] = json.loads(str(archive["metadata_json"]))
    metadata = result["metadata"]
    for key in ("q_mode", "q_axis_label", "q_path_labels", "num_blocks", "block_size", "frames_cropped"):
        result[key] = metadata[key]
    return result


_SPECTRUM_FILES = {
    "sqw": ".dsf", "coherent_sqw": ".dsf_coherent", "incoherent_sqw": ".dsf_incoherent",
    "self_sqw": ".dsf_self", "longitudinal_current": ".dsf_CL", "transverse_current": ".dsf_CT",
}


def save_dsf(result, params):
    """Save spectra, Q mappings, metadata and a self-contained plotting archive."""
    prefix = str(params.out_files_name)
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    metadata = result["metadata"]
    energy_density = metadata.get("spectral_density_axis") == "energy_mev"
    convention = (("two-sided density per meV" if energy_density else "two-sided density per THz") if metadata["normalization"] == "density"
                  else "pynamic-compatible arbitrary units; FFT norm='forward'; divided by N_atoms*N_frames_per_block")
    for name, suffix in _SPECTRUM_FILES.items():
        if name not in result:
            # Recomputing without currents must not leave stale current files.
            for extra in (suffix, suffix + "_sem"):
                Path(prefix + extra).unlink(missing_ok=True)
            continue
        unit_key = ("current_unit" if name.endswith("current") else
                    "self_unit" if name == "self_sqw" else "intensity_unit")
        description = ("number-weighted particle current" if name.endswith("current")
                       else "number self spectrum" if name == "self_sqw"
                       else metadata["experiment"] + " density spectrum")
        weighting = ("complex coherent b; incoherent |b|^2"
                     if metadata["experiment"] == "neutron" and unit_key == "intensity_unit"
                     else "unit particle weights")
        for extra, key in (("", name), ("_sem", name + "_sem")):
            np.savetxt(prefix + suffix + extra, result[key], fmt="%.18e", header=(
                f"{description}: {key}; units={metadata[unit_key]}; {convention}; "
                f"{weighting}; rows=Q groups, columns=frequency bins; "
                "see .dsf_metadata.json and .dsf_Qmap"
            ))
    for suffix, key, label in (
        (".dsf_THz", "freq_thz", "ordinary frequency (THz), fftshift order"),
        (".dsf_meV", "energy_mev", "energy transfer (meV); " + metadata["energy_conversion"]),
        (".dsf_Qcart", "q_cart", "individual Q vectors (rad/Angstrom), not shell means"),
        (".dsf_Qfrac", "q_frac", "individual Q vectors in primitive reciprocal coordinates; empty without prim_unitcell"),
        (".dsf_Qbox", "q_box", "individual Q vectors in simulation-box reciprocal coordinates (integers)"),
    ):
        np.savetxt(prefix + suffix, result[key], fmt="%.18e", header=label)
    np.savetxt(prefix + ".dsf_Qmap", np.c_[np.arange(len(result["q_group"])), result["q_group"]],
               fmt="%d", header="individual_Q_index output_row (zero-based)")
    np.savetxt(prefix + ".dsf_Qaxis",
               np.c_[result["q_axis"], result["q_counts"], result["q_available_counts"]],
               fmt=["%.18e", "%d", "%d"],
               header=result["q_axis_label"] + "; sampled_vectors; available_vectors")
    diagnostics_path = Path(prefix + ".dsf_Qdiagnostics")
    if "q_direction_counts" in result:
        np.savetxt(diagnostics_path, np.c_[result["q_axis"], result["q_sampled_range"],
                   result["q_magnitude_std"], result["q_direction_counts"],
                   result["q_direction_eigenvalues"], result["q_effective_count"],
                   result["q_coverage_poor"].astype(int)],
                   header="Qmean Qmin Qmax Qstd unoriented_directions moment_eigenvalues(3) effective_weight_count poor_coverage_flag")
    else:
        diagnostics_path.unlink(missing_ok=True)
    Path(prefix + ".dsf_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(prefix + ".dsf.npz", **arrays, metadata_json=json.dumps(metadata))
    print(f"  Saved DSF spectra and metadata: {prefix}.dsf*, archive: {prefix}.dsf.npz")
