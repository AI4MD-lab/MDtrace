# Copyright 2026 Ting Liang and MDTRACE development team
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plain-text DSF output, loading and computation-readiness checks."""

import hashlib
import json
from itertools import islice
from pathlib import Path

import numpy as np
from pySED.console import print_detail, print_section

from mdtrace.dsf.spectra import PLANCK_MEV_PS


def calculation_configuration(params):
    """Serializable numerical settings for reproducibility and thinking mode."""
    from mdtrace.parameters import dsf_params, get_density_weighting

    values = {name: getattr(params, name, definition.default)
              for name, definition in dsf_params.items()
              if not name.startswith("dsf_plot_") and name != "if_show_figures"}
    # Keep the saved configuration comparable to existing text results: a
    # spelling change must not make thinking mode recompute the same spectra.
    values.pop("density_weighting")
    values.pop("current_weighting")  # Only equal is implemented.
    values["experiment"] = "number" if get_density_weighting(params) == "equal" else "neutron"
    for name in ("time_step", "output_data_stride", "num_blocks", "prim_unitcell"):
        values[name] = getattr(params, name, None)
    values["lammps_unit"] = getattr(params, "lammps_unit", "metal")
    return {name: value.tolist() if isinstance(value, np.ndarray) else value
            for name, value in values.items()}


def source_signature(params):
    source = Path(getattr(params, "trajectory_file", getattr(params, "trajectory_path", ""))).resolve()
    stat = source.stat()
    return {"path": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


_SPECTRUM_FILES = {
    "sqw": (".dsf", "S"), "coherent_sqw": (".dsf_coherent", "S_coh"),
    "incoherent_sqw": (".dsf_incoherent", "S_inc"), "self_sqw": (".dsf_self", "S_self"),
    "longitudinal_current": (".dsf_CL", "CL"), "transverse_current": (".dsf_CT", "CT"),
    "total_current": (".dsf_Ctotal", "Ctotal"),
}


def _run_id(configuration, source):
    record = json.dumps([configuration, source], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(record.encode()).hexdigest()


def _shell_intervals(q_axis, grid):
    low, high, width = grid
    shells = np.minimum(np.floor(q_axis / width + 1e-10), np.ceil(high / width) - 1)
    return np.column_stack((np.maximum(low, shells * width),
                            np.minimum(high, (shells + 1) * width)))


def _compact_footer(result, spectra, keep_blocks):
    meta = result["metadata"]
    sampling = meta["q_mode"]
    lines = [f"Q sampling: {sampling}"]
    if sampling == "spherical":
        grid = meta.get("shell_grid")
        if grid is None and not meta.get("postprocessing"):
            config = meta["configuration"]
            grid = [config["dsf_q_min"], config["dsf_q_max"], config["dsf_q_bin_width"]]
        if grid is not None:
            lines.append("Shell grid (min max width): " + " ".join(map(str, grid)))
        # Preserve exact edges even for a mean Q rounded across a bin boundary.
        if grid is None or not np.array_equal(_shell_intervals(result["q_axis"], grid), result["q_shell_intervals"]):
            lines.append("Shell bounds: " + json.dumps(result["q_shell_intervals"].tolist()))
    elif sampling == "path":
        ticks = list(zip(result["q_path_ticks"].tolist(), meta["q_path_labels"], strict=True))
        lines.append("Path ticks: " + json.dumps(ticks, ensure_ascii=False))
    weighting = "neutron" if meta["experiment"] == "neutron" else "equal"
    lines += [f"Density: {weighting} {meta['normalization']}",
              "Spectra: " + " ".join(spectra.values()),
              f"Blocks (count frames cropped): {meta['num_blocks']} {meta['block_size']} {meta['frames_cropped']}"]
    if meta.get("spectral_density_axis") == "energy_mev":
        lines.append("Spectral density axis: energy_mev")
    if keep_blocks:
        lines.append("Per-Q data: .dsf_blocks")
    run_id = meta.get("run_id")
    if run_id is None:
        run_id = "-" if meta.get("postprocessing") else _run_id(meta["configuration"], meta["source_signature"])
    lines.append("Run ID: " + run_id)
    return "\n".join(lines)


def _read_header(prefix):
    notes = {}
    with open(str(prefix) + ".dsf_Qaxis", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("# {"):  # Older files stored their full record here.
                header = json.loads(line[2:])
                if header["text_format_version"] not in (1, 2):
                    raise ValueError("unsupported DSF text format; recompute with this version")
                return header
            if line.startswith("# "):
                key, sep, value = line[2:].strip().partition(": ")
                if sep:
                    notes[key] = value
                elif key.endswith("Available"):
                    notes["axis"] = key.rsplit(None, 2)[0]
    if "Run ID" not in notes:
        raise ValueError("missing DSF text metadata")
    weighting, normalization = notes["Density"].split()
    blocks, size, cropped = map(int, notes["Blocks (count frames cropped)"].split())
    suffixes = notes["Spectra"].split()
    spectra = {name + extra: suffix + extra for name, (suffix, _) in _SPECTRUM_FILES.items()
               for extra in ("", "_sem") if suffix + extra in suffixes}
    meta = dict(q_mode=notes["Q sampling"], q_axis_label=notes["axis"], q_path_labels=[],
                experiment="neutron" if weighting == "neutron" else "number",
                density_weighting=weighting, normalization=normalization, current_weighting="equal",
                currents="longitudinal_current" in spectra,
                spectral_density_axis=notes.get("Spectral density axis", "frequency_thz"),
                num_blocks=blocks, block_size=size, frames_cropped=cropped, run_id=notes["Run ID"])
    meta.update(spectrum_units(weighting, normalization, meta["spectral_density_axis"]))
    axes = {}
    if meta["q_mode"] == "path":
        ticks, labels = zip(*json.loads(notes["Path ticks"]))
        axes["q_path_ticks"] = np.asarray(ticks)
        meta["q_path_labels"] = list(labels)
    elif meta["q_mode"] == "spherical":
        if "Shell grid (min max width)" in notes:
            meta["shell_grid"] = list(map(float, notes["Shell grid (min max width)"].split()))
        if "Shell bounds" in notes:
            axes["q_shell_intervals"] = np.asarray(json.loads(notes["Shell bounds"]))
    return {"metadata": meta, "spectra": spectra, "blocks": "Per-Q data" in notes, "axes": axes, "q_arrays": {}}


def spectrum_units(weighting, normalization, axis="frequency_thz"):
    density = normalization == "density"
    unit = "meV" if axis == "energy_mev" else "THz"
    intensity = ("fm^2/" if weighting == "neutron" else "1/") + unit
    return {"intensity_unit": intensity if density else "legacy arbitrary units",
            "self_unit": "1/" + unit if density else "legacy arbitrary units",
            "current_unit": "(Å/ps)^2/" + unit if density else "legacy current units"}


def _spectrum_names(metadata, components=False, *, density=True, currents=True, for_plot=False):
    names = []
    if density:
        names.append("sqw")
        if components or (not for_plot and metadata["experiment"] == "neutron"):
            names += ["coherent_sqw", "incoherent_sqw"]
        if components:
            names.append("self_sqw")
    if currents and metadata["currents"]:
        names += ["longitudinal_current", "transverse_current"]
    return names


def _result_files(prefix, header, include_blocks, **selection):
    names = header["spectra"] if include_blocks else _spectrum_names(header["metadata"], for_plot=True, **selection)
    suffixes = [".dsf_THz", ".dsf_Qaxis", *(header["spectra"][name] for name in names)]
    if include_blocks and header["blocks"]:
        suffixes.append(".dsf_blocks")
    return [Path(str(prefix) + suffix) for suffix in suffixes]


def result_files(prefix, *, include_blocks=False, **selection):
    """Plot inputs, or all saved outputs when include_blocks is requested."""
    return _result_files(prefix, _read_header(prefix), include_blocks, **selection)


def spectra_ready(params, check_settings=False):
    """Check required text files and optionally their calculation settings."""
    try:
        header = _read_header(params.out_files_name)
        metadata = header["metadata"]
        selection = {} if check_settings else plot_selection(params)
        if not set(_spectrum_names(metadata, for_plot=not check_settings, **selection)).issubset(header["spectra"]):
            return False
        if not all(path.is_file() and path.stat().st_size for path in
                   _result_files(params.out_files_name, header, check_settings, **selection)):
            return False
        if check_settings:
            if "run_id" in metadata:
                return metadata["run_id"] == _run_id(calculation_configuration(params), source_signature(params))
            return (not metadata.get("postprocessing")
                    and metadata["configuration"] == calculation_configuration(params)
                    and metadata["source_signature"] == source_signature(params))
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def plot_selection(params):
    return dict(density=getattr(params, "dsf_plot_density", True),
                currents=getattr(params, "dsf_plot_currents", True))


def plot_files(params, metadata=None):
    """One PNG per selected spectrum available in the saved calculation."""
    prefix = str(params.out_files_name)
    if metadata is None:
        metadata = _read_header(prefix)["metadata"]
    names = _spectrum_names(metadata, for_plot=True, **plot_selection(params))
    if "longitudinal_current" in names:
        names.insert(names.index("longitudinal_current"), "total_current")
    return {name: Path(prefix + _SPECTRUM_FILES[name][0] + ".png") for name in names}


def plot_has_removed_dc(params, metadata):
    """Check whether the saved, unbroadened spectrum had its block mean removed."""
    if metadata.get("postprocessing") or metadata.get("run_id") == "-":
        return False
    if "subtract_mean" in metadata:
        return metadata["subtract_mean"]
    if not metadata.get("run_id"):
        return False
    # Compact text records the calculation hash rather than the individual flags.
    # Only mask DC if that hash confirms the settings used to produce the spectrum.
    try:
        source = source_signature(params)
    except OSError:
        return False
    config = calculation_configuration(params)
    config["dsf_subtract_mean"] = True
    return _run_id(config, source) == metadata["run_id"]


def plot_signature(params):
    from mdtrace.parameters import dsf_params

    options = {name: getattr(params, name, parameter.default)
               for name, parameter in dsf_params.items() if name.startswith("dsf_plot_")}
    return json.dumps({"layout": "single-spectrum-v9", **options}, sort_keys=True)


def load_spectra(prefix, *, for_plot=False, **selection):
    """Read a result prefix directly from text; plotting skips optional blocks."""
    header = _read_header(prefix)
    fields = (set(_spectrum_names(header["metadata"], for_plot=True, **selection)) | {"q_shell_intervals", "q_path_ticks"}
              if for_plot else None)
    result = {"metadata": header["metadata"], **header.get("axes", {})}
    qtable = np.loadtxt(str(prefix) + ".dsf_Qaxis", ndmin=2, encoding="utf-8")
    if not len(qtable) or qtable.shape[1] != 3 or not np.isfinite(qtable).all():
        raise ValueError("DSF Q-axis table must contain three finite columns")
    result.update(q_axis=qtable[:, 0], q_counts=qtable[:, 1].astype(int),
                  q_available_counts=qtable[:, 2].astype(int))
    if "shell_grid" in result["metadata"] and "q_shell_intervals" not in result:
        result["q_shell_intervals"] = _shell_intervals(result["q_axis"], result["metadata"]["shell_grid"])
    result["freq_thz"] = np.loadtxt(str(prefix) + ".dsf_THz", ndmin=1)
    if (result["freq_thz"].ndim != 1 or not len(result["freq_thz"])
            or not np.isfinite(result["freq_thz"]).all()
            or np.any(np.diff(result["freq_thz"]) <= 0)):
        raise ValueError("DSF frequency axis must be finite and strictly increasing")
    result["energy_mev"] = PLANCK_MEV_PS * result["freq_thz"]
    expected = (len(result["freq_thz"]), len(result["q_axis"]))
    for name, suffix in header["spectra"].items():
        if fields is None or name in fields:
            table = np.loadtxt(str(prefix) + suffix, ndmin=2, encoding="utf-8")
            if table.shape != expected:
                raise ValueError(f"{prefix}{suffix}: expected frequency/Q shape {expected}, got {table.shape}")
            result[name] = table.T
    if not for_plot and result["metadata"]["currents"] and "total_current" not in result:
        # Older exports store just CL and CT.
        result["total_current"] = result["longitudinal_current"] + result["transverse_current"]
    if fields is None and header["blocks"]:
        with open(str(prefix) + ".dsf_blocks", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("# {"):
                    detail = json.loads(line[2:])
                    result["metadata"] = detail["metadata"]
                    header["q_arrays"] = detail["q_arrays"]
                    continue
                name, dtype, shape = json.loads(line.removeprefix("# "))
                rows = int(np.prod(shape[:-1]))
                values = np.loadtxt(islice(stream, rows), dtype=dtype, ndmin=2)
                result[name] = values.reshape(shape)
    for name, spec in header["q_arrays"].items():
        if fields is None or name in fields:
            result[name] = np.asarray(spec["data"], dtype=spec["dtype"]).reshape(spec["shape"])
    metadata = result["metadata"]
    # Older Q labels use rad/Angstrom or rad/Å for the same wavevector.
    metadata["q_axis_label"] = metadata["q_axis_label"].replace("Angstrom", "Å").replace("rad/Å", "Å⁻¹")
    if "current_unit" in metadata:
        metadata["current_unit"] = metadata["current_unit"].replace("Angstrom", "Å")
    for key in ("q_mode", "q_axis_label", "q_path_labels", "num_blocks", "block_size", "frames_cropped"):
        result[key] = metadata[key]
    return result


def _write_spectrum(prefix, name, values, metadata, extra=""):
    suffix, symbol = _SPECTRUM_FILES[name]
    axis = "E" if metadata.get("spectral_density_axis") == "energy_mev" else "f"
    unit_key = ("current_unit" if name.endswith("current") else
                "self_unit" if name == "self_sqw" else "intensity_unit")
    title = ("Standard error of " if extra else "") + f"{symbol}(Q,{axis})"
    header = [f"{title} [{metadata[unit_key]}]"]
    if unit_key != "current_unit":
        weighting = ("neutron weights" if metadata["experiment"] == "neutron"
                     and unit_key == "intensity_unit" else "equal atom weights")
        header[0] += f"; {weighting}"
    if name == "transverse_current":
        header.append("Sum of two transverse components")
    if name == "total_current":
        header.append("Ctotal = CL + CT")
    header.append("Rows: frequency (.dsf_THz); columns: Q (.dsf_Qaxis)")
    np.savetxt(prefix + suffix + extra, values.T, fmt="%.18e", header="\n".join(header), encoding="utf-8")
    return suffix + extra


def save_dsf(result, params):
    """Save frequency rows/Q columns; optional detailed data remain plain text."""
    prefix = str(params.out_files_name)
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    metadata = result["metadata"]
    components = getattr(params, "dsf_output_components", False)
    sem = getattr(params, "dsf_output_sem", False)
    names = _spectrum_names(metadata, components)
    spectra = {}
    for name in names:
        for extra in (("", "_sem") if sem else ("",)):
            key = name + extra
            spectra[key] = _write_spectrum(prefix, name, result[key], metadata, extra)
    if metadata["currents"]:
        total = result["longitudinal_current"] + result["transverse_current"]
        spectra["total_current"] = _write_spectrum(prefix, "total_current", total, metadata)
    np.savetxt(prefix + ".dsf_THz", result["freq_thz"], fmt="%.18e",
               header="Frequency [THz]")
    keep_blocks = metadata.get("q_resolved") or metadata.get("partials")
    if keep_blocks:
        q_arrays = {name: {"data": value.tolist(), "shape": value.shape, "dtype": str(value.dtype)}
                    for name, value in result.items()
                    if name.startswith("q_") and name not in {"q_axis", "q_counts", "q_available_counts"}
                    and isinstance(value, np.ndarray)}
        # Only explicit advanced-output requests write this file. Section headers
        # identify array name/dtype/shape; the following rows are ordinary text.
        excluded = set(q_arrays) | {"q_axis", "q_counts", "q_available_counts", "freq_thz", "energy_mev"}
        with open(prefix + ".dsf_blocks", "w", encoding="utf-8") as stream:
            stream.write("# " + json.dumps({"metadata": metadata, "q_arrays": q_arrays},
                                           separators=(",", ":")) + "\n")
            for name, value in result.items():
                if isinstance(value, np.ndarray) and name not in excluded and name not in spectra:
                    stream.write("# " + json.dumps([name, str(value.dtype), value.shape]) + "\n")
                    np.savetxt(stream, value.reshape(-1, value.shape[-1]), fmt="%.18e")
    count_digits = len(str(int(np.max(result["q_available_counts"]))))
    used_width = max(len("Used"), count_digits)
    available_width = max(len("Available"), count_digits)
    # Write the index last so incomplete first-time outputs are not considered ready.
    np.savetxt(prefix + ".dsf_Qaxis",
               np.c_[result["q_axis"], result["q_counts"], result["q_available_counts"]],
               fmt=["  %-25.18e", f"%{used_width}d", f"%{available_width}d"], delimiter="  ",
               header=f"{result['q_axis_label']:<25}  {'Used':>{used_width}}  {'Available':>{available_width}}",
               encoding="utf-8",
               footer=_compact_footer(result, spectra, keep_blocks))
    print_section("Saved DSF text")
    print_detail("Output prefix", prefix)
    print_detail("Spectra", ", ".join(spectra.values()))
    print_detail("Axes", ".dsf_THz, .dsf_Qaxis")
    if keep_blocks:
        print_detail("Detailed blocks", ".dsf_blocks")
