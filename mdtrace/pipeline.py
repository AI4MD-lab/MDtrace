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


"""DSF workflow and trajectory preparation."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from mdtrace.io.netcdf import is_netcdf
from mdtrace.io.prepare import prepare_trajectory


def _dsf_ready(params, check_settings=False):
    from mdtrace.dsf.dsf import _configuration, _source_signature

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
            if check_settings and metadata["configuration"] != _configuration(params):
                return False
            if check_settings:
                if metadata.get("postprocessing"):
                    return False
                if metadata["source_signature"] != _source_signature(params):
                    return False
            if check_settings:
                return metadata["format_version"] == 2
            return metadata["format_version"] in {1, 2}
    except (OSError, ValueError, KeyError):
        return False


def step_prepare_trajectory(params):
    """Resolve a text trajectory to direct reading or a NetCDF cache."""
    source = Path(params.trajectory_file)
    input_directory = Path(params.input_file).parent
    converted_path = input_directory / f"{source.name}.mdtrace.nc"

    t0 = time.perf_counter()
    prepared, source_format = prepare_trajectory(
        source=source,
        converted_path=converted_path,
        lammps_unit=params.lammps_unit,
        batch_size=params.netcdf_batch_size,
        compression_level=params.netcdf_compression_level,
        text_mode=params.trajectory_read_mode,
    )
    params.source_format = source_format
    params.trajectory_path = str(prepared)
    elapsed = time.perf_counter() - t0
    if prepared == source and is_netcdf(prepared):
        print(f"  ✓  Reading NetCDF trajectory: {prepared}\n")
    elif prepared == source:
        print(f"  ✓  Reading text trajectory directly: {prepared}")
        print("      NetCDF cache         : disabled\n")
    else:
        print(f"  ✓  Trajectory ready: {prepared} ({elapsed:.1f} s)\n")


def step_compute_dsf(params):
    """Compute dynamic structure factor."""
    step_prepare_trajectory(params)
    print("  ▶  Computing DSF ...")
    t0 = time.perf_counter()
    from mdtrace.dsf import compute_dsf, save_dsf
    result = compute_dsf(params)
    save_dsf(result, params)
    elapsed = time.perf_counter() - t0
    print(f"  ✓  DSF compute done ({elapsed:.1f} s)")


def step_plot_dsf(params):
    """Plot the saved density and particle-current spectra."""
    if not _dsf_ready(params):
        raise ValueError("missing or incomplete .dsf.npz archive; run action = compute first")
    from mdtrace.dsf.plot import plot_dsf
    plot_dsf(params)


def step_compute_eels(params):
    """Report the status of the planned EELS method."""

    print("  ERROR: EELS calculation is not implemented yet.")
    sys.exit(1)


STEP_MAP = {
    "dsf": {"compute": step_compute_dsf, "plot": step_plot_dsf},
    "eels": {"compute": step_compute_eels},
}


def run(params):
    """Run DSF computation or plotting; EELS remains a planned method."""
    method = getattr(params, "method", "dsf")
    action = getattr(params, "action", "thinking")
    if method == "sed":
        raise ValueError("SED has been removed from MDtrace; use pySED (pysed)")
    if method not in STEP_MAP:
        raise ValueError(f"Unknown method '{method}'. Supported: {list(STEP_MAP)}")
    steps = STEP_MAP[method]
    if action not in steps and action != "thinking":
        raise ValueError(f"Unknown action '{action}' for {method}. Supported: thinking, {list(steps)}")
    if method == "eels":
        step_compute_eels(params)
    if action == "thinking":
        _run_thinking(params)
    else:
        steps[action](params)
    print("\n  ✅ MDtrace task finished successfully.\n")


def _run_thinking(params):
    """Compute changed DSF inputs, or regenerate a missing/outdated plot."""
    steps = STEP_MAP["dsf"]
    if not _dsf_ready(params, check_settings=True):
        print("  💡 thinking: missing DSF data or changed inputs → compute + plot")
        steps["compute"](params)
        steps["plot"](params)
        return
    figure = Path(params.out_files_name + ".dsf.png")
    if not figure.is_file() or figure.stat().st_mtime_ns < Path(params.out_files_name + ".dsf.npz").stat().st_mtime_ns:
        steps["plot"](params)
        return
    print("  ✅ DSF data and plot are ready. Use action = compute or plot to regenerate them.")
