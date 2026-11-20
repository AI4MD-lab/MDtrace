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
import sys
import time
from pathlib import Path

from pySED.console import print_section

from mdtrace.dsf.results import plot_files, plot_selection, plot_signature, spectra_ready as _dsf_ready, result_files
from mdtrace.io.prepare import default_converted_path, prepare_trajectory


def step_prepare_trajectory(params):
    """Select NetCDF input or convert text to a checked NetCDF cache."""
    source = Path(params.trajectory_file)
    input_directory = Path(params.input_file).parent
    converted_path = default_converted_path(input_directory / source.name)

    t0 = time.perf_counter()
    prepared, _ = prepare_trajectory(
        source=source,
        converted_path=converted_path,
        lammps_unit=params.lammps_unit,
        batch_size=params.netcdf_batch_size,
        compression_level=params.netcdf_compression_level,
        text_mode=params.trajectory_read_mode,
    )
    params.trajectory_path = str(prepared)
    elapsed = time.perf_counter() - t0
    if prepared == source:
        print(f"\n    ✓  Reading NetCDF trajectory: {prepared}")
    else:
        print(f"    ✓  Trajectory ready: {prepared} ({elapsed:.1f} s)")


def step_compute_dsf(params):
    """Compute dynamic structure factor."""
    step_prepare_trajectory(params)
    print_section("Computing DSF", "🚀")
    t0 = time.perf_counter()
    from mdtrace.dsf import compute_dsf, save_dsf
    result = compute_dsf(params)
    save_dsf(result, params)
    elapsed = time.perf_counter() - t0
    print_section(f"DSF compute done ({elapsed:.1f} s)", "✓")


def step_plot_dsf(params):
    """Plot the saved density and particle-current spectra."""
    if not any(plot_selection(params).values()):
        print_section("DSF plotting disabled")
        return
    if not _dsf_ready(params):
        raise ValueError("missing or incomplete DSF text files; run action = compute first")
    from mdtrace.dsf.plot import plot_dsf
    print_section("Plotting DSF from text")
    plot_dsf(params)


def step_compute_eels(params):
    """Report the status of the planned EELS method."""

    print("    ERROR: EELS calculation is not implemented yet.")
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
    print("\n    ✅ MDtrace task finished successfully.\n")


def _run_thinking(params):
    """Compute changed DSF inputs, or regenerate a missing/outdated plot."""
    steps = STEP_MAP["dsf"]
    if not _dsf_ready(params, check_settings=True):
        print_section("Thinking: missing DSF data or changed inputs → compute + plot", "💡")
        steps["compute"](params)
        steps["plot"](params)
        return
    selection = plot_selection(params)
    if not any(selection.values()):
        print_section("DSF data are ready. Plotting disabled.", "✓")
        return
    figures = plot_files(params)
    if figures:
        from PIL import Image

        latest_data = max(path.stat().st_mtime_ns for path in result_files(params.out_files_name, **selection))
        signature = plot_signature(params)
        try:
            for figure in figures.values():
                if figure.stat().st_mtime_ns < latest_data:
                    break
                with Image.open(figure) as image:
                    if image.info.get("MDtrace plot") != signature:
                        break
            else:
                print_section("DSF data and plots are ready. Use action = compute or plot to regenerate them.", "✓")
                return
        except OSError:
            pass  # Regenerate missing or unreadable figures from the saved spectra.
    steps["plot"](params)
