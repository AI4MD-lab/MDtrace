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


"""
mdtrace — Molecular Dynamics Trace.

Trace the physics inside your MD trajectory.
"""
import sys
import os

from mdtrace import __version__
from mdtrace.parser import read_input
from mdtrace.pipeline import run as run_pipeline


_LOGO_LINES = (
    r" __  __   ____    _____   ____       _       ____   _____",
    r"|  \/  | |  _ \  |_   _| |  _ \     / \     / ___| | ____|",
    r"| |\/| | | | | |   | |   | |_) |   / _ \   | |     | |",
    r"| |  | | | |_| |   | |   |  _ <   / ___ \  | |___  | |___",
    r"|_|  |_| |____/    |_|   |_| \_\ /_/   \_\  \____| |_____|",
)

_BANNER_INDENT = "  "
_BANNER_WIDTH = max(len(line) for line in _LOGO_LINES)


def _center_banner_line(text):
    """Center one line within the logo block without trailing whitespace."""
    return f"{_BANNER_INDENT}{text:^{_BANNER_WIDTH}}".rstrip()


LOGO = "\n".join(f"{_BANNER_INDENT}{line}" for line in _LOGO_LINES)
TAGLINE = "\n".join(
    _center_banner_line(line)
    for line in (
        f"Molecular Dynamics Trace   v{__version__}",
        "trace the physics inside your MD trajectory",
        "Author: liangting.zj@gmail.com",
    )
)


def _print_banner():
    print(f"\n{LOGO}\n\n{TAGLINE}\n")


def _print_run_summary(input_file, params):
    """Print one aligned panel describing the task about to run."""

    backend = "NumPy (serial DSF)" if params.method == "dsf" else "not implemented"

    border = f"{_BANNER_INDENT}{'=' * _BANNER_WIDTH}"
    divider = f"{_BANNER_INDENT}{'-' * _BANNER_WIDTH}"
    print(border)
    print(f"{_BANNER_INDENT}🚀 Starting MDtrace task")
    print(divider)
    print(f"{_BANNER_INDENT}Input   : {input_file}")
    print(f"{_BANNER_INDENT}Method  : {params.method.upper()}")
    print(f"{_BANNER_INDENT}Action  : {params.action}")
    print(f"{_BANNER_INDENT}Backend : {backend}")
    print(f"{_BANNER_INDENT}Output  : {params.out_files_name}")
    print(f"{border}\n")


def _show_help():
    _print_banner()
    print(r"""
USAGE:
    mdtrace [input.in]
    mdtrace -h

DESCRIPTION:
    MDtrace computes density dynamic structure factors (DSF) and
    longitudinal/transverse particle-current spectra from MD trajectories.
    Periodic crystals and amorphous systems require fixed simulation cells.
    EELS remains a planned extension.
    SED calculation, plotting, and fitting are provided separately by pySED.

CONTROL:
    method = dsf           # default analysis method
    action = thinking      # compute changed/missing DSF data, then plot
    action = compute       # always recompute DSF numerical output
    action = plot          # plot an existing .dsf.npz archive
    backend = numpy        # serial CPU DSF; GPU DSF is not implemented

TRAJECTORY:
    GPUMD XYZ, LAMMPS custom dumps, and compatible NetCDF are supported.
    trajectory_read_mode = cache
    Text input is converted to a reusable .mdtrace.nc cache.
    Currents require velocities; density DSF can use positions alone.

QUICK START:
    mdtrace input.in
    See examples/dsf/ and the DSF workflow in the manual for input templates.
    Without an explicit filename, MDtrace reads input.in only.
""")
    sys.exit(0)


def _detect_input_file():
    """Resolve input file from CLI args."""
    if len(sys.argv) == 1:
        if os.path.exists("input.in"):
            return "input.in"
        else:
            print("No input.in found in current directory. SED inputs should be run with pySED.")
            print("Run 'mdtrace -h' for help.")
            sys.exit(1)

    elif len(sys.argv) == 2:
        arg = sys.argv[1].lower()
        if arg in ("h", "help", "-h", "--help", "-help"):
            _show_help()
        return sys.argv[1]

    else:
        print("mdtrace takes 0 or 1 argument.")
        print("Run 'mdtrace -h' for help.")
        sys.exit(1)


def main():
    input_file = _detect_input_file()

    if not os.path.exists(input_file):
        print(f"\n  ERROR: file '{input_file}' not found.\n")
        sys.exit(1)

    _print_banner()

    # ── parse ──
    try:
        params = read_input(input_file)
        _print_run_summary(input_file, params)
        run_pipeline(params)
    except (ValueError, FileNotFoundError) as error:
        print(f"\n  ERROR: {error}\n", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
