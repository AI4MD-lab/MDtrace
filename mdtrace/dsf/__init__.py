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


"""mdtrace.dsf — Dynamic structure factor module."""

__all__ = ["compute_dsf", "save_dsf", "load_spectra"]


def __getattr__(name):
    """Load only the requested API, keeping numerical-only imports lightweight."""
    if name == "compute_dsf":
        from mdtrace.dsf.dsf import compute_dsf
        return compute_dsf
    if name in {"save_dsf", "load_spectra"}:
        from mdtrace.dsf import results
        return getattr(results, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
