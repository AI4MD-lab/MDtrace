# =============================================================================
#     Copyright 2025-2026 Ting Liang and MDTRACE development team
#     This file is part of MDTRACE.
#     MDTRACE is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#     MDTRACE is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#     GNU General Public License for more details.
#     You should have received a copy of the GNU General Public License
#     along with MDTRACE. If not, see <http://www.gnu.org/licenses/>.
# =============================================================================

"""MDtrace's trajectory boundary; all readers are provided by pySED."""

from pySED.io.schema import (
    ATOM_IDS, CELLS, POSITIONS, TIMES, TYPES, VELOCITIES,
    TrajectoryBatch, TrajectoryInfo,
)
from pySED.io.netcdf import NetCDFReader, is_netcdf


def require_current_cache(reader):
    """Reject unchecked text caches; native NetCDF needs no cache markers."""
    if hasattr(reader, "require_current_cache"):
        reader.require_current_cache(validate_headers=True)
        return
    # pySED 3.0 exposes the reader but not the cache-validation method.
    from netCDF4 import Dataset

    with Dataset(str(reader.path)) as dataset:
        for prefix in ("pysed", "mdtrace"):
            if not any(name.startswith(prefix + "_") for name in dataset.ncattrs()):
                continue
            if (getattr(dataset, prefix + "_schema_version", None) != 3
                    or getattr(dataset, prefix + "_conversion_complete", 0) != 1):
                raise ValueError("outdated or incomplete trajectory cache; select the original text trajectory to regenerate it")
        source_format = getattr(dataset, "pysed_source_format", None)
        if source_format == "lammps_dump" and getattr(dataset, "pysed_lammps_integer_ids", 0) != 1:
            raise ValueError("outdated atom-ID cache; select the original text trajectory to regenerate it")
        if source_format == "gpumd_xyz" and getattr(dataset, "pysed_headers_validated", 0) != 1:
            raise ValueError("GPUMD cache lacks per-frame header validation; select the original text trajectory to regenerate it")

__all__ = [
    "NetCDFReader", "TrajectoryBatch", "TrajectoryInfo", "is_netcdf", "require_current_cache",
    "POSITIONS", "VELOCITIES", "CELLS", "TYPES", "ATOM_IDS", "TIMES",
]
