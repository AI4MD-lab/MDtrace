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

"""Keep MDtrace cache names while delegating conversion to pySED."""

from inspect import signature
from itertools import islice
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from pySED.io.netcdf import converted_file_is_current
from pySED.io.prepare import detect_trajectory, prepare_trajectory as prepare_with_pysed
from pySED.io.text import _gpumd_columns, _lattice_from_comment

from mdtrace.io import NetCDFReader, is_netcdf, require_current_cache


_VALIDATES_HEADERS = "validate_headers" in signature(prepare_with_pysed).parameters


def _gpumd_cells(source):
    """Check headers omitted by pySED 3.0's fast parser, without parsing atoms."""
    cells = []
    previous = None
    columns = None
    expected_atoms = None
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            n_atoms = int(line)
            if n_atoms < 1 or (expected_atoms is not None and n_atoms != expected_atoms):
                raise ValueError("GPUMD atom count must be positive and fixed")
            expected_atoms = n_atoms
            comment = next(stream, "")
            if comment != previous:
                frame_columns, _ = _gpumd_columns(comment)
                if columns is not None and frame_columns != columns:
                    raise ValueError("GPUMD XYZ Properties changed between frames")
                columns = frame_columns
                cell = _lattice_from_comment(comment)
                if cell is None or not np.isfinite(cell).all():
                    raise ValueError("GPUMD XYZ requires a finite Lattice in every frame")
                previous = comment
            if sum(1 for _ in islice(stream, n_atoms)) != n_atoms:
                raise EOFError("truncated GPUMD XYZ atom block")
            cells.append(cell)
    return np.asarray(cells)


def default_converted_path(source: Path) -> Path:
    path = Path(source)
    return path.with_name(f"{path.name}.mdtrace.nc")


def prepare_trajectory(
    source: Path,
    converted_path: Path | None = None,
    batch_size: int = 32,
    compression_level: int = 1,
    storage_dtype: str = "float32",
    lammps_unit: str = "metal",
    text_mode: str = "cache",
    force: bool = False,
) -> tuple[Path, str]:
    """Prepare checked trajectories with pySED's streaming parser and writer.

    DSF needs positions and truthful per-frame cells. Keep header validation
    enabled, including when deciding whether an existing cache can be reused.
    """
    if text_mode != "cache":
        raise ValueError("DSF requires trajectory_read_mode = cache")
    source = Path(source).expanduser()
    output = Path(converted_path) if converted_path is not None else default_converted_path(source)
    options = dict(
        source=source, converted_path=output,
        batch_size=batch_size,
        compression_level=compression_level,
        storage_dtype=storage_dtype,
        lammps_unit=lammps_unit,
        text_mode=text_mode,
        force=force,
    )
    if _VALIDATES_HEADERS:
        return prepare_with_pysed(**options, validate_headers=True)

    # pySED 3.0 keeps the same array reader and compressor. Only GPUMD text
    # needs a header pass on conversion; native .nc and cache reuse do not.
    source_format = detect_trajectory(source)
    if is_netcdf(source):
        with NetCDFReader(source) as reader:
            require_current_cache(reader)
        return source, source_format
    if source_format != "gpumd_xyz":
        return prepare_with_pysed(**options)
    if not force and converted_file_is_current(
        output, source, compression_level=compression_level,
        storage_dtype=storage_dtype, lammps_unit=lammps_unit, source_format=source_format,
    ):
        with Dataset(str(output)) as dataset:
            if getattr(dataset, "pysed_headers_validated", 0) == 1:
                return output, source_format
    cells = _gpumd_cells(source)
    options["force"] = True
    prepared = prepare_with_pysed(**options)
    with Dataset(str(output), "a") as dataset:
        dataset.variables["cell_vectors"][:] = cells
        dataset.pysed_headers_validated = 1
    return prepared
