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

from pathlib import Path

from pySED.io import prepare_trajectory as prepare_with_pysed


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
    return prepare_with_pysed(
        source=source,
        converted_path=(converted_path if converted_path is not None
                        else default_converted_path(source)),
        batch_size=batch_size,
        compression_level=compression_level,
        storage_dtype=storage_dtype,
        lammps_unit=lammps_unit,
        text_mode=text_mode,
        force=force,
        validate_headers=True,
    )
