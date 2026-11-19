"""Tests for fast GPUMD and LAMMPS text-to-NetCDF conversion."""

import importlib.util
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mdtrace.io.netcdf import NetCDFReader
from mdtrace.io.prepare import default_converted_path, prepare_trajectory
from mdtrace.io.schema import ATOM_IDS, POSITIONS, TYPES, VELOCITIES
from mdtrace.io.text import GPUMD_XYZ, LAMMPS_DUMP, detect_text_format

NETCDF4_AVAILABLE = importlib.util.find_spec("netCDF4") is not None


GPUMD_TEXT = """\
2
Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3:vel:R:3
Sr 0 0 0 0.001 0.002 0.003
Ti 1 1 1 0.004 0.005 0.006
2
Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3:vel:R:3
Sr 0.1 0 0 0.002 0.003 0.004
Ti 1.1 1 1 0.005 0.006 0.007
"""


LAMMPS_TEXT = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 20
0 30
ITEM: ATOMS id type x y z vx vy vz
2 2 2 2 2 4 5 6
1 1 1 1 1 1 2 3
ITEM: TIMESTEP
10
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 20
0 30
ITEM: ATOMS id type x y z vx vy vz
1 1 1.1 1 1 2 3 4
2 2 2.1 2 2 5 6 7
"""


@unittest.skipUnless(NETCDF4_AVAILABLE, "netCDF4 is not installed")
class TextConversionTests(unittest.TestCase):
    def test_gpumd_properties_identify_velocity_without_confusing_force_or_mass(self) -> None:
        cases = (
            ("species:S:1:mass:R:1:pos:R:3:force:R:3:vel:R:3", "Si 28 1 2 3 99 98 97 .001 .002 .003", True),
            ("species:S:1:pos:R:3:force:R:3", "Si 1 2 3 99 98 97", False),
        )
        for properties, atoms, has_velocity in cases:
            with self.subTest(properties=properties), TemporaryDirectory() as directory:
                source = Path(directory) / "dump.xyz"
                source.write_text(f'1\nLattice="10 0 0 0 10 0 0 0 10" Properties={properties}\n{atoms}\n', encoding="utf-8")
                prepared, _ = prepare_trajectory(source, storage_dtype="float64")
                with NetCDFReader(prepared) as trajectory:
                    np.testing.assert_allclose(trajectory.read(POSITIONS)[0, 0], [1, 2, 3])
                    self.assertEqual(VELOCITIES in trajectory.info.fields, has_velocity)
                    if has_velocity:
                        np.testing.assert_allclose(trajectory.read(VELOCITIES)[0, 0], [100, 200, 300])

    def test_float64_conversion_preserves_precision_and_refreshes_old_cache(self) -> None:
        from netCDF4 import Dataset

        value = 1.123456789012345
        for filename, source_text, atom in (
            ("dump.xyz", GPUMD_TEXT, 1), ("dump.data", LAMMPS_TEXT, 0),
        ):
            with self.subTest(filename=filename), TemporaryDirectory() as directory:
                source = Path(directory) / filename
                source.write_text(source_text.replace("1.1 ", f"{value:.17g} "), encoding="utf-8")
                prepared, _ = prepare_trajectory(source, storage_dtype="float64", batch_size=1)
                with NetCDFReader(prepared) as trajectory:
                    self.assertEqual(trajectory.read(POSITIONS)[1, atom, 0], value)

                # Version 1 parsed into float32 even when the cache stored float64.
                with Dataset(str(prepared), "a") as dataset:
                    dataset.mdtrace_schema_version = 1
                    dataset.variables["coordinates"][1, atom, 0] = float(np.float32(value))
                prepare_trajectory(source, storage_dtype="float64", batch_size=1)
                with NetCDFReader(prepared) as trajectory:
                    self.assertEqual(trajectory.read(POSITIONS)[1, atom, 0], value)

    def test_conversion_prints_progress_and_destination(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "dump.xyz"
            source.write_text(GPUMD_TEXT, encoding="utf-8")
            output = io.StringIO()

            with redirect_stdout(output):
                prepared, _ = prepare_trajectory(source, batch_size=1)

            message = output.getvalue()
            self.assertIn("Converting text trajectory to NetCDF", message)
            self.assertIn(str(prepared), message)
            self.assertIn("100%", message)
            self.assertIn("2 frames", message)

    def test_gpumd_xyz_is_streamed_and_reused(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "dump.xyz"
            source.write_text(GPUMD_TEXT, encoding="utf-8")

            self.assertEqual(detect_text_format(source), GPUMD_XYZ)
            prepared, source_format = prepare_trajectory(
                source,
                batch_size=1,
            )
            self.assertEqual(source_format, GPUMD_XYZ)
            with NetCDFReader(prepared) as trajectory:
                np.testing.assert_allclose(
                    trajectory.read(POSITIONS)[:, 1, 0],
                    [1.0, 1.1],
                )
                np.testing.assert_allclose(
                    trajectory.read(VELOCITIES)[0, 0],
                    [100.0, 200.0, 300.0],
                    atol=1.0e-4,
                )
                np.testing.assert_array_equal(
                    trajectory.read(TYPES)[0],
                    [0, 1],
                )

            converted = default_converted_path(source)
            first_mtime = converted.stat().st_mtime_ns
            prepare_trajectory(source)
            self.assertEqual(converted.stat().st_mtime_ns, first_mtime)

    def test_lammps_dump_sorts_atom_ids_and_converts_units(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "dump.data"
            source.write_text(LAMMPS_TEXT, encoding="utf-8")

            self.assertEqual(detect_text_format(source), LAMMPS_DUMP)
            prepared, source_format = prepare_trajectory(
                source,
                batch_size=1,
                lammps_unit="metal",
            )
            self.assertEqual(source_format, LAMMPS_DUMP)
            with NetCDFReader(prepared) as trajectory:
                np.testing.assert_array_equal(
                    trajectory.read(ATOM_IDS),
                    [1, 2],
                )
                np.testing.assert_allclose(
                    trajectory.read(POSITIONS)[0, :, 0],
                    [1, 2],
                )
                np.testing.assert_allclose(
                    trajectory.read(VELOCITIES)[0, 0],
                    [100, 200, 300],
                )
                np.testing.assert_allclose(
                    np.diag(trajectory.read_cells()[0]),
                    [10, 20, 30],
                )


if __name__ == "__main__":
    unittest.main()
