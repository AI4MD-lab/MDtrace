"""Tests for direct/cached trajectory blocks and one-block prefetch."""

import importlib.util
import threading
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mdtrace.io.prepare import prepare_trajectory
from mdtrace.io.schema import CELLS, POSITIONS, VELOCITIES
from mdtrace.io.text import GPUMD_XYZ, LAMMPS_DUMP
from mdtrace.io.trajectory import (
    DirectTextBlockSource,
    NetCDFBlockSource,
    prefetch_one,
)
from tests.io.test_text import GPUMD_TEXT, LAMMPS_TEXT

NETCDF4_AVAILABLE = importlib.util.find_spec("netCDF4") is not None


def _gpumd_frames(count=8):
    frames = []
    for index in range(count):
        shift = 0.01 * index
        frames.extend(
            [
                "2",
                (
                    'Lattice="10 0 0 0 10 0 0 0 10" '
                    "Properties=species:S:1:pos:R:3:vel:R:3"
                ),
                f"Sr {shift} 0 0 {0.001 + shift} 0.002 0.003",
                f"Ti {1 + shift} 1 1 0.004 {0.005 + shift} 0.006",
            ]
        )
    return "\n".join(frames) + "\n"


class DirectTrajectoryTests(unittest.TestCase):
    def test_gpumd_and_lammps_use_the_same_block_interface(self):
        cases = (
            ("dump.xyz", GPUMD_TEXT, GPUMD_XYZ, "metal"),
            ("dump.lammpstrj", LAMMPS_TEXT, LAMMPS_DUMP, "metal"),
        )
        with TemporaryDirectory() as directory:
            for name, content, source_format, lammps_unit in cases:
                with self.subTest(source_format=source_format):
                    path = Path(directory) / name
                    path.write_text(content, encoding="utf-8")
                    source = DirectTextBlockSource(
                        path=path,
                        source_format=source_format,
                        required_frames=2,
                        parser_batch_size=1,
                        lammps_unit=lammps_unit,
                    )
                    blocks = list(source.iter_blocks(1, 2))

                    self.assertEqual([block.start for block in blocks], [0, 1])
                    self.assertEqual(source.info.n_atoms, 2)
                    for block in blocks:
                        self.assertEqual(
                            set(block.data),
                            {POSITIONS, VELOCITIES, CELLS},
                        )
                        self.assertEqual(block.data[POSITIONS].shape, (1, 2, 3))

                    with self.assertRaisesRegex(RuntimeError, "consumed once"):
                        next(source.iter_blocks(1, 2))

    def test_prefetch_runs_the_next_item_in_one_background_thread(self):
        thread_names = []

        def items():
            for value in range(3):
                thread_names.append(threading.current_thread().name)
                yield value

        with prefetch_one(items(), enabled=True) as prefetched:
            self.assertEqual(list(prefetched), [0, 1, 2])

        self.assertTrue(thread_names)
        self.assertTrue(
            all(name.startswith("mdtrace-trajectory") for name in thread_names)
        )


@unittest.skipUnless(NETCDF4_AVAILABLE, "netCDF4 is not installed")
class CachedTrajectoryParityTests(unittest.TestCase):
    def _read_blocks(self, source, prefetch):
        with prefetch_one(source.iter_blocks(4, 2), enabled=prefetch) as blocks:
            return list(blocks)

    def test_direct_and_cached_netcdf_produce_the_same_arrays(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dump.xyz"
            path.write_text(_gpumd_frames(), encoding="utf-8")
            cached_path, _ = prepare_trajectory(
                path,
                batch_size=3,
                compression_level=0,
            )

            direct = DirectTextBlockSource(
                path=path,
                source_format=GPUMD_XYZ,
                required_frames=8,
                parser_batch_size=3,
                lammps_unit="metal",
            )
            direct_blocks = self._read_blocks(
                direct,
                prefetch=False,
            )

            cached = NetCDFBlockSource(cached_path, required_frames=8)
            cached_blocks = self._read_blocks(
                cached,
                prefetch=True,
            )

            self.assertEqual(len(cached_blocks), len(direct_blocks))
            for cached_block, direct_block in zip(cached_blocks, direct_blocks):
                self.assertEqual(cached_block.start, direct_block.start)
                self.assertEqual(cached_block.data.keys(), direct_block.data.keys())
                for field in direct_block.data:
                    np.testing.assert_allclose(cached_block.data[field], direct_block.data[field], rtol=2e-6, atol=0)
