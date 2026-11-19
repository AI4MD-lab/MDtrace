"""Numerical and I/O tests for the minimal neutron DSF prototype."""

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from mdtrace.dsf.scattering import neutron_weights as _neutron_weights
from mdtrace.dsf.dsf import (
    PLANCK_MEV_PS,
    _map_atom_labels,
    compute_dsf,
    save_dsf,
)
from mdtrace.parameters import validate_dsf
from mdtrace.dsf.neutron_scattering_lengths import (
    coherent_scattering_lengths,
    incoherent_scattering_lengths,
)


NETCDF4_AVAILABLE = importlib.util.find_spec("netCDF4") is not None


def _params(trajectory_path: Path, output: Path, num_blocks: int = 1):
    return SimpleNamespace(
        backend="numpy",
        experiment="neutron",
        trajectory_path=str(trajectory_path),
        dsf_qpoints=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        prim_unitcell=np.eye(3),
        atom_types=["H", "O"],
        dsf_cell=np.eye(3),
        dsf_normalization="legacy",
        num_blocks=num_blocks,
        time_step=1000.0,
        output_data_stride=1,
        out_files_name=str(output),
    )


class DSFScatteringDataTests(unittest.TestCase):
    def test_matches_reviewed_sears_table(self) -> None:
        # Independently transcribed page images retain blank cells and uncertainty
        # notation, so column shifts and digits copied from parentheses are caught.
        def central_value(raw):
            if raw == "-":
                return None
            return complex(re.sub(r"\([^)]*\)", "", raw).replace("+/-", "").replace("E", ""))

        reference = Path(__file__).parent / "fixtures" / "sears1992_scattering_lengths.txt"
        expected_coherent = {"none": 0j}
        expected_incoherent = {"none": 0j}
        labels = set()
        for line in reference.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            label, raw_coherent, raw_incoherent, raw_sigma = line.split()
            self.assertNotIn(label, labels)
            labels.add(label)
            coherent = central_value(raw_coherent)
            incoherent = central_value(raw_incoherent)
            sigma = central_value(raw_sigma)
            if coherent is not None:
                expected_coherent[label] = coherent
            if incoherent is None and sigma is not None:
                self.assertEqual(sigma.imag, 0)
                self.assertGreaterEqual(sigma.real, 0)
                incoherent = np.sqrt(100 * sigma.real / (4 * np.pi)) + 0j
            if incoherent is not None:
                expected_incoherent[label] = incoherent

        self.assertEqual(len(labels), 371)
        for actual, expected in (
            (coherent_scattering_lengths, expected_coherent),
            (incoherent_scattering_lengths, expected_incoherent),
        ):
            self.assertEqual(actual.keys(), expected.keys())
            for label, value in expected.items():
                with self.subTest(label=label):
                    self.assertLessEqual(abs(actual[label] - value), 1e-12)

    def test_unavailable_source_lengths_are_rejected(self) -> None:
        # Sears leaves Xe b_inc and both Po lengths blank; neither means zero.
        for label, component in (("Xe", "incoherent"), ("Po", "coherent")):
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    ValueError, f"no {component} neutron scattering length"
                ):
                    _neutron_weights(np.array([label]))

    def test_sears_lengths_produce_consistent_incoherent_weights(self) -> None:
        # Sears Table 1, pp. 34 and 37; 147Sm's 14(19) barn is not 143 barn.
        _, weights = _neutron_weights(np.array(["147Sm", "Bi"]))
        np.testing.assert_allclose(weights, [11.0**2, 0.259**2])
        cross_sections = 4 * np.pi * weights / 100
        self.assertAlmostEqual(cross_sections[0], 14.0, delta=19.0)
        self.assertAlmostEqual(cross_sections[1], 0.0084, delta=0.0010)


class DSFTypeMappingTests(unittest.TestCase):
    def test_sorted_noncontiguous_type_ids_map_to_labels(self) -> None:
        labels, mapping = _map_atom_labels(
            np.array([42, 7, 42, 7]),
            ["H", "O"],
        )

        self.assertEqual(mapping, {7: "H", 42: "O"})
        np.testing.assert_array_equal(labels, ["O", "H", "O", "H"])

    def test_dsf_validation_rejects_deferred_modes(self) -> None:
        base = dict(
            action="compute",
            backend="numpy",
            experiment="neutron",
            trajectory_read_mode="cache",
            atom_types=["H"],
            dsf_qpoints=[0.0, 0.0, 0.0],
            prim_unitcell=np.eye(3),
        )
        for name, value, message in (
            ("backend", "cupy", "backend = numpy only"),
            ("experiment", "xray", "experiment = neutron or number only"),
            (
                "trajectory_read_mode",
                "direct",
                "trajectory_read_mode = cache",
            ),
        ):
            params = SimpleNamespace(**{**base, name: value})
            with self.assertRaisesRegex(ValueError, message):
                validate_dsf(params)


@unittest.skipUnless(NETCDF4_AVAILABLE, "netCDF4 is not installed")
class DSFIntegrationTests(unittest.TestCase):
    @staticmethod
    def _create_trajectory(path: Path, repeat: int = 1) -> None:
        from netCDF4 import Dataset

        base_x = np.array([0.0, 0.25, 0.5, 0.75])
        n_frames = 4 * repeat
        positions = np.zeros((n_frames, 2, 3), dtype=float)
        positions[:, 0, 0] = np.tile(base_x, repeat)
        type_ids = np.tile(np.array([[7, 42]], dtype=np.int32), (n_frames, 1))

        with Dataset(str(path), "w", format="NETCDF4") as dataset:
            dataset.createDimension("frame", n_frames)
            dataset.createDimension("atom", 2)
            dataset.createDimension("spatial", 3)
            coordinates = dataset.createVariable(
                "coordinates",
                "f8",
                ("frame", "atom", "spatial"),
            )
            coordinates.units = "angstrom"
            coordinates[:] = positions
            types = dataset.createVariable(
                "type",
                "i4",
                ("frame", "atom"),
            )
            types[:] = type_ids

    def test_matches_pynamic_estimator_and_legacy_normalization(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = root / "tiny.nc"
            self._create_trajectory(trajectory)

            result = compute_dsf(_params(trajectory, root / "result"))

        np.testing.assert_allclose(
            result["freq_thz"],
            [-0.5, -0.25, 0.0, 0.25],
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            result["energy_mev"],
            PLANCK_MEV_PS * result["freq_thz"],
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            result["q_cart"],
            [[0.0, 0.0, 0.0], [2.0 * np.pi, 0.0, 0.0]],
            atol=1.0e-15,
        )

        expected_coherent = np.zeros((2, 4))
        expected_incoherent = np.zeros((2, 4))
        # pynamic divides every block by N_atoms * N_frames = 2 * 4.
        # Sears p. 29: H sigma_inc = 80.26(6), O = 0.000(8) barn.
        hydrogen_inc_sq = 100.0 * 80.26 / (4.0 * np.pi)
        oxygen_inc_sq = 0.0  # Central value; the (8) is an uncertainty, not a digit.
        expected_coherent[0, 2] = (-3.739 + 5.803) ** 2 / 8.0
        expected_incoherent[0, 2] = (hydrogen_inc_sq + oxygen_inc_sq) / 8.0
        expected_coherent[1, 2] = 5.803**2 / 8.0
        expected_coherent[1, 3] = (-3.739) ** 2 / 8.0
        expected_incoherent[1, 2] = oxygen_inc_sq / 8.0
        expected_incoherent[1, 3] = hydrogen_inc_sq / 8.0

        np.testing.assert_allclose(
            result["coherent_sqw"],
            expected_coherent,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            result["incoherent_sqw"],
            expected_incoherent,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            result["sqw"],
            expected_coherent + expected_incoherent,
            atol=1.0e-12,
        )
        self.assertTrue(np.isnan(result["sqw_sem"]).all())
        self.assertEqual(result["type_mapping"], {7: "H", 42: "O"})

    def test_identical_blocks_have_zero_sem(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = root / "blocks.nc"
            self._create_trajectory(trajectory, repeat=2)

            result = compute_dsf(
                _params(trajectory, root / "result", num_blocks=2)
            )

        self.assertEqual(result["block_size"], 4)
        np.testing.assert_allclose(result["sqw_sem"], 0.0, atol=1.0e-15)
        np.testing.assert_allclose(
            result["sqw"],
            result["coherent_sqw"] + result["incoherent_sqw"],
            atol=1.0e-15,
        )

    def test_text_outputs_include_components_axes_and_units(self) -> None:
        suffixes = (
            ".dsf",
            ".dsf_coherent",
            ".dsf_incoherent",
            ".dsf_sem",
            ".dsf_coherent_sem",
            ".dsf_incoherent_sem",
            ".dsf_THz",
            ".dsf_meV",
            ".dsf_Qcart",
            ".dsf_Qfrac",
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = root / "tiny.nc"
            output = root / "result"
            self._create_trajectory(trajectory)
            params = _params(trajectory, output)

            result = compute_dsf(params)
            save_dsf(result, params)

            for suffix in suffixes:
                self.assertTrue(Path(str(output) + suffix).is_file(), suffix)
            header = Path(str(output) + ".dsf").read_text(
                encoding="utf-8"
            ).splitlines()[0]
            self.assertIn("pynamic-compatible arbitrary units", header)
            np.testing.assert_allclose(
                np.loadtxt(str(output) + ".dsf"),
                result["sqw"],
                atol=1.0e-15,
            )
            np.testing.assert_allclose(
                np.loadtxt(str(output) + ".dsf_meV"),
                result["energy_mev"],
                atol=1.0e-15,
            )


if __name__ == "__main__":
    unittest.main()
