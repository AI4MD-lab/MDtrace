"""Exact path intersections, reciprocal geometry, and SED phase invariance."""

from fractions import Fraction as F
import math
from random import Random
from types import SimpleNamespace
import unittest

import numpy as np

from mdtrace.sed.construct_BZ import BZPathHelper
from mdtrace.sed.Phonon import _calculate_q_batch


def enumerate_integer_planes(start, end, matrix):
    """Independent oracle: enumerate one coordinate, then check all coordinates."""
    origin = [sum(s * int(p) for s, p in zip(start, row)) for row in matrix]
    direction = [sum((e - s) * int(p) for s, e, p in zip(start, end, row))
                 for row in matrix]
    axis = next((i for i, value in enumerate(direction) if value), None)
    if axis is None:
        return [F(0)] if all(x.denominator == 1 for x in origin) else []
    low, high = sorted((origin[axis], origin[axis] + direction[axis]))
    answer = []
    for integer in range(math.ceil(low), math.floor(high) + 1):
        position = (integer - origin[axis]) / direction[axis]
        if all((a + position * b).denominator == 1 for a, b in zip(origin, direction)):
            answer.append(position)
    return sorted(answer)


class BZPathTests(unittest.TestCase):
    def assert_periodic(self, helper, qpoints):
        turns = qpoints @ helper.supercell.T / (2 * np.pi)
        np.testing.assert_allclose(turns, np.rint(turns), atol=1e-10, rtol=0)

    def test_diagonal_cell_known_points_and_list_input(self):
        helper = BZPathHelper(np.eye(3), np.diag([12, 8, 4]))
        points, positions = helper.build_commensurate_path([0, 0, 0], [.5, .5, 0])
        np.testing.assert_array_equal(positions, [0, .5, 1])
        np.testing.assert_allclose(points / (2*np.pi), [[0, 0, 0], [.25, .25, 0], [.5, .5, 0]])
        self.assert_periodic(helper, points)

    def test_nondiagonal_matrix_uses_transpose(self):
        primitive = np.array([[3, 0, 0], [1, 2.8, 0], [.4, -.3, 4]])
        matrix = np.array([[4, 1, 0], [0, 6, 0], [0, 0, 2]])
        helper = BZPathHelper(primitive, matrix @ primitive)
        points, positions = helper.build_commensurate_path([0, 0, 0], [.5, 0, 0])
        np.testing.assert_array_equal(positions, [0, .5, 1])
        self.assert_periodic(helper, points)

    def test_hexagonal_primitive_with_orthogonal_supercell(self):
        primitive = np.array([[2.5, 0, 0], [1.25, np.sqrt(3)*1.25, 0], [0, 0, 4]])
        matrix = np.array([[6, 0, 0], [-6, 12, 0], [0, 0, 4]])
        helper = BZPathHelper(primitive, matrix @ primitive)
        self.assertTrue(np.allclose(helper.supercell, np.diag(np.diag(helper.supercell))))
        points, positions = helper.build_commensurate_path([0, 0, 0], [F(1, 3), F(1, 3), 0])
        np.testing.assert_array_equal(positions, [0, .5, 1])
        self.assert_periodic(helper, points)

    def test_signed_dense_matrix_and_reversed_path(self):
        primitive = np.array([[3, .2, .1], [1, 2.8, .3], [.4, -.3, 4]])
        matrix = np.array([[-4, 2, 2], [2, 6, -2], [2, 2, 4]])
        helper = BZPathHelper(primitive, matrix @ primitive)
        forward, positions = helper.build_commensurate_path([0, 0, 0], [1, 1, 1])
        backward, reverse_positions = helper.build_commensurate_path([1, 1, 1], [0, 0, 0])
        np.testing.assert_allclose(backward, forward[::-1], atol=1e-14, rtol=0)
        np.testing.assert_allclose(reverse_positions, 1 - positions[::-1], atol=1e-15)
        self.assert_periodic(helper, forward)

    def test_noncommensurate_endpoints_can_have_internal_solutions(self):
        helper = BZPathHelper(np.eye(3), np.eye(3))
        points, positions = helper.build_commensurate_path([.5, .5, 0], [3.5, 3.5, 0])
        np.testing.assert_allclose(positions, [1/6, 1/2, 5/6])
        np.testing.assert_allclose(points / (2*np.pi), [[1, 1, 0], [2, 2, 0], [3, 3, 0]])

    def test_empty_line_and_empty_segment_have_stable_shapes(self):
        helper = BZPathHelper(np.eye(3), np.eye(3))
        for start, end in [([0, .5, 0], [4, .5, 0]), ([.2, 0, 0], [.4, 0, 0])]:
            with self.subTest(start=start, end=end):
                points, positions = helper.build_commensurate_path(start, end)
                self.assertEqual(points.shape, (0, 3))
                self.assertEqual(positions.shape, (0,))
                self.assertEqual(points.dtype, np.dtype(float))

    def test_zero_length_is_checked_for_commensurability(self):
        helper = BZPathHelper(np.eye(3), np.diag([4, 4, 4]))
        for coordinate, count in [(.25, 1), (.1, 0)]:
            with self.subTest(coordinate=coordinate):
                points, positions = helper.build_commensurate_path([coordinate, 0, 0], [coordinate, 0, 0])
                self.assertEqual(len(points), count)
                self.assertEqual(positions.tolist(), [0.0] if count else [])

    def test_rounded_third_uses_normalized_endpoints_for_output(self):
        helper = BZPathHelper(np.eye(3), np.diag([6, 6, 6]))
        points, positions = helper.build_commensurate_path([0, 0, 0], [.33333, 0, 0])
        exact_points, exact_positions = helper.build_commensurate_path([0, 0, 0], [F(1, 3), 0, 0])
        np.testing.assert_array_equal(points, exact_points)
        np.testing.assert_array_equal(positions, exact_positions)
        self.assert_periodic(helper, points)

    def test_distinct_inputs_that_normalize_to_one_point(self):
        helper = BZPathHelper(np.eye(3), np.diag([6, 6, 6]))
        points, positions = helper.build_commensurate_path([.33333, 0, 0], [1/3, 0, 0])
        np.testing.assert_array_equal(positions, [0])
        np.testing.assert_allclose(points / (2*np.pi), [[1/3, 0, 0]])

    def test_short_segment_is_not_discarded_by_allclose(self):
        helper = BZPathHelper(np.eye(3), np.diag([1000000, 1, 1]))
        points, positions = helper.build_commensurate_path([.123456, 0, 0], [.123457, 0, 0])
        np.testing.assert_array_equal(positions, [0, 1])
        self.assert_periodic(helper, points)

    def test_explicit_fractions_are_not_snapped(self):
        helper = BZPathHelper(np.eye(3), np.diag([1000000, 1, 1]))
        for start, end in [(F(1, 1000000), F(2, 1000000)), ("1/1000000", "2/1000000")]:
            with self.subTest(start=start):
                points, positions = helper.build_commensurate_path([start, 0, 0], [end, 0, 0])
                np.testing.assert_array_equal(positions, [0, 1])
                self.assert_periodic(helper, points)

    def test_decimal_normalization_policy(self):
        convert = BZPathHelper._as_commensurate_fraction
        self.assertEqual(convert(.33333), F(1, 3))
        self.assertEqual(convert(.3333), F("0.3333"))
        self.assertEqual(convert(.00001), F(0))
        self.assertEqual(convert(.000010001), F("0.000010001"))
        self.assertEqual(convert(F(1, 1000000)), F(1, 1000000))

    def test_invalid_endpoints(self):
        helper = BZPathHelper(np.eye(3), np.eye(3))
        for point in [[], [0, 0], [[0, 0, 0]], [np.nan, 0, 0], [np.inf, 0, 0], ["1/0", 0, 0]]:
            with self.subTest(point=point), self.assertRaises(ValueError):
                helper.build_commensurate_path(point, [0, 0, 0])

    def test_invalid_cells_and_noninteger_transformations(self):
        invalid = [np.eye(2), np.zeros((3, 3)), np.diag([1, 1, np.nan]), np.diag([1, 1, np.inf])]
        for cell in invalid:
            for primitive, supercell in [(cell, np.eye(3)), (np.eye(3), cell)]:
                with self.subTest(cell=cell), self.assertRaises(ValueError):
                    BZPathHelper(primitive, supercell)
        for matrix in [np.diag([1.2, 1, 1]), np.diag([100000.25, 1, 1]), np.diag([2**63, 1, 1]), np.diag([.00001, 1, 1])]:
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                BZPathHelper(np.eye(3), matrix)

    def test_cell_rounding_and_final_cartesian_check(self):
        helper = BZPathHelper(np.eye(3), np.diag([1 + 5e-5, 1, 1]))
        helper.build_commensurate_path([0, 0, 0], [1, 0, 0])
        with self.assertRaisesRegex(ValueError, "periodicity check"):
            helper.build_commensurate_path([100, 0, 0], [100, 0, 0])

    def test_coordinate_conversion_interfaces(self):
        primitive = np.array([[3, .2, .1], [1, 2.8, .3], [.4, -.3, 4]])
        helper = BZPathHelper(primitive, 2*primitive)
        for reduced in [[.25, .5, .75], [[.25, .5, .75], [0, 0, 0]], np.empty((0, 3))]:
            np.testing.assert_allclose(helper.cartesian_to_reduced(helper.reduced_to_cartesian(reduced)), reduced, atol=1e-15)

    def test_single_coordinate_compatibility_helper(self):
        helper = BZPathHelper(np.eye(3), np.eye(3))
        self.assertIsNone(helper._solve_linear_diophantine(F(2), F(0)))
        self.assertEqual(helper._solve_linear_diophantine(F(1, 2), F(0)), [])
        self.assertEqual(helper._solve_linear_diophantine(F(1, 2), F(3)), [F(1, 6), F(1, 2), F(5, 6)])
        self.assertEqual(helper._solve_linear_diophantine(F(7, 2), F(-3)), [F(5, 6), F(1, 2), F(1, 6)])

    def test_seeded_paths_against_independent_integer_enumeration(self):
        rng = Random(20261029)
        for case in range(240):
            primitive = np.diag([2.1, 3.2, 4.3])
            if case % 2:
                primitive[1, 0], primitive[2, 0], primitive[2, 1] = .7, -.3, .5
            matrix = np.diag([rng.randrange(2, 9) for _ in range(3)])
            if case % 4 >= 2:
                for i, j in [(0, 1), (1, 2), (2, 0)]:
                    matrix[i] += rng.choice([-1, 1]) * matrix[j]
            helper = BZPathHelper(primitive, matrix @ primitive)
            start = [F(rng.randrange(-4, 5), rng.choice([1, 2, 3, 4])) for _ in range(3)]
            end = [F(rng.randrange(-4, 5), rng.choice([1, 2, 3, 4])) for _ in range(3)]
            if case % 3 == 0:
                start = [F(0)] * 3
            expected = enumerate_integer_planes(start, end, matrix)
            with self.subTest(case=case):
                self.assertEqual(helper._find_allowed_fractions(start, end), expected)
                points, positions = helper.build_commensurate_path(start, end)
                self.assertEqual(len(points), len(expected))
                np.testing.assert_array_equal(positions, [float(t) for t in expected])
                if expected:
                    self.assert_periodic(helper, points)
                reversed_points, _ = helper.build_commensurate_path(end, start)
                np.testing.assert_allclose(reversed_points, points[::-1], atol=1e-13, rtol=0)

    def test_sed_is_invariant_under_supercell_image_changes(self):
        rng = np.random.default_rng(134)
        config = SimpleNamespace(output_partial=False, masses=np.array([1.0]), amu_2_kg=1.0)
        velocities = rng.normal(size=(32, 8, 3))
        for primitive in [np.diag([3., 3., 4.]), np.array([[3., 0, 0], [1., 2.8, 0], [.4, .2, 4.]])]:
            matrix = np.array([[4, 1, 0], [0, 6, 0], [0, 0, 2]])
            helper = BZPathHelper(primitive, matrix @ primitive)
            points, _ = helper.build_commensurate_path([0, 0, 0], [.5, 0, 0])
            positions = rng.integers(0, 3, size=(8, 3)) @ primitive
            shifted = positions + rng.integers(-2, 3, size=(8, 3)) @ helper.supercell
            original = _calculate_q_batch(velocities, positions, points, (np.arange(8),), config)
            images = _calculate_q_batch(velocities, shifted, points, (np.arange(8),), config)
            np.testing.assert_allclose(images, original, atol=1e-9, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
