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

# Python modules

"""Exact reciprocal-path geometry for periodic DSF sampling."""

import math
from fractions import Fraction

import numpy as np

class BZPathHelper:
    """Generate supercell-compatible q-points on straight reciprocal paths.

    Cell vectors are rows, with ``supercell = P @ primitive``. Reduced
    q-points refer to the primitive reciprocal basis, including 2*pi in
    Cartesian coordinates. Both orthogonal and non-orthogonal cells are
    supported, including non-diagonal integer repetition matrices.

    Path intersections with the reciprocal supercell lattice are obtained
    by a parameterization based on the greatest common divisor (GCD) and
    Bézout's identity, using exact integer arithmetic. The extended Euclidean
    algorithm computes the Bézout coefficients. See
    ``_commensurate_parameters`` for the derivation.
    """

    _CELL_TOLERANCE = 1e-4

    def __init__(self, primitive_cell, supercell_cell):
        """Accept finite, nonsingular (3, 3) primitive and supercell matrices."""
        self._primitive_cell = self._validate_cell(primitive_cell, "Primitive cell")
        self._supercell_cell = self._validate_cell(supercell_cell, "Supercell")
        self.P = self._compute_P_matrix()

    @staticmethod
    def _validate_cell(cell, name):
        cell = np.array(cell, dtype=float)
        if cell.shape != (3, 3) or not np.isfinite(cell).all():
            raise ValueError(f"{name} must be a finite (3, 3) matrix.")
        if np.linalg.slogdet(cell)[0] == 0:
            raise ValueError(f"{name} must be nonsingular.")
        return cell

    @property
    def primitive(self):
        """Primitive lattice vectors as rows, in Angstrom."""
        return self._primitive_cell

    @property
    def supercell(self):
        """Supercell lattice vectors as rows, in Angstrom."""
        return self._supercell_cell

    @property
    def reciprocal_primitive(self):
        """Primitive reciprocal vectors as rows, in rad/Angstrom."""
        return 2 * np.pi * np.linalg.inv(self.primitive.T)

    def _compute_P_matrix(self):
        """Infer P in S = P C, using an absolute integer tolerance of 1e-4."""
        repetitions = np.linalg.solve(self.primitive.T, self.supercell.T).T
        nearest = np.rint(repetitions)
        if (
            not np.isfinite(nearest).all()
            or np.any(nearest < -(2**63))
            or np.any(nearest >= 2**63)
            or not np.allclose(
                repetitions, nearest, rtol=0.0, atol=self._CELL_TOLERANCE
            )
        ):
            raise ValueError(
                "Supercell must be an integer transformation of the primitive "
                "cell (absolute tolerance 1e-4)."
            )
        matrix = nearest.astype(np.int64)
        a, b, c = [[int(x) for x in row] for row in matrix]
        determinant = (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
        if determinant == 0:
            raise ValueError("The inferred integer repetition matrix is singular.")
        return matrix

    def cartesian_to_reduced(self, qpoints_cart):
        """Convert Cartesian q-points to primitive reciprocal coordinates."""
        qpoints_cart = np.asarray(qpoints_cart, dtype=float)
        return np.linalg.solve(self.reciprocal_primitive.T, qpoints_cart.T).T

    def reduced_to_cartesian(self, qpoints_red):
        """Convert primitive reciprocal coordinates to Cartesian q-points."""
        return np.asarray(qpoints_red, dtype=float) @ self.reciprocal_primitive

    def build_commensurate_path(self, start_frac, end_frac):
        """Return all commensurate points on the normalized, closed segment.

        Parameters
        ----------
        start_frac, end_frac : array-like, shape (3,)
            Endpoints in primitive reciprocal coordinates. Nearby fractions
            with denominators up to 12 are recognized within 1e-5 for decimal
            input. Explicit Fraction values are preserved exactly.

        Returns
        -------
        q_cart : ndarray, shape (N, 3)
            Cartesian q-points in rad/Angstrom, ordered from start to end.
        fractions : ndarray, shape (N,)
            Positions in [0, 1] along the normalized segment. Empty paths
            return shapes (0, 3) and (0,). A zero-length segment returns its
            point at fraction 0 only when that point is commensurate.
        """
        start = self._normalize_endpoint(start_frac)
        end = self._normalize_endpoint(end_frac)
        fractions = self._find_allowed_fractions(start, end)
        if not fractions:
            return np.zeros((0, 3)), np.zeros((0,))

        # Use the very same rational endpoints for solving and interpolation.
        # Conversion to floating point happens only after exact interpolation.
        points = np.array([
            [float(s + f * (e - s)) for s, e in zip(start, end)]
            for f in fractions
        ])
        q_cart = self.reduced_to_cartesian(points)
        turns = q_cart @ self.supercell.T / (2 * np.pi)
        if not np.isfinite(turns).all() or not np.allclose(
            turns, np.rint(turns), rtol=0.0, atol=self._CELL_TOLERANCE
        ):
            raise ValueError(
                "Generated q-points fail the supercell periodicity check "
                "(absolute tolerance 1e-4). Check the cell matrices and "
                "their numerical precision."
            )
        return q_cart, np.array([float(f) for f in fractions])

    def _find_allowed_fractions(self, start_frac, end_frac):
        """Find exact path parameters f in [0, 1] satisfying periodicity.

        With row-vector cells C and S = P @ C, a reduced q-point is
        commensurate precisely when q @ P.T is an integer vector:

            [start + f * (end - start)] @ P.T = n,  n in Z^3.

        Set a = start @ P.T and b = (end - start) @ P.T. The helper
        _commensurate_parameters solves a + f*b in Z^3 using the greatest
        common divisor (GCD) and Bézout's identity, without enumerating
        separate coordinate sets.

        Parameters
        ----------
        start_frac, end_frac : array-like, shape (3,)
            Endpoints in reduced *primitive* reciprocal coordinates.
            Decimal endpoints use the same normalization as path output.

        Returns
        -------
        list(Fraction)
            All allowed parameters, sorted in increasing order. A constant
            path returns [Fraction(0)] only if its point is commensurate.
        """
        start = self._normalize_endpoint(start_frac)
        end = self._normalize_endpoint(end_frac)
        origin = []
        direction = []
        for row in self.P:
            origin.append(sum(s * int(p) for s, p in zip(start, row)))
            direction.append(sum((e - s) * int(p) for s, e, p in zip(start, end, row)))
        return self._commensurate_parameters(origin, direction)

    @staticmethod
    def _commensurate_parameters(origin, direction):
        """Parameterize a rational line's intersection with the integer lattice.

        Let a = origin and b = direction. Clear their denominators with a
        positive common multiple D, so A = D*a and B = D*b are integers.
        For B != 0, let g = gcd(|B_i|) > 0 be the greatest common divisor
        of the components of B, and write B = g*v with a primitive integer
        direction v. The extended Euclidean algorithm computes integers w_i
        in Bézout's identity, sum(w_i*B_i) = g, hence w dot v = 1.
        Define m = w dot A.

        If n = (A + f*B)/D is integer, k = w dot n must be integer, giving

            f = (D*k - m)/g,
            n = (A - m*v)/D + k*v.

        Conversely, these equations produce every solution whenever each
        component of A - m*v is divisible by D. Otherwise the infinite
        line misses the integer lattice. The finite segment further needs

            ceil(m/D) <= k <= floor((m + g)/D).

        These bounds and the divisibility test use Python integers only.
        The output spacing is D/g; an empty integer range means no point
        on the segment. For B = 0, test A/D directly and return at most one
        representative, f = 0. No approximate endpoint-equality test is used.
        """
        denominator = math.lcm(*(x.denominator for x in (*origin, *direction)))
        offset = [int(denominator * x) for x in origin]
        step = [int(denominator * x) for x in direction]
        divisor = 0
        coefficients = [0] * len(step)
        for axis, component in enumerate(step):
            divisor, left, right = BZPathHelper._extended_gcd(divisor, component)
            coefficients = [left * x for x in coefficients]
            coefficients[axis] += right

        if divisor == 0:
            return [Fraction(0)] if all(x % denominator == 0 for x in offset) else []

        direction_int = [x // divisor for x in step]
        shift = sum(w * a for w, a in zip(coefficients, offset))
        if any((a - shift * v) % denominator for a, v in zip(offset, direction_int)):
            return []
        first = -((-shift) // denominator)
        last = (shift + divisor) // denominator
        return [Fraction(denominator * k - shift, divisor) for k in range(first, last + 1)]

    @staticmethod
    def _extended_gcd(a, b):
        """Return nonnegative g and coefficients u, v with u*a + v*b = g."""
        remainder, next_remainder = int(a), int(b)
        u, next_u, v, next_v = 1, 0, 0, 1
        while next_remainder:
            quotient = remainder // next_remainder
            remainder, next_remainder = next_remainder, remainder - quotient * next_remainder
            u, next_u = next_u, u - quotient * next_u
            v, next_v = next_v, v - quotient * next_v
        sign = 1 if remainder >= 0 else -1
        return sign * remainder, sign * u, sign * v

    def _solve_linear_diophantine(self, a, b):
        """Compatibility helper for a single coordinate; path solving is joint."""
        a, b = Fraction(a), Fraction(b)
        if b == 0:
            return None if a.denominator == 1 else []
        fractions = self._commensurate_parameters([a], [b])
        return fractions if b > 0 else fractions[::-1]

    @classmethod
    def _normalize_endpoint(cls, endpoint):
        values = np.asarray(endpoint, dtype=object)
        if values.shape != (3,):
            raise ValueError("Each path endpoint must contain exactly three coordinates.")
        return tuple(cls._as_commensurate_fraction(value) for value in values)

    @staticmethod
    def _as_commensurate_fraction(value, max_denominator=12, atol=1e-5):
        """Preserve exact fractions; recognize nearby simple decimal inputs."""
        try:
            frac = value if isinstance(value, Fraction) else Fraction(str(value))
            if not math.isfinite(float(frac)):
                raise ValueError
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as error:
            raise ValueError("Path coordinates must be finite real numbers or fractions.") from error
        if isinstance(value, Fraction) or (isinstance(value, str) and "/" in value):
            return frac
        simple = frac.limit_denominator(max_denominator)
        return simple if abs(simple - frac) <= Fraction(str(atol)) else frac
