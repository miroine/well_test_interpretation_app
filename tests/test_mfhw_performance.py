"""Numerical parity and bounded work for the multi-fracture solver."""
import pathlib
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from engine import laplace as L


def reference_mfhw(u, n, spacing, conductivity):
    """Direct pairwise assembly, retained as an independent optimization reference."""
    edges, xc, dx, g = L._frac_geometry(10)
    z = np.sqrt(u)
    gx, gw = np.polynomial.legendre.leggauss(6)
    xq = (0.5 * (edges[:-1] + edges[1:])[:, None] + 0.5 * dx * gx).ravel()
    wq = np.tile(0.5 * dx * gw, 10)
    matrix = np.zeros((len(u), n * 10 + 1, n * 10 + 1))
    for k in range(n):
        for m in range(n):
            if k == m:
                block = L._self_matrix(z, edges, xc) + (2 * np.pi / conductivity) * (xc[:, None] * dx - g)
            else:
                distance = abs(k - m) * spacing
                r1 = np.sqrt((xc[:, None] - xq) ** 2 + distance ** 2)
                r2 = np.sqrt((xc[:, None] + xq) ** 2 + distance ** 2)
                kernel = (L.k0(z[:, None, None] * r1) + L.k0(z[:, None, None] * r2)) * wq
                block = kernel.reshape(len(u), 10, 10, 6).sum(axis=3)
            matrix[:, k*10:(k+1)*10, m*10:(m+1)*10] = block
    matrix[:, :-1, -1] = -1
    matrix[:, -1, :-1] = dx
    rhs = np.zeros((len(u), n * 10 + 1, 1))
    rhs[:, -1, 0] = 1 / (2 * u)
    return np.linalg.solve(matrix, rhs)[:, -1, 0]


class MultiFractureWorkTests(unittest.TestCase):
    def test_empty_time_batch_remains_empty(self):
        response = L.pws_mfhw(np.array([]), L.f_homogeneous, 0, 0, 10, 4, 2.5)
        self.assertEqual(response.shape, (0,))

    def test_pairwise_reference_parity(self):
        u = np.geomspace(1e-5, 1e3, 10)
        for n in (1, 4, 12):
            for spacing in (0.25, 2.5):
                with self.subTest(n=n, spacing=spacing):
                    ref = reference_mfhw(u, n, spacing, 10)
                    actual = L.pws_mfhw(u, L.f_homogeneous, 0, 0, 10, n, spacing)
                    np.testing.assert_allclose(actual, ref, rtol=2e-13)

    def test_cross_fracture_kernels_computed_once_per_distance(self):
        with patch.object(L, "k0", wraps=L.k0) as kernel:
            L.pws_mfhw(np.geomspace(1e-4, 10, 8), L.f_homogeneous, 0, 0, 10, 12, 2.5)
        self.assertEqual(kernel.call_count, 2 * (12 - 1))

    def test_pressure_derivative_share_laplace_evaluations(self):
        t = np.geomspace(1e-3, 1e3, 40)
        def response(u):
            return L.pws_mfhw(u, L.f_homogeneous, 0.01, 2, 10, 4, 2.5)
        expected_p = L.invert(response, t)
        expected_d = t * L.invert(lambda u: u * response(u), t)
        with patch.object(L, "pws_mfhw", wraps=L.pws_mfhw) as solver:
            p, der = L.unit_response(response, t)
        self.assertEqual(solver.call_count, 12)
        np.testing.assert_array_equal(p, expected_p)
        np.testing.assert_array_equal(der, expected_d)


if __name__ == "__main__":
    unittest.main()
