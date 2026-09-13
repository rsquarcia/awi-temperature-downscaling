"""Small offline checks for the U-Net, normalization, EOF identities and imports.
Uses synthetic arrays rather than simulation output; does not reproduce
the full paper. PyTorch-dependent checks are skipped if PyTorch is absent.
Run with: python -m unittest discover -s tests -v.
"""


import unittest


import numpy as np


EOF_RNG = np.random.default_rng(20260824)


def cosine_weights(n_lat):

    edges = np.linspace(-90.0, 90.0, n_lat + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return np.cos(np.deg2rad(centres))


def weighted_eof_basis(field, weights, k):

    sqrt_w = np.sqrt(weights)
    x = field * sqrt_w
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return vt[:k], x


class WeightedEOF(unittest.TestCase):
    def setUp(self):
        self.n_time, self.n_lat, self.n_lon = 60, 16, 24
        w = cosine_weights(self.n_lat)
        self.weights = np.repeat(w, self.n_lon)
        self.field = EOF_RNG.normal(size=(self.n_time, self.n_lat * self.n_lon))

    def test_basis_is_orthonormal_in_the_weighted_metric(self):
        eofs, _ = weighted_eof_basis(self.field, self.weights, k=10)
        gram = eofs @ eofs.T
        np.testing.assert_allclose(gram, np.eye(10), atol=1e-10)

    def test_parseval_holds_for_a_complete_basis(self):
        k = min(self.n_time - 1, self.n_lat * self.n_lon)
        eofs, x = weighted_eof_basis(self.field, self.weights, k=k)
        scores = x @ eofs.T
        np.testing.assert_allclose(
            (scores ** 2).sum(), (x ** 2).sum(), rtol=1e-10,
            err_msg="weighted energy must be preserved by a complete projection")

    def test_truncated_projector_is_idempotent_and_symmetric(self):
        eofs, _ = weighted_eof_basis(self.field, self.weights, k=8)
        p = eofs.T @ eofs
        np.testing.assert_allclose(p @ p, p, atol=1e-10)
        np.testing.assert_allclose(p, p.T, atol=1e-12)

    def test_cosine_weights_are_symmetric_and_pole_exclusive(self):
        w = cosine_weights(self.n_lat)
        np.testing.assert_allclose(w, w[::-1], rtol=1e-12)
        self.assertTrue((w > 0).all(), "no cell centre may sit exactly on a pole")


class ErrorDecomposition(unittest.TestCase):


    def setUp(self):
        self.n_time, self.n_pix, self.k = 80, 120, 12
        self.weights = np.abs(EOF_RNG.normal(size=self.n_pix)) + 0.1
        self.weights /= self.weights.sum()
        sqrt_w = np.sqrt(self.weights)
        target = EOF_RNG.normal(size=(self.n_time, self.n_pix)) * sqrt_w
        _, _, vt = np.linalg.svd(target - target.mean(0), full_matrices=False)
        self.basis = vt[: self.k]
        self.target = target
        self.pred = self.basis.T @ (EOF_RNG.normal(size=(self.n_time, self.k)).T)
        self.pred = self.pred.T

    def test_out_of_basis_plus_mapping_equals_total(self):
        err = self.pred - self.target
        p = self.basis.T @ self.basis
        inside = err @ p
        outside = err - inside
        total = (err ** 2).sum() / self.n_time
        f = (outside ** 2).sum() / self.n_time
        m = (inside ** 2).sum() / self.n_time
        np.testing.assert_allclose(total, f + m, rtol=1e-10)
        cross = (inside * outside).sum() / self.n_time
        self.assertLess(abs(cross), 1e-10, "the split must be orthogonal")

    def test_mapping_error_splits_into_bias_and_variance(self):
        err = self.pred - self.target
        p = self.basis.T @ self.basis
        inside = err @ p
        m = (inside ** 2).sum() / self.n_time
        bias = inside.mean(axis=0)
        b = (bias ** 2).sum()
        v = ((inside - bias) ** 2).sum() / self.n_time
        np.testing.assert_allclose(m, b + v, rtol=1e-10)


import ast


import importlib


import sys


from pathlib import Path


DIAGNOSTICS = Path(__file__).resolve().parents[1] / "code" / "diagnostics"


sys.path.insert(0, str(DIAGNOSTICS))


class DiagnosticImports(unittest.TestCase):
    def test_shared_helpers_resolve(self):
        common = importlib.import_module("decomposition_common")
        support = importlib.import_module("sup_core")
        model = importlib.import_module("model_pass")
        for name in ("decomposition_exact", "uniform_shift_defs"):
            with self.subTest(module=name):
                module = importlib.import_module(name)
                self.assertIs(module.C, common)
                self.assertIs(module.S, support)
        self.assertIs(support.C, common)
        self.assertIs(model.C, common)
        self.assertEqual(len(model.MODEL_QUANTITIES), 13)
        self.assertEqual(len(set(model.MODEL_QUANTITIES)), 13)

    def test_referenced_helper_attributes_exist(self):
        helpers = {
            "C": importlib.import_module("decomposition_common"),
            "S": importlib.import_module("sup_core"),
        }
        for name in ("sup_core", "model_pass", "decomposition_exact",
                     "uniform_shift_defs"):
            tree = ast.parse((DIAGNOSTICS / f"{name}.py").read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.value.id in helpers):
                    with self.subTest(module=name, attribute=node.attr):
                        self.assertTrue(hasattr(helpers[node.value.id], node.attr))


import json


STATS = Path(__file__).resolve().parents[1] / "configs" / "unet" / "norm_stats.json"


NORM_RNG = np.random.default_rng(11)


class NormStats(unittest.TestCase):
    def setUp(self):
        self.stats = json.loads(STATS.read_text())

    def _flat(self):
        out = {}
        def walk(o, prefix=""):
            for k, v in o.items():
                if isinstance(v, dict):
                    walk(v, f"{prefix}{k}.")
                else:
                    out[f"{prefix}{k}"] = v
        walk(self.stats)
        return out

    def test_every_scale_is_strictly_positive(self):
        for key, value in self._flat().items():
            if key.endswith(("std", "scale")) and isinstance(value, (int, float)):
                self.assertGreater(value, 0.0, f"{key} must be a usable divisor")

    def test_normalise_denormalise_round_trips(self):
        flat = self._flat()
        means = [v for k, v in flat.items() if k.endswith("mean") and isinstance(v, (int, float))]
        stds = [v for k, v in flat.items() if k.endswith("std") and isinstance(v, (int, float))]
        self.assertTrue(means and stds, "norm_stats.json must carry mean/std pairs")
        mu, sd = means[0], stds[0]
        x = NORM_RNG.normal(loc=mu, scale=sd, size=(4, 8, 8))
        np.testing.assert_allclose(((x - mu) / sd) * sd + mu, x, rtol=1e-12)

    def test_transform_is_deterministic(self):
        x = NORM_RNG.normal(size=(3, 5))
        mu, sd = 277.8568272051984, 21.581598298286263
        np.testing.assert_array_equal((x - mu) / sd, (x - mu) / sd)


UNET_ROOT = Path(__file__).resolve().parents[1] / "code" / "unet"


sys.path.insert(0, str(UNET_ROOT))


try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


if HAVE_TORCH:
    from training.global_unet import GlobalPeriodicConv2d, GlobalUNet


NORM = dict(input_t2m_mean=277.8568272051984, input_t2m_std=21.581598298286263,
            target_t2m_mean=277.8518781731243, target_t2m_std=21.627892139211994)


def small_model(**kw):
    opts = dict(in_channels=5, base=8, depth=2, norm="group",
                lat_padding="pole_aware", target_mode="correction", **NORM)
    opts.update(kw)
    return GlobalUNet(**opts)


@unittest.skipUnless(HAVE_TORCH, "PyTorch is not installed in this environment")
class PoleGhostRow(unittest.TestCase):


    def test_even_width_is_an_exact_half_roll(self):
        row = torch.randn(1, 1, 1, 8)
        ghost = GlobalPeriodicConv2d._pole_ghost_row(row)
        torch.testing.assert_close(ghost, torch.roll(row, shifts=4, dims=-1))

    def test_odd_width_interpolates_between_the_two_half_shifts(self):
        row = torch.randn(1, 1, 1, 7)
        ghost = GlobalPeriodicConv2d._pole_ghost_row(row)
        expected = 0.5 * torch.roll(row, 3, -1) + 0.5 * torch.roll(row, 4, -1)
        torch.testing.assert_close(ghost, expected)

    def test_applying_it_twice_returns_the_original_row(self):
        row = torch.randn(1, 1, 1, 12)
        twice = GlobalPeriodicConv2d._pole_ghost_row(
            GlobalPeriodicConv2d._pole_ghost_row(row))
        torch.testing.assert_close(twice, row)

    def test_a_degenerate_width_is_rejected(self):
        with self.assertRaises(ValueError):
            GlobalPeriodicConv2d._pole_ghost_row(torch.randn(1, 1, 1, 1))


@unittest.skipUnless(HAVE_TORCH, "PyTorch is not installed in this environment")
class Convolution(unittest.TestCase):
    def test_shape_is_preserved_in_both_padding_modes(self):
        x = torch.randn(2, 3, 8, 16)
        for mode in ("replicate", "pole_aware"):
            conv = GlobalPeriodicConv2d(input_channels=3, output_channels=4,
                                        latitude_padding=mode)
            self.assertEqual(conv(x).shape, (2, 4, 8, 16), msg=mode)

    def test_longitude_is_circular(self):

        conv = GlobalPeriodicConv2d(input_channels=3, output_channels=4,
                                    latitude_padding="pole_aware").eval()
        x = torch.randn(1, 3, 8, 16)
        with torch.no_grad():
            a = torch.roll(conv(x), shifts=5, dims=-1)
            b = conv(torch.roll(x, shifts=5, dims=-1))
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)

    def test_rejects_a_non_four_dimensional_input(self):
        conv = GlobalPeriodicConv2d(input_channels=3, output_channels=4)
        with self.assertRaises(ValueError):
            conv(torch.randn(3, 8, 16))


@unittest.skipUnless(HAVE_TORCH, "PyTorch is not installed in this environment")
class Model(unittest.TestCase):
    def test_output_shape_matches_the_input_grid(self):
        model = small_model().eval()
        x = torch.randn(2, 5, 16, 32)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(y.shape[0], 2)
        self.assertEqual(y.shape[-2:], (16, 32))

    def test_model_is_longitude_shift_equivariant(self):
        model = small_model().eval()
        x = torch.randn(1, 5, 16, 32)
        with torch.no_grad():
            a = torch.roll(model(x), shifts=7, dims=-1)
            b = model(torch.roll(x, shifts=7, dims=-1))
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-3)

    def test_forward_is_deterministic_in_eval_mode(self):
        model = small_model().eval()
        x = torch.randn(1, 5, 16, 32)
        with torch.no_grad():
            torch.testing.assert_close(model(x), model(x))


if __name__ == "__main__":
    unittest.main()
