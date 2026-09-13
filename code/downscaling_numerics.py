"""Shared area weights, spectra, regions, seasons, and block resampling.
The function bodies are retained unchanged from the numerical evaluation code.
Cosine-latitude and exact-cell-area weights remain distinct conventions.
The legacy bootstrap_quantities reports MSE ratios; the SSP5-8.5 reducer
retains its separate final RMSE-ratio calculations."""

from __future__ import annotations

import math
import numpy as np


def season_name(date_string: str) -> str:
    month = int(date_string[5:7])
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def area_weights_from_lat(lat_centers) -> tuple[np.ndarray, np.ndarray]:

    w = np.cos(np.deg2rad(np.asarray(lat_centers, dtype=np.float64)))
    w = w / w.mean()
    return w, np.sqrt(w)


def unet_pipeline_row_weights(lat_file: np.ndarray) -> np.ndarray:


    w = np.cos(np.deg2rad(np.asarray(lat_file, dtype=np.float64))).astype(np.float32)
    w /= w.mean()
    return np.asarray(w, dtype=np.float64)


def region_masks(lat_deg, lsm, oro_m) -> dict:


    height, width = lsm.shape
    lat = np.broadcast_to(np.asarray(lat_deg, dtype=np.float64)[:, None],
                          (height, width))
    ones = np.ones((height, width), dtype=bool)
    return {
        "global": ones,
        "land": lsm >= 0.5,
        "ocean": lsm < 0.5,
        "elevation_gt_1000m": oro_m > 1000.0,
        "elevation_le_1000m": oro_m <= 1000.0,
        "tropics": np.abs(lat) < 23.5,
        "midlat_north": (lat >= 23.5) & (lat < 60.0),
        "midlat_south": (lat <= -23.5) & (lat > -60.0),
        "highlat_north": (lat >= 60.0) & (lat < 80.0),
        "highlat_south": (lat <= -60.0) & (lat > -80.0),
        "arctic_gt_80N": lat >= 80.0,
        "antarctic_lt_80S": lat <= -80.0,
    }


def exact_cell_area_row_weights(latitude_deg: np.ndarray) -> np.ndarray:

    lat = np.asarray(latitude_deg, dtype=np.float64)
    step = 180.0 / float(len(lat))
    expected = -90.0 + (np.arange(len(lat)) + 0.5) * step
    if np.max(np.abs(lat - expected)) > 1.0e-5:
        raise SystemExit("latitude centers depart from the regular grid")
    edges = -90.0 + np.arange(len(lat) + 1, dtype=np.float64) * step
    weights = np.sin(np.deg2rad(edges[1:])) - np.sin(np.deg2rad(edges[:-1]))
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 2.0, atol=2e-14):
        raise SystemExit("invalid regular-grid cell-area weights")
    return weights


def cubic_latitude_interpolation(source_latitude_deg, target_latitude_deg):
    source = np.asarray(source_latitude_deg, dtype=np.float64)
    target = np.asarray(target_latitude_deg, dtype=np.float64)
    indices = np.empty((len(target), 4), dtype=np.int64)
    weights = np.empty((len(target), 4), dtype=np.float64)
    for row, value in enumerate(target):
        j = int(np.searchsorted(source, value))
        base = min(max(j - 2, 0), len(source) - 4)
        idx = np.arange(base, base + 4, dtype=np.int64)
        x = source[idx]
        w = np.ones(4, dtype=np.float64)
        for p in range(4):
            for q in range(4):
                if p != q:
                    w[p] *= (value - x[q]) / (x[p] - x[q])
        indices[row], weights[row] = idx, w
    if np.max(np.abs(weights.sum(axis=1) - 1.0)) > 2e-13:
        raise SystemExit("cubic interpolation does not preserve constants")
    return indices, weights


def derive_spherical_ell_max(latitude_deg, longitude_deg) -> int:

    lat = np.asarray(latitude_deg, dtype=np.float64)
    lon = np.asarray(longitude_deg, dtype=np.float64)
    dlat = float(np.median(np.diff(lat)))
    dlon = 360.0 / float(len(lon))
    if abs(abs(dlat) * len(lat) - 180.0) > 1e-5:
        raise SystemExit("latitude sampling does not cover 180 degrees")
    coarsest = max(abs(dlat), abs(dlon))
    return int(math.floor(360.0 / (6.0 * coarsest) + 1e-12))


class GaussLegendreSHTPlan:


    def __init__(self, latitude_deg, longitude_deg, *, ell_max=None,
                 gauss_latitudes: int = 640, basis_dtype=np.float32) -> None:
        self.latitude = np.asarray(latitude_deg, dtype=np.float64)
        self.longitude = np.asarray(longitude_deg, dtype=np.float64)
        self.ell_max = int(derive_spherical_ell_max(self.latitude, self.longitude)
                           if ell_max is None else ell_max)
        self.n_gauss = int(gauss_latitudes)
        if self.n_gauss <= self.ell_max:
            raise SystemExit("Gauss quadrature needs more rows than ell_max")
        self.gauss_x, self.gauss_weights = np.polynomial.legendre.leggauss(self.n_gauss)
        self.gauss_latitude = np.rad2deg(np.arcsin(self.gauss_x))
        self.interp_indices, self.interp_weights = cubic_latitude_interpolation(
            self.latitude, self.gauss_latitude)
        self.source_area_weights = exact_cell_area_row_weights(self.latitude)
        self.basis_dtype = np.dtype(basis_dtype)
        basis = np.zeros((self.ell_max + 1, self.ell_max + 1, self.n_gauss),
                         dtype=self.basis_dtype)


        sin_theta = np.sqrt(np.maximum(1.0 - self.gauss_x ** 2, 0.0))
        previous_diagonal = np.full(self.n_gauss, 1.0 / math.sqrt(4.0 * math.pi),
                                    dtype=np.float64)
        for m in range(self.ell_max + 1):
            if m == 0:
                diagonal = previous_diagonal
            else:
                diagonal = (-math.sqrt((2.0 * m + 1.0) / (2.0 * m))
                            * sin_theta * previous_diagonal)
                previous_diagonal = diagonal
            normalized = np.empty((self.ell_max - m + 1, self.n_gauss),
                                  dtype=np.float64)
            normalized[0] = diagonal
            if m < self.ell_max:
                normalized[1] = math.sqrt(2.0 * m + 3.0) * self.gauss_x * diagonal
            for ell in range(m + 2, self.ell_max + 1):
                denominator = float(ell * ell - m * m)
                coefficient_a = math.sqrt((4.0 * ell * ell - 1.0) / denominator)
                coefficient_b = math.sqrt(
                    (2.0 * ell + 1.0) * ((ell - 1.0) ** 2 - m * m)
                    / ((2.0 * ell - 3.0) * denominator))
                local = ell - m
                normalized[local] = (
                    coefficient_a * self.gauss_x * normalized[local - 1]
                    - coefficient_b * normalized[local - 2])
            weighted = 2.0 * np.pi * normalized * self.gauss_weights[None, :]
            if not np.isfinite(weighted).all():
                raise SystemExit(f"non-finite Legendre basis at m={m}")
            basis[m, m:, :] = weighted.astype(self.basis_dtype)
        self.basis = basis

    def interpolate_numpy(self, fields: np.ndarray) -> np.ndarray:
        gathered = np.asarray(fields)[:, self.interp_indices, :]
        return np.einsum("fgjw,gj->fgw", gathered, self.interp_weights,
                         optimize=True)

    def prepare_numpy(self, fields: np.ndarray, *, remove_global_mean: bool):
        value = np.asarray(fields, dtype=np.float64)
        if value.ndim != 3 or value.shape[1:] != (len(self.latitude),
                                                  len(self.longitude)):
            raise SystemExit("spherical fields must be (field, lat, lon)")
        denom = len(self.longitude) * self.source_area_weights.sum()
        source_mean = np.einsum("frw,r->f", value, self.source_area_weights,
                                optimize=True) / denom
        if remove_global_mean:
            value = value - source_mean[:, None, None]
        gl = self.interpolate_numpy(value)
        if remove_global_mean:
            gl_mean = 0.5 * np.einsum("fgw,g->f", gl, self.gauss_weights,
                                      optimize=True) / len(self.longitude)
            gl = gl - gl_mean[:, None, None]
        return gl

    @staticmethod
    def degree_statistics_numpy(coefficients: np.ndarray):
        coeff = np.asarray(coefficients)
        nf, nl, _ = coeff.shape
        energy = np.zeros((nf, nl), dtype=np.float64)
        c_ell = np.zeros((nf, nl), dtype=np.float64)
        for ell in range(nl):
            values = np.abs(coeff[:, ell, : ell + 1]) ** 2
            total = values[:, 0]
            if ell:
                total = total + 2.0 * values[:, 1:].sum(axis=1, dtype=np.float64)
            energy[:, ell] = total / (4.0 * np.pi)
            c_ell[:, ell] = total / (2 * ell + 1)
        return energy, c_ell

    def transform_c_ell(self, fields: np.ndarray, *,
                        remove_global_mean: bool = True) -> np.ndarray:

        gl = np.asarray(self.prepare_numpy(fields,
                                           remove_global_mean=remove_global_mean),
                        dtype=np.float32)
        fourier = (np.fft.rfft(gl, axis=-1).astype(np.complex64)
                   / np.float32(len(self.longitude)))
        nf = gl.shape[0]
        coeff = np.zeros((nf, self.ell_max + 1, self.ell_max + 1),
                         dtype=np.complex64)
        for m in range(self.ell_max + 1):
            coeff[:, m:, m] = fourier[:, :, m] @ self.basis[m, m:, :].T
        _energy, c_ell = self.degree_statistics_numpy(coeff)
        return c_ell


def one_sided_zonal_power(field: np.ndarray, *, detrend_constant: bool = True):


    value = np.asarray(field, dtype=np.float64)
    if detrend_constant:
        value = value - value.mean(axis=-1, keepdims=True)
    width = value.shape[-1]
    coeff = np.fft.rfft(value, axis=-1, norm="ortho")
    power = coeff.real * coeff.real + coeff.imag * coeff.imag
    if width % 2 == 0:
        power[..., 1:-1] *= 2.0
    else:
        power[..., 1:] *= 2.0
    return power


def zonal_row_weighted_power(fields: np.ndarray, row_weights: np.ndarray,
                             k_max: int) -> np.ndarray:


    power = one_sided_zonal_power(fields, detrend_constant=True)
    return np.einsum("frk,r->fk", power[..., : k_max + 1], row_weights,
                     optimize=True)


def circular_block_bootstrap_indices(n_days, block_len, n_resamples, rng):


    if block_len < 1 or block_len > n_days:
        raise SystemExit("block length must be in [1, n_days]")
    n_blocks = math.ceil(n_days / block_len)
    starts = rng.integers(0, n_days, size=(n_resamples, n_blocks), dtype=np.int64)
    idx = (starts[:, :, None]
           + np.arange(block_len, dtype=np.int64)[None, None, :]) % n_days
    return idx.reshape(n_resamples, n_blocks * block_len)[:, :n_days]


def bootstrap_quantities(daily: dict, idx: np.ndarray, std: float) -> dict:


    def rs(x):
        return x[idx].sum(axis=1)

    sw_c = rs(daily["w_c"])
    sw_u = rs(daily["w_u"])
    mse_bil_c = rs(daily["bil_e2_c"]) / sw_c
    mse_cca = rs(daily["cca_e2"]) / sw_c
    mse_bil_u = rs(daily["bil_e2_u"]) / sw_u
    mse_unet = rs(daily["unet_e2"]) / sw_u
    rmse_cca = np.sqrt(mse_cca) * std
    rmse_unet = np.sqrt(mse_unet) * std
    return {
        "bilinear_rmse_K": np.sqrt(mse_bil_c) * std,
        "bilinear_rmse_unet_pipeline_K": np.sqrt(mse_bil_u) * std,
        "cca_rmse_K": rmse_cca,
        "unet_rmse_K": rmse_unet,
        "cca_skill_vs_bilinear": 1.0 - mse_cca / mse_bil_c,
        "unet_skill_vs_bilinear": 1.0 - mse_unet / mse_bil_u,
        "delta_rmse_K": rmse_cca - rmse_unet,
        "skill_vs_cca": 1.0 - mse_unet / mse_cca,
    }
