import math
from typing import Tuple

import numpy as np


# ----------- Core PDFs and samplers -----------

def pdf_gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-9)
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (math.sqrt(2.0 * math.pi) * sigma)


def sample_gaussian(n: int, rng: np.random.Generator, mu: float, sigma: float) -> np.ndarray:
    return rng.normal(mu, max(sigma, 1e-9), size=n)


def pdf_laplace(x: np.ndarray, mu: float, b: float) -> np.ndarray:
    b = max(float(b), 1e-9)
    return 0.5 / b * np.exp(-np.abs(x - mu) / b)


def sample_laplace(n: int, rng: np.random.Generator, mu: float, b: float) -> np.ndarray:
    return rng.laplace(mu, max(b, 1e-9), size=n)


def pdf_student_t(x: np.ndarray, nu: float, mu: float, sigma: float) -> np.ndarray:
    nu = max(float(nu), 1.0)
    sigma = max(float(sigma), 1e-9)
    z = (x - mu) / sigma
    c = math.gamma((nu + 1) / 2) / (math.sqrt(nu * math.pi) * math.gamma(nu / 2))
    return c * (1 + (z * z) / nu) ** (-(nu + 1) / 2) / sigma


def sample_student_t(n: int, rng: np.random.Generator, nu: float, mu: float, sigma: float) -> np.ndarray:
    nu = max(float(nu), 1.0)
    sigma = max(float(sigma), 1e-9)
    return mu + sigma * rng.standard_t(df=nu, size=n)


def pdf_lognormal(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-9)
    x_safe = np.maximum(x, 1e-12)
    z = (np.log(x_safe) - mu) / sigma
    return np.exp(-0.5 * z * z) / (x_safe * sigma * math.sqrt(2.0 * math.pi))


def sample_lognormal(n: int, rng: np.random.Generator, mu: float, sigma: float) -> np.ndarray:
    return rng.lognormal(mean=mu, sigma=max(sigma, 1e-9), size=n)


def pdf_beta(x: np.ndarray, a: float, b: float) -> np.ndarray:
    a = max(float(a), 1e-3)
    b = max(float(b), 1e-3)
    x_clamped = np.clip(x, 1e-12, 1 - 1e-12)
    B = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
    return (x_clamped ** (a - 1)) * ((1 - x_clamped) ** (b - 1)) / B


def sample_beta(n: int, rng: np.random.Generator, a: float, b: float) -> np.ndarray:
    a = max(float(a), 1e-3)
    b = max(float(b), 1e-3)
    return rng.beta(a, b, size=n)


def pdf_uniform(x: np.ndarray, a: float, b: float) -> np.ndarray:
    a, b = (a, b) if a <= b else (b, a)
    y = np.zeros_like(x, dtype=float)
    inside = (x >= a) & (x <= b)
    width = max(b - a, 1e-12)
    y[inside] = 1.0 / width
    return y


def sample_uniform(n: int, rng: np.random.Generator, a: float, b: float) -> np.ndarray:
    a, b = (a, b) if a <= b else (b, a)
    return rng.uniform(a, b, size=n)


def pdf_triangular(x: np.ndarray, a: float, b: float, m: float) -> np.ndarray:
    a, b = (a, b) if a <= b else (b, a)
    if m < a:
        m = a
    if m > b:
        m = b
    y = np.zeros_like(x, dtype=float)
    denom_left = (b - a) * max(m - a, 1e-12)
    denom_right = (b - a) * max(b - m, 1e-12)
    left = (x >= a) & (x <= m)
    right = (x >= m) & (x <= b)
    y[left] = 2 * (x[left] - a) / denom_left
    y[right] = 2 * (b - x[right]) / denom_right
    return y


def sample_triangular(n: int, rng: np.random.Generator, a: float, b: float, m: float) -> np.ndarray:
    a, b = (a, b) if a <= b else (b, a)
    m = min(max(m, a), b)
    return rng.triangular(a, m, b, size=n)


def _erf_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sign = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * np.exp(-a * a)
    return sign * y


def pdf_skew_normal(x: np.ndarray, mu: float, sigma: float, alpha: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-9)
    z = (x - mu) / sigma
    c = 2.0 / sigma
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    t = alpha * z
    Phi = 0.5 * (1.0 + _erf_vec(t / math.sqrt(2.0)))
    return c * phi * Phi


# ----------- MoG (PDF + sampler) -----------

def pdf_mog(x: np.ndarray, mus: np.ndarray, sigmas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    K = len(mus)
    px = np.zeros_like(x, dtype=float)
    for k in range(K):
        px += float(weights[k]) * pdf_gaussian(x, float(mus[k]), float(sigmas[k]))
    return px


def sample_mog(n: int, rng: np.random.Generator, mus: np.ndarray, sigmas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    K = len(mus)
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 1e-9, None)
    weights = weights / weights.sum()
    ks = rng.choice(K, size=n, p=weights)
    samples = np.empty(n, dtype=float)
    for k in range(K):
        idx = np.where(ks == k)[0]
        if idx.size:
            samples[idx] = rng.normal(float(mus[k]), max(float(sigmas[k]), 1e-9), size=idx.size)
    return samples


# ----------- 2D -> 1D projected fun shapes -----------

def pdf_proj_twomoons_x(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    p = np.zeros_like(x)
    mask_u = (x > -1.0) & (x < 1.0)
    if np.any(mask_u):
        p[mask_u] += 0.5 * (1.0 / (math.pi * np.sqrt(np.clip(1.0 - x[mask_u] * x[mask_u], 1e-12, None))))
    mask_l = (x > 0.0) & (x < 2.0)
    if np.any(mask_l):
        u = 1.0 - x[mask_l]
        p[mask_l] += 0.5 * (1.0 / (math.pi * np.sqrt(np.clip(1.0 - u * u, 1e-12, None))))
    return p


def sample_proj_twomoons_x(n: int, rng: np.random.Generator) -> np.ndarray:
    phi = rng.uniform(0.0, math.pi, size=n)
    branch = rng.uniform(0.0, 1.0, size=n) < 0.5
    x_u = np.cos(phi)
    x_l = 1.0 - np.cos(phi)
    return np.where(branch, x_u, x_l)


def sample_proj_checkerboard_x(
    n: int, rng: np.random.Generator, m: int = 4, delta: float = 0.4, noise: float = 0.05
) -> np.ndarray:
    i = rng.integers(0, m, size=n)
    cx = -1.0 + (i + 0.5) * (2.0 / m)
    sgn = np.where(rng.uniform(0.0, 1.0, size=n) < 0.5, 1.0, -1.0)
    x = cx + sgn * delta
    if noise > 0:
        x = x + rng.normal(0.0, noise, size=n)
    return x


def sample_proj_spiral_x(
    n: int, rng: np.random.Generator, r0: float = 0.3, k: float = 0.06, phi_max: float = 4.0 * math.pi
) -> np.ndarray:
    phi = rng.uniform(0.0, float(phi_max), size=n)
    r = r0 + k * phi
    return r * np.cos(phi)


def pdf_arcsine_projected_circle(x: np.ndarray, r: float) -> np.ndarray:
    r = max(float(r), 1e-9)
    y = np.zeros_like(x, dtype=float)
    inside = np.abs(x) < r
    denom = np.sqrt(np.maximum(0.0, r * r - (x[inside] * x[inside])))
    y[inside] = 1.0 / (math.pi * np.maximum(denom, 1e-12))
    return y


def sample_arcsine_projected_circle(n: int, rng: np.random.Generator, r: float) -> np.ndarray:
    r = max(float(r), 1e-9)
    phi = rng.uniform(0.0, 2.0 * math.pi, size=n)
    return r * np.cos(phi)


def pdf_mixture_arcsine(x: np.ndarray, radii: np.ndarray, weights: np.ndarray) -> np.ndarray:
    radii = np.asarray(radii, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 1e-12, None)
    weights = weights / weights.sum()
    px = np.zeros_like(x, dtype=float)
    for r, w in zip(radii, weights):
        px += float(w) * pdf_arcsine_projected_circle(x, float(r))
    return px


def sample_mixture_arcsine(n: int, rng: np.random.Generator, radii: np.ndarray, weights: np.ndarray) -> np.ndarray:
    K = len(radii)
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 1e-12, None)
    weights = weights / weights.sum()
    ks = rng.choice(K, size=n, p=weights)
    out = np.empty(n, dtype=float)
    for k in range(K):
        idx = np.where(ks == k)[0]
        if idx.size:
            out[idx] = sample_arcsine_projected_circle(idx.size, rng, float(radii[k]))
    return out

