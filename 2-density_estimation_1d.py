import math
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

import numpy as np
import streamlit as st


try:
    import plotly.graph_objects as go  # type: ignore
    _PLOTLY_AVAILABLE = True
except Exception:
    _PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt  # type: ignore
    _MATPLOTLIB_AVAILABLE = True
except Exception:
    _MATPLOTLIB_AVAILABLE = False


st.set_page_config(page_title="1D Density Estimation Demo", layout="wide")


# ---------- True 1D densities ----------

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
    # Standardized t scaled and shifted
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


# Additional 1D densities

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


def pdf_skew_normal(x: np.ndarray, mu: float, sigma: float, alpha: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-9)
    z = (x - mu) / sigma
    c = 2.0 / sigma
    # standard normal pdf and cdf
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    t = alpha * z
    Phi = 0.5 * (1.0 + _erf_vec(t / math.sqrt(2.0)))
    return c * phi * Phi


def sample_skew_normal(n: int, rng: np.random.Generator, mu: float, sigma: float, alpha: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-9)
    delta = alpha / math.sqrt(1.0 + alpha * alpha)
    u0 = rng.normal(0.0, 1.0, size=n)
    u1 = rng.normal(0.0, 1.0, size=n)
    y = delta * np.abs(u0) + math.sqrt(max(1e-12, 1.0 - delta * delta)) * u1
    return mu + sigma * y


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


# ---------- 2D -> 1D projected fun shapes ----------

def pdf_proj_twomoons_x(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    p = np.zeros_like(x)
    # Upper moon contribution on [-1,1]
    mask_u = (x > -1.0) & (x < 1.0)
    if np.any(mask_u):
        p[mask_u] += 0.5 * (1.0 / (math.pi * np.sqrt(np.clip(1.0 - x[mask_u] * x[mask_u], 1e-12, None))))
    # Lower moon contribution on [0,2]; transform u = 1 - x in [-1,1]
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


def mc_pdf_from_sampler(
    sampler_fn, support: Tuple[float, float], cache_key: str, rng: np.random.Generator, n_mc: int = 30000
):
    cache = st.session_state.setdefault("_mc_cache", {})
    if cache_key not in cache:
        x_mc = sampler_fn(n_mc)
        h = silverman_bandwidth(x_mc)
        cache[cache_key] = (x_mc, h)

    x_mc, h = cache[cache_key]

    def _pdf(xs: np.ndarray) -> np.ndarray:
        return kde_gaussian(x_mc, xs, h)

    return _pdf

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


# ---------- KDE (Gaussian) ----------

def silverman_bandwidth(x: np.ndarray) -> float:
    n = max(1, x.size)
    std = float(np.std(x, ddof=1)) if n > 1 else 1.0
    iqr = float(np.subtract(*np.percentile(x, [75, 25]))) if n > 1 else 0.0
    sigma = min(std, iqr / 1.349) if (std > 0 or iqr > 0) else 1.0
    return 0.9 * sigma * n ** (-1 / 5)


def scott_bandwidth(x: np.ndarray) -> float:
    n = max(1, x.size)
    std = float(np.std(x, ddof=1)) if n > 1 else 1.0
    return std * n ** (-1 / 5)


def kde_gaussian(x: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    h = max(float(bandwidth), 1e-9)
    diffs = (grid[:, None] - x[None, :]) / h
    K = np.exp(-0.5 * diffs * diffs) / math.sqrt(2.0 * math.pi)
    return K.mean(axis=1) / h


# ---------- MoG fit via simple EM ----------

@dataclass
class GMMParams:
    mus: np.ndarray
    sigmas: np.ndarray
    weights: np.ndarray


def em_fit_gmm(x: np.ndarray, K: int, max_iter: int = 100, tol: float = 1e-6, rng: Optional[np.random.Generator] = None) -> GMMParams:
    rng = rng or np.random.default_rng(0)
    x = x.reshape(-1)
    n = x.size
    # Init means by percentiles, weights uniform, variances to overall std
    mus = np.percentile(x, np.linspace(5, 95, K))
    sig = np.std(x) if n > 1 else 1.0
    sigmas = np.full(K, max(sig, 1e-3))
    weights = np.ones(K) / K

    def loglik(mus_: np.ndarray, sigmas_: np.ndarray, weights_: np.ndarray) -> float:
        px = np.zeros(n, dtype=float)
        for k in range(K):
            px += float(weights_[k]) * pdf_gaussian(x, float(mus_[k]), float(sigmas_[k]))
        return float(np.log(np.clip(px, 1e-300, None)).sum())

    prev_ll = loglik(mus, sigmas, weights)
    for _ in range(max_iter):
        # E-step
        resp = np.zeros((n, K))
        for k in range(K):
            resp[:, k] = weights[k] * pdf_gaussian(x, mus[k], sigmas[k])
        s = resp.sum(axis=1, keepdims=True)
        s = np.clip(s, 1e-300, None)
        resp /= s

        # M-step
        Nk = resp.sum(axis=0)
        Nk = np.clip(Nk, 1e-12, None)
        weights = Nk / n
        mus = (resp * x[:, None]).sum(axis=0) / Nk
        diffs2 = (x[:, None] - mus[None, :]) ** 2
        sigmas = np.sqrt((resp * diffs2).sum(axis=0) / Nk)
        sigmas = np.clip(sigmas, 1e-3, None)

        # Log-likelihood of the observed data under current params
        ll = loglik(mus, sigmas, weights)
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return GMMParams(mus=mus, sigmas=sigmas, weights=weights)


# ---------- MLE for simple families ----------

def fit_gaussian_mle(x: np.ndarray) -> Tuple[float, float]:
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=0))
    return mu, max(sigma, 1e-9)


def fit_laplace_mle(x: np.ndarray) -> Tuple[float, float]:
    mu = float(np.median(x))
    b = float(np.mean(np.abs(x - mu)))
    return mu, max(b, 1e-9)


# ---------- UI State ----------


def ensure_state():
    if "rng_seed" not in st.session_state:
        st.session_state.rng_seed = 0
    if "rng" not in st.session_state:
        st.session_state.rng = np.random.default_rng(int(st.session_state.rng_seed))
    if "samples" not in st.session_state:
        st.session_state.samples = np.array([], dtype=float)


def get_rng() -> np.random.Generator:
    return st.session_state.rng


def main():
    ensure_state()
    st.title("1D Density Estimation Demo")
    st.caption("Draw samples from a chosen 1D distribution, then fit parametric or non-parametric models and compare.")

    with st.sidebar:
        st.header("Sampling")
        # Top: manual resample button
        resample_clicked = st.button("Draw new samples")
        # Sample size first (seed moved to bottom)
        N = st.slider("Sample size N", min_value=20, max_value=5000, value=500, step=10)

        st.markdown("**True density**")
        true_name = st.selectbox("Family", [
            "Gaussian",
            "Mixture of Gaussians (2)",
            "Mixture of Gaussians (3)",
            "Laplace",
            "Student-t",
            "Lognormal",
            "Beta [0,1]",
            "Uniform",
            "Triangular",
            "Projected Circle (arcsine)",
            "Projected Rings (mixture of arcsine)",
            "Projected Two Moons (x)",
            "Projected Checkerboard (x)",
            "Projected Spiral (x)",
        ], index=0)

        # Parameter controls per family (collapsed by default). Hide expander for families with no params.
        if true_name == "Projected Two Moons (x)":
            sampler = lambda n: sample_proj_twomoons_x(n, get_rng())
            pdf = pdf_proj_twomoons_x
            support = (-1.0, 2.0)
        else:
            with st.expander("True density parameters", expanded=False):
                if true_name == "Gaussian":
                    mu = st.number_input("μ", value=0.0)
                    sigma = st.number_input("σ", value=1.0, min_value=0.0)
                    sampler = lambda n: sample_gaussian(n, get_rng(), mu, sigma)
                    pdf = lambda xs: pdf_gaussian(xs, mu, sigma)
                    support = (mu - 5 * max(sigma, 1e-9), mu + 5 * max(sigma, 1e-9))
                elif true_name == "Mixture of Gaussians (2)":
                    c1, c2 = st.columns(2)
                    with c1:
                        mu1 = st.number_input("μ₁", value=-2.0)
                        s1 = st.number_input("σ₁", value=0.8, min_value=0.0)
                        w1 = st.slider("w₁", min_value=0.0, max_value=1.0, value=0.5)
                    with c2:
                        mu2 = st.number_input("μ₂", value=2.0)
                        s2 = st.number_input("σ₂", value=0.8, min_value=0.0)
                        w2 = 1.0 - w1
                    mus = np.array([mu1, mu2])
                    sigmas = np.array([s1, s2])
                    weights = np.array([w1, w2])
                    sampler = lambda n: sample_mog(n, get_rng(), mus, sigmas, weights)
                    pdf = lambda xs: pdf_mog(xs, mus, sigmas, weights)
                    support = (float(mus.min() - 5 * np.max(sigmas)), float(mus.max() + 5 * np.max(sigmas)))
                elif true_name == "Mixture of Gaussians (3)":
                    mu1 = st.number_input("μ₁", value=-4.0)
                    mu2 = st.number_input("μ₂", value=0.0)
                    mu3 = st.number_input("μ₃", value=3.0)
                    s1 = st.number_input("σ₁", value=0.6, min_value=0.0)
                    s2 = st.number_input("σ₂", value=0.9, min_value=0.0)
                    s3 = st.number_input("σ₃", value=0.5, min_value=0.0)
                    w1 = st.slider("w₁", 0.0, 1.0, 0.3)
                    w2 = st.slider("w₂", 0.0, 1.0, 0.5)
                    w3 = max(1.0 - w1 - w2, 1e-6)
                    wsum = w1 + w2 + w3
                    w1, w2, w3 = w1 / wsum, w2 / wsum, w3 / wsum
                    mus = np.array([mu1, mu2, mu3])
                    sigmas = np.array([s1, s2, s3])
                    weights = np.array([w1, w2, w3])
                    sampler = lambda n: sample_mog(n, get_rng(), mus, sigmas, weights)
                    pdf = lambda xs: pdf_mog(xs, mus, sigmas, weights)
                    support = (float(mus.min() - 5 * np.max(sigmas)), float(mus.max() + 5 * np.max(sigmas)))
                elif true_name == "Laplace":
                    mu = st.number_input("μ", value=0.0)
                    b = st.number_input("b", value=1.0, min_value=0.0)
                    sampler = lambda n: sample_laplace(n, get_rng(), mu, b)
                    pdf = lambda xs: pdf_laplace(xs, mu, b)
                    support = (mu - 8 * max(b, 1e-9), mu + 8 * max(b, 1e-9))
                elif true_name == "Student-t":
                    nu = st.number_input("ν (dof)", value=5.0, min_value=1.0)
                    mu = st.number_input("μ", value=0.0)
                    sigma = st.number_input("σ", value=1.0, min_value=0.0)
                    sampler = lambda n: sample_student_t(n, get_rng(), nu, mu, sigma)
                    pdf = lambda xs: pdf_student_t(xs, nu, mu, sigma)
                    support = (mu - 10 * max(sigma, 1e-9), mu + 10 * max(sigma, 1e-9))
                elif true_name == "Lognormal":
                    mu = st.number_input("μ (log)", value=0.0)
                    sigma = st.number_input("σ (log)", value=0.5, min_value=0.0)
                    sampler = lambda n: sample_lognormal(n, get_rng(), mu, sigma)
                    pdf = lambda xs: pdf_lognormal(xs, mu, sigma)
                    support = (0.0, np.exp(mu + 5 * max(sigma, 1e-9)))
                elif true_name == "Beta [0,1]":
                    a = st.number_input("α", value=2.0, min_value=0.1)
                    b = st.number_input("β", value=5.0, min_value=0.1)
                    sampler = lambda n: sample_beta(n, get_rng(), a, b)
                    pdf = lambda xs: pdf_beta(xs, a, b)
                    support = (0.0, 1.0)
                elif true_name == "Uniform":
                    a = st.number_input("a", value=-3.0)
                    b = st.number_input("b", value=3.0)
                    sampler = lambda n: sample_uniform(n, get_rng(), a, b)
                    pdf = lambda xs: pdf_uniform(xs, a, b)
                    support = (min(a, b), max(a, b))
                elif true_name == "Triangular":
                    a = st.number_input("a", value=-3.0)
                    b = st.number_input("b", value=3.0)
                    m = st.number_input("mode", value=0.0)
                    sampler = lambda n: sample_triangular(n, get_rng(), a, b, m)
                    pdf = lambda xs: pdf_triangular(xs, a, b, m)
                    support = (min(a, b), max(a, b))
                elif true_name == "Projected Circle (arcsine)":
                    r = st.number_input("radius r", value=1.0, min_value=0.0)
                    sampler = lambda n: sample_arcsine_projected_circle(n, get_rng(), r)
                    pdf = lambda xs: pdf_arcsine_projected_circle(xs, r)
                    support = (-abs(r), abs(r))
                elif true_name == "Projected Rings (mixture of arcsine)":
                    r1 = st.number_input("r₁", value=0.7, min_value=0.0)
                    r2 = st.number_input("r₂", value=1.2, min_value=0.0)
                    r3 = st.number_input("r₃", value=1.7, min_value=0.0)
                    w1 = st.slider("w₁", 0.0, 1.0, 0.5)
                    w2 = st.slider("w₂", 0.0, 1.0, 0.3)
                    w3 = max(1.0 - w1 - w2, 1e-6)
                    wsum = w1 + w2 + w3
                    w1, w2, w3 = w1 / wsum, w2 / wsum, w3 / wsum
                    radii = np.array([r1, r2, r3])
                    weights = np.array([w1, w2, w3])
                    sampler = lambda n: sample_mixture_arcsine(n, get_rng(), radii, weights)
                    pdf = lambda xs: pdf_mixture_arcsine(xs, radii, weights)
                    rmax = float(np.max(radii))
                    support = (-rmax, rmax)
                elif true_name == "Projected Checkerboard (x)":
                    cb_m = st.slider("m (grid size)", 2, 8, 4)
                    cb_delta = st.slider("δ (offset)", 0.0, 0.8, 0.4)
                    cb_noise = st.slider("σ (noise)", 0.0, 0.2, 0.05)
                    sampler = lambda n, m=cb_m, d=cb_delta, s=cb_noise: sample_proj_checkerboard_x(n, get_rng(), m=m, delta=d, noise=s)
                    pdf = mc_pdf_from_sampler(
                        lambda n, m=cb_m, d=cb_delta, s=cb_noise: sample_proj_checkerboard_x(n, get_rng(), m=m, delta=d, noise=s),
                        (-1.5, 1.5), f"cb_{cb_m}_{cb_delta}_{cb_noise}", get_rng()
                    )
                    support = (-1.5, 1.5)
                else:  # Projected Spiral (x)
                    sp_r0 = st.number_input("r0", value=0.3)
                    sp_k = st.number_input("k", value=0.06)
                    sp_phi = st.number_input("φ_max", value=4.0 * math.pi)
                    sampler = lambda n, r0=sp_r0, k=sp_k, pm=sp_phi: sample_proj_spiral_x(n, get_rng(), r0=r0, k=k, phi_max=pm)
                    rmax = float(sp_r0 + sp_k * sp_phi)
                    support = (-rmax, rmax)
                    pdf = mc_pdf_from_sampler(
                        lambda n, r0=sp_r0, k=sp_k, pm=sp_phi: sample_proj_spiral_x(n, get_rng(), r0=r0, k=k, phi_max=pm),
                        support, f"spiral_{sp_r0}_{sp_k}_{sp_phi}", get_rng()
                    )

        # Auto-resample on family change, button click, or if empty
        if "_prev_true_name" not in st.session_state:
            st.session_state._prev_true_name = true_name
        family_changed = (st.session_state._prev_true_name != true_name)
        if family_changed or resample_clicked or st.session_state.samples.size == 0:
            st.session_state.samples = sampler(int(N))
            st.session_state._prev_true_name = true_name

        st.header("Fit")
        fit_name = st.selectbox("Estimator", [
            "Gaussian (MLE)",
            "Gaussian (User)",
            "Mixture of Gaussians (EM)",
            "KDE (Gaussian)"
        ], index=0)
        # Advanced fit options in an expander
        fit_params = {}
        with st.expander("Fit options", expanded=False):
            if fit_name == "Gaussian (User)":
                mu0 = st.number_input("μ (user)", value=0.0)
                sigma0 = st.number_input("σ (user)", value=1.0, min_value=0.0)
                fit_params.update(dict(mu=mu0, sigma=sigma0))
            elif fit_name == "Mixture of Gaussians (EM)":
                K = st.slider("Components K", 1, 6, 2)
                iters = st.slider("EM iters", 10, 300, 120, step=10)
                fit_params.update(dict(K=K, iters=iters))
            elif fit_name == "KDE (Gaussian)":
                bw_mode = st.selectbox("Bandwidth", ["Silverman", "Scott", "Manual"], index=0)
                bw = None
                if bw_mode == "Manual":
                    bw = st.number_input("h (bandwidth)", value=0.3, min_value=0.0)
                fit_params.update(dict(bw_mode=bw_mode, bw=bw))

        st.markdown("---")
        st.subheader("Display")
        st.checkbox("Show histogram", value=st.session_state.get("show_hist", True), key="show_hist")
        complex_true = true_name in (
            "Projected Circle (arcsine)",
            "Projected Rings (mixture of arcsine)",
            "Projected Two Moons (x)",
            "Projected Checkerboard (x)",
            "Projected Spiral (x)",
        )
        if not complex_true:
            st.checkbox("Show true density", value=st.session_state.get("show_true", False), key="show_true")
        else:
            st.session_state.show_true = False
        st.slider("Histogram bins", 10, 120, int(st.session_state.get("hist_bins", 50)), key="hist_bins")

        st.markdown("---")
        # Random seed at bottom for reproducibility
        seed = st.number_input("Random seed", value=int(st.session_state.rng_seed), step=1)
        if seed != st.session_state.rng_seed:
            st.session_state.rng_seed = int(seed)
            st.session_state.rng = np.random.default_rng(int(seed))

    # ---- SINGLE PLOT (samples, rug, true (opt), fit) ----

    x = st.session_state.samples
    if x.size == 0:
        st.stop()

    # Grid for PDF curves
    lo, hi = float(np.min(x)), float(np.max(x))
    span = hi - lo if hi > lo else 1.0
    pad = 0.1 * span
    gmin = min(support[0], lo - pad)
    gmax = max(support[1], hi + pad)
    grid = np.linspace(gmin, gmax, 800)

    # Session-backed bins, control placed below for alignment
    hist_n = int(st.session_state.get("hist_bins", 50))
    true_pdf = pdf(grid)
    # Rug markers (subset of samples), colored and lightly jittered for visibility
    subset_n = int(min(100, x.size))
    rng = get_rng()
    idx = rng.choice(x.size, size=subset_n, replace=False) if x.size > 0 else np.array([], dtype=int)
    x_rug = x[idx] if idx.size else np.array([], dtype=float)
    jitter_scale = max(float(np.max(true_pdf)), 1e-6)
    y_rug = rng.uniform(0.0, 0.015 * jitter_scale, size=x_rug.size) if x_rug.size else np.array([], dtype=float)

    st.subheader("Samples and densities")
    if _PLOTLY_AVAILABLE:
        fig = go.Figure()
        if st.session_state.get("show_hist", True):
            fig.add_trace(go.Histogram(x=x, nbinsx=hist_n, histnorm="probability density",
                                       marker_color="rgba(31,119,180,0.55)", name="samples"))
        if x_rug.size:
            fig.add_trace(go.Scatter(
                x=x_rug, y=y_rug, mode="markers",
                marker=dict(size=8, color="#FFD700", line=dict(color="rgba(0,0,0,0.6)", width=0.5)),
                name="samples (subset)"
            ))
        # fit curve added below after computing fit_pdf
        # true curve optionally
        # We'll append curves after computing fit below
        pass
    elif _MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        if st.session_state.get("show_hist", True):
            ax.hist(x, bins=hist_n, density=True, color=(0.12, 0.47, 0.71, 0.55))
        if x_rug.size:
            ax.scatter(x_rug, y_rug, s=24, c="#FFD700", edgecolors="k", linewidths=0.4)
        # curves appended below
        pass
    else:
        st.info("Install plotly or matplotlib to see plots.")

    # Compute fit curve
    if fit_name == "Gaussian (MLE)":
        mu_hat, sig_hat = fit_gaussian_mle(x)
        fit_pdf = pdf_gaussian(grid, mu_hat, sig_hat)
        desc = f"μ̂={mu_hat:.3f}, σ̂={sig_hat:.3f}"
    elif fit_name == "Gaussian (User)":
        mu0 = float(fit_params.get("mu", 0.0))
        sig0 = max(float(fit_params.get("sigma", 1.0)), 1e-9)
        fit_pdf = pdf_gaussian(grid, mu0, sig0)
        # Mean log-likelihood over all samples
        z = (x - mu0) / sig0
        mean_ll = float((-0.5 * (z * z) - math.log(math.sqrt(2.0 * math.pi) * sig0)).mean())
        desc = f"μ={mu0:.3f}, σ={sig0:.3f}; mean log-lik={mean_ll:.4f}"
    elif fit_name == "Mixture of Gaussians (EM)":
        K = int(fit_params["K"]) ; iters = int(fit_params["iters"])
        gmm = em_fit_gmm(x, K=K, max_iter=iters, rng=get_rng())
        fit_pdf = pdf_mog(grid, gmm.mus, gmm.sigmas, gmm.weights)
        desc = f"K={K}; weights={np.round(gmm.weights,3)}, mus={np.round(gmm.mus,3)}, sigmas={np.round(gmm.sigmas,3)}"
    else:
        mode = fit_params["bw_mode"]
        if mode == "Silverman":
            h = silverman_bandwidth(x)
        elif mode == "Scott":
            h = scott_bandwidth(x)
        else:
            h = float(fit_params["bw"]) if fit_params["bw"] is not None else silverman_bandwidth(x)
        fit_pdf = kde_gaussian(x, grid, h)
        desc = f"h={h:.4f} ({mode})"

    # Now add curves to the existing figure/context above
    if _PLOTLY_AVAILABLE:
        # Recreate full figure for clarity
        fig = go.Figure()
        if st.session_state.get("show_hist", True):
            fig.add_trace(go.Histogram(x=x, nbinsx=hist_n, histnorm="probability density",
                                       marker_color="rgba(31,119,180,0.55)", name="samples"))
        if x_rug.size:
            fig.add_trace(go.Scatter(x=x_rug, y=y_rug, mode="markers",
                                     marker=dict(size=8, color="#FFD700", line=dict(color="rgba(0,0,0,0.6)", width=0.5)),
                                     name="samples (subset)"))
        fig.add_trace(go.Scatter(x=grid, y=fit_pdf, mode="lines", line=dict(color="#d62728", width=2), name="fit"))
        if st.session_state.get("show_true", True):
            fig.add_trace(go.Scatter(x=grid, y=true_pdf, mode="lines", line=dict(color="#2ca02c", width=2), name="true"))
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=420)
        st.plotly_chart(fig, use_container_width=True)
    elif _MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        if st.session_state.get("show_hist", True):
            ax.hist(x, bins=hist_n, density=True, color=(0.12, 0.47, 0.71, 0.55))
        if x_rug.size:
            ax.scatter(x_rug, y_rug, s=24, c="#FFD700", edgecolors="k", linewidths=0.4)
        ax.plot(grid, fit_pdf, c="#d62728", lw=2, label="fit")
        if st.session_state.get("show_true", True):
            ax.plot(grid, true_pdf, c="#2ca02c", lw=2, label="true")
        ax.legend(loc="best")
        st.pyplot(fig, clear_figure=True)

    # Bins control moved to sidebar

    st.caption(desc)


if __name__ == "__main__":
    main()
# ----------- Utilities -----------

def _erf_vec(x: np.ndarray) -> np.ndarray:
    """Vectorized erf approximation (Numerical Recipes). Accurate and fast without scipy."""
    x = np.asarray(x, dtype=float)
    sign = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    # Horner's method for polynomial
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * np.exp(-a * a)
    return sign * y
