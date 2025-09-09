import math
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


from demo2_plotting import make_plotly_figure, make_matplotlib_figure

from density1d_models import (
    pdf_gaussian,
    sample_gaussian,
    pdf_laplace,
    sample_laplace,
    pdf_student_t,
    sample_student_t,
    pdf_lognormal,
    sample_lognormal,
    pdf_beta,
    sample_beta,
    pdf_uniform,
    sample_uniform,
    pdf_triangular,
    sample_triangular,
    pdf_skew_normal,
    pdf_proj_twomoons_x,
    sample_proj_twomoons_x,
    sample_proj_checkerboard_x,
    sample_proj_spiral_x,
    pdf_arcsine_projected_circle,
    sample_arcsine_projected_circle,
    pdf_mixture_arcsine,
    sample_mixture_arcsine,
    pdf_mog,
    sample_mog,
)


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





# ---------- UI State ----------


def ensure_state():
    if "rng_seed" not in st.session_state:
        st.session_state.rng_seed = 0
    if "rng" not in st.session_state:
        st.session_state.rng = np.random.default_rng(int(st.session_state.rng_seed))
    if "samples" not in st.session_state:
        st.session_state.samples = np.array([], dtype=float)
    if "fitted" not in st.session_state:
        # Holds last fitted model independent of new samples
        st.session_state.fitted = None  # dict with keys: kind, params
    if "fixed_grids" not in st.session_state:
        # Persisted plot grids per estimator kind so curves don't rescale
        st.session_state.fixed_grids = {}


def get_rng() -> np.random.Generator:
    return st.session_state.rng


def main():
    ensure_state()
    st.title("1D Density Estimation Demo")
    st.caption(
        "Draw samples from a chosen 1D distribution, then fit parametric or non-parametric models and compare. "
        "Tip: with ‘Gaussian (User)’, try optimizing the mean log-likelihood by adjusting μ and σ."
    )

    with st.sidebar:
        # Order: Fit at top, then Sampling and True density
        st.header("Fit")
        fit_name = st.selectbox("Estimator", [
            "Gaussian (User)",
            "Mixture of Gaussians (User, 2)",
            "Gaussian (MLE)",
            "Mixture of Gaussians (EM)",
            "KDE (Gaussian)"
        ], index=0)
        # Advanced fit options
        fit_params = {}
        with st.expander("Fit options", expanded=False):
            if fit_name == "Gaussian (User)":
                mu_default = float(st.session_state.get("mu_user", 0.0))
                sigma_default = float(st.session_state.get("sigma_user", 1.0))
                mu0 = st.number_input("μ (user)", value=mu_default, key="mu_user")
                sigma0 = st.number_input("σ (user)", value=sigma_default, min_value=0.0, key="sigma_user")
                fit_params.update(dict(mu=float(st.session_state.get("mu_user", mu0)),
                                       sigma=float(st.session_state.get("sigma_user", sigma0))))
            elif fit_name == "Mixture of Gaussians (User, 2)":
                # User-defined 2-component MoG
                mu1 = st.number_input("μ₁ (user)", value=float(st.session_state.get("mog2_mu1", -2.0)), key="mog2_mu1")
                s1 = st.number_input("σ₁ (user)", value=float(st.session_state.get("mog2_sigma1", 0.8)), min_value=0.0, key="mog2_sigma1")
                mu2 = st.number_input("μ₂ (user)", value=float(st.session_state.get("mog2_mu2", 2.0)), key="mog2_mu2")
                s2 = st.number_input("σ₂ (user)", value=float(st.session_state.get("mog2_sigma2", 0.8)), min_value=0.0, key="mog2_sigma2")
                w1 = st.slider("w₁ (user)", 0.0, 1.0, float(st.session_state.get("mog2_w1", 0.5)), key="mog2_w1")
                w2 = 1.0 - w1
                fit_params.update(dict(
                    mus=np.array([float(mu1), float(mu2)]),
                    sigmas=np.array([max(float(s1), 1e-9), max(float(s2), 1e-9)]),
                    weights=np.array([float(w1), float(w2)])
                ))
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
        # No nudge buttons; user edits μ, σ directly above
        # Explicit refit button (does not auto-refit on new samples)
        # Disabled for user-defined models
        if st.button("Refit model", disabled=(fit_name in ("Gaussian (User)", "Mixture of Gaussians (User, 2)"))):
            if fit_name == "Gaussian (User)":
                pass
            else:
                x_fit = st.session_state.samples
                if x_fit.size == 0:
                    st.warning("Draw samples first to fit a model.")
                else:
                    if fit_name == "Gaussian (MLE)":
                        mu_hat, sig_hat = fit_gaussian_mle(x_fit)
                        st.session_state.fitted = {"kind": "Gaussian (MLE)", "mu": float(mu_hat), "sigma": float(sig_hat)}
                    elif fit_name == "Mixture of Gaussians (EM)":
                        K = int(fit_params.get("K", 2))
                        iters = int(fit_params.get("iters", 120))
                        gmm = em_fit_gmm(x_fit, K=K, max_iter=iters, rng=get_rng())
                        st.session_state.fitted = {
                            "kind": "Mixture of Gaussians (EM)",
                            "mus": gmm.mus.tolist(),
                            "sigmas": gmm.sigmas.tolist(),
                            "weights": gmm.weights.tolist(),
                        }
                    else:  # KDE
                        mode = fit_params.get("bw_mode", "Silverman")
                        if mode == "Silverman":
                            h = silverman_bandwidth(x_fit)
                        elif mode == "Scott":
                            h = scott_bandwidth(x_fit)
                        else:
                            h = float(fit_params.get("bw", 0.3))
                        st.session_state.fitted = {
                            "kind": "KDE (Gaussian)",
                            "train_x": x_fit.tolist(),
                            "bandwidth": float(h),
                        }
                # Clear any previous fixed grid; it will be set on next render
                if "fitted" in st.session_state and isinstance(st.session_state.fitted, dict):
                    st.session_state.fitted.pop("grid", None)
                # Also clear the fixed grid for this estimator kind
                fg = st.session_state.get("fixed_grids", {})
                if fit_name in fg:
                    del fg[fit_name]
        st.markdown("---")

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
            # Track a generation counter so rug subset stays stable across UI tweaks
            st.session_state["samples_gen"] = int(st.session_state.get("samples_gen", 0)) + 1
            # Invalidate cached rug subset
            for _k in ("rug_idx", "rug_u", "rug_idx_gen", "rug_subset_n"):
                if _k in st.session_state:
                    del st.session_state[_k]
            st.session_state._prev_true_name = true_name

        # Fit is configured at the top; proceed to display controls
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

    # Grid for PDF curves (persist grid per estimator kind even for User Gaussian)
    fitted = st.session_state.get("fitted")
    grid = None
    # First preference: per-kind fixed grid cache
    fg = st.session_state.get("fixed_grids", {})
    if fit_name in fg:
        grid = np.array(fg[fit_name], dtype=float)
    # Second: grid attached to a fitted model of same kind
    elif fitted is not None and fitted.get("kind") == fit_name and "grid" in fitted:
        grid = np.array(fitted["grid"], dtype=float)
        # sync into per-kind cache
        st.session_state.fixed_grids[fit_name] = grid.tolist()
    else:
        lo, hi = float(np.min(x)), float(np.max(x))
        span = hi - lo if hi > lo else 1.0
        pad = 0.1 * span
        gmin = min(support[0], lo - pad)
        gmax = max(support[1], hi + pad)
        grid = np.linspace(gmin, gmax, 800)
        # cache this grid for the current estimator kind
        st.session_state.fixed_grids[fit_name] = grid.tolist()

    # Session-backed bins, control placed below for alignment
    hist_n = int(st.session_state.get("hist_bins", 50))
    true_pdf = pdf(grid)
    # Rug markers (subset of samples), colored and lightly jittered for visibility
    subset_n = int(min(100, x.size))
    # Keep the subset of yellow rug points stable unless samples change
    samples_gen = int(st.session_state.get("samples_gen", 0))
    need_new_rug = (
        "rug_idx" not in st.session_state or
        int(st.session_state.get("rug_idx_gen", -1)) != samples_gen or
        int(st.session_state.get("rug_subset_n", -1)) != subset_n or
        (subset_n > 0 and len(st.session_state.get("rug_idx", [])) != subset_n)
    )
    if need_new_rug:
        if subset_n > 0:
            rng_local = np.random.default_rng(int(st.session_state.rng_seed) + samples_gen)
            st.session_state.rug_idx = rng_local.choice(x.size, size=subset_n, replace=False)
            st.session_state.rug_u = rng_local.uniform(0.0, 1.0, size=subset_n)
        else:
            st.session_state.rug_idx = np.array([], dtype=int)
            st.session_state.rug_u = np.array([], dtype=float)
        st.session_state.rug_idx_gen = samples_gen
        st.session_state.rug_subset_n = subset_n
    idx = st.session_state.rug_idx
    x_rug = x[idx] if subset_n > 0 else np.array([], dtype=float)
    jitter_scale = max(float(np.max(true_pdf)), 1e-6)
    y_rug = (st.session_state.rug_u if subset_n > 0 else np.array([], dtype=float)) * (0.015 * jitter_scale)

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
                name="samples"
            ))
        # Fit curve and overlays will be added after fit computation below
    elif _MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        if st.session_state.get("show_hist", True):
            ax.hist(x, bins=hist_n, density=True, color=(0.12, 0.47, 0.71, 0.55))
        if x_rug.size:
            ax.scatter(x_rug, y_rug, s=24, c="#FFD700", edgecolors="k", linewidths=0.4, label="samples")
        # Fit curve and overlays will be added after fit computation below
    else:
        st.info("Install plotly or matplotlib to see plots.")

    # Compute fit curve
    mean_ll_for_display = None
    if fit_name == "Gaussian (User)":
        mu0 = float(fit_params.get("mu", 0.0))
        sig0 = max(float(fit_params.get("sigma", 1.0)), 1e-9)
        fit_pdf = pdf_gaussian(grid, mu0, sig0)
        # Mean log-likelihood over all samples
        z = (x - mu0) / sig0
        mean_ll = float((-0.5 * (z * z) - math.log(math.sqrt(2.0 * math.pi) * sig0)).mean())
        mean_ll_for_display = mean_ll
        # Track best-so-far mean log-likelihood for current samples
        samples_gen = int(st.session_state.get("samples_gen", 0))
        if st.session_state.get("user_gauss_best_gen") != samples_gen:
            st.session_state["user_gauss_best_gen"] = samples_gen
            st.session_state["user_gauss_best_ll"] = -float("inf")
            st.session_state["user_gauss_best_params"] = (mu0, sig0)
        if mean_ll > float(st.session_state.get("user_gauss_best_ll", -float("inf"))):
            st.session_state["user_gauss_best_ll"] = mean_ll
            st.session_state["user_gauss_best_params"] = (mu0, sig0)
        desc = f"μ={mu0:.3f}, σ={sig0:.3f}; mean log-lik={mean_ll:.4f} (best: {st.session_state['user_gauss_best_ll']:.4f})"
    elif fit_name == "Mixture of Gaussians (User, 2)":
        mus = np.asarray(fit_params.get("mus", np.array([-2.0, 2.0])), dtype=float)
        sigmas = np.asarray(fit_params.get("sigmas", np.array([0.8, 0.8])), dtype=float)
        weights = np.asarray(fit_params.get("weights", np.array([0.5, 0.5])), dtype=float)
        # normalize weights just in case
        weights = np.clip(weights, 1e-12, None)
        weights = weights / weights.sum()
        fit_pdf = pdf_mog(grid, mus, sigmas, weights)
        # Mean log-likelihood under the user mixture
        px = np.zeros_like(x, dtype=float)
        for k in range(2):
            px += float(weights[k]) * pdf_gaussian(x, float(mus[k]), float(sigmas[k]))
        mean_ll = float(np.log(np.clip(px, 1e-300, None)).mean())
        mean_ll_for_display = mean_ll
        desc = (
            f"K=2 (user); weights={np.round(weights,3)}, mus={np.round(mus,3)}, sigmas={np.round(sigmas,3)}; "
            f"mean log-lik={mean_ll:.4f}"
        )
    elif fit_name == "Gaussian (MLE)":
        # Use existing fitted model if available; do not refit automatically
        fitted = st.session_state.get("fitted")
        if fitted is not None and fitted.get("kind") == "Gaussian (MLE)":
            mu_hat = float(fitted["mu"]) ; sig_hat = max(float(fitted["sigma"]), 1e-9)
            fit_pdf = pdf_gaussian(grid, mu_hat, sig_hat)
            z = (x - mu_hat) / sig_hat
            mean_ll = float((-0.5 * (z * z) - math.log(math.sqrt(2.0 * math.pi) * sig_hat)).mean())
            mean_ll_for_display = mean_ll
            desc = f"μ̂={mu_hat:.3f}, σ̂={sig_hat:.3f}; mean log-lik={mean_ll:.4f}"
        else:
            fit_pdf = None
            desc = "No fitted Gaussian (MLE). Click ‘Refit model’."
    elif fit_name == "Mixture of Gaussians (EM)":
        fitted = st.session_state.get("fitted")
        if fitted is not None and fitted.get("kind") == "Mixture of Gaussians (EM)":
            mus = np.array(fitted["mus"], dtype=float)
            sigmas = np.array(fitted["sigmas"], dtype=float)
            weights = np.array(fitted["weights"], dtype=float)
            fit_pdf = pdf_mog(grid, mus, sigmas, weights)
            px = np.zeros_like(x, dtype=float)
            for k in range(len(mus)):
                px += float(weights[k]) * pdf_gaussian(x, float(mus[k]), float(sigmas[k]))
            mean_ll = float(np.log(np.clip(px, 1e-300, None)).mean())
            mean_ll_for_display = mean_ll
            desc = (
                f"K={len(mus)}; weights={np.round(weights,3)}, mus={np.round(mus,3)}, sigmas={np.round(sigmas,3)}; "
                f"mean log-lik={mean_ll:.4f}"
            )
        else:
            fit_pdf = None
            desc = "No fitted mixture. Click ‘Refit model’."
    else:  # KDE (Gaussian)
        fitted = st.session_state.get("fitted")
        if fitted is not None and fitted.get("kind") == "KDE (Gaussian)":
            train_x = np.array(fitted["train_x"], dtype=float)
            h = max(float(fitted["bandwidth"]), 1e-9)
            fit_pdf = kde_gaussian(train_x, grid, h)
            px = kde_gaussian(train_x, x, h)
            mean_ll = float(np.log(np.clip(px, 1e-300, None)).mean())
            mean_ll_for_display = mean_ll
            desc = f"KDE h={h:.4f}; mean log-lik={mean_ll:.4f}"
        else:
            fit_pdf = None
            desc = "No fitted KDE. Click ‘Refit model’."

    # Render figures via helpers
    if _PLOTLY_AVAILABLE:
        fig = make_plotly_figure(
            x=x,
            hist_n=hist_n,
            x_rug=x_rug,
            y_rug=y_rug,
            grid=grid,
            fit_pdf=(fit_pdf if 'fit_pdf' in locals() else None),
            true_pdf=true_pdf,
            fit_name=fit_name,
            fit_params=fit_params,
            show_hist=bool(st.session_state.get("show_hist", True)),
            show_true=bool(st.session_state.get("show_true", True)),
        )
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    elif _MATPLOTLIB_AVAILABLE:
        figm = make_matplotlib_figure(
            x=x,
            hist_n=hist_n,
            x_rug=x_rug,
            y_rug=y_rug,
            grid=grid,
            fit_pdf=(fit_pdf if 'fit_pdf' in locals() else None),
            true_pdf=true_pdf,
            fit_name=fit_name,
            fit_params=fit_params,
            show_hist=bool(st.session_state.get("show_hist", True)),
            show_true=bool(st.session_state.get("show_true", True)),
        )
        if figm is not None:
            st.pyplot(figm, clear_figure=True)
    else:
        st.info("Install plotly or matplotlib to see plots.")

    # Bins control moved to sidebar

    # Prominent metrics
    if mean_ll_for_display is not None:
        if fit_name == "Gaussian (User)" and "user_gauss_best_ll" in st.session_state:
            c1, c2 = st.columns(2)
            with c1:
                st.metric(label="Mean log-likelihood", value=f"{mean_ll_for_display:.4f}")
            with c2:
                best_ll = float(st.session_state.get("user_gauss_best_ll", float("nan")))
                st.metric(label="Best so far (mean log-lik)", value=f"{best_ll:.4f}")
            best_mu, best_sig = st.session_state.get("user_gauss_best_params", (None, None))
            if best_mu is not None:
                st.caption(f"Best at μ={best_mu:.3f}, σ={best_sig:.3f}")
        else:
            st.metric(label="Mean log-likelihood", value=f"{mean_ll_for_display:.4f}")
    st.caption(desc)

    # --- More details (theory and models) ---
    with st.expander("More details (theory and models)"):
        st.markdown(
            """
            Purpose and big picture
            - Goal: build intuition for 1D density estimation. You pick a true distribution, draw samples x₁:ₙ, then fit a model p̂(x|θ) and inspect both the curve and the mean log‑likelihood 1/n Σ log p̂(xᵢ|θ).
            - Posterior‑free: this demo focuses on frequentist fits (MLE/EM/KDE) and direct likelihood evaluation, complementary to the ABC demo.

            True families (generators)
            - Gaussian, Mixture of Gaussians (2 or 3 components), Laplace, Student‑t, Lognormal, Beta [0,1], Uniform, Triangular.
            - 2D→1D projections: Projected Circle (arcsine), Projected Rings (mixture of arcsine), Projected Two Moons (x), Projected Checkerboard (x), Projected Spiral (x).

            Estimators (fits)
            - Gaussian (User): you set μ and σ; the app reports the mean log‑likelihood on the current samples and tracks best‑so‑far per sample set. The curve does not auto‑change when you draw new samples.
            - Gaussian (MLE): closed‑form estimates μ̂ = mean(x), σ̂ = std(x). Click “Refit model” to recompute; after that, the curve persists while you explore.
            - Mixture of Gaussians (EM): expectation‑maximization with configurable K and iterations. “Refit model” learns {weights, means, stds}; the stored fit is then used to evaluate and plot.
            - KDE (Gaussian): p̂(x) = 1/n Σ ϕ((x − xᵢ)/h)/h with bandwidth h via Silverman/Scott or manual. “Refit model” stores the training samples and bandwidth; evaluation uses the stored set.

            Likelihood and evaluation
            - Mean log‑likelihood: 1/n Σ log p̂(xᵢ). Higher is better (less negative). For mixtures: log Σ wₖ N(x | μₖ, σₖ²). For KDE: log p̂(xᵢ) with numerical safeguards.
            - Caution: Evaluating on the same samples used to fit (especially for KDE) can overestimate performance. For rigor, split into train/validation.

            Persistence model
            - Curves and their x‑grids persist across new samples and true‑family changes. MLE/EM/KDE only change after you press “Refit model”. Gaussian (User) uses exactly your current μ,σ and a fixed x‑grid.

            Suggested explorations
            - Manually tune μ, σ to maximize the mean log‑likelihood for various true families; compare with MLE.
            - Change K for the MoG fit and watch how the mean log‑likelihood and shape respond on multi‑modal truths.
            - Vary KDE bandwidth (Silverman/Scott/Manual) on heavy‑tailed vs. compact distributions; find h values that balance bias/variance.
            """
        )


if __name__ == "__main__":
    main()
# ----------- Utilities -----------

from density1d_fit import (
    silverman_bandwidth,
    scott_bandwidth,
    kde_gaussian,
    GMMParams,
    em_fit_gmm,
    fit_gaussian_mle,
    fit_laplace_mle,
)
