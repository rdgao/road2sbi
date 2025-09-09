import math
from typing import Callable, Dict, Tuple

import numpy as np
import streamlit as st


# Optional plotting backends
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


# Reuse some fun 2D->1D projection samplers from existing module
from density1d_models import (
    sample_arcsine_projected_circle,
    sample_proj_twomoons_x,
    sample_proj_spiral_x,
    sample_proj_checkerboard_x,
)


st.set_page_config(page_title="Conditional Density Demo", layout="wide")


# -------- Session state --------
def ensure_state():
    if "rng_seed" not in st.session_state:
        st.session_state.rng_seed = 123
    if "rng" not in st.session_state:
        st.session_state.rng = np.random.default_rng(int(st.session_state.rng_seed))
    # Defaults for theta range so noise UI can reference it early
    if "theta_min" not in st.session_state:
        st.session_state.theta_min = -2.5
    if "theta_max" not in st.session_state:
        st.session_state.theta_max = 2.5
    # Conditioning defaults
    st.session_state.setdefault("cond_theta_star", 0.0)
    st.session_state.setdefault("cond_d_theta", 0.2)
    st.session_state.setdefault("cond_x_star", 0.0)
    st.session_state.setdefault("cond_d_x", 0.2)
    st.session_state.setdefault("cond_mode", "Window")
    st.session_state.setdefault("cond_show_marginals", True)
    st.session_state.setdefault("cond_h_theta_star", 0.3)
    st.session_state.setdefault("cond_h_x_star", 0.3)
    st.session_state.setdefault("cond_kde_bw_scale", 0.25)


def get_rng() -> np.random.Generator:
    return st.session_state.rng


# -------- Noise models --------
def make_noise_sigma_fn(kind: str, base: float, theta0: float, alpha: float, period: float) -> Callable[[np.ndarray], np.ndarray]:
    base = max(float(base), 1e-9)
    alpha = float(alpha)
    theta0 = float(theta0)
    period = max(float(period), 1e-9)

    if kind == "Homoscedastic":
        def sigma_fn(theta: np.ndarray) -> np.ndarray:
            return np.full_like(theta, base, dtype=float)

    elif kind == "Linear |θ|":
        def sigma_fn(theta: np.ndarray) -> np.ndarray:
            return base * (1.0 + alpha * np.abs(theta))

    elif kind == "U-shape (quadratic)":
        def sigma_fn(theta: np.ndarray) -> np.ndarray:
            return base * (1.0 + alpha * (theta - theta0) ** 2)

    elif kind == "Sinusoidal":
        def sigma_fn(theta: np.ndarray) -> np.ndarray:
            return base * (1.0 + alpha * 0.5 * (1.0 + np.sin(2.0 * math.pi * theta / period)))

    else:
        def sigma_fn(theta: np.ndarray) -> np.ndarray:
            return np.full_like(theta, base, dtype=float)

    return sigma_fn


# -------- Conditional simulators p(x|theta) --------
def sim_linear(theta: np.ndarray, params: Dict[str, float], rng: np.random.Generator, sigma_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    a = float(params.get("a", 1.0))
    b = float(params.get("b", 0.0))
    mu = a * theta + b
    sig = sigma_fn(theta)
    return mu + rng.normal(0.0, 1.0, size=theta.shape) * np.maximum(sig, 1e-9)


def sim_sine(theta: np.ndarray, params: Dict[str, float], rng: np.random.Generator, sigma_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    A = float(params.get("A", 1.0))
    w = float(params.get("w", 1.0))
    phi = float(params.get("phi", 0.0))
    c = float(params.get("c", 0.0))
    mu = A * np.sin(w * theta + phi) + c
    sig = sigma_fn(theta)
    return mu + rng.normal(0.0, 1.0, size=theta.shape) * np.maximum(sig, 1e-9)


def sim_sine_plus_line(theta: np.ndarray, params: Dict[str, float], rng: np.random.Generator, sigma_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    A = float(params.get("A", 1.0))
    w = float(params.get("w", 1.0))
    b = float(params.get("b", 0.5))
    c = float(params.get("c", 0.0))
    mu = A * np.sin(w * theta) + b * theta + c
    sig = sigma_fn(theta)
    return mu + rng.normal(0.0, 1.0, size=theta.shape) * np.maximum(sig, 1e-9)


def _shape_additive(theta: np.ndarray, draw_shape: Callable[[int, np.random.Generator], np.ndarray], params: Dict[str, float], rng: np.random.Generator, sigma_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    # Generic helper: shift(theta) + scale * shape_sample + measurement noise
    shift = float(params.get("shift", 0.0))
    scale = float(params.get("scale", 1.0))
    z = draw_shape(theta.size, rng)
    mu = shift + scale * z
    sig = sigma_fn(theta)
    return mu + rng.normal(0.0, 1.0, size=theta.shape) * np.maximum(sig, 1e-9)


def sim_circle_curve(theta: np.ndarray, params: Dict[str, float], rng: np.random.Generator, sigma_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    # Draw points on a circle in (theta,x): (theta - c_th)^2 + (x - c_x)^2 = r^2
    c_th = float(params.get("c_th", 0.0))
    c_x = float(params.get("c_x", 0.0))
    r = max(float(params.get("r0", 1.0)), 1e-6)
    shift = float(params.get("shift", 0.0))
    scale = float(params.get("scale", 1.0))
    dx = theta - c_th
    inside = np.abs(dx) <= r
    x0 = np.full_like(theta, np.nan, dtype=float)
    # Randomly choose upper/lower branch where valid
    sgn = np.where(rng.uniform(0.0, 1.0, size=theta.shape) < 0.5, 1.0, -1.0)
    rad = np.sqrt(np.clip(r * r - dx * dx, 0.0, None))
    x0[inside] = c_x + sgn[inside] * rad[inside]
    x0 = shift + scale * x0
    sig = sigma_fn(theta)
    x = x0 + rng.normal(0.0, 1.0, size=theta.shape) * np.maximum(sig, 1e-9)
    return x


def sim_twomoons_curve(theta: np.ndarray, params: Dict[str, float], rng: np.random.Generator, sigma_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    # Two arcs in (theta,x): upper: (theta)^2 + x^2 = 1 on [-1,1]; lower: (theta-1)^2 + x^2 = 1 on [0,2], with negative x.
    shift = float(params.get("shift", 0.0))
    scale = float(params.get("scale", 1.0))
    x0 = np.full_like(theta, np.nan, dtype=float)
    # Determine validity for both arcs
    dx_u = theta  # centered at 0
    inside_u = (dx_u > -1.0) & (dx_u < 1.0)
    y_u = np.sqrt(np.clip(1.0 - dx_u * dx_u, 0.0, None))

    dx_l = theta - 1.0  # centered at 1
    inside_l = (dx_l > -1.0) & (dx_l < 1.0)
    y_l = -np.sqrt(np.clip(1.0 - dx_l * dx_l, 0.0, None))

    # Randomly pick branch where both valid; otherwise choose the valid one
    both = inside_u & inside_l
    pick_upper = rng.uniform(0.0, 1.0, size=theta.shape) < 0.5
    x0[both] = np.where(pick_upper[both], y_u[both], y_l[both])
    only_u = inside_u & (~inside_l)
    x0[only_u] = y_u[only_u]
    only_l = inside_l & (~inside_u)
    x0[only_l] = y_l[only_l]

    x0 = shift + scale * x0
    sig = sigma_fn(theta)
    x = x0 + rng.normal(0.0, 1.0, size=theta.shape) * np.maximum(sig, 1e-9)
    return x


def sim_spiral_curve(theta: np.ndarray, params: Dict[str, float], rng: np.random.Generator, sigma_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    # Archimedean-like arm in (theta, x): treat theta as angle φ and grow radius with θ
    # x(θ) ≈ (A0 + k·θ) · sin(w·θ), then add measurement noise.
    A0 = float(params.get("A0", 0.2))
    k = float(params.get("k", 0.15))
    w = float(params.get("w", 1.5))
    shift = float(params.get("shift", 0.0))
    scale = float(params.get("scale", 1.0))
    amp = A0 + k * theta
    x0 = shift + scale * (amp * np.sin(w * theta))
    sig = sigma_fn(theta)
    return x0 + rng.normal(0.0, 1.0, size=theta.shape) * np.maximum(sig, 1e-9)


def sim_checkerboard_curve(theta: np.ndarray, params: Dict[str, float], rng: np.random.Generator, sigma_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    # Vertical stripes along theta, with two x levels ±delta per stripe
    m = max(int(params.get("m", 4)), 1)
    delta = float(params.get("delta", 0.4))
    shift = float(params.get("shift", 0.0))
    scale = float(params.get("scale", 1.0))
    th_min = float(np.min(theta))
    th_max = float(np.max(theta))
    width = max(th_max - th_min, 1e-9)
    # Stripe index in [0, m-1]
    idx = np.floor((theta - th_min) / width * m).astype(int)
    idx = np.clip(idx, 0, m - 1)
    sgn = np.where(rng.uniform(0.0, 1.0, size=theta.shape) < 0.5, 1.0, -1.0)
    x0 = shift + scale * (sgn * delta)
    sig = sigma_fn(theta)
    return x0 + rng.normal(0.0, 1.0, size=theta.shape) * np.maximum(sig, 1e-9)


SIMS: Dict[str, Callable[[np.ndarray, Dict[str, float], np.random.Generator, Callable[[np.ndarray], np.ndarray]], np.ndarray]] = {
    "Linear": sim_linear,
    "Sine": sim_sine,
    "Sine + Line": sim_sine_plus_line,
    "Circle": sim_circle_curve,
    "Two Moons": sim_twomoons_curve,
    "Spiral": sim_spiral_curve,
}


# -------- Plotting --------
def plot_scatter(
    theta: np.ndarray,
    x: np.ndarray,
    use_plotly: bool = True,
    theta_star: float = None,
    x_star: float = None,
    show_guides: bool = True,
    shade_windows: bool = True,
    cond_mode: str = "Window",
    d_theta: float = 0.0,
    d_x: float = 0.0,
):
    # Default horizontal bounds for theta axis (visual zoom only)
    x_min, x_max = -5.0, 5.0
    # Default vertical bounds for data x to align with side histogram
    y_min, y_max = -2.5, 2.5
    if use_plotly and _PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=theta,
                y=x,
                mode="markers",
                marker=dict(size=6, opacity=0.6, color="#1f77b4"),
                name="samples",
            )
        )
        fig.update_layout(
            xaxis_title="θ",
            yaxis_title="x",
            margin=dict(l=10, r=10, t=30, b=10),
            height=544,
        )
        # Apply axis bounds
        fig.update_xaxes(range=[x_min, x_max])
        fig.update_yaxes(range=[y_min, y_max])
        # Optional guide lines at θ* and x*
        if show_guides and (theta_star is not None):
            fig.add_shape(
                type="line",
                x0=float(theta_star),
                x1=float(theta_star),
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color="rgba(44,160,44,0.9)", width=1.5, dash="dash"),
            )
        if show_guides and (x_star is not None):
            fig.add_shape(
                type="line",
                x0=x_min,
                x1=x_max,
                y0=float(x_star),
                y1=float(x_star),
                xref="x",
                yref="y",
                line=dict(color="rgba(214,39,40,0.8)", width=1.5, dash="dash"),
            )
        # Optional shaded windows (only in Window mode)
        if shade_windows and cond_mode == "Window":
            if theta_star is not None and d_theta > 0:
                fig.add_shape(
                    type="rect",
                    x0=float(theta_star - d_theta),
                    x1=float(theta_star + d_theta),
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    fillcolor="rgba(255,165,0,0.15)",
                    line=dict(width=0),
                )
            if x_star is not None and d_x > 0:
                fig.add_shape(
                    type="rect",
                    x0=x_min,
                    x1=x_max,
                    y0=float(x_star - d_x),
                    y1=float(x_star + d_x),
                    xref="x",
                    yref="y",
                    fillcolor="rgba(255,165,0,0.15)",
                    line=dict(width=0),
                )
        st.plotly_chart(fig, use_container_width=True)
        return

    # Fallback to Matplotlib
    if _MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        ax.scatter(theta, x, s=18, c="#1f77b4", alpha=0.65, edgecolors="none")
        ax.set_xlabel("θ")
        ax.set_ylabel("x")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if show_guides and (theta_star is not None):
            ax.axvline(float(theta_star), color="#2ca02c", linestyle="--", linewidth=1.2)
        if show_guides and (x_star is not None):
            ax.axhline(float(x_star), color="#d62728", linestyle="--", linewidth=1.2)
        if shade_windows and cond_mode == "Window":
            if theta_star is not None and d_theta > 0:
                ax.axvspan(float(theta_star - d_theta), float(theta_star + d_theta), color=(1.0, 0.65, 0.0, 0.15))
            if x_star is not None and d_x > 0:
                ax.axhspan(float(x_star - d_x), float(x_star + d_x), color=(1.0, 0.65, 0.0, 0.15))
        st.pyplot(fig)
    else:
        st.write("Neither Plotly nor Matplotlib available for plotting.")


# -------- Simple 1D KDE helpers --------
def silverman_bandwidth(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = max(int(x.size), 1)
    std = float(np.std(x))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    s = min(std, iqr / 1.349) if (std > 0 and iqr > 0) else max(std, 1e-9)
    return 0.9 * s * (n ** (-1.0 / 5.0)) if s > 0 else 1.0


def kde_gaussian_1d(samples: np.ndarray, grid: np.ndarray, h: float) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    grid = np.asarray(grid, dtype=float)
    h = max(float(h), 1e-9)
    if samples.size == 0:
        return np.zeros_like(grid)
    diffs = (grid[:, None] - samples[None, :]) / h  # (G, N)
    vals = np.exp(-0.5 * diffs * diffs) / math.sqrt(2.0 * math.pi)
    return np.sum(vals, axis=1) / (samples.size * h + 1e-12)


def kde_gaussian_weighted_1d(samples: np.ndarray, weights: np.ndarray, grid: np.ndarray, h: float) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    weights = np.asarray(weights, dtype=float)
    grid = np.asarray(grid, dtype=float)
    h = max(float(h), 1e-9)
    if samples.size == 0:
        return np.zeros_like(grid)
    w = np.clip(weights, 0.0, None)
    sw = float(np.sum(w))
    if sw <= 0:
        return np.zeros_like(grid)
    w = w / sw
    diffs = (grid[:, None] - samples[None, :]) / h  # (G, N)
    vals = np.exp(-0.5 * diffs * diffs) / math.sqrt(2.0 * math.pi)
    return (vals @ w) / h


# -------- UI and main --------
def main():
    ensure_state()

    st.title("Conditional Density Estimation")
    st.caption(
        "Visualize samples from p(x | θ) for various simulators. "
        "Choose a mapping and a noise model; the plot shows (θ, x) pairs."
    )

    # Top-of-sidebar: conditioning controls (hideable)
    with st.sidebar:
        st.subheader("Conditioning")
        with st.expander("Conditioning slices", expanded=False):
            # θ* slice controls
            th_min0 = float(st.session_state.theta_min)
            th_max0 = float(st.session_state.theta_max)
            st.slider("θ*", th_min0, th_max0, float(st.session_state.cond_theta_star), 0.01, key="cond_theta_star")
            st.slider("Δθ (window)", 0.01, max(1.0, th_max0 - th_min0), float(st.session_state.cond_d_theta), 0.01, key="cond_d_theta")
            # x* slice controls (use fixed display y-lims for bounds)
            st.slider("x*", -2.5, 2.5, float(st.session_state.cond_x_star), 0.01, key="cond_x_star")
            st.slider("Δx (window)", 0.01, 5.0, float(st.session_state.cond_d_x), 0.01, key="cond_d_x")
            st.radio("Conditioning mode", ["Window", "Kernel"], index=(0 if st.session_state.cond_mode == "Window" else 1), horizontal=True, key="cond_mode")
            st.checkbox("Show marginal comparisons", value=bool(st.session_state.cond_show_marginals), key="cond_show_marginals")
            if st.session_state.cond_mode == "Kernel":
                st.slider("kernel h(θ*)", 0.01, 2.0, float(st.session_state.cond_h_theta_star), 0.01, key="cond_h_theta_star")
                st.slider("kernel h(x*)", 0.01, 2.0, float(st.session_state.cond_h_x_star), 0.01, key="cond_h_x_star")
            st.slider("KDE bandwidth scale", 0.2, 1.5, float(st.session_state.cond_kde_bw_scale), 0.05, key="cond_kde_bw_scale")

    with st.sidebar:
        # Simulator: main option exposed
        st.subheader("Simulator")
        _sim_options = list(SIMS.keys())
        _default_sim = "Sine + Line"
        _default_idx = _sim_options.index(_default_sim) if _default_sim in _sim_options else 0
        sim_name = st.selectbox(
            "Mapping f: θ → x (with noise)",
            _sim_options,
            index=_default_idx,
            key="sim_name",
        )
        # Simulator-specific parameters inside collapsible
        sim_params: Dict[str, float] = {}
        with st.expander("Simulator parameters", expanded=False):
            if sim_name == "Linear":
                sim_params["a"] = st.slider("slope a", -3.0, 3.0, 1.0, 0.05)
                sim_params["b"] = st.slider("intercept b", -3.0, 3.0, 0.0, 0.05)
            elif sim_name == "Sine":
                sim_params["A"] = st.slider("amplitude A", 0.0, 3.0, 1.0, 0.05)
                sim_params["w"] = st.slider("frequency w", 0.1, 4.0, 1.0, 0.05)
                sim_params["phi"] = st.slider("phase φ", -math.pi, math.pi, 0.0, 0.05)
                sim_params["c"] = st.slider("offset c", -3.0, 3.0, 0.0, 0.05)
            elif sim_name == "Sine + Line":
                sim_params["A"] = st.slider("amplitude A", 0.0, 3.0, 1.0, 0.05)
                sim_params["w"] = st.slider("frequency w", 0.1, 4.0, 3.0, 0.05)
                sim_params["b"] = st.slider("slope b", -2.0, 2.0, 0.3, 0.05)
                sim_params["c"] = st.slider("offset c", -3.0, 3.0, 0.0, 0.05)
            elif sim_name == "Circle":
                sim_params["c_th"] = st.slider("center θ", -3.0, 3.0, 0.0, 0.05)
                sim_params["c_x"] = st.slider("center x", -3.0, 3.0, 0.0, 0.05)
                sim_params["r0"] = st.slider("radius r", 0.05, 3.0, 1.0, 0.05)
                sim_params["shift"] = st.slider("x shift", -2.0, 2.0, 0.0, 0.05)
                sim_params["scale"] = st.slider("x scale", 0.1, 3.0, 1.0, 0.05)
            elif sim_name == "Two Moons":
                sim_params["shift"] = st.slider("x shift", -2.0, 2.0, 0.0, 0.05)
                sim_params["scale"] = st.slider("x scale", 0.1, 3.0, 1.0, 0.05)
            elif sim_name == "Spiral":
                sim_params["A0"] = st.slider("base amplitude A0", 0.0, 2.0, 0.2, 0.01)
                sim_params["k"] = st.slider("amplitude slope k", 0.0, 0.5, 0.15, 0.005)
                sim_params["w"] = st.slider("frequency w", 0.1, 6.0, 2.0, 0.05)
                sim_params["shift"] = st.slider("x shift", -2.0, 2.0, 0.0, 0.05)
                sim_params["scale"] = st.slider("x scale", 0.1, 3.0, 1.0, 0.05)
            

        # Noise model: main option exposed
        st.subheader("Noise model")
        noise_kind = st.selectbox("type", ["Homoscedastic", "Linear |θ|", "U-shape (quadratic)", "Sinusoidal"], index=0, key="noise_kind")
        # Initialize defaults so variables exist regardless of branch
        base_sigma = 0.2
        alpha = 0.5
        theta0 = 0.0
        period = 4.0
        with st.expander("Noise parameters", expanded=False):
            base_sigma = st.slider("base σ", 0.0, 1.5, 0.3, 0.01, key="noise_base_sigma")
            if noise_kind in ("Linear |θ|", "U-shape (quadratic)", "Sinusoidal"):
                alpha = st.slider("hetero α", 0.0, 2.0, 0.5, 0.05, key="noise_alpha")
            # Dynamic theta0 bounds from current theta range in state
            thmin = float(st.session_state.get("theta_min", -2.5))
            thmax = float(st.session_state.get("theta_max", 2.5))
            if noise_kind == "U-shape (quadratic)":
                theta0 = st.slider("center θ0", thmin, thmax, min(max(0.0, thmin), thmax), 0.05, key="noise_theta0")
            if noise_kind == "Sinusoidal":
                period = st.slider("period (sin)", 0.2, 8.0, 4.0, 0.1, key="noise_period")

        # Theta sampling in dropdown
        st.subheader("Parameter (θ) sampling")
        with st.expander("θ sampling", expanded=False):
            theta_min, theta_max = st.slider("θ range", -5.0, 5.0, (float(st.session_state.theta_min), float(st.session_state.theta_max)), 0.1)
            sample_mode = st.radio("mode", ["Uniform", "Grid"], horizontal=True)
            n_samples = st.slider("# samples", 100, 5000, 1000, 50)
            if sample_mode == "Grid":
                n_grid = st.slider("grid size", 10, 200, 80, 5)
            else:
                n_grid = None  # type: ignore

        # Misc settings
        st.subheader("Misc")
        st.session_state.rng_seed = st.number_input("random seed", value=int(st.session_state.rng_seed), step=1)
        # Refresh RNG each render to keep reproducible for same seed and settings
        st.session_state.rng = np.random.default_rng(int(st.session_state.rng_seed))


    # Build theta and sigma(theta)
    rng = get_rng()
    # Persist theta range for other UI sections (noise center, etc.)
    st.session_state.theta_min = float(theta_min)
    st.session_state.theta_max = float(theta_max)

    if sample_mode == "Grid":
        assert n_grid is not None
        theta_grid = np.linspace(theta_min, theta_max, int(n_grid))
        reps = int(np.ceil(n_samples / int(n_grid)))
        theta = np.tile(theta_grid, reps)[: int(n_samples)]
    else:
        theta = rng.uniform(theta_min, theta_max, size=int(n_samples))

    # Now that theta range is known, adjust theta0 to this domain if needed (if used)
    theta0 = float(np.clip(theta0, theta_min, theta_max))
    sigma_fn = make_noise_sigma_fn(noise_kind, base_sigma, theta0, alpha, period)

    # Run simulator
    sim_fn = SIMS[sim_name]
    x = sim_fn(theta, sim_params, rng, sigma_fn)

    # Read conditioning values from state
    theta_star = float(st.session_state.cond_theta_star)
    d_theta = float(st.session_state.cond_d_theta)
    x_star = float(st.session_state.cond_x_star)
    d_x = float(st.session_state.cond_d_x)
    cond_mode = str(st.session_state.cond_mode)
    show_marginals = bool(st.session_state.cond_show_marginals)
    h_theta_star = float(st.session_state.cond_h_theta_star)
    h_x_star = float(st.session_state.cond_h_x_star)
    kde_bw_scale = float(st.session_state.cond_kde_bw_scale)

    # Select or weight samples near conditioning values
    if cond_mode == "Window":
        mask_theta = np.abs(theta - float(theta_star)) <= float(d_theta)
        x_sel = x[mask_theta]
        mask_x = np.abs(x - float(x_star)) <= float(d_x)
        th_sel = theta[mask_x]
    else:
        # Kernel weights
        w_theta = np.exp(-0.5 * ((theta - float(theta_star)) / float(h_theta_star)) ** 2)
        w_x = np.exp(-0.5 * ((x - float(x_star)) / float(h_x_star)) ** 2)

    # Row layout: top θ|x* above the main scatter in the same column, and right x|θ*
    col_main, col_right = st.columns([5, 2])
    with col_main:
        # Top panel: p(θ | x*) aligned width with main
        if _PLOTLY_AVAILABLE:
            fig_top = go.Figure()
            if show_marginals:
                grid_th_m = np.linspace(theta_min, theta_max, 400)
                h_th_m = silverman_bandwidth(theta)
                kde_th_m = kde_gaussian_1d(theta, grid_th_m, h_th_m)
                fig_top.add_trace(go.Scatter(x=grid_th_m, y=kde_th_m, mode="lines", line=dict(color="rgba(100,100,100,0.9)", width=1.5), name="marginal θ"))
            if cond_mode == "Window":
                if th_sel.size:
                    fig_top.add_trace(
                        go.Histogram(x=th_sel, histnorm="probability density", nbinsx=40, marker_color="rgba(31,119,180,0.55)", name="hist")
                    )
                    grid_th = np.linspace(theta_min, theta_max, 400)
                    h_th = silverman_bandwidth(th_sel) * float(kde_bw_scale)
                    kde_th = kde_gaussian_1d(th_sel, grid_th, h_th)
                    fig_top.add_trace(go.Scatter(x=grid_th, y=kde_th, mode="lines", line=dict(color="#d62728", width=2), name="θ|x*"))
            else:
                grid_th = np.linspace(theta_min, theta_max, 400)
                kde_th = kde_gaussian_weighted_1d(theta, w_x, grid_th, silverman_bandwidth(theta) * float(kde_bw_scale))
                fig_top.add_trace(go.Scatter(x=grid_th, y=kde_th, mode="lines", line=dict(color="#d62728", width=2), name="θ|x* (kernel)"))
            fig_top.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=160,
                xaxis_title="θ",
                yaxis_title="density",
                legend=dict(
                    x=0.99,
                    xanchor="right",
                    y=0.98,
                    yanchor="top",
                    orientation="h",
                    bgcolor="rgba(255,255,255,0.6)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                ),
            )
            fig_top.update_xaxes(range=[-5.0, 5.0], showticklabels=False, ticks="outside", ticklen=6)
            st.plotly_chart(fig_top, use_container_width=True)
        elif _MATPLOTLIB_AVAILABLE:
            fig_top, ax_top = plt.subplots(figsize=(6.0, 1.6))
            if show_marginals:
                grid_th_m = np.linspace(theta_min, theta_max, 400)
                h_th_m = silverman_bandwidth(theta)
                kde_th_m = kde_gaussian_1d(theta, grid_th_m, h_th_m)
                ax_top.plot(grid_th_m, kde_th_m, c="0.4", lw=1.5, label="marginal θ")
            if cond_mode == "Window":
                if th_sel.size:
                    ax_top.hist(th_sel, bins=40, density=True, color=(0.12, 0.47, 0.71, 0.55), label="hist")
                    grid_th = np.linspace(theta_min, theta_max, 400)
                    h_th = silverman_bandwidth(th_sel) * float(kde_bw_scale)
                    kde_th = kde_gaussian_1d(th_sel, grid_th, h_th)
                    ax_top.plot(grid_th, kde_th, c="#d62728", lw=2, label="θ|x*")
            else:
                grid_th = np.linspace(theta_min, theta_max, 400)
                kde_th = kde_gaussian_weighted_1d(theta, w_x, grid_th, silverman_bandwidth(theta) * float(kde_bw_scale))
                ax_top.plot(grid_th, kde_th, c="#d62728", lw=2, label="θ|x* (kernel)")
            ax_top.set_xlabel("θ")
            ax_top.set_ylabel("density")
            if show_marginals:
                ax_top.legend(loc="upper right", framealpha=0.6)
            ax_top.set_xlim(-5.0, 5.0)
            ax_top.tick_params(axis='x', labelbottom=False)
            st.pyplot(fig_top)

        plot_scatter(
            theta,
            x,
            use_plotly=True,
            theta_star=theta_star,
            x_star=x_star,
            show_guides=True,
            shade_windows=True,
            cond_mode=cond_mode,
            d_theta=float(d_theta),
            d_x=float(d_x),
        )
    with col_right:
        # Spacer to align right panel with the top plot above the main scatter
        top_height_px = 160
        spacer_extra_px = 24  # small tweak to account for plot margins/padding
        st.markdown(
            f"<div style='height: {top_height_px + spacer_extra_px}px'></div>",
            unsafe_allow_html=True,
        )
        if _PLOTLY_AVAILABLE:
            fig_right = go.Figure()
            # Marginal x overlay
            if show_marginals:
                xg_m = np.linspace(float(np.min(x)), float(np.max(x)), 400)
                kde_x_m = kde_gaussian_1d(x, xg_m, silverman_bandwidth(x))
                fig_right.add_trace(go.Scatter(x=kde_x_m, y=xg_m, mode="lines", line=dict(color="rgba(100,100,100,0.9)", width=1.5), name="marginal x"))
            if cond_mode == "Window":
                if x_sel.size:
                    fig_right.add_trace(
                        go.Histogram(y=x_sel, histnorm="probability density", nbinsy=40, marker_color="rgba(31,119,180,0.55)", orientation="h", name="hist")
                    )
                    xg = np.linspace(float(np.min(x_sel)), float(np.max(x_sel)), 400)
                    h_x = silverman_bandwidth(x_sel) * float(kde_bw_scale)
                    kde_x = kde_gaussian_1d(x_sel, xg, h_x)
                    fig_right.add_trace(go.Scatter(x=kde_x, y=xg, mode="lines", line=dict(color="#2ca02c", width=2), name="x|θ*"))
            else:
                # Kernel-weighted using θ distances
                xg = np.linspace(float(np.min(x)), float(np.max(x)), 400)
                kde_x = kde_gaussian_weighted_1d(x, w_theta, xg, silverman_bandwidth(x) * float(kde_bw_scale))
                fig_right.add_trace(go.Scatter(x=kde_x, y=xg, mode="lines", line=dict(color="#2ca02c", width=2), name="x|θ* (kernel)"))
            fig_right.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=544, xaxis_title="density", yaxis_title="x")
            # Match 2D panel vertical x-lims and hide y tick labels
            fig_right.update_yaxes(range=[-2.5, 2.5], showticklabels=False, ticks="outside", ticklen=6)
            st.plotly_chart(fig_right, use_container_width=True)
        elif _MATPLOTLIB_AVAILABLE:
            fig_r, ax_r = plt.subplots(figsize=(2.6, 5.2))
            if show_marginals:
                xg_m = np.linspace(float(np.min(x)), float(np.max(x)), 400)
                kde_x_m = kde_gaussian_1d(x, xg_m, silverman_bandwidth(x))
                ax_r.plot(kde_x_m, xg_m, c="0.4", lw=1.5, label="marginal x")
            if cond_mode == "Window":
                if x_sel.size:
                    ax_r.hist(x_sel, bins=40, density=True, orientation="horizontal", color=(0.12, 0.47, 0.71, 0.55), label="hist")
                    xg = np.linspace(float(np.min(x_sel)), float(np.max(x_sel)), 400)
                    h_x = silverman_bandwidth(x_sel) * float(kde_bw_scale)
                    kde_x = kde_gaussian_1d(x_sel, xg, h_x)
                    ax_r.plot(kde_x, xg, c="#2ca02c", lw=2, label="x|θ*")
            else:
                xg = np.linspace(float(np.min(x)), float(np.max(x)), 400)
                kde_x = kde_gaussian_weighted_1d(x, w_theta, xg, silverman_bandwidth(x) * float(kde_bw_scale))
                ax_r.plot(kde_x, xg, c="#2ca02c", lw=2, label="x|θ* (kernel)")
            ax_r.set_xlabel("density")
            ax_r.set_ylabel("x")
            ax_r.set_ylim(-2.5, 2.5)
            ax_r.tick_params(axis='y', labelleft=False)
            if show_marginals:
                ax_r.legend(loc="best")
            st.pyplot(fig_r)

if __name__ == "__main__":
    try:
        main()
    except Exception as _e:
        st.exception(_e)
