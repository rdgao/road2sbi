from typing import Optional, Dict, Any

import numpy as np

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

from density1d_models import pdf_gaussian


def make_plotly_figure(
    x: np.ndarray,
    hist_n: int,
    x_rug: np.ndarray,
    y_rug: np.ndarray,
    grid: np.ndarray,
    fit_pdf: Optional[np.ndarray],
    true_pdf: Optional[np.ndarray],
    fit_name: str,
    fit_params: Dict[str, Any],
    show_hist: bool = True,
    show_true: bool = True,
):
    if not _PLOTLY_AVAILABLE:
        return None
    fig = go.Figure()
    if show_hist:
        fig.add_trace(
            go.Histogram(
                x=x,
                nbinsx=int(hist_n),
                histnorm="probability density",
                marker_color="rgba(31,119,180,0.55)",
                name="samples",
            )
        )
    if x_rug.size:
        fig.add_trace(
            go.Scatter(
                x=x_rug,
                y=y_rug,
                mode="markers",
                marker=dict(size=8, color="#FFD700", line=dict(color="rgba(0,0,0,0.6)", width=0.5)),
                name="samples",
            )
        )
    if fit_pdf is not None:
        fig.add_trace(
            go.Scatter(x=grid, y=fit_pdf, mode="lines", line=dict(color="#d62728", width=2), name="fit")
        )
    if show_true and (true_pdf is not None):
        fig.add_trace(
            go.Scatter(x=grid, y=true_pdf, mode="lines", line=dict(color="#2ca02c", width=2), name="true")
        )
    # Overlays for Gaussian (User)
    if fit_name == "Gaussian (User)":
        mu0 = float(fit_params.get("mu", 0.0))
        sig0 = max(float(fit_params.get("sigma", 1.0)), 1e-9)
        fig.add_shape(
            type="line", x0=mu0, x1=mu0, y0=0, y1=1, xref="x", yref="paper",
            line=dict(color="rgba(214,39,40,0.7)", width=1.5, dash="dash")
        )
        y_sigma = float(pdf_gaussian(np.array([mu0 + sig0]), mu0, sig0)[0])
        fig.add_shape(
            type="line", x0=mu0 - sig0, x1=mu0 + sig0, y0=y_sigma, y1=y_sigma, xref="x", yref="y",
            line=dict(color="rgba(214,39,40,0.7)", width=1.5, dash="dash")
        )

    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=420)
    return fig


def make_matplotlib_figure(
    x: np.ndarray,
    hist_n: int,
    x_rug: np.ndarray,
    y_rug: np.ndarray,
    grid: np.ndarray,
    fit_pdf: Optional[np.ndarray],
    true_pdf: Optional[np.ndarray],
    fit_name: str,
    fit_params: Dict[str, Any],
    show_hist: bool = True,
    show_true: bool = True,
):
    if not _MATPLOTLIB_AVAILABLE:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    if show_hist:
        ax.hist(x, bins=int(hist_n), density=True, color=(0.12, 0.47, 0.71, 0.55))
    if x_rug.size:
        ax.scatter(x_rug, y_rug, s=24, c="#FFD700", edgecolors="k", linewidths=0.4, label="samples")
    if fit_pdf is not None:
        ax.plot(grid, fit_pdf, c="#d62728", lw=2, label="fit")
    if show_true and (true_pdf is not None):
        ax.plot(grid, true_pdf, c="#2ca02c", lw=2, label="true")
    if fit_name == "Gaussian (User)":
        mu0 = float(fit_params.get("mu", 0.0))
        sig0 = max(float(fit_params.get("sigma", 1.0)), 1e-9)
        ax.axvline(mu0, color="#d62728", linestyle="--", linewidth=1.2)
        y_sigma = float(pdf_gaussian(np.array([mu0 + sig0]), mu0, sig0)[0])
        ax.hlines(y=y_sigma, xmin=mu0 - sig0, xmax=mu0 + sig0, colors="#d62728", linestyles="--", linewidth=1.2)
    ax.legend(loc="best")
    return fig

# ---- Deprecated shim: re-export from consolidated module ----
try:
    from road2sbi.plot_utils import *  # type: ignore  # noqa: F401,F403
except Exception:
    pass
