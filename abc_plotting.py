from typing import List, Tuple, Optional

try:
    import plotly.graph_objects as go  # type: ignore
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt  # type: ignore
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

from abc_utils import Bounds2D


def plot_scatter_plotly(points: List[Tuple[float, float]], title: str, bounds: Bounds2D | None = None,
                        gt_point: Tuple[float, float] | None = None,
                        eps: float | None = None,
                        accepted_mask: Optional[List[bool]] = None,
                        prev_name: str | None = None,
                        curr_name: str | None = None,
                        crosshair_xy: Tuple[float, float] | None = None,
                        fade_prev_alphas: Optional[List[float]] = None,
                        only_accepted: bool = False):
    if not PLOTLY_AVAILABLE:
        return None
    n = len(points)
    xs_prev = [p[0] for p in points[:-1]] if n > 1 else []
    ys_prev = [p[1] for p in points[:-1]] if n > 1 else []
    xs_curr = [points[-1][0]] if n >= 1 else []
    ys_curr = [points[-1][1]] if n >= 1 else []
    fig = go.Figure()
    if n > 1:
        if accepted_mask is not None and len(accepted_mask) == n:
            if only_accepted:
                xs_prev_acc, ys_prev_acc, cols_prev_acc = [], [], []
                for i in range(0, n - 1):
                    if not accepted_mask[i]:
                        continue
                    x_i, y_i = points[i]
                    alpha = 1.0
                    if fade_prev_alphas is not None and i < len(fade_prev_alphas):
                        alpha = float(fade_prev_alphas[i])
                    xs_prev_acc.append(x_i); ys_prev_acc.append(y_i)
                    cols_prev_acc.append(f"rgba(44,160,44,{alpha})")
                if xs_prev_acc:
                    fig.add_trace(
                        go.Scatter(x=xs_prev_acc, y=ys_prev_acc, mode="markers",
                                   marker=dict(size=8, color=cols_prev_acc), name="accepted")
                    )
            else:
                xs_prev_acc, ys_prev_acc, cols_prev_acc = [], [], []
                xs_prev_rej, ys_prev_rej, cols_prev_rej = [], [], []
                for i in range(0, n - 1):
                    x_i, y_i = points[i]
                    alpha = 1.0
                    if fade_prev_alphas is not None and i < len(fade_prev_alphas):
                        alpha = float(fade_prev_alphas[i])
                    if accepted_mask[i]:
                        xs_prev_acc.append(x_i); ys_prev_acc.append(y_i)
                        cols_prev_acc.append(f"rgba(44,160,44,{alpha})")
                    else:
                        xs_prev_rej.append(x_i); ys_prev_rej.append(y_i)
                        cols_prev_rej.append(f"rgba(31,119,180,{alpha})")
                if xs_prev_rej:
                    fig.add_trace(
                        go.Scatter(x=xs_prev_rej, y=ys_prev_rej, mode="markers",
                                   marker=dict(size=7, color=cols_prev_rej), name="rejected")
                    )
                if xs_prev_acc:
                    fig.add_trace(
                        go.Scatter(x=xs_prev_acc, y=ys_prev_acc, mode="markers",
                                   marker=dict(size=8, color=cols_prev_acc), name="accepted")
                    )
        else:
            cols_prev = None
            if fade_prev_alphas is not None and len(fade_prev_alphas) >= (n - 1):
                cols_prev = [f"rgba(31,119,180,{float(fade_prev_alphas[i])})" for i in range(0, n - 1)]
            fig.add_trace(
                go.Scatter(x=xs_prev, y=ys_prev, mode="markers",
                           marker=dict(size=8, color=cols_prev or "#1f77b4"), name=prev_name or "previous")
            )
    if n >= 1:
        fig.add_trace(
            go.Scatter(x=xs_curr, y=ys_curr, mode="markers",
                       marker=dict(size=12, color="#d62728"), name=curr_name)
        )
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=30, b=10),
        height=420,
    )
    if crosshair_xy is not None:
        cx, cy = crosshair_xy
        fig.add_shape(type="line", x0=cx, x1=cx, xref="x", y0=0, y1=1, yref="paper",
                      line=dict(color="#d62728", width=1, dash="dot"))
        fig.add_shape(type="line", x0=0, x1=1, xref="paper", y0=cy, y1=cy, yref="y",
                      line=dict(color="#d62728", width=1, dash="dot"))
    if gt_point is not None:
        fig.add_trace(
            go.Scatter(x=[gt_point[0]], y=[gt_point[1]], mode="markers",
                       marker=dict(size=14, color="#FFD700", symbol="star"), name="ground truth")
        )
        if eps is not None and eps > 0:
            gx, gy = gt_point
            fig.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=gx - eps, x1=gx + eps,
                y0=gy - eps, y1=gy + eps,
                line=dict(color="rgba(44,160,44,0.4)", width=1),
                fillcolor="rgba(44,160,44,0.08)",
                layer="below",
            )
    if bounds is not None:
        fig.update_xaxes(range=[bounds.x_min, bounds.x_max], title_text="x", showgrid=True)
        fig.update_yaxes(range=[bounds.y_min, bounds.y_max], title_text="y", scaleanchor="x", scaleratio=1, showgrid=True)
    else:
        fig.update_xaxes(title_text="x", showgrid=True)
        fig.update_yaxes(title_text="y", scaleanchor="x", scaleratio=1, showgrid=True)
    return fig


def plot_scatter_matplotlib(points: List[Tuple[float, float]], title: str, bounds: Bounds2D | None = None,
                            gt_point: Tuple[float, float] | None = None,
                            eps: float | None = None,
                            accepted_mask: Optional[List[bool]] = None,
                            prev_name: str | None = None,
                            curr_name: str | None = None,
                            crosshair_xy: Tuple[float, float] | None = None,
                            fade_prev_alphas: Optional[List[float]] = None,
                            only_accepted: bool = False):
    if not MATPLOTLIB_AVAILABLE:
        return None
    fig, ax = plt.subplots(figsize=(5, 4))
    n = len(points)
    if n > 1:
        if accepted_mask is not None and len(accepted_mask) == n:
            if only_accepted:
                xs_prev_acc, ys_prev_acc, cols_prev_acc = [], [], []
                for i in range(0, n - 1):
                    if not accepted_mask[i]:
                        continue
                    x_i, y_i = points[i]
                    alpha = 1.0
                    if fade_prev_alphas is not None and i < len(fade_prev_alphas):
                        alpha = float(fade_prev_alphas[i])
                    xs_prev_acc.append(x_i); ys_prev_acc.append(y_i)
                    cols_prev_acc.append((0.17, 0.63, 0.17, alpha))
                if xs_prev_acc:
                    ax.scatter(xs_prev_acc, ys_prev_acc, s=30, c=cols_prev_acc, label="accepted")
            else:
                xs_prev_acc, ys_prev_acc, cols_prev_acc = [], [], []
                xs_prev_rej, ys_prev_rej, cols_prev_rej = [], [], []
                for i in range(0, n - 1):
                    x_i, y_i = points[i]
                    alpha = 1.0
                    if fade_prev_alphas is not None and i < len(fade_prev_alphas):
                        alpha = float(fade_prev_alphas[i])
                    if accepted_mask[i]:
                        xs_prev_acc.append(x_i); ys_prev_acc.append(y_i)
                        cols_prev_acc.append((0.17, 0.63, 0.17, alpha))
                    else:
                        xs_prev_rej.append(x_i); ys_prev_rej.append(y_i)
                        cols_prev_rej.append((0.12, 0.47, 0.71, alpha))
                if xs_prev_rej:
                    ax.scatter(xs_prev_rej, ys_prev_rej, s=30, c=cols_prev_rej, label="rejected")
                if xs_prev_acc:
                    ax.scatter(xs_prev_acc, ys_prev_acc, s=30, c=cols_prev_acc, label="accepted")
        else:
            xs_prev = [p[0] for p in points[:-1]]
            ys_prev = [p[1] for p in points[:-1]]
            if fade_prev_alphas is not None and len(fade_prev_alphas) >= (n - 1):
                cols_prev = [(0.12, 0.47, 0.71, float(fade_prev_alphas[i])) for i in range(0, n - 1)]
            else:
                cols_prev = [(0.12, 0.47, 0.71, 1.0)] * (n - 1)
            ax.scatter(xs_prev, ys_prev, s=30, c=cols_prev, label=prev_name)
    if n >= 1:
        ax.scatter([points[-1][0]], [points[-1][1]], s=60, c="#d62728", label=curr_name)
    if bounds is not None:
        ax.set_xlim(bounds.x_min, bounds.x_max)
        ax.set_ylim(bounds.y_min, bounds.y_max)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3, which="major")
    if gt_point is not None:
        ax.scatter([gt_point[0]], [gt_point[1]], s=100, c="#FFD700", marker="*")
        if eps is not None and eps > 0:
            try:
                from matplotlib.patches import Circle  # type: ignore
                circ = Circle((gt_point[0], gt_point[1]), eps,
                              edgecolor=(0.17, 0.63, 0.17, 0.4), facecolor=(0.17, 0.63, 0.17, 0.08), linewidth=1)
                ax.add_patch(circ)
            except Exception:
                pass
    if crosshair_xy is not None:
        cx, cy = crosshair_xy
        ax.axvline(cx, color="#d62728", linestyle=":", linewidth=1)
        ax.axhline(cy, color="#d62728", linestyle=":", linewidth=1)
    if prev_name or curr_name:
        ax.legend(loc="best")
    return fig

