import math
import time
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st


# Optional: click capture via drawable canvas
try:
    from streamlit_drawable_canvas import st_canvas  # type: ignore
    _CANVAS_AVAILABLE = True
except Exception:
    _CANVAS_AVAILABLE = False
    st_canvas = None

# Optional: use plotly for nicer scatter plots; fallback to matplotlib if unavailable
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

# Optional: Plotly click capture fallback
try:
    from streamlit_plotly_events import plotly_events  # type: ignore
    _PLOTLY_EVENTS_AVAILABLE = True
except Exception:
    _PLOTLY_EVENTS_AVAILABLE = False
    plotly_events = None


st.set_page_config(page_title="Rejection ABC Demo", layout="wide")

# Import modular utilities
try:
    from utils import (
        Bounds2D,
        nice_ticks,
        canvas_to_theta,
        theta_to_canvas,
        ellipse_points,
        kde2d_grid,
        compute_acceptance_mask,
        compute_distances,
    )
except Exception:  # Support running from project root
    from road2sbi.utils import (
        Bounds2D,
        nice_ticks,
        canvas_to_theta,
        theta_to_canvas,
        ellipse_points,
        kde2d_grid,
        compute_acceptance_mask,
        compute_distances,
    )


try:
    from simulators import get_simulator, preprocess_theta, sim_checkerboard
except Exception:
    from road2sbi.simulators import get_simulator, preprocess_theta, sim_checkerboard


def ensure_state():
    if "thetas" not in st.session_state:
        st.session_state.thetas = []  # List[Tuple[float, float]]
    if "ys" not in st.session_state:
        st.session_state.ys = []  # List[Tuple[float, float]]
    if "rng_seed" not in st.session_state:
        st.session_state.rng_seed = 123
    if "rng" not in st.session_state:
        st.session_state.rng = np.random.default_rng(int(st.session_state.rng_seed))
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0
    if "last_sim_name" not in st.session_state:
        st.session_state.last_sim_name = None
    if "gt_theta" not in st.session_state:
        st.session_state.gt_theta = None  # type: ignore
    if "gt_y" not in st.session_state:
        st.session_state.gt_y = None  # type: ignore
    if "show_gt_theta" not in st.session_state:
        st.session_state.show_gt_theta = False
    if "fade_old" not in st.session_state:
        st.session_state.fade_old = True
    if "_canvas_objs_count" not in st.session_state:
        st.session_state._canvas_objs_count = 0
    if "last_click_time" not in st.session_state:
        st.session_state.last_click_time = 0


def get_rng() -> np.random.Generator:
    return st.session_state.rng



def add_sample(theta: np.ndarray, sim_name: str, noise_sigma: float, bounds: Bounds2D) -> None:
    rng = get_rng()
    sim_fn = get_simulator(sim_name)
    theta_used = preprocess_theta(theta, sim_name, bounds)
    if sim_name == "Checkerboard":
        y = sim_checkerboard(theta, noise_sigma, rng, bounds=bounds)
    else:
        y = sim_fn(theta_used, noise_sigma, rng)
    st.session_state.thetas.append((float(theta[0]), float(theta[1])))
    st.session_state.ys.append((float(y[0]), float(y[1])))


try:
    from plot_utils import (
        plot_scatter_plotly,
        plot_scatter_matplotlib,
    )
except Exception:
    from road2sbi.plot_utils import (
        plot_scatter_plotly,
        plot_scatter_matplotlib,
    )


# utilities already imported above via utils/road2sbi.utils


def main():
    ensure_state()

    st.title("Rejection ABC Demo")
    st.caption(
        "Click in parameter space (left) or click the buttons in the side panel to draw samples randomly to simulate and plot in data space (right)."
    )

    with st.sidebar:
        st.header("Settings")
        # Placeholders to control visual order
        s_sampling = st.container()
        s_gt = st.container()
        s_abc = st.container()
        s_sim = st.container()

        # --- Simulator (defined first for dependency) ---
        with s_sim:
            st.markdown("---")
            st.subheader("Simulator")
            with st.expander("Simulator parameters", expanded=False):
                sim_name = st.selectbox(
                    "Simulator",
                    ["Linear Gaussian", "Banana", "Two Moons", "Circle", "Spiral", "Rings", "Pinwheel", "S-Curve", "Checkerboard"],
                    index=2,
                )
                # If simulator changed, clear all existing samples and refresh canvas
                if st.session_state.last_sim_name is None:
                    st.session_state.last_sim_name = sim_name
                elif st.session_state.last_sim_name != sim_name:
                    st.session_state.last_sim_name = sim_name
                    st.session_state.thetas.clear()
                    st.session_state.ys.clear()
                    st.session_state.canvas_key += 1
                    st.session_state._canvas_objs_count = 0

                noise_sigma = st.slider("Noise σ", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

                st.caption("Prior bounds (θ-space)")
                theta1_range = st.slider("θ₁ range", min_value=-10.0, max_value=10.0, value=(-3.0, 3.0), step=0.5)
                theta2_range = st.slider("θ₂ range", min_value=-10.0, max_value=10.0, value=(-3.0, 3.0), step=0.5)
            x_min, x_max = theta1_range
            y_min, y_max = theta2_range
            bounds = Bounds2D(float(x_min), float(x_max), float(y_min), float(y_max))

        # --- Sampling (top) ---
        with s_sampling:
            st.subheader("Sampling")
            with st.expander("Sampling controls", expanded=False):
                # Compact row: Fade toggle + seed input
                row1c1, row1c2 = st.columns([1.2, 0.8])
                with row1c1:
                    st.session_state.fade_old = st.checkbox("Fade older samples", value=st.session_state.fade_old)
                with row1c2:
                    seed = st.number_input("Random seed", value=int(st.session_state.rng_seed), step=1)
                    if seed != st.session_state.rng_seed:
                        st.session_state.rng_seed = int(seed)
                        st.session_state.rng = np.random.default_rng(int(seed))

                # Batch size slider
                batch_n = st.slider("Batch size (N)", min_value=10, max_value=2000, value=100, step=10)

                # Two buttons in one row
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("Sample 1 θ"):
                        theta = bounds.sample(get_rng())
                        add_sample(theta, sim_name, noise_sigma, bounds)
                with b2:
                    if st.button("Sample N θ"):
                        with st.spinner(f"Sampling {batch_n} θ and simulating..."):
                            for _ in range(batch_n):
                                theta = bounds.sample(get_rng())
                                add_sample(theta, sim_name, noise_sigma, bounds)

                st.markdown("---")
                if st.button("Clear history"):
                    st.session_state.thetas.clear()
                    st.session_state.ys.clear()
                    st.session_state.canvas_key += 1  # refresh canvas
                    st.session_state._canvas_objs_count = 0

        # --- Ground truth (middle) ---
        with s_gt:
            st.subheader("Ground truth")
            with st.expander("Ground truth controls", expanded=False):
                gt_col1, gt_col2 = st.columns(2)
                with gt_col1:
                    show_gt = st.checkbox("Show GT θ", value=st.session_state.show_gt_theta)
                with gt_col2:
                    if st.button("Sample new GT"):
                        st.session_state.gt_theta = bounds.sample(get_rng())
                st.session_state.show_gt_theta = show_gt

                if st.session_state.gt_theta is None:
                    st.session_state.gt_theta = bounds.sample(get_rng())
                # Compute GT observation deterministically (no noise)
                gt_theta_used = preprocess_theta(st.session_state.gt_theta, sim_name, bounds)
                st.session_state.gt_y = get_simulator(sim_name)(gt_theta_used, 0.0, get_rng())
                if st.session_state.show_gt_theta:
                    st.caption(
                        f"Ground truth θ ≈ ({float(st.session_state.gt_theta[0]):.3f}, {float(st.session_state.gt_theta[1]):.3f}); "
                        f"x* ≈ ({float(st.session_state.gt_y[0]):.3f}, {float(st.session_state.gt_y[1]):.3f})"
                    )

        # --- ABC controls ---
        with s_abc:
            st.subheader("ABC")
            with st.expander("Distance and acceptance", expanded=False):
                if "abc_epsilon" not in st.session_state:
                    st.session_state.abc_epsilon = 0.5
                # Distance options
                dist_metric = st.selectbox("Distance", ["L2", "L1", "Mahalanobis (diag)"])
                w1 = 1.0
                w2 = 1.0
                if dist_metric.startswith("Mahalanobis"):
                    c_w1, c_w2 = st.columns(2)
                    with c_w1:
                        w1 = st.number_input("w₁", value=1.0, min_value=0.0)
                    with c_w2:
                        w2 = st.number_input("w₂", value=1.0, min_value=0.0)
                st.session_state.abc_epsilon = st.slider("ε (acceptance radius)", min_value=0.0, max_value=5.0, value=float(st.session_state.abc_epsilon), step=0.05)

                # Optional: set epsilon by quantile of distances
                set_eps_q = st.checkbox("Set ε by distance quantile", value=False)
                q = 0.1
                if set_eps_q:
                    q = st.slider("Quantile q", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
                # Compute distances and (if enabled) update epsilon
                dists = compute_distances(st.session_state.ys, st.session_state.gt_y, dist_metric, w1, w2)
                if set_eps_q and dists is not None and len(dists) > 0:
                    try:
                        st.session_state.abc_epsilon = float(np.quantile(np.array(dists), q))
                    except Exception:
                        pass
                    st.caption(f"ε set to q={q:.2f} quantile: {st.session_state.abc_epsilon:.3f}")

                # Acceptance stats
                if dists is not None and len(dists) > 0:
                    acc_count = int(np.sum(np.array(dists) <= float(st.session_state.abc_epsilon)))
                    total = len(dists)
                    rate = acc_count / total if total > 0 else 0.0
                    st.caption(f"Accepted: {acc_count}/{total} (rate {100*rate:.1f}%)")

                # Option to show only accepted
                if "only_accepted" not in st.session_state:
                    st.session_state.only_accepted = False
                st.session_state.only_accepted = st.checkbox("Show only accepted", value=st.session_state.only_accepted)

                # PPC controls
                if "ppc_n" not in st.session_state:
                    st.session_state.ppc_n = 500
                st.session_state.ppc_n = st.slider("N posterior predictive samples", min_value=50, max_value=5000, value=int(st.session_state.ppc_n), step=50)

        # Canvas info
        if not _CANVAS_AVAILABLE:
            st.info(
                "Click capture uses optional dependency 'streamlit-drawable-canvas'.\n"
                "Install with: pip install streamlit-drawable-canvas"
            )

    

    col_left, col_right = st.columns(2)

    # Compute acceptance mask for current samples
    acceptance_mask = compute_acceptance_mask(
        st.session_state.ys,
        st.session_state.gt_y if st.session_state.gt_y is not None else None,
        float(st.session_state.abc_epsilon) if "abc_epsilon" in st.session_state else None,
        metric=dist_metric,
        w1=w1,
        w2=w2,
    )

    # Compute fading alphas for previous points (older → more transparent)
    def compute_fade_alphas(n: int, min_alpha: float = 0.2, max_alpha: float = 1.0) -> List[float]:
        if n <= 0:
            return []
        if n == 1:
            return [max_alpha]
        return list(np.linspace(min_alpha, max_alpha, n))

    n_pts = len(st.session_state.thetas)
    fade_prev_alphas = compute_fade_alphas(max(0, n_pts - 1)) if st.session_state.fade_old else None
    fade_all_theta_alphas = compute_fade_alphas(n_pts) if st.session_state.fade_old else None

    # Left: Parameter space capture + plot
    with col_left:
        st.subheader("Parameter space (θ)")

        # Optional canvas for clicks
        W, H = 420, 420
        new_theta_from_click = None

        if _CANVAS_AVAILABLE:
            # Pre-populate with existing points
            init_objs = []
            # ----- Axes, ticks, and labels on canvas -----
            # Border (four lines)
            init_objs.append({"type": "line", "x1": 0, "y1": 0, "x2": W, "y2": 0,
                              "stroke": "#999999", "strokeWidth": 1, "selectable": False, "evented": False})
            init_objs.append({"type": "line", "x1": W, "y1": 0, "x2": W, "y2": H,
                              "stroke": "#999999", "strokeWidth": 1, "selectable": False, "evented": False})
            init_objs.append({"type": "line", "x1": W, "y1": H, "x2": 0, "y2": H,
                              "stroke": "#999999", "strokeWidth": 1, "selectable": False, "evented": False})
            init_objs.append({"type": "line", "x1": 0, "y1": H, "x2": 0, "y2": 0,
                              "stroke": "#999999", "strokeWidth": 1, "selectable": False, "evented": False})

            # Compute axis baselines in pixels
            left_px_min, top_px_min = theta_to_canvas((bounds.x_min, bounds.y_min), W, H, bounds)
            left_px_max, top_px_max = theta_to_canvas((bounds.x_max, bounds.y_max), W, H, bounds)
            bottom_y = top_px_min
            left_x = left_px_min

            # Ticks
            try:
                x_ticks = nice_ticks(bounds.x_min, bounds.x_max, 5)
                y_ticks = nice_ticks(bounds.y_min, bounds.y_max, 5)
            except Exception:
                x_ticks = [bounds.x_min, bounds.x_max]
                y_ticks = [bounds.y_min, bounds.y_max]

            # X ticks and labels
            for xv in x_ticks:
                xp, yp = theta_to_canvas((float(xv), bounds.y_min), W, H, bounds)
                init_objs.append({
                    "type": "line", "x1": xp, "y1": bottom_y, "x2": xp, "y2": bottom_y - 6,
                    "stroke": "#888888", "strokeWidth": 1, "selectable": False, "evented": False
                })
                init_objs.append({
                    "type": "textbox", "left": xp - 10, "top": bottom_y - 20,
                    "text": f"{xv:g}", "fontSize": 10, "fill": "#666666", "editable": False,
                    "selectable": False, "evented": False
                })

            # Y ticks and labels
            for yv in y_ticks:
                xp, yp = theta_to_canvas((bounds.x_min, float(yv)), W, H, bounds)
                init_objs.append({
                    "type": "line", "x1": left_x, "y1": yp, "x2": left_x + 6, "y2": yp,
                    "stroke": "#888888", "strokeWidth": 1, "selectable": False, "evented": False
                })
                init_objs.append({
                    "type": "textbox", "left": max(0, left_x + 8), "top": yp - 8,
                    "text": f"{yv:g}", "fontSize": 10, "fill": "#666666", "editable": False,
                    "selectable": False, "evented": False
                })

            # Axis labels on canvas
            init_objs.append({
                "type": "textbox", "left": W/2 - 10, "top": H - 24, "text": "θ₁",
                "fontSize": 14, "fill": "#555555", "editable": False,
            })
            init_objs.append({
                "type": "textbox", "left": 6, "top": 6, "text": "θ₂",
                "fontSize": 14, "fill": "#555555", "editable": False,
            })

            # Existing sample points (color by acceptance)
            # Prepare acceptance mask in main before this block.
            for i, th in enumerate(st.session_state.thetas):
                if st.session_state.only_accepted and (acceptance_mask is not None):
                    if i >= len(acceptance_mask) or not acceptance_mask[i]:
                        continue
                left_px, top_px = theta_to_canvas(th, W, H, bounds)
                # Default color rejected/neutral
                # Fading alpha per point
                alpha = 1.0
                if fade_all_theta_alphas is not None and i < len(fade_all_theta_alphas):
                    alpha = float(fade_all_theta_alphas[i])
                color = f"rgba(31,119,180,{alpha})"
                try:
                    if acceptance_mask is not None and i < len(acceptance_mask) and acceptance_mask[i]:
                        color = f"rgba(44,160,44,{alpha})"
                except Exception:
                    pass
                init_objs.append(
                    {
                        "type": "circle",
                        "left": left_px,
                        "top": top_px,
                        "radius": 4,
                        "fill": color,
                        "stroke": color,
                    }
                )

            # Ground truth θ on canvas (optional)
            if st.session_state.show_gt_theta and st.session_state.gt_theta is not None:
                gt_left, gt_top = theta_to_canvas(tuple(st.session_state.gt_theta), W, H, bounds)
                init_objs.append({
                    "type": "textbox",
                    "left": gt_left - 8,
                    "top": gt_top - 12,
                    "text": "★",
                    "fontSize": 18,
                    "fill": "#FFD700",
                    "editable": False,
                    "selectable": False,
                    "evented": False,
                })

            # Crosshair through origin (θ=(0,0)) always shown if origin within bounds
            # Vertical line at θ1 = 0 (only if within bounds)
            if bounds.x_min <= 0.0 <= bounds.x_max:
                cx, _ = theta_to_canvas((0.0, bounds.y_min), W, H, bounds)
                init_objs.append({
                    "type": "line", "x1": cx, "y1": 0, "x2": cx, "y2": H,
                    "stroke": "#d62728", "strokeWidth": 1, "strokeDashArray": [4, 4],
                    "selectable": False, "evented": False
                })
            # Horizontal line at θ2 = 0 (only if within bounds)
            if bounds.y_min <= 0.0 <= bounds.y_max:
                _, cy = theta_to_canvas((bounds.x_min, 0.0), W, H, bounds)
                init_objs.append({
                    "type": "line", "x1": 0, "y1": cy, "x2": W, "y2": cy,
                    "stroke": "#d62728", "strokeWidth": 1, "strokeDashArray": [4, 4],
                    "selectable": False, "evented": False
                })

            if _CANVAS_AVAILABLE and st_canvas is not None:
                canvas = st_canvas(
                    key=f"param-canvas-{st.session_state.canvas_key}",
                    fill_color="#1f77b466",
                    stroke_width=2,
                    stroke_color="#1f77b4",
                    background_color="#FFFFFF",
                    update_streamlit=True,
                    height=H,
                    width=W,
                    drawing_mode="point",
                    initial_drawing={"version": "4.4.0", "objects": init_objs},
                    display_toolbar=True,
                )
                # Handle canvas clicks more robustly to prevent flashing
                if canvas.json_data is not None:
                    try:
                        data = canvas.json_data
                        objs = data.get("objects", [])
                        current_count = len(objs)
                        current_time = time.time()
                        
                        # Only process new clicks, not every rerun, and throttle rapid clicks
                        if (current_count > st.session_state._canvas_objs_count and 
                            current_time - st.session_state.last_click_time > 0.5):  # 500ms throttle
                            # Get the most recent object (the new click)
                            last = objs[-1]
                            if last.get("type") == "circle":  # Only process actual click points
                                left_px = float(last.get("left", 0.0))
                                top_px = float(last.get("top", 0.0))
                                new_theta_from_click = canvas_to_theta(left_px, top_px, W, H, bounds)
                                st.session_state._canvas_objs_count = current_count
                                st.session_state.last_click_time = current_time
                    except Exception:
                        pass
            else:
                st.warning("Canvas drawing not available. Install streamlit-drawable-canvas for interactive parameter selection.")

        else:
            # Fallback: render θ scatter with Plotly/Matplotlib when canvas is unavailable
            gt_theta_plot = (
                tuple(st.session_state.gt_theta)
                if (st.session_state.show_gt_theta and st.session_state.gt_theta is not None)
                else None
            )
            eps = float(st.session_state.abc_epsilon) if "abc_epsilon" in st.session_state else None
            if _PLOTLY_AVAILABLE:
                fig_theta = plot_scatter_plotly(
                    st.session_state.thetas,
                    "Parameter space (θ)",
                    bounds,
                    gt_theta_plot,
                    eps,
                    acceptance_mask,
                    prev_name="previous",
                    curr_name="current",
                    crosshair_xy=None,
                    fade_prev_alphas=fade_prev_alphas,
                    only_accepted=bool(st.session_state.only_accepted),
                )
                if fig_theta is not None:
                    st.caption("Click on the plot to add a θ.")
                    if _PLOTLY_EVENTS_AVAILABLE and plotly_events is not None:
                        # Capture click in data coordinates
                        events = plotly_events(
                            fig_theta,
                            click_event=True,
                            hover_event=False,
                            select_event=False,
                            key=f"theta-plot-events-{st.session_state.canvas_key}",
                        )
                        if events:
                            ev = events[0]
                            try:
                                x = float(ev.get("x"))
                                y = float(ev.get("y"))
                                # Clamp to bounds and accept
                                x = min(max(x, bounds.x_min), bounds.x_max)
                                y = min(max(y, bounds.y_min), bounds.y_max)
                                new_theta_from_click = np.array([x, y], dtype=float)
                            except Exception:
                                pass
                    else:
                        st.plotly_chart(fig_theta, use_container_width=True)
            elif _MATPLOTLIB_AVAILABLE:
                fig_theta_m = plot_scatter_matplotlib(
                    st.session_state.thetas,
                    "Parameter space (θ)",
                    bounds,
                    gt_theta_plot,
                    eps,
                    acceptance_mask,
                    prev_name="previous",
                    curr_name="current",
                    crosshair_xy=None,
                    fade_prev_alphas=fade_prev_alphas,
                    only_accepted=bool(st.session_state.only_accepted),
                )
                if fig_theta_m is not None:
                    st.pyplot(fig_theta_m, clear_figure=True)

        # If we captured a click, add it and rerun to update display
        if new_theta_from_click is not None:
            add_sample(new_theta_from_click, sim_name, noise_sigma, bounds)
            st.rerun()
        # Removed 1D parameter histograms; projections added in Posterior view

        # Posterior view (accepted θ only)
        st.subheader("Posterior view (accepted θ)")
        if acceptance_mask is not None and any(acceptance_mask):
                acc_idx = [i for i, a in enumerate(acceptance_mask) if a]
                th_arr = np.array([st.session_state.thetas[i] for i in acc_idx], dtype=float)
                mu = th_arr.mean(axis=0)
                cov = np.cov(th_arr.T) if th_arr.shape[0] > 1 else np.eye(2) * 1e-6
                pts = ellipse_points(mu, cov, n=200, k_sigma=1.0)
                if _PLOTLY_AVAILABLE:
                    import plotly.graph_objects as go  # type: ignore
                    from plotly.subplots import make_subplots  # type: ignore
                    fig_post = make_subplots(rows=2, cols=2,
                                             row_heights=[0.25, 0.75], column_widths=[0.75, 0.25],
                                             specs=[[{"type": "xy"}, {"type": "histogram"}],
                                                    [{"type": "xy"}, {"type": "histogram"}]],
                                             horizontal_spacing=0.04, vertical_spacing=0.04,
                                             shared_xaxes=True, shared_yaxes=True)
                    # Main density (row2,col1)
                    if th_arr.shape[0] >= 3:
                        Xg, Yg, Zg = kde2d_grid(th_arr, bounds, gridsize=70)
                        fig_post.add_trace(go.Contour(
                            x=Xg[0, :], y=Yg[:, 0], z=Zg,
                            colorscale='Viridis', showscale=False, contours_coloring='heatmap',
                            opacity=0.85, ncontours=20, line_smoothing=1.0
                        ), row=2, col=1)
                    fig_post.add_trace(go.Scatter(x=th_arr[:, 0], y=th_arr[:, 1], mode='markers',
                                                  marker=dict(size=5, color='rgba(44,160,44,0.8)'), name='accepted'),
                                       row=2, col=1)
                    fig_post.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode='lines',
                                                  line=dict(color='#2ca02c', width=2), name='1σ ellipse'),
                                       row=2, col=1)
                    # Top histogram for θ1 (row1,col1)
                    fig_post.add_trace(go.Histogram(x=th_arr[:, 0], nbinsx=40,
                                                    marker=dict(color='rgba(44,160,44,0.75)'), showlegend=False,
                                                    xbins=dict(start=bounds.x_min, end=bounds.x_max)),
                                       row=1, col=1)
                    # Right histogram for θ2 (row2,col2)
                    fig_post.add_trace(go.Histogram(y=th_arr[:, 1], nbinsy=40, orientation='h',
                                                    marker=dict(color='rgba(44,160,44,0.75)'), showlegend=False,
                                                    ybins=dict(start=bounds.y_min, end=bounds.y_max)),
                                       row=2, col=2)
                    fig_post.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=380,
                                           title="Accepted θ density + 1σ ellipse + marginals", bargap=0.02)
                    # Keep bounds consistent with parameter space, tidy axes
                    fig_post.update_xaxes(range=[bounds.x_min, bounds.x_max], showgrid=True, row=2, col=1, title_text="θ₁")
                    fig_post.update_yaxes(range=[bounds.y_min, bounds.y_max], showgrid=True, row=2, col=1, title_text="θ₂", scaleanchor="x", scaleratio=1)
                    # Turn off grid on 1D marginal plots for a cleaner look
                    fig_post.update_xaxes(range=[bounds.x_min, bounds.x_max], showgrid=False, row=1, col=1, title_text="")
                    fig_post.update_yaxes(showticklabels=False, showgrid=False, row=1, col=1)
                    fig_post.update_yaxes(range=[bounds.y_min, bounds.y_max], showgrid=False, row=2, col=2)
                    fig_post.update_xaxes(showticklabels=False, showgrid=False, row=2, col=2)
                    st.plotly_chart(fig_post, use_container_width=True)
                elif _MATPLOTLIB_AVAILABLE:
                    import matplotlib.pyplot as plt  # type: ignore
                    from matplotlib.gridspec import GridSpec  # type: ignore
                    figp = plt.figure(figsize=(5.2, 3.6))
                    gs = GridSpec(2, 2, figure=figp, width_ratios=[4, 1.2], height_ratios=[1.2, 4], wspace=0.05, hspace=0.05)
                    ax_histx = figp.add_subplot(gs[0, 0])
                    ax_main = figp.add_subplot(gs[1, 0])
                    ax_histy = figp.add_subplot(gs[1, 1], sharey=ax_main)
                    # Main density
                    if th_arr.shape[0] >= 3:
                        Xg, Yg, Zg = kde2d_grid(th_arr, bounds, gridsize=100)
                        ax_main.imshow(Zg.T, extent=(bounds.x_min, bounds.x_max, bounds.y_min, bounds.y_max),
                                       origin='lower', cmap='viridis', alpha=0.8, aspect='auto')
                    ax_main.scatter(th_arr[:, 0], th_arr[:, 1], s=10, c=(0.17, 0.63, 0.17, 0.8))
                    ax_main.plot(pts[:, 0], pts[:, 1], c="#2ca02c", lw=2)
                    ax_main.set_xlim(bounds.x_min, bounds.x_max)
                    ax_main.set_ylim(bounds.y_min, bounds.y_max)
                    ax_main.set_aspect('equal', adjustable='box')
                    ax_main.grid(True, alpha=0.3)
                    # Top histogram θ1
                    ax_histx.hist(th_arr[:, 0], bins=40, range=(bounds.x_min, bounds.x_max), color=(0.17, 0.63, 0.17, 0.7), density=True)
                    ax_histx.set_xlim(bounds.x_min, bounds.x_max)
                    ax_histx.set_xticklabels([])
                    ax_histx.grid(False)
                    # Right histogram θ2
                    ax_histy.hist(th_arr[:, 1], bins=40, range=(bounds.y_min, bounds.y_max), color=(0.17, 0.63, 0.17, 0.7), density=True, orientation='horizontal')
                    ax_histy.set_ylim(bounds.y_min, bounds.y_max)
                    ax_histy.set_yticklabels([])
                    ax_histy.grid(False)
                    figp.suptitle("Accepted θ density + 1σ ellipse + marginals", y=0.98)
                    st.pyplot(figp, clear_figure=True)
                else:
                    st.info("Install plotly or matplotlib to see posterior view.")
                st.caption(f"Posterior mean θ ≈ ({mu[0]:.3f}, {mu[1]:.3f})")
        else:
            st.info("No accepted samples yet.")

    # Right: Data space
    with col_right:
        st.subheader("Data space (x)")

        # For Two Moons and Circle, it helps if θ₁ behaves like an angle in [0, π] or [0, 2π].
        # Our simulator wrappers already map inside, so just plot results.
        if _PLOTLY_AVAILABLE:
            gt_y_plot = tuple(st.session_state.gt_y) if st.session_state.gt_y is not None else None
            eps = float(st.session_state.abc_epsilon) if st.session_state.gt_y is not None else None
            fig_y = plot_scatter_plotly(st.session_state.ys, "Simulated x", None, gt_y_plot, eps,
                                        accepted_mask=acceptance_mask,
                                        prev_name="previous", curr_name="current", crosshair_xy=None,
                                        fade_prev_alphas=fade_prev_alphas,
                                        only_accepted=st.session_state.only_accepted)
            st.plotly_chart(fig_y, use_container_width=True)
        elif _MATPLOTLIB_AVAILABLE:
            gt_y_plot = tuple(st.session_state.gt_y) if st.session_state.gt_y is not None else None
            eps = float(st.session_state.abc_epsilon) if st.session_state.gt_y is not None else None
            fig_y = plot_scatter_matplotlib(st.session_state.ys, "Simulated x", None, gt_y_plot, eps,
                                            accepted_mask=acceptance_mask,
                                            prev_name="previous", curr_name="current", crosshair_xy=None,
                                            fade_prev_alphas=fade_prev_alphas,
                                            only_accepted=st.session_state.only_accepted)
            st.pyplot(fig_y, clear_figure=True)
        else:
            st.warning("Install plotly or matplotlib to see plots.")

        # Last-sample distance summary
        if st.session_state.gt_y is not None and len(st.session_state.ys) > 0:
            lastx, lasty = st.session_state.ys[-1]
            # Recompute last distance using chosen metric
            d_list = compute_distances([(lastx, lasty)], st.session_state.gt_y, dist_metric, w1, w2)
            d_last = d_list[0] if d_list is not None and len(d_list) == 1 else math.hypot(lastx - float(st.session_state.gt_y[0]), lasty - float(st.session_state.gt_y[1]))
            acc_last = None
            if acceptance_mask is not None and len(acceptance_mask) == len(st.session_state.ys):
                acc_last = acceptance_mask[-1]
            status = "accepted" if acc_last else "rejected" if acc_last is not None else ""
            st.write(f"Last distance ||x - x*|| = {d_last:.3f} {f'({status})' if status else ''}")

        # Posterior predictive check
        st.subheader("Posterior predictive check (PPC)")
        # Read PPC sample size from sidebar
        ppc_n = int(st.session_state.ppc_n) if "ppc_n" in st.session_state else 500
        if acceptance_mask is not None and any(acceptance_mask):
            try:
                acc_idx = [i for i, a in enumerate(acceptance_mask) if a]
                # Draw θ from accepted set uniformly and simulate x
                rng_ppc = get_rng()
                sim_fn = get_simulator(sim_name)
                thetas_acc = np.array([st.session_state.thetas[i] for i in acc_idx], dtype=float)
                choose_idx = rng_ppc.integers(0, len(acc_idx), size=ppc_n)
                ppc_x = []
                for j in range(ppc_n):
                    th = thetas_acc[choose_idx[j]]
                    th_used = preprocess_theta(th, sim_name, bounds)
                    x_pred = sim_fn(th_used, noise_sigma, rng_ppc)
                    ppc_x.append((float(x_pred[0]), float(x_pred[1])))
                ppc_x = np.array(ppc_x)

                # Determine data-plot extents to match PPC axes
                if len(st.session_state.ys) > 0:
                    ys_arr = np.array(st.session_state.ys, dtype=float)
                    x_min_d = float(np.min(ys_arr[:, 0]))
                    x_max_d = float(np.max(ys_arr[:, 0]))
                    y_min_d = float(np.min(ys_arr[:, 1]))
                    y_max_d = float(np.max(ys_arr[:, 1]))
                    # include ground truth point in extents
                    if st.session_state.gt_y is not None:
                        x_min_d = min(x_min_d, float(st.session_state.gt_y[0]))
                        x_max_d = max(x_max_d, float(st.session_state.gt_y[0]))
                        y_min_d = min(y_min_d, float(st.session_state.gt_y[1]))
                        y_max_d = max(y_max_d, float(st.session_state.gt_y[1]))
                    # pad a bit
                    pad_x = 0.05 * max(1e-6, x_max_d - x_min_d)
                    pad_y = 0.05 * max(1e-6, y_max_d - y_min_d)
                    ppc_bounds = Bounds2D(x_min_d - pad_x, x_max_d + pad_x, y_min_d - pad_y, y_max_d + pad_y)
                else:
                    # fallback to PPC extents if no data yet
                    ppc_bounds = Bounds2D(float(np.min(ppc_x[:, 0])), float(np.max(ppc_x[:, 0])),
                                          float(np.min(ppc_x[:, 1])), float(np.max(ppc_x[:, 1])))

                if _PLOTLY_AVAILABLE:
                    import plotly.graph_objects as go  # type: ignore
                    fig_ppc = go.Figure()
                    # Density via 2D KDE grid over PPC samples
                    Xg, Yg, Zg = kde2d_grid(ppc_x, ppc_bounds, gridsize=80)
                    fig_ppc.add_trace(go.Contour(x=Xg[0, :], y=Yg[:, 0], z=Zg, colorscale='Plasma', showscale=False,
                                                 contours_coloring='heatmap', opacity=0.85, ncontours=20))
                    # Overlay PPC points lightly
                    fig_ppc.add_trace(go.Scatter(x=ppc_x[:,0], y=ppc_x[:,1], mode='markers',
                                                 marker=dict(size=4, color='rgba(31,119,180,0.35)'), name='ppc'))
                    # GT
                    if st.session_state.gt_y is not None:
                        fig_ppc.add_trace(go.Scatter(x=[float(st.session_state.gt_y[0])], y=[float(st.session_state.gt_y[1])],
                                                     mode='markers', marker=dict(size=12, color='#FFD700', symbol='star'), name='x*'))
                    fig_ppc.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=320, title="PPC density and samples")
                    fig_ppc.update_xaxes(range=[ppc_bounds.x_min, ppc_bounds.x_max], showgrid=True)
                    fig_ppc.update_yaxes(range=[ppc_bounds.y_min, ppc_bounds.y_max], showgrid=True)
                    st.plotly_chart(fig_ppc, use_container_width=True)
                elif _MATPLOTLIB_AVAILABLE:
                    import matplotlib.pyplot as plt  # type: ignore
                    figppc, axppc = plt.subplots(figsize=(5.0, 3.0))
                    Xg, Yg, Zg = kde2d_grid(ppc_x, ppc_bounds, gridsize=100)
                    im = axppc.imshow(Zg.T, extent=(ppc_bounds.x_min, ppc_bounds.x_max, ppc_bounds.y_min, ppc_bounds.y_max), origin='lower', cmap='plasma', alpha=0.85, aspect='auto')
                    axppc.scatter(ppc_x[:,0], ppc_x[:,1], s=6, c=(0.12,0.47,0.71,0.35))
                    if st.session_state.gt_y is not None:
                        axppc.scatter([float(st.session_state.gt_y[0])], [float(st.session_state.gt_y[1])], s=80, c='#FFD700', marker='*')
                    axppc.set_title("PPC density and samples")
                    axppc.grid(True, alpha=0.3)
                    axppc.set_xlim(ppc_bounds.x_min, ppc_bounds.x_max)
                    axppc.set_ylim(ppc_bounds.y_min, ppc_bounds.y_max)
                    st.pyplot(figppc, clear_figure=True)
            except Exception as e:
                st.warning(f"PPC plotting error: {e}")
        else:
            st.info("No accepted samples yet — cannot run PPC.")

        # Parameter histograms moved to left panel

        with st.expander("History (last 10)"):
            n = len(st.session_state.thetas)
            for i in range(max(0, n - 10), n):
                th = st.session_state.thetas[i]
                yy = st.session_state.ys[i]
                d_i = None
                if st.session_state.gt_y is not None:
                    d_list_i = compute_distances([yy], st.session_state.gt_y, dist_metric if 'dist_metric' in locals() else 'L2', w1 if 'w1' in locals() else 1.0, w2 if 'w2' in locals() else 1.0)
                    d_i = d_list_i[0] if d_list_i is not None else None
                acc_i = None
                if acceptance_mask is not None and i < len(acceptance_mask):
                    acc_i = acceptance_mask[i]
                suffix = ""
                if d_i is not None:
                    suffix = f", d={d_i:.3f}"
                    if acc_i is not None:
                        suffix += " ✓" if acc_i else " ✗"
                st.write(f"{i+1}. θ=({th[0]:.3f}, {th[1]:.3f}) → x=({yy[0]:.3f}, {yy[1]:.3f}){suffix}")

    # ---- Equations at bottom (depends on sim and bounds) ----
    st.markdown("---")
    st.markdown("**Simulator Equations**")
    st.caption("ε ∼ N(0, σ² I)")
    if sim_name == "Linear Gaussian":
        st.latex(r"x = A\,\theta + b + \varepsilon,\quad A = \begin{bmatrix}1 & 0.5\\ -0.3 & 1.2\end{bmatrix},\ b=\begin{bmatrix}0\\0\end{bmatrix}")
    elif sim_name == "Banana":
        st.latex(r"x_1 = \theta_1,\quad x_2 = \theta_2 + a\,\theta_1^2 + \varepsilon,\quad a=0.2")
    elif sim_name == "Two Moons":
        st.latex(r"\varphi = (\theta_1 - x_{\min})/(x_{\max}-x_{\min})\cdot \pi")
        st.latex(r"x = \begin{cases}(\cos\varphi,\ \sin\varphi), & \theta_2 < (y_{\min}+y_{\max})/2\\ (1-\cos\varphi,\ -\sin\varphi - 0.5), & \text{otherwise}\end{cases} + \varepsilon")
    elif sim_name == "Circle":
        st.latex(r"\varphi = (\theta_1 - x_{\min})/(x_{\max}-x_{\min})\cdot 2\pi,\quad r = 0.75 + 0.25\tanh(0.5\,\theta_2)")
        st.latex(r"x = r\,(\cos\varphi,\ \sin\varphi) + \varepsilon")
    elif sim_name == "Spiral":
        st.latex(r"\varphi = (\theta_1 - x_{\min})/(x_{\max}-x_{\min})\cdot 4\pi")
        st.latex(r"r = r_0 + k\,\varphi + c\tanh(b\,\theta_2),\quad x = r(\cos\varphi,\ \sin\varphi) + \varepsilon")
        st.caption("Defaults: r0=0.3, k=0.06, c=0.25, b=0.5")
    elif sim_name == "Rings":
        st.latex(r"\varphi = (\theta_1 - x_{\min})/(x_{\max}-x_{\min})\cdot 2\pi,\quad k \in \{0,1,2\}\text{ by tertiles of }\theta_2")
        st.latex(r"r_k = r_{\min} + k\,\Delta r,\quad x = r_k(\cos\varphi,\ \sin\varphi) + \varepsilon")
        st.caption("Defaults: r_min=0.5, Δr=0.25")
    elif sim_name == "Pinwheel":
        st.latex(r"r = \sqrt{\theta_1^2 + \theta_2^2},\ \alpha = \kappa r,\ R(\alpha)=\begin{bmatrix}\cos\alpha & -\sin\alpha\\ \sin\alpha & \cos\alpha\end{bmatrix},\ x = R(\alpha)\,\theta + \varepsilon")
        st.caption("Default: κ = 0.8")
    elif sim_name == "S-Curve":
        st.latex(r"x_1 = \theta_1,\quad x_2 = \theta_2 + a\tanh(b\,\theta_1) + \varepsilon")
        st.caption("Defaults: a=1.0, b=1.0")
    elif sim_name == "Checkerboard":
        st.latex(r"i=\lfloor m u\rfloor,\ j=\lfloor m v\rfloor,\ u=\frac{\theta_1-x_{\min}}{x_{\max}-x_{\min}},\ v=\frac{\theta_2-y_{\min}}{y_{\max}-y_{\min}}")
        st.latex(r"\mu_{ij} = (c_x(i)+(-1)^{i+j}\,\delta,\ c_y(j)-(-1)^{i+j}\,\delta),\quad x=\mu_{ij}+\varepsilon")
        st.caption("Defaults: m=4, δ=0.4")

    # ---- About / Help ----
    with st.expander("About this app (how to use + tips)"):
        st.markdown(
            """
            How to use this app (suggested workflow)
            - Pick a simulator and noise σ (sigma, the standard deviation of the Gaussian simulator noise) in the sidebar (bottom section). Equations appear at the page bottom and reflect any angle preprocessing.
            - Ground truth (GT): optionally show θ* (yellow star in θ‑space) and see the corresponding x* (yellow star in x‑space), computed without noise.
            - Sampling (top of sidebar):
              - Click in θ‑space (left panel) to add individual θ, or use the buttons to draw θ from the prior (one or a batch of size N).
              - Use “Fade older samples” to reveal temporal ordering. “Clear history” resets all samples.
            - ABC controls (middle of sidebar): choose the distance d(·,·), set ε (epsilon) directly or via the q‑quantile of current distances, optionally hide rejected (“Show only accepted”).
            - Inspect results across panels:
              - Data space (right): blue = rejected, green = accepted, red = current; the ε‑disk around x* shows the acceptance radius.
              - Posterior view (left, below θ‑space): a 2D KDE (Kernel Density Estimate) over accepted θ on a fixed grid, with a 1σ (one standard deviation) covariance ellipse and marginal histograms for θ₁, θ₂.
              - PPC (posterior predictive check, right): draws x̃ by sampling θ uniformly from accepted θ (the empirical ABC posterior) and simulating with the current σ; the KDE of x̃ with x* shows predictive fit.

            Practical tips
            - Start with a moderate ε to see some accepted points, then tighten ε to sharpen the posterior (at the cost of acceptance rate).
            - If one x dimension varies more, prefer Mahalanobis (diag) and increase the weight for the less variable dimension to balance the metric.
            - Tune batch size N to trade off speed vs granularity. Use “Show only accepted” to de‑clutter and view posterior support directly.
            - Multimodal posteriors (e.g., Rings or Two Moons) appear as multiple modes in the KDE and marginals; the PPC should reflect the corresponding mixture in x.
            """
        )

    with st.expander("More details (theory and simulators)"):
        st.markdown(
            """
            Purpose and big picture
            - Goal: build intuition for Approximate Bayesian Computation (ABC) using simple 2D parameter spaces (θ ∈ R²) and 2D observations (x ∈ R²).
            - Simulator model: x = f(θ) + ε where ε ∼ N(0, σ² I) (isotropic Gaussian noise). We treat f as a black box.
            - Prior: uniform over the user‑selected rectangular bounds in θ‑space.

            What is ABC (rejection ABC)
            - Intuition: if simulated x is close to the observed x* under a distance d, the θ that produced it is plausible under the posterior.
            - Step: sample θ ~ prior; simulate x ~ p(x | θ); accept if d(x, x*) ≤ ε (epsilon).
            - Approximate posterior: the accepted samples approximate p_ABC(θ | x*) ∝ 1{d(x, x*) ≤ ε} p(θ). Smaller ε → tighter but lower acceptance.

            Distance d(·,·) and choices
            - L2 (Euclidean): d(x, x*) = √((x₁−x₁*)² + (x₂−x₂*)²).
            - L1 (Manhattan): d(x, x*) = |x₁−x₁*| + |x₂−x₂*|.
            - Mahalanobis (diagonal): d(x, x*) = √(w₁(x₁−x₁*)² + w₂(x₂−x₂*)²), with user‑set weights w₁, w₂.

            Choosing ε (epsilon)
            - Manual: use the ε slider; the green disk around x* visualizes the acceptance region.
            - Quantile: set ε to the q‑quantile of current distances {d(x, x*)}; smaller q → stricter ε.
            - Monitor acceptance: Accepted/Total and % provide feedback on inference tightness and sampling efficiency.

            Mathematical notes and acronyms
            - ABC: Approximate Bayesian Computation — likelihood‑free inference via simulate‑and‑compare.
            - KDE: Kernel Density Estimate — here a Gaussian KDE on a fixed grid for stability and full‑range coverage.
            - PPC: Posterior Predictive Check — visualize p(x̃ | x*) ≈ ∫ p(x̃ | θ) p_ABC(θ | x*) dθ by drawing θ from the ABC posterior.
            - Covariance ellipse (1σ): the ellipse of Mahalanobis distance 1 around the empirical mean using the empirical covariance of accepted θ.
            - Symbols: θ (parameters), x (data), x* (observed/ground‑truth data), σ (noise std), ε (acceptance threshold).

            Simulators in brief (f differs per choice)
            - Linear Gaussian: x = A θ + b + ε.
            - Banana: x₁ = θ₁; x₂ = θ₂ + a θ₁² + ε.
            - Two Moons: θ₁→ angle φ ∈ [0, π]; θ₂ picks the upper/lower moon.
            - Circle: θ₁→ φ ∈ [0, 2π]; radius r(θ₂); x = r[cosφ, sinφ] + ε.
            - Spiral: θ₁→ φ ∈ [0, 4π]; r increases with φ and θ₂.
            - Rings: θ₁→ φ ∈ [0, 2π]; θ₂ tertiles select ring index k ∈ {0,1,2}.
            - Pinwheel: rotate θ by κ ‖θ‖; x = R(κ ‖θ‖) θ + ε.
            - S‑Curve: x₁ = θ₁; x₂ = θ₂ + a tanh(b θ₁) + ε.
            - Checkerboard: bin θ into an m×m grid (bounds‑aware), map to alternating centers μᵢⱼ, then add ε.

            Implementation details
            - θ‑space clicking uses `streamlit-drawable-canvas` (optional). Plots prefer Plotly; Matplotlib is a fallback.
            - θ‑space bounds are shared across all θ views (including the posterior). The posterior/PPC densities use grid‑based Gaussian KDEs.
            """
        )


if __name__ == "__main__":
    main()
