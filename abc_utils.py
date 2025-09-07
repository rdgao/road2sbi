import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


@dataclass
class Bounds2D:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        x = rng.uniform(self.x_min, self.x_max)
        y = rng.uniform(self.y_min, self.y_max)
        return np.array([x, y], dtype=float)


def _nice_num(x: float, round_result: bool = True) -> float:
    if x <= 0:
        return 1.0
    exp = math.floor(math.log10(x))
    f = x / (10 ** exp)
    if round_result:
        if f < 1.5:
            nf = 1.0
        elif f < 3.0:
            nf = 2.0
        elif f < 7.0:
            nf = 5.0
        else:
            nf = 10.0
    else:
        if f <= 1.0:
            nf = 1.0
        elif f <= 2.0:
            nf = 2.0
        elif f <= 5.0:
            nf = 5.0
        else:
            nf = 10.0
    return nf * (10 ** exp)


def nice_ticks(vmin: float, vmax: float, nticks: int = 5) -> List[float]:
    if nticks < 2 or not math.isfinite(vmin) or not math.isfinite(vmax):
        return [vmin, vmax]
    if vmin == vmax:
        return [vmin]
    if vmin > vmax:
        vmin, vmax = vmax, vmin
    span = vmax - vmin
    step = _nice_num(span / (nticks - 1), round_result=True)
    tick_start = math.ceil(vmin / step) * step
    ticks: List[float] = []
    t = tick_start
    for _ in range(100):
        if t > vmax + 1e-12:
            break
        ticks.append(round(t, max(0, -int(math.floor(math.log10(step)))) + 2))
        t += step
    if not ticks:
        ticks = [vmin, vmax]
    return ticks


def canvas_to_theta(left_px: float, top_px: float, w: int, h: int, bounds: Bounds2D) -> np.ndarray:
    x_norm = np.clip(left_px / max(w, 1), 0.0, 1.0)
    y_norm_from_top = np.clip(top_px / max(h, 1), 0.0, 1.0)
    y_norm = 1.0 - y_norm_from_top
    x_val = bounds.x_min + x_norm * (bounds.x_max - bounds.x_min)
    y_val = bounds.y_min + y_norm * (bounds.y_max - bounds.y_min)
    return np.array([x_val, y_val], dtype=float)


def theta_to_canvas(theta: Tuple[float, float], w: int, h: int, bounds: Bounds2D) -> Tuple[float, float]:
    x, y = theta
    x_norm = 0.0 if bounds.x_max == bounds.x_min else (x - bounds.x_min) / (bounds.x_max - bounds.x_min)
    y_norm = 0.0 if bounds.y_max == bounds.y_min else (y - bounds.y_min) / (bounds.y_max - bounds.y_min)
    left_px = x_norm * w
    top_px = (1.0 - y_norm) * h
    return float(left_px), float(top_px)


def ellipse_points(mean: np.ndarray, cov: np.ndarray, n: int = 120, k_sigma: float = 1.0) -> np.ndarray:
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    axes = vecs @ (np.sqrt(vals) * k_sigma * np.eye(2))
    ts = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.cos(ts), np.sin(ts)], axis=0)
    pts = (mean.reshape(2, 1) + axes @ circle).T
    return pts


def kde2d_grid(data: np.ndarray, bounds: Bounds2D, gridsize: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = data.shape[0]
    gx = np.linspace(bounds.x_min, bounds.x_max, gridsize)
    gy = np.linspace(bounds.y_min, bounds.y_max, gridsize)
    X, Y = np.meshgrid(gx, gy, indexing='xy')
    if n < 2:
        Z = np.zeros_like(X)
        return X, Y, Z
    std = np.std(data, axis=0)
    factor = n ** (-1.0 / 6.0)
    hx = std[0] * factor if std[0] > 0 else (bounds.x_max - bounds.x_min) / 50.0
    hy = std[1] * factor if std[1] > 0 else (bounds.y_max - bounds.y_min) / 50.0
    hx = max(hx, 1e-6)
    hy = max(hy, 1e-6)
    xs = X.ravel()[:, None]
    ys = Y.ravel()[:, None]
    dx = (xs - data[:, 0][None, :]) / hx
    dy = (ys - data[:, 1][None, :]) / hy
    expo = -0.5 * (dx * dx + dy * dy)
    Z = np.exp(expo).sum(axis=1)
    Z = Z.reshape(X.shape)
    Z *= (1.0 / (2.0 * np.pi * hx * hy * n))
    return X, Y, Z


def compute_distances(
    xs: List[Tuple[float, float]], gt_x: Optional[np.ndarray], metric: str = "L2", w1: float = 1.0, w2: float = 1.0
) -> Optional[List[float]]:
    if gt_x is None:
        return None
    gx, gy = float(gt_x[0]), float(gt_x[1])
    dists: List[float] = []
    for (x, y) in xs:
        dx, dy = x - gx, y - gy
        if metric == "L1":
            d = abs(dx) + abs(dy)
        elif metric.startswith("Mahalanobis"):
            d = math.sqrt(max(0.0, w1 * dx * dx + w2 * dy * dy))
        else:
            d = math.hypot(dx, dy)
        dists.append(float(d))
    return dists


def compute_acceptance_mask(
    xs: List[Tuple[float, float]], gt_x: Optional[np.ndarray], epsilon: Optional[float],
    metric: str = "L2", w1: float = 1.0, w2: float = 1.0
) -> Optional[List[bool]]:
    if gt_x is None or epsilon is None:
        return None
    dists = compute_distances(xs, gt_x, metric, w1, w2)
    if dists is None:
        return None
    eps = float(epsilon)
    return [d <= eps for d in dists]

