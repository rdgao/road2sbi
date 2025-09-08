import math
from typing import Callable

import numpy as np

from abc_utils import Bounds2D


def sim_linear_gaussian(theta: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    A = np.array([[1.0, 0.5], [-0.3, 1.2]])
    b = np.array([0.0, 0.0])
    y = A @ theta + b
    if sigma > 0:
        y = y + rng.normal(0.0, sigma, size=2)
    return y


def sim_banana(theta: np.ndarray, sigma: float, rng: np.random.Generator, a: float = 0.2) -> np.ndarray:
    x, y = float(theta[0]), float(theta[1])
    out = np.array([x, y + a * (x ** 2)])
    if sigma > 0:
        out = out + rng.normal(0.0, sigma, size=2)
    return out


def sim_circle(theta: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    r = 0.75 + 0.25 * math.tanh(0.5 * float(theta[1]))
    phi = float(theta[0])
    out = np.array([r * math.cos(phi), r * math.sin(phi)])
    # Clean tiny numerical residue to zero for exact cardinal angles
    if sigma == 0.0:
        if abs(out[0]) < 1e-12:
            out[0] = 0.0
        if abs(out[1]) < 1e-12:
            out[1] = 0.0
    if sigma > 0:
        out = out + rng.normal(0.0, sigma, size=2)
    return out


def sim_spiral(theta: np.ndarray, sigma: float, rng: np.random.Generator,
               r0: float = 0.3, k: float = 0.06, c: float = 0.25, b: float = 0.5) -> np.ndarray:
    phi = float(theta[0])
    r = r0 + k * phi + c * math.tanh(b * float(theta[1]))
    y = np.array([r * math.cos(phi), r * math.sin(phi)])
    if sigma > 0:
        y = y + rng.normal(0.0, sigma, size=2)
    return y


def sim_rings(theta: np.ndarray, sigma: float, rng: np.random.Generator,
              r_min: float = 0.5, dr: float = 0.25) -> np.ndarray:
    phi = float(theta[0])
    k_idx = int(np.clip(round(float(theta[1])), 0, 2))
    r = r_min + k_idx * dr
    y = np.array([r * math.cos(phi), r * math.sin(phi)])
    if sigma > 0:
        y = y + rng.normal(0.0, sigma, size=2)
    return y


def sim_pinwheel(theta: np.ndarray, sigma: float, rng: np.random.Generator, kappa: float = 0.8) -> np.ndarray:
    x1, x2 = float(theta[0]), float(theta[1])
    r = math.hypot(x1, x2)
    alpha = kappa * r
    ca, sa = math.cos(alpha), math.sin(alpha)
    R = np.array([[ca, -sa], [sa, ca]])
    y = R @ np.array([x1, x2])
    if sigma > 0:
        y = y + rng.normal(0.0, sigma, size=2)
    return y


def sim_s_curve(theta: np.ndarray, sigma: float, rng: np.random.Generator, a: float = 1.0, b: float = 1.0) -> np.ndarray:
    t1, t2 = float(theta[0]), float(theta[1])
    y = np.array([t1, t2 + a * math.tanh(b * t1)])
    if sigma > 0:
        y = y + rng.normal(0.0, sigma, size=2)
    return y


from typing import Optional


def sim_checkerboard(theta: np.ndarray, sigma: float, rng: np.random.Generator, m: int = 4, delta: float = 0.4,
                     bounds: Optional[Bounds2D] = None) -> np.ndarray:
    t1, t2 = float(theta[0]), float(theta[1])
    if bounds is None:
        bx = Bounds2D(-3, 3, -3, 3)
    else:
        bx = bounds
    u = 0.0 if bx.x_max == bx.x_min else (t1 - bx.x_min) / (bx.x_max - bx.x_min)
    v = 0.0 if bx.y_max == bx.y_min else (t2 - bx.y_min) / (bx.y_max - bx.y_min)
    # Guard against exact 1.0 mapping to out-of-range bin due to floating boundaries
    if u >= 1.0:
        u = np.nextafter(1.0, 0.0)
    if v >= 1.0:
        v = np.nextafter(1.0, 0.0)
    i = int(math.floor(u * m))
    j = int(math.floor(v * m))
    cx = -1.0 + (i + 0.5) * (2.0 / m)
    cy = -1.0 + (j + 0.5) * (2.0 / m)
    sx = delta if ((i + j) % 2 == 0) else -delta
    sy = -delta if ((i + j) % 2 == 0) else delta
    y = np.array([cx + sx, cy + sy])
    if sigma > 0:
        y = y + rng.normal(0.0, sigma, size=2)
    return y


def get_simulator(name: str) -> Callable[[np.ndarray, float, np.random.Generator], np.ndarray]:
    if name == "Linear Gaussian":
        return sim_linear_gaussian
    if name == "Banana":
        return sim_banana
    if name == "Two Moons":
        def wrapper(theta: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
            phi = float(theta[0])
            choose_lower = float(theta[1]) >= 0.0
            upper = np.array([math.cos(phi), math.sin(phi)])
            lower = np.array([1.0 - math.cos(phi), -math.sin(phi) - 0.5])
            out = lower if choose_lower else upper
            if sigma > 0:
                out = out + rng.normal(0.0, sigma, size=2)
            return out

        return wrapper
    if name == "Circle":
        return sim_circle
    if name == "Spiral":
        return sim_spiral
    if name == "Rings":
        return sim_rings
    if name == "Pinwheel":
        return sim_pinwheel
    if name == "S-Curve":
        return sim_s_curve
    if name == "Checkerboard":
        def wrapper(theta: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
            return sim_checkerboard(theta, sigma, rng)

        return wrapper
    return sim_linear_gaussian


def preprocess_theta(theta: np.ndarray, sim_name: str, bounds: Bounds2D) -> np.ndarray:
    if sim_name == "Two Moons":
        if bounds.x_max != bounds.x_min:
            phi = (float(theta[0]) - bounds.x_min) / (bounds.x_max - bounds.x_min) * math.pi
        else:
            phi = 0.0
        mid_y = 0.5 * (bounds.y_min + bounds.y_max)
        moon_var = float(theta[1]) - mid_y
        return np.array([phi, moon_var], dtype=float)
    if sim_name == "Circle":
        if bounds.x_max != bounds.x_min:
            phi = (float(theta[0]) - bounds.x_min) / (bounds.x_max - bounds.x_min) * (2.0 * math.pi)
        else:
            phi = 0.0
        return np.array([phi, float(theta[1])], dtype=float)
    if sim_name == "Spiral":
        if bounds.x_max != bounds.x_min:
            phi = (float(theta[0]) - bounds.x_min) / (bounds.x_max - bounds.x_min) * (4.0 * math.pi)
        else:
            phi = 0.0
        return np.array([phi, float(theta[1])], dtype=float)
    if sim_name == "Rings":
        if bounds.x_max != bounds.x_min:
            phi = (float(theta[0]) - bounds.x_min) / (bounds.x_max - bounds.x_min) * (2.0 * math.pi)
        else:
            phi = 0.0
        t1 = bounds.y_min + (bounds.y_max - bounds.y_min) / 3.0
        t2 = bounds.y_min + 2.0 * (bounds.y_max - bounds.y_min) / 3.0
        yv = float(theta[1])
        if yv < t1:
            k_idx = 0
        elif yv < t2:
            k_idx = 1
        else:
            k_idx = 2
        return np.array([phi, float(k_idx)], dtype=float)
    if sim_name == "Checkerboard":
        return theta
    return theta
