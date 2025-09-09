import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from density1d_models import pdf_gaussian


# ----------- KDE (Gaussian) -----------

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


# ----------- MoG fit via simple EM -----------

@dataclass
class GMMParams:
    mus: np.ndarray
    sigmas: np.ndarray
    weights: np.ndarray


def em_fit_gmm(x: np.ndarray, K: int, max_iter: int = 100, tol: float = 1e-6, rng: Optional[np.random.Generator] = None) -> GMMParams:
    rng = rng or np.random.default_rng(0)
    x = x.reshape(-1)
    n = x.size
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
        resp = np.zeros((n, K))
        for k in range(K):
            resp[:, k] = weights[k] * pdf_gaussian(x, mus[k], sigmas[k])
        s = resp.sum(axis=1, keepdims=True)
        s = np.clip(s, 1e-300, None)
        resp /= s

        Nk = resp.sum(axis=0)
        Nk = np.clip(Nk, 1e-12, None)
        weights = Nk / n
        mus = (resp * x[:, None]).sum(axis=0) / Nk
        diffs2 = (x[:, None] - mus[None, :]) ** 2
        sigmas = np.sqrt((resp * diffs2).sum(axis=0) / Nk)
        sigmas = np.clip(sigmas, 1e-3, None)

        ll = loglik(mus, sigmas, weights)
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return GMMParams(mus=mus, sigmas=sigmas, weights=weights)


# ----------- MLE for simple families -----------

def fit_gaussian_mle(x: np.ndarray) -> Tuple[float, float]:
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=0))
    return mu, max(sigma, 1e-9)


def fit_laplace_mle(x: np.ndarray) -> Tuple[float, float]:
    mu = float(np.median(x))
    b = float(np.mean(np.abs(x - mu)))
    return mu, max(b, 1e-9)

