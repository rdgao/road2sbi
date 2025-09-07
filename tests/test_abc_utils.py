import math
import numpy as np

from abc_utils import (
    Bounds2D,
    nice_ticks,
    canvas_to_theta,
    theta_to_canvas,
    kde2d_grid,
    compute_distances,
    compute_acceptance_mask,
)


def test_bounds2d_sample_within_bounds():
    rng = np.random.default_rng(0)
    b = Bounds2D(-2.0, 3.0, -1.0, 4.0)
    for _ in range(100):
        th = b.sample(rng)
        assert b.x_min <= th[0] <= b.x_max
        assert b.y_min <= th[1] <= b.y_max


def test_nice_ticks_monotonic_and_within():
    ticks = nice_ticks(-3.0, 7.0, 6)
    assert len(ticks) >= 2
    assert all(ticks[i] <= ticks[i + 1] for i in range(len(ticks) - 1))
    assert ticks[0] >= -3.0 - 1e-9 and ticks[-1] <= 7.0 + 1e-9


def test_canvas_theta_roundtrip():
    b = Bounds2D(-5.0, 5.0, -2.0, 2.0)
    W, H = 400, 400
    points = [(-5.0, -2.0), (0.0, 0.0), (5.0, 2.0), (2.5, -1.0)]
    for th in points:
        left, top = theta_to_canvas(th, W, H, b)
        th_back = canvas_to_theta(left, top, W, H, b)
        assert np.allclose(th_back, np.array(th), atol=1e-6)


def test_kde2d_grid_basic_properties():
    b = Bounds2D(-1.0, 1.0, -1.0, 1.0)
    data = np.array([[0.0, 0.0], [0.5, 0.5], [-0.5, -0.25]])
    X, Y, Z = kde2d_grid(data, b, gridsize=20)
    assert X.shape == Y.shape == Z.shape
    assert Z.min() >= 0.0


def test_compute_distances_variants():
    gt = np.array([1.0, -1.0])
    xs = [(1.0, -1.0), (2.0, -1.0), (1.0, 1.0)]
    d_l2 = compute_distances(xs, gt, metric="L2")
    d_l1 = compute_distances(xs, gt, metric="L1")
    d_mah = compute_distances(xs, gt, metric="Mahalanobis (diag)", w1=4.0, w2=1.0)
    assert d_l2[0] == 0.0
    assert math.isclose(d_l2[1], 1.0)
    assert math.isclose(d_l1[2], 2.0)
    # Mahalanobis with w1=4 makes dx weighted twice in distance
    assert math.isclose(d_mah[1], 2.0)  # sqrt(4*(1)^2 + 0)


def test_compute_acceptance_mask_threshold():
    gt = np.array([0.0, 0.0])
    xs = [(0.0, 0.0), (0.6, 0.0), (0.3, 0.4)]
    mask = compute_acceptance_mask(xs, gt, epsilon=0.5, metric="L2")
    assert mask == [True, False, False]

