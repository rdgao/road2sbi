import numpy as np

from abc_utils import Bounds2D
from abc_simulators import (
    get_simulator,
    preprocess_theta,
    sim_checkerboard,
)


def _rng():
    return np.random.default_rng(0)


def test_linear_gaussian_deterministic_sigma0():
    sim = get_simulator("Linear Gaussian")
    theta = np.array([1.0, 2.0])
    y = sim(theta, 0.0, _rng())
    # A = [[1,0.5],[-0.3,1.2]], b = [0,0]
    expected = np.array([1.0 + 0.5 * 2.0, -0.3 * 1.0 + 1.2 * 2.0])
    assert np.allclose(y, expected)


def test_banana_mapping_sigma0():
    sim = get_simulator("Banana")
    theta = np.array([2.0, 1.0])
    y = sim(theta, 0.0, _rng())
    assert np.allclose(y, np.array([2.0, 1.0 + 0.2 * 4.0]))


def test_circle_preprocess_and_sim():
    b = Bounds2D(-3.0, 3.0, -3.0, 3.0)
    theta = np.array([0.0, 0.0])  # left edge maps to phi ~ 0
    th_used = preprocess_theta(theta, "Circle", b)
    sim = get_simulator("Circle")
    y = sim(th_used, 0.0, _rng())
    # At phi ~ 0, y ~ [r, 0]
    assert y[1] == 0.0


def test_two_moons_branches():
    b = Bounds2D(0.0, 1.0, -1.0, 1.0)
    # angle from theta[0], branch from theta[1] sign around midpoint 0
    theta_upper = np.array([0.5, -0.5])
    theta_lower = np.array([0.5, 0.5])
    th_u = preprocess_theta(theta_upper, "Two Moons", b)
    th_l = preprocess_theta(theta_lower, "Two Moons", b)
    sim = get_simulator("Two Moons")
    yu = sim(th_u, 0.0, _rng())
    yl = sim(th_l, 0.0, _rng())
    # Upper y2 is +sin(phi); lower y2 is negative (shifted)
    assert yu[1] > yl[1]


def test_checkerboard_uses_bounds_grid():
    b = Bounds2D(-2.0, 2.0, -2.0, 2.0)
    # Same raw theta but different bounds would map differently; here we just check deterministic and within [-1.4,1.4]^2 approx
    theta = np.array([0.1, -0.3])
    y = sim_checkerboard(theta, 0.0, _rng(), bounds=b)
    assert y.shape == (2,)
    assert np.all(y <= 1.5) and np.all(y >= -1.5)

