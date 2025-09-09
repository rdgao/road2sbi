import numpy as np

from density1d_models import sample_mog, pdf_gaussian
from density1d_fit import fit_gaussian_mle, em_fit_gmm, silverman_bandwidth, kde_gaussian


def _rng():
    return np.random.default_rng(0)


def test_gaussian_mle_recovers_params():
    rng = _rng()
    mu_true, sig_true = 1.2, 0.7
    x = rng.normal(mu_true, sig_true, size=2000)
    mu_hat, sig_hat = fit_gaussian_mle(x)
    assert abs(mu_hat - mu_true) < 0.1
    assert abs(sig_hat - sig_true) < 0.1


def test_em_gmm_two_components_close():
    rng = _rng()
    mus = np.array([-2.0, 2.5])
    sigmas = np.array([0.6, 0.9])
    weights = np.array([0.4, 0.6])
    x = sample_mog(3000, rng, mus, sigmas, weights)
    gmm = em_fit_gmm(x, K=2, max_iter=200, rng=_rng())
    # Compare sorted means/weights to true
    order = np.argsort(gmm.mus)
    mu_hat = gmm.mus[order]
    w_hat = gmm.weights[order]
    assert np.allclose(np.sort(mus), mu_hat, atol=0.3)
    assert np.allclose(np.sort(weights), w_hat, atol=0.1)


def test_kde_gaussian_basic_properties():
    rng = _rng()
    x = rng.normal(0.0, 1.0, size=1000)
    h = silverman_bandwidth(x)
    grid = np.linspace(-5, 5, 400)
    px = kde_gaussian(x, grid, h)
    assert px.shape == grid.shape
    assert np.all(px >= 0.0)
