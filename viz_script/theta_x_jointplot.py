"""Joint plot of theta vs x for a sine+linear simulator with homoscedastic Gaussian noise.

Simulator: x = A * sin(w * theta + phi) + b * theta + c + N(0, sigma^2)

Reproduces a matplotlib jointplot: 2D KDE heatmap (viridis) + scatter in the
main panel, with marginal histograms on top (theta) and right (x).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

# ---- simulator params ----
A = 3.2                 # sine amplitude
PERIOD = 4.6             # sine period (in theta units)
PHI = 0.6                # sine phase offset
B = 0.55                 # linear slope
C = 0.0                  # linear intercept
SIGMA = 0.9              # homoscedastic noise std

THETA_RANGE = (-10.0, 10.0)
N_SAMPLES = 900
SEED = 0


def f(theta: np.ndarray) -> np.ndarray:
    w = 2.0 * np.pi / PERIOD
    return A * np.sin(w * theta + PHI) + B * theta + C


def simulate(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return f(theta) + rng.normal(0.0, SIGMA, size=theta.shape)


def make_jointplot(theta: np.ndarray, x: np.ndarray, save_path: str) -> None:
    theta_min, theta_max = THETA_RANGE
    x_min, x_max = x.min() - 1.5, x.max() + 1.5

    fig = plt.figure(figsize=(6, 7))
    gs = GridSpec(
        2, 2, figure=fig,
        width_ratios=[4, 1], height_ratios=[1, 4],
        wspace=0.05, hspace=0.05,
    )
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Joint density heatmap (tight bandwidth so it hugs the curve rather than
    # smoothing into a diffuse diagonal blob)
    kde = gaussian_kde(np.vstack([theta, x]), bw_method=0.15)
    xx, yy = np.mgrid[theta_min:theta_max:200j, x_min:x_max:200j]
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax_main.imshow(
        zz.T, extent=(theta_min, theta_max, x_min, x_max),
        origin="lower", cmap="viridis", aspect="auto",
    )
    ax_main.scatter(theta, x, s=20, facecolors="0.15", edgecolors="k", linewidths=0.4, alpha=0.6)
    ax_main.set_xlim(theta_min, theta_max)
    ax_main.set_ylim(x_min, x_max)
    ax_main.set_xlabel(r"$\theta$")
    ax_main.set_ylabel("x")
    ax_main.grid(True, linestyle="--", alpha=0.4, color="0.8")

    ax_histx.hist(theta, bins=40, range=(theta_min, theta_max), color="0.6")
    ax_histx.set_xlim(theta_min, theta_max)
    ax_histx.tick_params(labelbottom=False)
    ax_histx.grid(True, linestyle="--", alpha=0.4)

    ax_histy.hist(x, bins=40, range=(x_min, x_max), orientation="horizontal", color="0.6")
    ax_histy.set_ylim(x_min, x_max)
    ax_histy.tick_params(labelleft=False)
    ax_histy.grid(True, linestyle="--", alpha=0.4)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(SEED)
    theta = rng.uniform(*THETA_RANGE, size=N_SAMPLES)
    x = simulate(theta, rng)

    import os
    out_path = os.path.join(os.path.dirname(__file__), "theta_x_jointplot.png")
    make_jointplot(theta, x, out_path)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
