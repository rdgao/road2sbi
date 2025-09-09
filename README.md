# SBI: Parameters → Simulation (Streamlit)

Interactive demo where you start with a hidden ground-truth parameter θ_true and a random observation xₒ. Click in the parameter panel to propose θ and see the corresponding simulation x; aim to get close to xₒ. Toggle to reveal θ_true.

## Fast setup (conda + uv, Python 3.12)

```bash
cd sbi-tut
conda create -n sbi-tut python=3.12 -y   # minimal, just Python
conda activate sbi-tut
python -m pip install -U uv               # bootstrap uv once
uv pip install --system -r requirements.txt
streamlit run 1-rej_abc_demo.py
```

Optionally, start notebooks in the same env:

```bash
jupyter lab
```

## Alternative (pip virtualenv + uv)

```bash
cd sbi-tut
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
python -m pip install -U uv
uv pip install -r requirements.txt
streamlit run 1-rej_abc_demo.py
```

## Notes

- The Streamlit demo needs `numpy`, `plotly`, `streamlit`, `streamlit-plotly-events`; the conda env also ships
  `matplotlib`, `jupyterlab`, and `sbi` (with CPU-only PyTorch) for your broader tutorial work.
- Use the sidebar to pick a simulator, set the prior box, and noise σ.
- Click in θ-space (or use 'Random guess') to simulate x; compare to xₒ.
- Use 'New observation' to resample θ_true and xₒ; toggle 'Show ground truth θ_true' to reveal the answer.

## Rejection ABC demo (1-rej_abc_demo.py)

This app now uses a lightweight modular structure to keep concerns separated:

- `abc_utils.py`:
  - Core data structures and helpers: `Bounds2D`, nice ticks, canvas↔θ mapping, 2D KDE (`kde2d_grid`), covariance ellipse points, distances and acceptance masks.
- `abc_simulators.py`:
  - All 2D→2D simulators and preprocessing: Linear Gaussian, Banana, Two Moons, Circle, Spiral, Rings, Pinwheel, S‑Curve, Checkerboard. Also `get_simulator` and `preprocess_theta`.
- `abc_plotting.py`:
  - Plot helpers for both Plotly and Matplotlib, returning figure objects. Used by the main app to render θ and x panels with consistent styles.

Run the demo:

```bash
cd sbi-tut
streamlit run 1-rej_abc_demo.py
```

Feature highlights:
- Click‑to‑sample in θ‑space, sampling buttons (1 or N), fading of older samples, and clear history.
- Multiple simulators with equations rendered at the bottom; ground truth θ and x* overlay (yellow stars).
- ABC controls: distance choices (L2/L1/Mahalanobis), ε slider and optional quantile setting, acceptance stats, “show only accepted”.
- Posterior view: accepted θ KDE on a fixed grid, 1σ ellipse, and marginals (top/right); bounds match θ‑space.
- PPC: posterior predictive density and points with the same axes as the x panel; N PPC draws slider in the sidebar.

## uv quick installs

For faster dependency management, this repo includes `uv`.

- In the conda env (recommended):

```bash
conda activate sbi-tut
uv pip install --system -r requirements.txt -U
```

- In a pip virtualenv:

```bash
python -m pip install -U uv
uv pip install -r requirements.txt
```

## New: 1D Density Estimation Demo

Explore simple 1D densities, draw N samples, then fit parametric/non‑parametric estimators (Gaussian MLE, GMM via EM, Gaussian KDE) and compare fits. Tip: with "Gaussian (User)", try optimizing the mean log‑likelihood by choosing μ and σ interactively.

```bash
cd sbi-tut
streamlit run 2-density_estimation_1d.py
```

What’s included
- Estimators: Gaussian (User), Gaussian (MLE), Mixture of Gaussians (EM), and Gaussian KDE.
- Likelihoods: Displays the mean log-likelihood of the current samples under the selected model. For Gaussian (User), also tracks the best‑so‑far mean log-likelihood per sample set.
- Persistence: Fitted curves and their x‑grids persist across new samples and true‑family changes. Use “Refit model” (top of sidebar) to update MLE/EM/KDE fits. “Gaussian (User)” does not refit; it uses your μ and σ directly.
- Overlays (Gaussian User): Dashed vertical line at μ, and a dashed horizontal segment at y = pdf(μ ± σ) from μ−σ to μ+σ.

Usage notes
- Start in the sidebar: choose a true density and sample size, draw samples, then configure and (if applicable) refit your estimator at the top.
- “Refit model” is available for Gaussian (MLE), MoG (EM), and KDE; it’s disabled for Gaussian (User).
- The yellow points labeled “samples” are a stable subset for rug-like visualization and do not change unless you redraw samples.

### More details (theory and models)

Purpose and big picture
- Goal: build intuition for 1D density estimation. You pick a true distribution, draw samples x₁:ₙ, then fit a model p̂(x|θ) and inspect both the curve and the mean log‑likelihood 1/n Σ log p̂(xᵢ|θ).
- Posterior‐free: this demo focuses on frequentist fits (MLE/EM/KDE) and direct likelihood evaluation, complementary to the ABC demo.

True families (generators)
- Gaussian, Mixture of Gaussians (2 or 3 components), Laplace, Student‑t, Lognormal, Beta [0,1], Uniform, Triangular.
- 2D→1D projections: Projected Circle (arcsine), Projected Rings (mixture of arcsine), Projected Two Moons (x), Projected Checkerboard (x), Projected Spiral (x). These produce non‑Gaussian shapes (multi‑modal, heavy tails, bounded, etc.).

Estimators (fits)
- Gaussian (User): you set μ and σ; the app reports the mean log‑likelihood on the current samples and tracks best‑so‑far per sample set. The curve does not auto‑change when you draw new samples.
- Gaussian (MLE): closed‑form estimates μ̂ = mean(x), σ̂ = std(x) (population version). Click “Refit model” to recompute; after that, the curve persists while you explore.
- Mixture of Gaussians (EM): expectation‑maximization with configurable K and iterations. “Refit model” learns {weights, means, stds}; the stored fit is then used to evaluate and plot.
- KDE (Gaussian): p̂(x) = 1/n Σ ϕ((x − xᵢ)/h)/h with bandwidth h chosen by Silverman/Scott or set manually. “Refit model” stores the training samples and bandwidth; evaluation uses the stored set.

Likelihood and evaluation
- Mean log‑likelihood: 1/n Σ log p̂(xᵢ). Higher is better (less negative). For mixtures: log Σ wₖ N(x | μₖ, σₖ²). For KDE: log p̂(xᵢ) with numerical safeguards.
- Caution: Evaluating on the same samples used to fit (especially for KDE) can overestimate performance. For rigorous comparison, split into train/validation; this demo keeps things interactive and fast.

Persistence model
- Curves and their x‑grids persist across new samples and true‑family changes. MLE/EM/KDE only change after you press “Refit model”. Gaussian (User) uses exactly your current μ,σ and a fixed x‑grid.

Suggested explorations
- Manually tune μ, σ to maximize the mean log‑likelihood for various true families; compare with MLE.
- Change K for the MoG fit and watch how the mean log‑likelihood and shape respond on multi‑modal truths.
- Vary KDE bandwidth (Silverman/Scott/Manual) on heavy‑tailed vs. compact distributions; find h values that balance bias/variance.
