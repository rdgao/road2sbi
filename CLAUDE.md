# road2sbi

Streamlit demos (`1-rej_abc_app.py`, `3-density_estimation_1d_app.py`, `4-conditional_density_app.py`) and teaching material leading up to simulation-based inference. Environment: conda env `road2sbi` (`/opt/miniconda3/envs/road2sbi`, Python 3.12, numpy 2.3, torch 2.5, sbi 0.25).

- `drafts/`: shelved earlier attempts (`5-mdn_walkthrough.ipynb`, `tutorial.ipynb`, `2-sbi_workflow.ipynb`). Context only; do not copy code or text from them into `exercises/`.
- `viz_script/theta_x_jointplot.py`: a sine+line joint-density figure. It is NOT the script behind the user's talk figures (those used a different curve, and the original script was never found). The talk figures are set aside for now.

## `exercises/`: Göttingen SBI summer-school tutorial

A 3-hour hands-on tutorial following lectures on Bayesian basics → rejection ABC → conditional density estimation (MDN) → modern SBI. The original build spec is at `~/Downloads/sbi_tutorial_notebook_specs.md`. Where this file and the spec disagree, this file wins.

| Notebook | Content | Exercise gaps |
|---|---|---|
| `exercise_0_mle.ipynb` | Gaussian MLE: pen-and-paper derivation, closed form, then the same fit with `mu`/`log_sigma` + Adam. No simulator. | 0a derivation (answer in the last markdown cell), 0b closed form, 0c `gaussian_nll`, 0d training loop |
| `exercise_1_abc.ipynb` | Rejection ABC, ε sweep, `abc_posterior_samples` | 1a `rejection_abc`, 1b choose `EPSILON` (target <1% acceptance; solution `EPSILON = 0.15`) |
| `exercise_2_mdn.ipynb` | Warm-up: a `RegressionNet` baseline (single mean, Gaussian NLL with fixed σ=1, i.e. squared error) shows why one Gaussian can't follow multiple branches, before introducing the MDN (K=6, tanh, `x_scale=10`, hidden attribute `self.mlp` not `self.body`), log-sum-exp NLL, same loop shape as 0d, overlay with ABC, amortization panels | 2a `MDN` module, 2b `mdn_nll`, 2c training loop, 2d `mdn_density` |
| `exercise_3_sbi.ipynb` | `sbi` NPE with manual simulation (not `simulate_for_sbi`), 3-way comparison (ABC/MDN/NPE), knobs (`posterior_nn`, train args), ABC vs NPE at matched budgets (TV distance), optional SBC, failure modes (too-narrow prior, out-of-range x), `run_npe` template for your own simulator | 3a NPE train/build, 3b sample, 3c NPE inside the budget loop |

### Decisions (fixed unless the user changes them)

- **Simulator** (chosen by the user): `x = 7·sin(0.75·θ) + 1·θ + ε`, `ε ~ N(0, σ(θ)²)`, `σ(θ) = 1.5·(sin θ + 1.5)`, prior `θ ~ U(-10.5, 10.5)`. Keep the noise model (heteroscedastic in θ) exactly.
- **Prior and RNG (redesigned 2026-09-16):** `prior` is a single `sbi.utils.BoxUniform` object, defined once in the shared setup and reused unchanged as the ABC/MDN sampling distribution *and* the `sbi` prior in Exercise 3 (no second prior definition there anymore). `simulate(theta)` is a pure-torch function (`torch.sin`, `torch.randn_like`) with no explicit RNG argument — it draws from whatever the global torch RNG state happens to be. Seeding is just `torch.manual_seed(SEED)` (no numpy `rng` object anywhere in 1/2/3). This means exact sampled values can shift if cells are run out of order or extra sampling calls are added/removed before a given cell — the user is fine with that; no exercise depends on exact reproducibility, only on qualitative behavior (acceptance rate thresholds, mode counts, etc.).
- `THETA_TRUE = -1.0` → `x_obs ≈ -5.44` (with the torch-based `simulate`). The true posterior has 4 modes (θ ≈ -8, -3.7, -1, 6.5; the last is small). The user tunes this by hand. Don't run parameter sweeps or grid analyses unless asked.
- **Shared setup cell**: the code cell starting `# ---------------- Shared setup` (constants, `mean_x`, `noise_std`, `simulate`, `THETA_TRUE`/`x_obs`, `SHOW_GROUND_TRUTH` + `grid_posterior`, `COLORS`) and its markdown cell must stay **byte-identical in exercises 1, 2 and 3**. Copy-paste, no shared module. Any edit to one must be applied to all three.
- `grid_posterior(x)` is the numerical ground-truth posterior, shown when `SHOW_GROUND_TRUTH = True`. The TV-distance budget comparison in Ex 3 always uses it.
- **MDN input scaling (2026-09-16):** `x_scaled = x / self.x_scale` sits in `MDN.forward` *outside* the EXERCISE 2a SOLUTION block (given scaffold code) — the student's gap is just "pass `x_scaled` through `self.mlp`", so they never have to reason about or write the rescaling themselves.
- **Colors** (Okabe–Ito): ABC `#E69F00`, MDN `#009E73`, NPE `#0072B2`, truth `0.3` grey, prior `0.75` grey, `theta_true` a black dashed line.
- Ex 2 and Ex 3 regenerate ABC (and Ex 3 the MDN) inline from solution code. No files are passed between notebooks.
- sbi 0.25 API: `from sbi.inference import NPE` (not `SNPE`), `NPE(prior, density_estimator="nsf")`, tensors shaped `(n, 1)` float32, `posterior.sample((n,), x=x_obs_t)`.

### Authoring philosophy (from the user)

Write each notebook as a complete, coherent **tutorial first**. Then mark the key lines students should write. Each gap already has its prompt: a markdown instruction plus a commented verbal description or hints above the code. The solution lines stay in place for now, wrapped in markers:

```python
# EXERCISE 2b: <instructions and hints>
# --- SOLUTION START (remove for exercise version) ---
...solution...
# --- SOLUTION END ---
```

Markdown solutions use `<!-- SOLUTION START ... -->` / `<!-- SOLUTION END -->`. Every gap is followed by an `assert`-style sanity-check cell where feasible. Each notebook opens with a goal/connection cell and ends with 3–4 notebook-specific "Questions to think about".

### Status (2026-09-16)

- All four exercises are reviewed and split into student/solution pairs, each committed and re-executed top to bottom: `exercise_0_mle.ipynb` / `_solution.ipynb`, `exercise_1_abc.ipynb` / `_solution.ipynb`, `exercise_2_mdn.ipynb` / `_solution.ipynb`, `exercise_3_sbi.ipynb` / `_solution.ipynb`. Format: the `_solution.ipynb` file keeps the full worked solution; the plain-named file strips each SOLUTION block down to variable-name hints (e.g. `mu_hat = ...`, `posterior = ...`), with bare `...`/inline comments for calls that return nothing (e.g. `...  # optimizer.step()`). The two files are otherwise byte-identical — diff them cell-by-cell after any edit to confirm only the intended gaps differ. There is no build script; **edit both notebooks directly** and keep them in sync by hand.
- **Next:** open-ended — further review/polish as the user requests it, or move on to the deferred ideas below.

### Open review points (raised, not yet decided)

1. Ex 3 budget comparison: with the current seed/RNG, NPE now has lower TV distance than ABC at all three budgets (500/2k/10k), including a wide margin at N=500. (Previously, before the 2026-09-16 RNG redesign, ABC and NPE were about equal at 2k/10k.) Still worth watching if this flips with future edits — the 0.5-wide TV bins can't resolve the sharp modes well.
2. Ex 2 MDN occasionally produces a narrow spike (a collapsed component, e.g. θ≈6.5 at `x_obs`, θ≈1.2 at x=10). Keep it as a discussion point, or add a lower bound on `log_sigma`.
3. ~~Ex 3 SBC rank plot (`sbc_rank_plot(..., plot_type="hist")`) is hard to read.~~ Resolved 2026-09-16: switched to `plot_type="cdf"` — the histogram version overlaid a semi-transparent uniform-reference band on the bars in a way that read as a solid block; the CDF version clearly shows the empirical CDF tracking the uniform band.

### Deferred ideas (not in these four notebooks)

- Grid-posterior exercise (Poisson likelihood + Gaussian prior on firing rate, from `drafts/tutorial.ipynb`): a good idea, kept for later.
