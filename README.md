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
