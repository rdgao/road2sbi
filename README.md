# road2sbi: From zero to hero (almost)

A sequence of demos and interactive apps that walk through the foundational concepts leading up to Simulation-Based Inference. It's not so much of a *guide to sbi*, but the prerequisite concepts to understanding and using `sbi` in a more informed way.

### Covered concepts (so far)
1. **Approximate Bayesian Computation:** rejection ABC, and why thresholding acceptance is inefficient.
2. **`sbi` Getting started:** start using `sbi` in 10 minutes.
3. **Density estimation (in 1D):** fundamental concept required to engage with probabilistic models.
4. **Conditional density estimation:** don't regress.
5. **Mixture density network:** OG conditional neural density estimator.

## Fast setup (conda + uv, Python 3.12)

**Option 1: Using environment.yml (easiest, but slower)**
```bash
conda env create -f environment.yml
conda activate road2sbi
streamlit run 1-rej_abc_app.py
```

**Option 2: Manual conda + uv (uv go vrooom)**
```bash
conda create -n road2sbi python=3.12 -y   # minimal, just Python
conda activate road2sbi
python -m pip install -U uv               # bootstrap uv once
uv pip install --system -r requirements.txt
streamlit run 1-rej_abc_app.py
```