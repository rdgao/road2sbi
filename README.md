# road2sbi: From zero to hero (almost)

A sequence of demos and interactive apps that walk through the foundational concepts leading up to Simulation-Based Inference. It's not so much of a *guide to sbi*, but the prerequisite concepts to understanding and using `sbi` in a more informed way.

### Covered concepts (so far)
1. **Approximate Bayesian Computation:** rejection ABC, and why thresholding acceptance is inefficient.
2. **`sbi` Getting started:** start using `sbi` in 10 minutes.
3. **Density estimation (in 1D):** fundamental concept required to engage with probabilistic models.
4. **Conditional density estimation:** don't regress.
5. **Mixture density network:** OG conditional neural density estimator.

## Setup

```bash
conda env create -f environment_conda.yml
conda activate road2sbi
streamlit run 1-rej_abc_app.py
```

## Online apps
The `streamlit` interactive apps are also hosted online at Streamlit Cloud:
- https://road2sbi-1-abc.streamlit.app/
- https://road2sbi-3-density-est.streamlit.app/
- https://road2sbi-4-cde.streamlit.app/ 