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
```

This installs everything needed for both the interactive apps and the exercise notebooks below (Python 3.12, numpy, matplotlib, PyTorch, `sbi`, JupyterLab). You don't strictly need `streamlit` to work through the exercises, but it comes with the same environment, so there's no separate setup for it.

Run an interactive app:

```bash
streamlit run 1-rej_abc_app.py
```

## Exercises: Göttingen SBI summer-school tutorial

A self-contained, hands-on introduction to Bayesian basics, rejection ABC, conditional density estimation, and modern SBI. After the setup above:

```bash
jupyter lab exercises/
```

Work through the notebooks in order: `exercise_0_mle.ipynb`, `exercise_1_abc.ipynb`, `exercise_2_mdn.ipynb`, `exercise_3_sbi.ipynb`. Each has marked `EXERCISE` gaps for you to fill in, with a sanity-check cell after most of them so you know if your answer is on the right track. If you get stuck, the matching `_solution.ipynb` file (e.g. `exercise_0_mle_solution.ipynb`) has the full worked solution.

## Online apps
The `streamlit` interactive apps are also hosted online at Streamlit Cloud:
- https://road2sbi-1-abc.streamlit.app/
- https://road2sbi-3-density-est.streamlit.app/
- https://road2sbi-4-cde.streamlit.app/ 