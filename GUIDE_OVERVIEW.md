# Project Overview

## File Tree

```text
credit-copula/
├── GUIDE_OVERVIEW.md
├── GUIDE_ROOT.md
├── README.md
├── docs/
│   └── reference/
│       ├── GUIDE_reference.md
│       ├── credit-copula.ipynb
│       ├── notebook_reference.md
│       └── notebook_split.md
├── logs/
├── notebooks/
│   ├── GUIDE_notebooks.md
│   └── demo.ipynb
├── scripts/
│   ├── GUIDE_scripts.md
│   └── run_pipeline.py
├── src/
│   ├── GUIDE_src.md
│   └── credit_copula/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── model.py
│       ├── pipeline.py
│       └── steps/
├── tests/
│   └── test_model.py
├── blog/
│   ├── data/
│   ├── images/
│   ├── generate_charts.py
│   ├── index.md
│   └── index.fr.md
├── data/
│   └── processed/
└── outputs/
    ├── figures/
    └── tables/
```

## Purpose

Compares continuous matched-moment losses with discrete rating-migration losses under Gaussian dependence. The original notebook workflow remains executable, while exact two-name integration verifies the threshold map, attainable loss quantiles, and the difference between latent asset correlation and realized default or loss correlation.

## Flow

1. `uv run credit-copula`, `scripts/run_pipeline.py`, or `notebooks/demo.ipynb` calls the package pipeline.
2. The pipeline builds a shared execution context with project paths and optional smoke-test overrides.
3. The step scripts in `src/credit_copula/steps/` execute in notebook order.
4. Outputs are written under `outputs/`, while `docs/reference/` holds the original notebook, the notebook copy-out, and the split map.
5. The analytic model integrates each two-name rating-state cell from a conditional normal distribution, producing a deterministic benchmark for Monte Carlo tests and blog evidence.
6. The project-local blog script freezes exact, simulated, convergence, and portfolio-scaling tables before regenerating its analytical figures.

## Main Assumptions

- The generated step scripts should stay close to the notebook code instead of being deeply refactored.
- Notebook state is preserved through one shared execution context.
- The `0.5` input is latent Gaussian asset correlation. Thresholding converts it into smaller, nonlinear default and dollar-loss correlations.
- Default is the final rating state and therefore occupies the upper latent tail. A global sign reversal produces the equivalent lower-tail convention.
- Portfolio scaling assumes homogeneous BBB names, one common factor, constant pairwise latent correlation, deterministic rating-state values, and a one-year horizon.
- Any data bundled in `data/processed/` is local to this project copy and does not mutate the original `one-time-projects` files.
