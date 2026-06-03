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
│       ├── pipeline.py
│       └── steps/
├── data/
│   └── processed/
└── outputs/
    ├── figures/
    └── tables/
```

## Purpose

Splits the original credit copula notebook into sequential Python step scripts while preserving the original simulation logic and figures.

## Flow

1. `uv run credit-copula`, `scripts/run_pipeline.py`, or `notebooks/demo.ipynb` calls the package pipeline.
2. The pipeline builds a shared execution context with project paths and optional smoke-test overrides.
3. The step scripts in `src/credit_copula/steps/` execute in notebook order.
4. Outputs are written under `outputs/`, while `docs/reference/` holds the original notebook, the notebook copy-out, and the split map.

## Main Assumptions

- The generated step scripts should stay close to the notebook code instead of being deeply refactored.
- Notebook state is preserved through one shared execution context.
- Any data bundled in `data/processed/` is local to this project copy and does not mutate the original `one-time-projects` files.
