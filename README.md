# Credit Copula

Compares two ways of modelling one-year credit loss on BBB bonds under rating migration: a Gaussian approximation matched to the loss mean and variance, and a Gaussian copula applied to the discrete rating-migration table. It exists to show where a smooth normal-loss model and a copula that respects discrete rating states agree and where they diverge, especially in the tail.

## What it does

- Takes a one-year BBB rating-migration table (probability of ending in each rating, AAA through Default) and a latent asset correlation between two bonds.
- Method A: fits a multivariate Gaussian directly to portfolio losses (matches mean and variance only).
- Method B: a Gaussian copula — correlated latent normal variables are thresholded into the discrete rating states, so migration probabilities are preserved exactly.
- Adds an exact two-name benchmark (`src/credit_copula/model.py`): deterministic numerical integration of the two-name joint distribution, used to check Monte Carlo loss quantiles, default correlation, and convergence against a closed-form answer instead of another simulation.
- The original notebook analysis is preserved unchanged at `docs/reference/credit-copula.ipynb`; the executable version is split into ordered step scripts under `src/credit_copula/steps/` that run in the same sequence, sharing one namespace, as the original notebook cells.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- No external services or API keys are required.

## Setup

```bash
uv sync
```

## Usage

```bash
uv run credit-copula                        # run the full notebook-derived pipeline
uv run credit-copula --smoke                 # same pipeline with a smaller Monte Carlo sample, for a quick check
uv run credit-copula --print-keys            # also print the final execution-context keys
uv run python scripts/run_pipeline.py        # thin script wrapper, equivalent to `credit-copula`
uv run pytest -q                             # run the hand-checked tests in tests/test_model.py
MPLBACKEND=Agg uv run python blog/generate_charts.py   # regenerate blog evidence tables and figures under blog/
```

`blog/generate_charts.py` only regenerates the project-local bilingual blog evidence in `blog/data/` and `blog/images/`; it does not publish to the website.

## Dependence convention

The input `correlation = 0.5` is the correlation between the two bonds' *latent* Gaussian asset variables, not the correlation between their observed defaults or dollar losses — thresholding turns 0.5 latent correlation into much smaller default and loss correlations. Rating states are ordered AAA to Default, so default sits above the upper threshold $\Phi^{-1}(1-p_D)$, where $p_D$ is the probability of default and $\Phi^{-1}$ is the inverse standard normal CDF. Reversing every latent sign gives the equivalent lower-tail convention used in some credit models. See `docs/reference/GUIDE_reference.md` and the frozen derivation in `blog/index.md` for the full walk-through.

## Layout

```text
src/credit_copula/         package: cli.py, config.py, pipeline.py, model.py, steps/
scripts/run_pipeline.py    thin CLI wrapper
notebooks/demo.ipynb       thin notebook wrapper around the same pipeline
docs/reference/            original notebook, its Markdown copy-out, and the section-to-script map
tests/test_model.py        hand checks for moments, thresholds, marginals, dependence, tail quantiles
data/processed/            local project inputs
outputs/                   generated figures and tables
blog/                      bilingual write-up, chart script, and frozen evidence tables (project-local, not a deploy bundle)
```

## Output

Running the pipeline writes:

- `outputs/figures/` — PNG figures (loss distribution, VaR comparison, tail risk, VaR vs. portfolio size).
- `outputs/tables/` — generated summary tables.

`blog/generate_charts.py` separately writes frozen evidence tables to `blog/data/` and figures to `blog/images/` for the write-up in `blog/index.md` / `blog/index.fr.md`.

## License

All rights reserved. See [LICENSE](LICENSE).
