# Credit Risk Portfolio Analysis: Copula vs Gaussian Dependence

Compares a matched Gaussian loss approximation with a Gaussian copula that preserves discrete BBB rating migrations. The repository keeps the original notebook analysis in sequential step scripts and adds exact two-name integration to audit latent correlation, default thresholds, loss quantiles, and Monte Carlo convergence.

The original notebook is preserved unchanged in `docs/reference/credit-copula.ipynb`. The new execution notebook in `notebooks/demo.ipynb` only calls the Python backend under `src/`.

## Layout

- `docs/reference/`: original notebook, pasted notebook content, and split map
- `docs/reference/notebook_reference.md`: full notebook content copied into Markdown
- `src/credit_copula/steps/`: notebook-derived Python section scripts
- `src/credit_copula/model.py`: deterministic two-name Gaussian-copula checks
- `tests/test_model.py`: hand checks for moments, thresholds, marginals, dependence, and tail quantiles
- `scripts/run_pipeline.py`: thin CLI wrapper
- `notebooks/demo.ipynb`: thin notebook wrapper
- `data/processed/`: local project inputs
- `outputs/`: generated figures and tables
- `docs/reference/notebook_split.md`: section-to-script map

## Run

```bash
uv sync
uv run credit-copula
uv run credit-copula --smoke
uv run python scripts/run_pipeline.py --smoke
uv run pytest -q
MPLBACKEND=Agg uv run python blog/generate_charts.py
```

The chart command regenerates the project-local bilingual blog evidence under `blog/data/` and `blog/images/`. It does not publish to the website.

## Dependence convention

The input `correlation = 0.5` is the correlation of latent Gaussian asset variables. It is not a 0.5 correlation between observed defaults or dollar losses. Rating states are ordered from AAA to Default, so default lies above the upper threshold $\Phi^{-1}(1-p_D)$, where $p_D$ is probability of default. Reversing every latent sign gives the equivalent lower-tail convention used in some credit models.
