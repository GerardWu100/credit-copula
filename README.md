# Credit Risk Portfolio Analysis: Copula vs Gaussian Dependence

Splits the original credit copula notebook into sequential Python step scripts while preserving the original simulation logic and figures.

The original notebook is preserved unchanged in `docs/reference/credit-copula.ipynb`. The new execution notebook in `notebooks/demo.ipynb` only calls the Python backend under `src/`.

## Layout

- `docs/reference/`: original notebook, pasted notebook content, and split map
- `docs/reference/notebook_reference.md`: full notebook content copied into Markdown
- `src/credit_copula/steps/`: notebook-derived Python section scripts
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
```
