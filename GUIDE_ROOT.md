# GUIDE_ROOT.md

## Part 1: Conceptual Explanation

This repository is a notebook-to-project conversion. The notebook reference materials live under `docs/reference/`, and the executable workflow is split into ordered Python step scripts under `src/credit_copula/steps/`. The root folder keeps the project thin: `scripts/run_pipeline.py` and the `credit-copula` console entrypoint execute the notebook-derived pipeline, `pyproject.toml` defines the Python 3.13 environment, `notebooks/` holds the new thin execution notebook, `docs/` records both the section split and a direct notebook copy-out, and `data/` plus `outputs/` hold local inputs and generated artifacts.

The execution model intentionally mirrors notebook semantics. Each step script is executed in order inside one shared namespace, so variables, functions, and imported modules persist across sections just as they did in the original notebook. This keeps the code close to the source notebook while moving the reusable logic out of the new notebook wrapper.

## Part 2: Code Reference

- `scripts/run_pipeline.py`: Thin CLI wrapper around `credit_copula.cli.main()`.
- `pyproject.toml`: Python 3.13 package metadata, dependencies for `uv`, and the `credit-copula` console script.
- `src/credit_copula/cli.py`: Command-line entrypoint. Supports a `--smoke` mode where the project defines smaller verification overrides.
- `src/credit_copula/config.py`: Defines project paths and builds the shared execution context.
- `src/credit_copula/pipeline.py`: Runs each generated step script in notebook order.
- `src/credit_copula/steps/`: Contains the notebook-derived Python scripts, one file per major notebook section.
- `notebooks/demo.ipynb`: Thin notebook that only calls the backend pipeline.
- `docs/reference/credit-copula.ipynb`: Unchanged copy of the original source notebook.
- `docs/reference/notebook_reference.md`: Markdown copy-out of the source notebook in notebook order.
- `docs/reference/notebook_split.md`: Maps notebook sections to generated script files.

## Part 3: Short Journal

- 2026-04-16: Split the original notebook into ordered step scripts while preserving the raw notebook under `docs/reference/`.
- 2026-04-16: Added a direct notebook copy-out under `docs/reference/` so the source analysis is readable without opening the `.ipynb` file.
- 2026-05-20: Moved the CLI wrapper to `scripts/` and exposed `credit-copula` as a package console script.
