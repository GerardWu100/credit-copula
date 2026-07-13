# GUIDE_src.md

## Part 1: Conceptual Explanation

The `src/credit_copula` package has two related layers. The execution layer preserves the original notebook logic: `config.py` defines paths, `pipeline.py` runs the ordered step files, `cli.py` exposes the command, and `steps/` contains the copied procedural analysis. The audit layer in `model.py` validates rating inputs and calculates exact two-name results without random sampling.

The `steps/` folder is intentionally procedural. Each file corresponds to a notebook section and expects the shared context created by the pipeline. That context includes filesystem paths, runtime overrides, and any variables created by earlier sections. The result is notebook parity without keeping the operational logic inside `.ipynb` files.

The exact calculation partitions each latent normal into rating intervals. For interval bounds $a_i<b_i$ and $a_j<b_j$, latent correlation $\rho$, and standard normal density $\phi$ and cumulative distribution $\Phi$, one joint cell is

$$
\Pr(a_i<Z_1\le b_i,\ a_j<Z_2\le b_j)
=\int_{a_i}^{b_i}\phi(z)
\left[
\Phi\left(\frac{b_j-\rho z}{\sqrt{1-\rho^2}}\right)
-\Phi\left(\frac{a_j-\rho z}{\sqrt{1-\rho^2}}\right)
\right]dz.
$$

The matrix of these cells must sum to one and reproduce the rating probabilities on both margins. Dollar losses then determine exact portfolio moments, Value-at-Risk quantiles, loss correlation, joint-default probability, and default-indicator correlation.

## Part 2: Code Reference

- `config.py`: Path configuration and context assembly for step execution.
- `pipeline.py`: Sequentially executes all files in `steps/` with one shared namespace.
- `model.py`: Provides `marginal_loss_moments()`, `latent_rating_edges()`, `bivariate_state_probabilities()`, `weighted_quantile()`, and `exact_two_name_metrics()` for deterministic verification.
- `cli.py`: Command-line entrypoint with `--smoke` and `--print-keys` flags.
- `steps/*.py`: Notebook-derived code files ordered by notebook section.

## Part 3: Short Journal

- 2026-04-16: Kept notebook semantics by executing ordered step scripts inside one shared context instead of rewriting the workflow into new abstractions.
- 2026-05-20: Added `cli.py` and moved the root script wrapper to `scripts/run_pipeline.py`.
- 2026-07-13: Kept the notebook pipeline intact and added a separate exact audit layer so tests can distinguish latent asset correlation from realized default and loss correlation.
