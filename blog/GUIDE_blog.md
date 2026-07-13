# GUIDE_blog.md

## Part 1: Conceptual Explanation

This folder is the canonical, project-local workspace for the bilingual credit-copula article. The English and French pages share a cover and two technical figures. The chart script reproduces the rating table, two-name simulation, exact two-name benchmark, convergence diagnostic, and portfolio-size experiment before writing any figure. Frozen comma-separated values files under `data/` make every published number inspectable without rerunning one million scenarios.

The two-name audit separates numerical error from model behavior. The Gaussian loss approximation has an analytic portfolio distribution. The copula has only 64 joint rating cells, so conditional-normal quadrature calculates its moments, attainable quantiles, loss correlation, and joint-default probability without Monte Carlo error. Nested simulation prefixes then show which empirical estimates converge quickly and which remain noisy because joint defaults are rare.

The size experiment uses a one-factor construction. A common normal factor creates constant latent asset correlation across homogeneous BBB names, while independent name shocks represent diversifiable migration risk. The Gaussian comparison is analytic. The discrete copula curve remains simulated because its portfolio loss is a sum of thresholded rating states.

The files in this folder are not copied to or built in the website repository during the current drafting stage.

## Part 2: Code Reference

- `generate_charts.py`: Recreates frozen data and the two analytical figures. Run it with `MPLBACKEND=Agg uv run python blog/generate_charts.py` from the project root.
- `index.md`: Canonical English Hugo article.
- `index.fr.md`: Canonical French Hugo article using the same math, code, numbers, links, and image paths.
- `outline.md`: Adaptive risk-model outline and publication-scope decision.
- `data/rating_states.csv`: Rating probabilities, end values, and one-name losses.
- `data/two_bond_metrics.csv`: Monte Carlo comparison of matched Gaussian and copula losses.
- `data/exact_two_bond_benchmark.csv`: Deterministic copula loss, quantile, and default metrics.
- `data/copula_convergence.csv`: Diagnostics from nested simulation prefixes.
- `data/portfolio_scaling.csv`: Analytic Gaussian and simulated copula 99% Value-at-Risk by name count.
- `images/01_loss_tail.png`: Two-name loss distribution and upper-quantile comparison.
- `images/02_portfolio_scaling.png`: Absolute and per-name 99% Value-at-Risk by portfolio size.
- `images/cover-credit-copula.png`: Text-free editorial cover illustrating a smooth distribution mapped to discrete states.

## Part 3: Short Journal

- 2026-07-13: Replaced simulated Gaussian portfolio-size quantiles with their closed form and added exact copula integration because fixed seeds do not remove rare-event sampling error.
