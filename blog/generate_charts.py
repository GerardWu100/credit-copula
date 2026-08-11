"""Reproduce the blog's credit-loss estimates and publication charts.

The script is self-contained by design. It freezes every reported simulation
summary under ``blog/data`` and writes only the two analytical figures used by
the bilingual article under ``blog/images``.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from credit_copula.model import exact_two_name_metrics
from scipy import stats
from scipy.stats import norm

BLOG_DIR = Path(__file__).resolve().parent
DATA_DIR = BLOG_DIR / "data"
IMAGES_DIR = BLOG_DIR / "images"

RATINGS = np.array(["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "Default"])
PROBABILITIES = np.array([0.02, 0.33, 5.95, 86.93, 5.30, 1.17, 0.12, 0.18]) / 100
STATE_VALUES = np.array([109.37, 109.19, 108.66, 107.55, 102.02, 98.10, 83.64, 51.13])
INITIAL_BOND_VALUE = 107.55
LATENT_CORRELATION = 0.5
TWO_BOND_SIMULATIONS = 1_000_000
SCALING_SIMULATIONS = 200_000
BOND_COUNTS = np.array([1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100])
TWO_BOND_SEED = 111_111
SCALING_SEED = 42
CHUNK_SIZE = 20_000
VAR_LEVEL = 99
FIGURE_DPI = 240
CONVERGENCE_SAMPLE_SIZES = np.array([10_000, 100_000, 1_000_000])


def marginal_loss_moments() -> tuple[np.ndarray, float, float]:
    """Calculate exact losses, mean loss, and standard deviation for one bond.

    Returns
    -------
    state_losses : numpy.ndarray
        One-year loss in dollars for each rating state, in ``RATINGS`` order.
    mean_loss : float
        Probability-weighted one-year loss in dollars.
    standard_deviation : float
        Probability-weighted loss standard deviation in dollars.
    """
    state_losses = INITIAL_BOND_VALUE - STATE_VALUES
    mean_loss = float(PROBABILITIES @ state_losses)
    variance = float(PROBABILITIES @ (state_losses - mean_loss) ** 2)
    return state_losses, mean_loss, float(np.sqrt(variance))


def simulate_two_bond_losses() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate two-bond losses under the matched Gaussian and copula methods.

    Returns
    -------
    gaussian_portfolio : numpy.ndarray
        Shape ``(TWO_BOND_SIMULATIONS,)``; continuous portfolio loss in dollars.
    copula_portfolio : numpy.ndarray
        Shape ``(TWO_BOND_SIMULATIONS,)``; discrete portfolio loss in dollars.
    gaussian_names : numpy.ndarray
        Shape ``(TWO_BOND_SIMULATIONS, 2)``; matched-Gaussian name losses.
    copula_names : numpy.ndarray
        Shape ``(TWO_BOND_SIMULATIONS, 2)``; rating-state name losses.
    """
    state_losses, mean_loss, standard_deviation = marginal_loss_moments()
    correlation_matrix = np.array(
        [[1.0, LATENT_CORRELATION], [LATENT_CORRELATION, 1.0]]
    )
    cholesky = np.linalg.cholesky(correlation_matrix)
    rng = np.random.default_rng(TWO_BOND_SEED)

    # Independent draws are transformed into correlated latent normals first.
    gaussian_latent = rng.standard_normal((TWO_BOND_SIMULATIONS, 2)) @ cholesky.T
    gaussian_names = mean_loss + standard_deviation * gaussian_latent

    # A separate draw keeps the two Monte Carlo estimators independent, matching
    # the source pipeline's use of new random numbers for the copula method.
    copula_latent = rng.standard_normal((TWO_BOND_SIMULATIONS, 2)) @ cholesky.T
    uniforms = norm.cdf(copula_latent)
    rating_indices = np.searchsorted(np.cumsum(PROBABILITIES), uniforms, side="right")
    copula_names = state_losses[rating_indices]

    return (
        gaussian_names.sum(axis=1),
        copula_names.sum(axis=1),
        gaussian_names,
        copula_names,
    )


def summarize_two_bond_results(
    gaussian_portfolio: np.ndarray,
    copula_portfolio: np.ndarray,
    gaussian_names: np.ndarray,
    copula_names: np.ndarray,
) -> pd.DataFrame:
    """Build and save the two-bond metrics used in the article.

    Parameters
    ----------
    gaussian_portfolio, copula_portfolio : numpy.ndarray
        Portfolio loss arrays in dollars, each with one row per simulation.
    gaussian_names, copula_names : numpy.ndarray
        Per-name loss matrices in dollars with two columns.

    Returns
    -------
    pandas.DataFrame
        One row per method and columns for location, tail, and dependence metrics.
    """
    rows: list[dict[str, float | str]] = []
    for method, portfolio, names in (
        ("Matched Gaussian loss", gaussian_portfolio, gaussian_names),
        ("Gaussian copula", copula_portfolio, copula_names),
    ):
        rows.append(
            {
                "method": method,
                "mean_loss_dollars": float(np.mean(portfolio)),
                "median_loss_dollars": float(np.median(portfolio)),
                "standard_deviation_dollars": float(np.std(portfolio)),
                "skewness": float(stats.skew(portfolio)),
                "excess_kurtosis": float(stats.kurtosis(portfolio)),
                "var_95_dollars": float(np.percentile(portfolio, 95)),
                "var_99_dollars": float(np.percentile(portfolio, 99)),
                "var_99_9_dollars": float(np.percentile(portfolio, 99.9)),
                "pearson_loss_correlation": float(np.corrcoef(names.T)[0, 1]),
                "spearman_loss_correlation": float(
                    stats.spearmanr(names[:, 0], names[:, 1]).statistic
                ),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(DATA_DIR / "two_bond_metrics.csv", index=False, float_format="%.8f")
    return results


def save_exact_two_bond_benchmark() -> pd.DataFrame:
    """Integrate and save the exact two-name Gaussian-copula benchmark.

    Returns
    -------
    pandas.DataFrame
        One-row table containing exact loss, quantile, and default metrics.
    """
    metrics = exact_two_name_metrics(
        probabilities=PROBABILITIES,
        state_values=STATE_VALUES,
        initial_value=INITIAL_BOND_VALUE,
        latent_correlation=LATENT_CORRELATION,
    )
    benchmark = pd.DataFrame([asdict(metrics)])
    benchmark.to_csv(
        DATA_DIR / "exact_two_bond_benchmark.csv",
        index=False,
        float_format="%.10f",
    )
    return benchmark


def save_copula_convergence_check(
    copula_portfolio: np.ndarray,
    copula_names: np.ndarray,
) -> pd.DataFrame:
    """Save prefix-sample diagnostics for the rare-event Monte Carlo estimates.

    Parameters
    ----------
    copula_portfolio : numpy.ndarray
        Shape ``(TWO_BOND_SIMULATIONS,)``; discrete portfolio loss in dollars.
    copula_names : numpy.ndarray
        Shape ``(TWO_BOND_SIMULATIONS, 2)``; rating-state name losses.

    Returns
    -------
    pandas.DataFrame
        Sample size, mean, standard deviation, tail quantiles, and name-loss
        correlation for nested simulation prefixes.
    """
    rows: list[dict[str, float | int]] = []
    for sample_size in CONVERGENCE_SAMPLE_SIZES:
        portfolio_prefix = copula_portfolio[:sample_size]
        name_prefix = copula_names[:sample_size]
        rows.append(
            {
                "simulations": int(sample_size),
                "mean_loss_dollars": float(np.mean(portfolio_prefix)),
                "standard_deviation_dollars": float(np.std(portfolio_prefix)),
                "var_99_dollars": float(np.percentile(portfolio_prefix, 99)),
                "var_99_9_dollars": float(np.percentile(portfolio_prefix, 99.9)),
                "pearson_loss_correlation": float(np.corrcoef(name_prefix.T)[0, 1]),
            }
        )

    convergence = pd.DataFrame(rows)
    convergence.to_csv(
        DATA_DIR / "copula_convergence.csv",
        index=False,
        float_format="%.8f",
    )
    return convergence


def simulate_scaling_results(
    mean_loss: float, standard_deviation: float
) -> pd.DataFrame:
    """Estimate 99% portfolio VaR for homogeneous portfolios of several sizes.

    Parameters
    ----------
    mean_loss : float
        Exact one-name expected loss in dollars.
    standard_deviation : float
        Exact one-name loss standard deviation in dollars.

    Returns
    -------
    pandas.DataFrame
        Bond count, absolute 99% VaR, and per-bond 99% VaR for both methods.

    Notes
    -----
    The equicorrelated latent variable is generated as
    ``sqrt(rho) * common_factor + sqrt(1-rho) * idiosyncratic_factor``.
    Chunking bounds memory without changing the Gaussian factor model.
    """
    state_losses, _, _ = marginal_loss_moments()
    cumulative_probabilities = np.cumsum(PROBABILITIES)
    rows: list[dict[str, float | int]] = []

    for n_bonds in BOND_COUNTS:
        copula_losses = np.empty(SCALING_SIMULATIONS)
        # A size-specific stream makes each estimate reproducible even if the
        # BOND_COUNTS grid is reordered or expanded later.
        rng = np.random.default_rng(SCALING_SEED + int(n_bonds))

        for start in range(0, SCALING_SIMULATIONS, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, SCALING_SIMULATIONS)
            n_rows = stop - start

            common = rng.standard_normal((n_rows, 1))
            idiosyncratic = rng.standard_normal((n_rows, int(n_bonds)))
            latent = (
                np.sqrt(LATENT_CORRELATION) * common
                + np.sqrt(1.0 - LATENT_CORRELATION) * idiosyncratic
            )
            rating_indices = np.searchsorted(
                cumulative_probabilities,
                norm.cdf(latent),
                side="right",
            )
            copula_losses[start:stop] = state_losses[rating_indices].sum(axis=1)

        # The matched-loss model is Gaussian, so its portfolio quantile is
        # analytic. This removes needless Monte Carlo noise from its curve.
        portfolio_variance_multiplier = n_bonds + LATENT_CORRELATION * n_bonds * (
            n_bonds - 1
        )
        gaussian_var = float(
            n_bonds * mean_loss
            + norm.ppf(VAR_LEVEL / 100)
            * standard_deviation
            * np.sqrt(portfolio_variance_multiplier)
        )
        copula_var = float(np.percentile(copula_losses, VAR_LEVEL))
        rows.append(
            {
                "number_of_bonds": int(n_bonds),
                "gaussian_var_99_dollars": gaussian_var,
                "copula_var_99_dollars": copula_var,
                "gaussian_var_99_per_bond_dollars": gaussian_var / n_bonds,
                "copula_var_99_per_bond_dollars": copula_var / n_bonds,
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(DATA_DIR / "portfolio_scaling.csv", index=False, float_format="%.8f")
    return results


def plot_loss_tail(
    gaussian_portfolio: np.ndarray,
    copula_portfolio: np.ndarray,
) -> None:
    """Plot the central distribution and upper quantiles for two-bond loss.

    Parameters
    ----------
    gaussian_portfolio, copula_portfolio : numpy.ndarray
        Simulated portfolio losses in dollars for the two methods.
    """
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    bins = np.linspace(-15, 65, 130)
    axes[0].hist(
        gaussian_portfolio,
        bins=bins,
        density=True,
        alpha=0.72,
        color="#4C78A8",
        label="Matched Gaussian loss",
    )
    axes[0].hist(
        copula_portfolio,
        bins=bins,
        density=True,
        alpha=0.68,
        color="#F5855B",
        label="Gaussian copula",
    )
    axes[0].set_title("Two-bond portfolio loss distribution")
    axes[0].set_xlabel("One-year portfolio loss (dollars)")
    axes[0].set_ylabel("Probability density")
    axes[0].legend(frameon=False)

    confidence_levels = np.array([90, 95, 97.5, 99, 99.5, 99.9])
    axes[1].plot(
        confidence_levels,
        np.percentile(gaussian_portfolio, confidence_levels),
        "o-",
        color="#4C78A8",
        linewidth=2.2,
        label="Matched Gaussian loss",
    )
    axes[1].plot(
        confidence_levels,
        np.percentile(copula_portfolio, confidence_levels),
        "o-",
        color="#E45756",
        linewidth=2.2,
        label="Gaussian copula",
    )
    axes[1].set_title("The gap opens in the far tail")
    axes[1].set_xlabel("Confidence level (percent)")
    axes[1].set_ylabel("Value-at-Risk (dollars)")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.grid(alpha=0.24)

    figure.savefig(IMAGES_DIR / "01_loss_tail.png", dpi=FIGURE_DPI)
    plt.close(figure)


def plot_portfolio_scaling(results: pd.DataFrame) -> None:
    """Plot absolute and per-name 99% VaR as portfolio size increases.

    Parameters
    ----------
    results : pandas.DataFrame
        Output from :func:`simulate_scaling_results`.
    """
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    x = results["number_of_bonds"]

    for label, prefix, color in (
        ("Matched Gaussian loss", "gaussian", "#4C78A8"),
        ("Gaussian copula", "copula", "#E45756"),
    ):
        axes[0].plot(
            x,
            results[f"{prefix}_var_99_dollars"],
            "o-",
            color=color,
            linewidth=2.2,
            label=label,
        )
        axes[1].plot(
            x,
            results[f"{prefix}_var_99_per_bond_dollars"],
            "o-",
            color=color,
            linewidth=2.2,
            label=label,
        )

    axes[0].set_title("Absolute 99% VaR grows with portfolio size")
    axes[0].set_xlabel("Number of BBB bonds")
    axes[0].set_ylabel("99% Value-at-Risk (dollars)")
    axes[1].set_title("Per-bond VaR falls unevenly under discrete losses")
    axes[1].set_xlabel("Number of BBB bonds")
    axes[1].set_ylabel("99% Value-at-Risk per bond (dollars)")

    for axis in axes:
        axis.grid(alpha=0.24)
        axis.legend(frameon=False)

    figure.savefig(IMAGES_DIR / "02_portfolio_scaling.png", dpi=FIGURE_DPI)
    plt.close(figure)


def main() -> None:
    """Run the deterministic analysis, freeze its outputs, and create charts."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    state_losses, mean_loss, standard_deviation = marginal_loss_moments()
    pd.DataFrame(
        {
            "rating": RATINGS,
            "probability": PROBABILITIES,
            "end_value_dollars": STATE_VALUES,
            "loss_dollars": state_losses,
        }
    ).to_csv(DATA_DIR / "rating_states.csv", index=False, float_format="%.8f")

    gaussian_portfolio, copula_portfolio, gaussian_names, copula_names = (
        simulate_two_bond_losses()
    )
    summarize_two_bond_results(
        gaussian_portfolio,
        copula_portfolio,
        gaussian_names,
        copula_names,
    )
    save_exact_two_bond_benchmark()
    save_copula_convergence_check(copula_portfolio, copula_names)
    scaling_results = simulate_scaling_results(mean_loss, standard_deviation)
    plot_loss_tail(gaussian_portfolio, copula_portfolio)
    plot_portfolio_scaling(scaling_results)


if __name__ == "__main__":
    main()
