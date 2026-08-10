"""Notebook section: statistical summary."""

LOSS_METRICS = [
    "Mean Loss",
    "Median Loss",
    "Std Dev",
    "Skewness",
    # scipy.stats.kurtosis defaults to fisher=True, which subtracts 3, so the
    # reported number is excess kurtosis: a normal distribution scores 0, not 3.
    "Excess Kurtosis",
    "95% VaR",
    "99% VaR",
    "99.9% VaR",
    "Min Loss",
    "Max Loss",
]


def portfolio_loss_profile(
    credit_losses: np.ndarray, expected_loss: float, var_99: float
) -> list[float]:
    """Return the standard loss distribution statistics for one simulation method."""
    return [
        expected_loss,
        np.median(credit_losses),
        np.std(credit_losses),
        stats.skew(credit_losses),
        stats.kurtosis(credit_losses),
        np.percentile(credit_losses, 95),
        var_99,
        np.percentile(credit_losses, 99.9),
        np.min(credit_losses),
        np.max(credit_losses),
    ]


stats_summary = pd.DataFrame(
    {
        "Metric": LOSS_METRICS,
        "Method A": portfolio_loss_profile(
            credit_losses_mvn, expected_loss_mvn, var_99_mvn
        ),
        "Method B": portfolio_loss_profile(
            credit_losses_copula, expected_loss_copula, var_99_copula
        ),
    }
)

stats_summary["Difference (B-A)"] = (
    stats_summary["Method B"] - stats_summary["Method A"]
)
stats_summary["% Diff"] = (
    stats_summary["Difference (B-A)"] / stats_summary["Method A"].abs()
) * 100

print(stats_summary.to_string(index=False))
