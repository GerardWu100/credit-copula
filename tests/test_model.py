"""Hand-checks and deterministic tests for the Gaussian copula model."""

from __future__ import annotations

import numpy as np
import pytest
from credit_copula.model import (
    bivariate_state_probabilities,
    exact_two_name_metrics,
    latent_rating_edges,
    marginal_loss_moments,
)
from scipy.stats import norm

PROBABILITIES = np.array([0.02, 0.33, 5.95, 86.93, 5.30, 1.17, 0.12, 0.18]) / 100
STATE_VALUES = np.array([109.37, 109.19, 108.66, 107.55, 102.02, 98.10, 83.64, 51.13])
INITIAL_VALUE = 107.55
LATENT_CORRELATION = 0.5


def test_marginal_moments_match_manual_weighted_calculation() -> None:
    """The helper must reproduce a direct probability-weighted hand check."""
    state_losses, mean_loss, standard_deviation = marginal_loss_moments(
        PROBABILITIES,
        STATE_VALUES,
        INITIAL_VALUE,
    )

    assert state_losses[0] == pytest.approx(-1.82)
    assert state_losses[-1] == pytest.approx(56.42)
    assert mean_loss == pytest.approx(0.462082)
    assert standard_deviation == pytest.approx(2.99178383)


def test_zero_latent_correlation_produces_independent_states() -> None:
    """At zero latent correlation, every joint cell must factor into marginals."""
    joint = bivariate_state_probabilities(PROBABILITIES, latent_correlation=0.0)

    np.testing.assert_allclose(
        joint, np.outer(PROBABILITIES, PROBABILITIES), atol=1e-12
    )


def test_positive_latent_correlation_preserves_marginals() -> None:
    """The copula may change joint probabilities but must preserve each marginal."""
    joint = bivariate_state_probabilities(PROBABILITIES, LATENT_CORRELATION)

    np.testing.assert_allclose(joint.sum(axis=0), PROBABILITIES, atol=1e-12)
    np.testing.assert_allclose(joint.sum(axis=1), PROBABILITIES, atol=1e-12)
    assert joint.sum() == pytest.approx(1.0)


def test_default_uses_the_upper_tail_with_the_expected_probability() -> None:
    """The final inverse-CDF state must be above ``Phi^-1(1 - PD)``."""
    edges = latent_rating_edges(PROBABILITIES)
    default_probability = PROBABILITIES[-1]

    assert edges[-2] == pytest.approx(norm.ppf(1.0 - default_probability))
    assert 1.0 - norm.cdf(edges[-2]) == pytest.approx(default_probability)


def test_exact_two_name_metrics_match_attainable_loss_levels() -> None:
    """Exact integration must verify the article's tail quantiles and dependence."""
    metrics = exact_two_name_metrics(
        PROBABILITIES,
        STATE_VALUES,
        INITIAL_VALUE,
        LATENT_CORRELATION,
    )

    assert metrics.mean_loss_dollars == pytest.approx(0.924164, abs=1e-10)
    assert metrics.var_95_dollars == pytest.approx(5.53)
    assert metrics.var_99_dollars == pytest.approx(14.98)
    assert metrics.var_99_9_dollars == pytest.approx(61.95)
    assert metrics.pearson_loss_correlation == pytest.approx(0.2013969, abs=1e-7)
    assert metrics.conditional_default_probability == pytest.approx(
        0.06754213, abs=1e-8
    )
    assert metrics.default_indicator_correlation == pytest.approx(0.06586068, abs=1e-8)


def test_independent_defaults_have_zero_indicator_correlation() -> None:
    """With zero latent correlation, conditional default probability equals PD."""
    metrics = exact_two_name_metrics(
        PROBABILITIES,
        STATE_VALUES,
        INITIAL_VALUE,
        latent_correlation=0.0,
    )

    assert metrics.conditional_default_probability == pytest.approx(PROBABILITIES[-1])
    assert metrics.default_indicator_correlation == pytest.approx(0.0, abs=1e-12)


def test_matched_gaussian_portfolio_variance_scales_with_equicorrelation() -> None:
    """The analytic variance must include every ordered covariance term once."""
    _, _, one_name_standard_deviation = marginal_loss_moments(
        PROBABILITIES,
        STATE_VALUES,
        INITIAL_VALUE,
    )
    name_count = 2
    variance_multiplier = name_count + LATENT_CORRELATION * name_count * (
        name_count - 1
    )
    portfolio_standard_deviation = one_name_standard_deviation * np.sqrt(
        variance_multiplier
    )

    # For two names and rho=0.5: Var(L1 + L2) = sigma^2 * (2 + 2*rho)
    assert variance_multiplier == pytest.approx(3.0)
    assert portfolio_standard_deviation == pytest.approx(5.18192161, abs=1e-8)
