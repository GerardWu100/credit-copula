"""Analytic checks for the homogeneous Gaussian credit-copula model.

The notebook pipeline remains the executable research artifact.  This module
collects the small, reusable calculations needed to test its probability
mapping without relying on Monte Carlo output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.stats import norm


PROBABILITY_TOLERANCE = 1e-12
QUADRATURE_ABSOLUTE_TOLERANCE = 1e-13
QUADRATURE_RELATIVE_TOLERANCE = 1e-11
QUADRATURE_LIMIT = 200


@dataclass(frozen=True, slots=True)
class TwoNameMetrics:
    """Exact loss and dependence metrics for two homogeneous names.

    Attributes
    ----------
    mean_loss_dollars : float
        Expected two-name portfolio loss in dollars.
    standard_deviation_dollars : float
        Standard deviation of two-name portfolio loss in dollars.
    var_95_dollars, var_99_dollars, var_99_9_dollars : float
        Left-continuous loss quantiles in dollars.
    pearson_loss_correlation : float
        Linear correlation between the two discrete name losses.
    joint_default_probability : float
        Probability that both names default over the horizon.
    conditional_default_probability : float
        Probability that name 2 defaults conditional on name 1 defaulting.
    default_indicator_correlation : float
        Pearson correlation between the two binary default indicators.
    default_threshold : float
        Upper latent-normal threshold above which default occurs.
    """

    mean_loss_dollars: float
    standard_deviation_dollars: float
    var_95_dollars: float
    var_99_dollars: float
    var_99_9_dollars: float
    pearson_loss_correlation: float
    joint_default_probability: float
    conditional_default_probability: float
    default_indicator_correlation: float
    default_threshold: float


def validate_rating_inputs(
    probabilities: NDArray[np.float64],
    state_values: NDArray[np.float64],
) -> None:
    """Validate a one-name rating distribution and its state values.

    Parameters
    ----------
    probabilities : numpy.ndarray
        One-dimensional state probabilities in ascending cumulative order.
    state_values : numpy.ndarray
        One-dimensional end-of-horizon dollar values in the same order.

    Raises
    ------
    ValueError
        If shapes differ, probabilities are invalid, or their sum is not one.
    """
    if probabilities.ndim != 1 or state_values.ndim != 1:
        raise ValueError("probabilities and state_values must be one-dimensional")
    if probabilities.shape != state_values.shape or probabilities.size < 2:
        raise ValueError(
            "probabilities and state_values must have equal non-trivial shapes"
        )
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
        raise ValueError("probabilities must be finite and strictly positive")
    if not np.isclose(probabilities.sum(), 1.0, atol=PROBABILITY_TOLERANCE):
        raise ValueError("probabilities must sum to one")
    if not np.all(np.isfinite(state_values)):
        raise ValueError("state_values must be finite")


def marginal_loss_moments(
    probabilities: NDArray[np.float64],
    state_values: NDArray[np.float64],
    initial_value: float,
) -> tuple[NDArray[np.float64], float, float]:
    """Calculate exact one-name state losses and their first two moments.

    Parameters
    ----------
    probabilities : numpy.ndarray
        State probabilities, with shape ``(K,)`` for ``K`` rating states.
    state_values : numpy.ndarray
        End-of-horizon dollar values, with shape ``(K,)``.
    initial_value : float
        Initial bond value in dollars.

    Returns
    -------
    state_losses : numpy.ndarray
        Dollar loss in each state; positive values are economic losses.
    mean_loss : float
        Probability-weighted loss in dollars.
    standard_deviation : float
        Loss standard deviation in dollars.
    """
    validate_rating_inputs(probabilities, state_values)
    if not np.isfinite(initial_value):
        raise ValueError("initial_value must be finite")

    state_losses = initial_value - state_values
    mean_loss = float(probabilities @ state_losses)
    variance = float(probabilities @ (state_losses - mean_loss) ** 2)
    return state_losses, mean_loss, float(np.sqrt(variance))


def latent_rating_edges(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert cumulative rating probabilities to latent-normal interval edges.

    Parameters
    ----------
    probabilities : numpy.ndarray
        State probabilities in the inverse-transform sampling order.

    Returns
    -------
    numpy.ndarray
        ``K + 1`` interval edges. The first is negative infinity and the last
        is positive infinity. Default is the final, upper-tail state.
    """
    placeholder_values = np.zeros_like(probabilities)
    validate_rating_inputs(probabilities, placeholder_values)
    interior_edges = norm.ppf(np.cumsum(probabilities)[:-1])
    return np.concatenate(([-np.inf], interior_edges, [np.inf]))


def bivariate_state_probabilities(
    probabilities: NDArray[np.float64],
    latent_correlation: float,
) -> NDArray[np.float64]:
    """Integrate the exact two-name Gaussian-copula state probabilities.

    Parameters
    ----------
    probabilities : numpy.ndarray
        One-name state probabilities in cumulative-threshold order.
    latent_correlation : float
        Correlation of the two latent standard normal variables. This is an
        asset-model parameter, not observed loss or default correlation.

    Returns
    -------
    numpy.ndarray
        Joint state probability matrix with shape ``(K, K)``. Rows index name
        1 states and columns index name 2 states.

    Notes
    -----
    Conditional on ``Z1 = z``, ``Z2`` is normal with mean ``rho * z`` and
    variance ``1 - rho**2``. One-dimensional quadrature over each ``Z1``
    interval avoids Monte Carlo error in the two-name benchmark.
    """
    if not -1.0 < latent_correlation < 1.0:
        raise ValueError("latent_correlation must lie strictly between -1 and 1")

    edges = latent_rating_edges(probabilities)
    conditional_scale = np.sqrt(1.0 - latent_correlation**2)
    state_count = probabilities.size
    joint = np.empty((state_count, state_count), dtype=float)

    for first_state, (first_lower, first_upper) in enumerate(
        zip(edges[:-1], edges[1:], strict=True)
    ):
        for second_state, (second_lower, second_upper) in enumerate(
            zip(edges[:-1], edges[1:], strict=True)
        ):

            def integrand(first_latent: float) -> float:
                """Return the joint density integrated over name 2's interval."""
                upper_z = (
                    second_upper - latent_correlation * first_latent
                ) / conditional_scale
                lower_z = (
                    second_lower - latent_correlation * first_latent
                ) / conditional_scale
                conditional_probability = norm.cdf(upper_z) - norm.cdf(lower_z)
                return float(norm.pdf(first_latent) * conditional_probability)

            joint[first_state, second_state] = quad(
                integrand,
                first_lower,
                first_upper,
                epsabs=QUADRATURE_ABSOLUTE_TOLERANCE,
                epsrel=QUADRATURE_RELATIVE_TOLERANCE,
                limit=QUADRATURE_LIMIT,
            )[0]

    return joint


def weighted_quantile(
    values: NDArray[np.float64],
    probabilities: NDArray[np.float64],
    confidence_level: float,
) -> float:
    """Return the left-continuous quantile of a finite probability distribution.

    Parameters
    ----------
    values : numpy.ndarray
        Possible scalar outcomes.
    probabilities : numpy.ndarray
        Probability mass attached to each outcome.
    confidence_level : float
        Cumulative probability in the open interval ``(0, 1)``.

    Returns
    -------
    float
        Smallest outcome whose cumulative probability reaches the confidence
        level.
    """
    if values.ndim != 1 or values.shape != probabilities.shape:
        raise ValueError("values and probabilities must be equal-length vectors")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")

    order = np.argsort(values)
    ordered_values = values[order]
    ordered_probabilities = probabilities[order]
    unique_values, inverse = np.unique(ordered_values, return_inverse=True)
    unique_probabilities = np.bincount(inverse, weights=ordered_probabilities)
    cumulative_probability = np.cumsum(unique_probabilities)
    quantile_index = np.searchsorted(cumulative_probability, confidence_level)
    return float(unique_values[quantile_index])


def exact_two_name_metrics(
    probabilities: NDArray[np.float64],
    state_values: NDArray[np.float64],
    initial_value: float,
    latent_correlation: float,
) -> TwoNameMetrics:
    """Calculate exact two-name loss, quantile, and default metrics.

    Parameters
    ----------
    probabilities : numpy.ndarray
        One-name rating probabilities; the final state is default.
    state_values : numpy.ndarray
        End-of-horizon dollar values in rating-state order.
    initial_value : float
        Initial value of each homogeneous bond in dollars.
    latent_correlation : float
        Pairwise correlation of the latent Gaussian asset variables.

    Returns
    -------
    TwoNameMetrics
        Deterministic benchmark for the two-name Gaussian copula.
    """
    state_losses, one_name_mean, one_name_standard_deviation = marginal_loss_moments(
        probabilities,
        state_values,
        initial_value,
    )
    joint = bivariate_state_probabilities(probabilities, latent_correlation)
    portfolio_losses = state_losses[:, None] + state_losses[None, :]
    mean_loss = float(np.sum(joint * portfolio_losses))
    variance = float(np.sum(joint * (portfolio_losses - mean_loss) ** 2))
    cross_moment = float(np.sum(joint * state_losses[:, None] * state_losses[None, :]))
    loss_correlation = (
        cross_moment - one_name_mean**2
    ) / one_name_standard_deviation**2

    default_probability = float(probabilities[-1])
    joint_default_probability = float(joint[-1, -1])
    conditional_default_probability = joint_default_probability / default_probability
    default_indicator_variance = default_probability * (1.0 - default_probability)
    default_indicator_correlation = (
        joint_default_probability - default_probability**2
    ) / default_indicator_variance
    default_threshold = float(latent_rating_edges(probabilities)[-2])

    flattened_losses = portfolio_losses.ravel()
    flattened_probabilities = joint.ravel()
    return TwoNameMetrics(
        mean_loss_dollars=mean_loss,
        standard_deviation_dollars=float(np.sqrt(variance)),
        var_95_dollars=weighted_quantile(
            flattened_losses, flattened_probabilities, 0.95
        ),
        var_99_dollars=weighted_quantile(
            flattened_losses, flattened_probabilities, 0.99
        ),
        var_99_9_dollars=weighted_quantile(
            flattened_losses, flattened_probabilities, 0.999
        ),
        pearson_loss_correlation=float(loss_correlation),
        joint_default_probability=joint_default_probability,
        conditional_default_probability=conditional_default_probability,
        default_indicator_correlation=float(default_indicator_correlation),
        default_threshold=default_threshold,
    )
