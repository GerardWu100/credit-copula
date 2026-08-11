"""Notebook section: method b gaussian copula on rating thresholds."""

print("Method B: Gaussian Copula with Rating-Based Marginals")

# Same latent correlation as Method A; reuse Cholesky factor L from the prior step.
z_independent = np.random.randn(n_simulations, 2)
z_correlated = z_independent @ L.T

# Gaussian copula: correlated normals -> uniforms -> discrete ratings via inverse CDF.
uniforms = norm.cdf(z_correlated)
rating_idx1_copula = np.searchsorted(cumulative_probs, uniforms[:, 0], side="right")
rating_idx2_copula = np.searchsorted(cumulative_probs, uniforms[:, 1], side="right")

bond1_values_copula = values[rating_idx1_copula]
bond2_values_copula = values[rating_idx2_copula]

losses_copula = np.column_stack(
    (
        initial_bond_value - bond1_values_copula,
        initial_bond_value - bond2_values_copula,
    )
)
credit_losses_copula = losses_copula.sum(axis=1)
portfolio_values_copula = initial_portfolio_value - credit_losses_copula
