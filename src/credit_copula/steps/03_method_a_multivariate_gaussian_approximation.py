"""Notebook section: method a multivariate gaussian approximation."""

n_simulations = OVERRIDES.get('n_simulations', 1_000_000)
correlation = 0.5

# Two identical BBB bonds: portfolio value is twice the par-equivalent BBB value.
initial_portfolio_value = 2 * initial_bond_value

# Loss if the bond lands in each rating state (positive means economic loss).
single_bond_losses = initial_bond_value - values
mean_loss_single = float(np.dot(probabilities, single_bond_losses))
var_loss_single = float(np.dot(probabilities, (single_bond_losses - mean_loss_single) ** 2))
std_loss_single = np.sqrt(var_loss_single)

# Equicorrelated Gaussian shocks for the two-bond portfolio.
correlation_matrix_mvn = np.array([[1.0, correlation], [correlation, 1.0]])
L = np.linalg.cholesky(correlation_matrix_mvn)

z_independent = np.random.randn(n_simulations, 2)
z_correlated = z_independent @ L.T

# Method A: approximate each bond loss as Gaussian with matched mean and variance.
losses_mvn = mean_loss_single + std_loss_single * z_correlated
credit_losses_mvn = losses_mvn.sum(axis=1)
portfolio_values_mvn = initial_portfolio_value - credit_losses_mvn

print(f'Cholesky factor L:\n{L}')
