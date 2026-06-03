"""Notebook section: results analysis."""

expected_loss_mvn = np.mean(credit_losses_mvn)
var_99_mvn = np.percentile(credit_losses_mvn, 99)

expected_loss_copula = np.mean(credit_losses_copula)
var_99_copula = np.percentile(credit_losses_copula, 99)

loss_corr_mvn = np.corrcoef(losses_mvn[:, 0], losses_mvn[:, 1])[0, 1]
loss_corr_copula = np.corrcoef(losses_copula[:, 0], losses_copula[:, 1])[0, 1]

# Spearman on a subsample keeps rank-correlation cost bounded for large n_simulations.
rng_diag = np.random.default_rng(20251025)
diagnostic_sample_size = min(n_simulations, 200_000)
diagnostic_idx = rng_diag.choice(n_simulations, size=diagnostic_sample_size, replace=False)
spearman_mvn = stats.spearmanr(
    losses_mvn[diagnostic_idx, 0],
    losses_mvn[diagnostic_idx, 1],
).correlation
spearman_copula = stats.spearmanr(
    losses_copula[diagnostic_idx, 0],
    losses_copula[diagnostic_idx, 1],
).correlation

results_df = pd.DataFrame({
    'Method': ['A: Multivariate Gaussian Approx.', 'B: Gaussian Copula'],
    'Expected Credit Loss ($)': [expected_loss_mvn, expected_loss_copula],
    '99% VaR ($)': [var_99_mvn, var_99_copula],
    'Linear Corr (loss1, loss2)': [loss_corr_mvn, loss_corr_copula],
    'Spearman Corr (loss1, loss2)': [spearman_mvn, spearman_copula],
})

print('\n' + '=' * 70)
print('CREDIT RISK ANALYSIS RESULTS')
print('=' * 70)
print(f'\nInitial Portfolio Value: ${initial_portfolio_value:.2f}')
print(f'Number of Simulations: {n_simulations:,}')
print(f'Latent Correlation Parameter: {correlation}')
print('\n' + results_df.to_string(index=False, float_format=lambda x: f'{x:,.4f}'))
print('=' * 70)
