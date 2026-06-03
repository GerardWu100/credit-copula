"""Notebook section: portfolio size scaling analysis."""

SCALING_SIMULATIONS = 200_000
VAR_PERCENTILE = 99
BOND_COUNTS = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100]


def correlated_standard_normals(n_sims: int, n_bonds: int, correlation: float) -> np.ndarray:
    """Draw (n_sims, n_bonds) standard normals with constant pairwise correlation."""
    corr_matrix = np.full((n_bonds, n_bonds), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    cholesky = np.linalg.cholesky(corr_matrix)
    z_independent = np.random.randn(n_sims, n_bonds)
    return z_independent @ cholesky.T


def portfolio_var_99_copula(n_bonds: int, n_sims: int, correlation: float) -> float:
    """99th percentile portfolio credit loss under the Gaussian copula with rating marginals."""
    z_correlated = correlated_standard_normals(n_sims, n_bonds, correlation)
    uniforms = norm.cdf(z_correlated)
    rating_indices = np.searchsorted(cumulative_probs, uniforms, side='right')
    bond_values = values[rating_indices]
    per_bond_losses = initial_bond_value - bond_values
    portfolio_losses = per_bond_losses.sum(axis=1)
    return float(np.percentile(portfolio_losses, VAR_PERCENTILE))


def portfolio_var_99_mvn(n_bonds: int, n_sims: int, correlation: float) -> float:
    """99th percentile portfolio credit loss under the Gaussian loss approximation."""
    z_correlated = correlated_standard_normals(n_sims, n_bonds, correlation)
    per_bond_losses = mean_loss_single + std_loss_single * z_correlated
    portfolio_losses = per_bond_losses.sum(axis=1)
    return float(np.percentile(portfolio_losses, VAR_PERCENTILE))


print('Computing 99% VaR for different portfolio sizes...')
print('-' * 50)

np.random.seed(42)

var_99_copula_by_size: list[float] = []
var_99_mvn_by_size: list[float] = []

for n_bonds in BOND_COUNTS:
    var_copula = portfolio_var_99_copula(n_bonds, SCALING_SIMULATIONS, correlation)
    var_mvn = portfolio_var_99_mvn(n_bonds, SCALING_SIMULATIONS, correlation)
    var_99_copula_by_size.append(var_copula)
    var_99_mvn_by_size.append(var_mvn)
    print(f'n={n_bonds:3d} bonds | Copula 99% VaR: ${var_copula:10.2f} | MVN 99% VaR: ${var_mvn:10.2f}')

print('-' * 50)
print('Done!')

bond_counts = BOND_COUNTS
scaling_results = pd.DataFrame({
    'Number of Bonds': bond_counts,
    'Copula 99% VaR ($)': var_99_copula_by_size,
    'MVN 99% VaR ($)': var_99_mvn_by_size,
    'Copula VaR per Bond ($)': np.array(var_99_copula_by_size) / bond_counts,
    'MVN VaR per Bond ($)': np.array(var_99_mvn_by_size) / bond_counts,
})

print('\nPortfolio Size Scaling Results:')
print('=' * 80)
print(scaling_results.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
print('=' * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: absolute 99% VaR grows with portfolio size
ax1 = axes[0]
ax1.plot(
    bond_counts,
    var_99_copula_by_size,
    'o-',
    color='coral',
    linewidth=2.5,
    markersize=8,
    label='Gaussian Copula',
    markeredgecolor='darkred',
    markeredgewidth=1.5,
)
ax1.plot(
    bond_counts,
    var_99_mvn_by_size,
    's--',
    color='steelblue',
    linewidth=2.5,
    markersize=8,
    label='Multivariate Gaussian',
    markeredgecolor='darkblue',
    markeredgewidth=1.5,
)

var_one_bond_copula = var_99_copula_by_size[0]
sqrt_scaling = [var_one_bond_copula * np.sqrt(n) for n in bond_counts]
ax1.plot(bond_counts, sqrt_scaling, ':', color='gray', linewidth=2, label=r'$\sqrt{n}$ scaling (no correlation)')

ax1.set_xlabel('Number of Bonds in Portfolio', fontsize=13, fontweight='bold')
ax1.set_ylabel('99th Percentile Credit Loss ($)', fontsize=13, fontweight='bold')
ax1.set_title('99% VaR vs Portfolio Size', fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=11, loc='upper left', framealpha=0.95)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.tick_params(labelsize=11)
ax1.set_xlim(0, 105)

idx_10 = bond_counts.index(10)
ax1.annotate(
    f'10 bonds: ${var_99_copula_by_size[idx_10]:.0f}',
    xy=(10, var_99_copula_by_size[idx_10]),
    xytext=(20, var_99_copula_by_size[idx_10] + 50),
    fontsize=10,
    arrowprops=dict(arrowstyle='->', color='coral'),
)
ax1.annotate(
    f'100 bonds: ${var_99_copula_by_size[-1]:.0f}',
    xy=(100, var_99_copula_by_size[-1]),
    xytext=(75, var_99_copula_by_size[-1] - 200),
    fontsize=10,
    arrowprops=dict(arrowstyle='->', color='coral'),
)

# Right: VaR per bond shows diversification when correlation is imperfect
ax2 = axes[1]
var_per_bond_copula = scaling_results['Copula VaR per Bond ($)']
var_per_bond_mvn = scaling_results['MVN VaR per Bond ($)']

ax2.plot(
    bond_counts,
    var_per_bond_copula,
    'o-',
    color='coral',
    linewidth=2.5,
    markersize=8,
    label='Gaussian Copula',
    markeredgecolor='darkred',
    markeredgewidth=1.5,
)
ax2.plot(
    bond_counts,
    var_per_bond_mvn,
    's--',
    color='steelblue',
    linewidth=2.5,
    markersize=8,
    label='Multivariate Gaussian',
    markeredgecolor='darkblue',
    markeredgewidth=1.5,
)
ax2.axhline(
    var_per_bond_copula.iloc[0],
    color='coral',
    linestyle=':',
    alpha=0.5,
    label=f'Single bond VaR: ${var_per_bond_copula.iloc[0]:.2f}',
)

ax2.set_xlabel('Number of Bonds in Portfolio', fontsize=13, fontweight='bold')
ax2.set_ylabel('99% VaR per Bond ($)', fontsize=13, fontweight='bold')
ax2.set_title('Diversification Effect: VaR per Bond', fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=10, loc='upper right', framealpha=0.95)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.tick_params(labelsize=11)
ax2.set_xlim(0, 105)

plt.tight_layout()
plt.savefig(figures_dir / '006_var_vs_portfolio_size.png', dpi=300, bbox_inches='tight')
print('Saved: 006_var_vs_portfolio_size.png')
plt.show()
