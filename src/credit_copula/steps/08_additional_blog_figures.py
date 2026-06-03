"""Notebook section: additional blog figures."""

# Standalone versions of selected panels from the overview figure (blog export sizes).

loss_min = min(credit_losses_mvn.min(), credit_losses_copula.min())
loss_max = max(credit_losses_mvn.max(), credit_losses_copula.max())
var_percentiles = [90, 95, 97.5, 99, 99.5, 99.9]

# Figure 3: loss distribution comparison
fig, ax = plt.subplots(figsize=(10, 6))
bins = np.linspace(loss_min, loss_max, 100)
ax.hist(
    credit_losses_mvn,
    bins=bins,
    alpha=0.6,
    label='Multivariate Gaussian',
    color='steelblue',
    density=True,
    edgecolor='darkblue',
    linewidth=0.5,
)
ax.hist(
    credit_losses_copula,
    bins=bins,
    alpha=0.6,
    label='Gaussian Copula',
    color='coral',
    density=True,
    edgecolor='darkred',
    linewidth=0.5,
)
ax.axvline(expected_loss_mvn, color='blue', linestyle='--', linewidth=2.5, label=f'MVN Mean: ${expected_loss_mvn:.2f}')
ax.axvline(expected_loss_copula, color='red', linestyle='--', linewidth=2.5, label=f'Copula Mean: ${expected_loss_copula:.2f}')
ax.set_xlabel('Portfolio Credit Loss ($)', fontsize=13, fontweight='bold')
ax.set_ylabel('Probability Density', fontsize=13, fontweight='bold')
ax.set_title('Portfolio Loss Distribution: MVN vs Gaussian Copula', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig(figures_dir / '003_loss_distribution.png', dpi=300, bbox_inches='tight')
print('Saved: 003_loss_distribution.png')
plt.show()

# Figure 4: VaR comparison across confidence levels
fig, ax = plt.subplots(figsize=(10, 6))
var_mvn_plot = [np.percentile(credit_losses_mvn, p) for p in var_percentiles]
var_copula_plot = [np.percentile(credit_losses_copula, p) for p in var_percentiles]
x_pos = np.arange(len(var_percentiles))
bar_width = 0.35

bars1 = ax.bar(
    x_pos - bar_width / 2,
    var_mvn_plot,
    bar_width,
    label='Multivariate Gaussian',
    color='steelblue',
    alpha=0.8,
    edgecolor='darkblue',
    linewidth=1.5,
)
bars2 = ax.bar(
    x_pos + bar_width / 2,
    var_copula_plot,
    bar_width,
    label='Gaussian Copula',
    color='coral',
    alpha=0.8,
    edgecolor='darkred',
    linewidth=1.5,
)

for bars in (bars1, bars2):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'${height:.2f}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
        )

ax.set_xlabel('Confidence Level', fontsize=13, fontweight='bold')
ax.set_ylabel('Value-at-Risk ($)', fontsize=13, fontweight='bold')
ax.set_title('Value-at-Risk Comparison Across Confidence Levels', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{p}%' for p in var_percentiles], fontsize=11)
ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig(figures_dir / '004_var_comparison.png', dpi=300, bbox_inches='tight')
print('Saved: 004_var_comparison.png')
plt.show()

# Figure 5: tail risk focus
fig, ax = plt.subplots(figsize=(10, 6))
tail_cutoff = np.percentile(credit_losses_mvn, 90)
tail_mvn = credit_losses_mvn[credit_losses_mvn >= tail_cutoff]
tail_copula = credit_losses_copula[credit_losses_copula >= tail_cutoff]
bins_tail = np.linspace(tail_cutoff, max(tail_mvn.max(), tail_copula.max()), 80)

ax.hist(tail_mvn, bins=bins_tail, alpha=0.6, label='Multivariate Gaussian', color='steelblue', density=True, edgecolor='darkblue', linewidth=0.5)
ax.hist(tail_copula, bins=bins_tail, alpha=0.6, label='Gaussian Copula', color='coral', density=True, edgecolor='darkred', linewidth=0.5)
ax.axvline(var_99_mvn, color='blue', linestyle='--', linewidth=2.5, label=f'MVN 99% VaR: ${var_99_mvn:.2f}')
ax.axvline(var_99_copula, color='red', linestyle='--', linewidth=2.5, label=f'Copula 99% VaR: ${var_99_copula:.2f}')
ax.set_xlabel('Credit Loss ($)', fontsize=13, fontweight='bold')
ax.set_ylabel('Probability Density', fontsize=13, fontweight='bold')
ax.set_title('Tail Risk Distribution (>90th Percentile)', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig(figures_dir / '005_tail_risk.png', dpi=300, bbox_inches='tight')
print('Saved: 005_tail_risk.png')
plt.show()
