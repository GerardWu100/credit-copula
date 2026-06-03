"""Notebook section: visualizations."""

figures_dir = FIGURES_DIR

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Loss distribution histogram
ax1 = axes[0, 0]
loss_min = min(credit_losses_mvn.min(), credit_losses_copula.min())
loss_max = max(credit_losses_mvn.max(), credit_losses_copula.max())
bins = np.linspace(loss_min, loss_max, 150)
ax1.hist(credit_losses_mvn, bins=bins, alpha=0.5, label='Method A', color='steelblue', density=True)
ax1.hist(credit_losses_copula, bins=bins, alpha=0.5, label='Method B', color='coral', density=True)
ax1.axvline(expected_loss_mvn, color='blue', linestyle='--', linewidth=2, label=f'A: Mean=${expected_loss_mvn:.2f}')
ax1.axvline(expected_loss_copula, color='red', linestyle='--', linewidth=2, label=f'B: Mean=${expected_loss_copula:.2f}')
ax1.set_xlabel('Credit Loss ($)', fontsize=11)
ax1.set_ylabel('Probability Density', fontsize=11)
ax1.set_title('Loss Distribution', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. VaR levels comparison
ax2 = axes[0, 1]
var_percentiles = [90, 95, 97.5, 99, 99.5, 99.9]
var_mvn = [np.percentile(credit_losses_mvn, p) for p in var_percentiles]
var_copula = [np.percentile(credit_losses_copula, p) for p in var_percentiles]
x_pos = np.arange(len(var_percentiles))
bar_width = 0.35
ax2.bar(x_pos - bar_width / 2, var_mvn, bar_width, label='Method A', color='steelblue', alpha=0.8)
ax2.bar(x_pos + bar_width / 2, var_copula, bar_width, label='Method B', color='coral', alpha=0.8)
ax2.set_xlabel('Confidence Level', fontsize=11)
ax2.set_ylabel('Value-at-Risk ($)', fontsize=11)
ax2.set_title('VaR at Different Confidence Levels', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'{p}%' for p in var_percentiles])
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# 3. Tail risk (losses above the 90th percentile of Method A)
ax3 = axes[0, 2]
tail_cutoff = np.percentile(credit_losses_mvn, 90)
tail_mvn = credit_losses_mvn[credit_losses_mvn >= tail_cutoff]
tail_copula = credit_losses_copula[credit_losses_copula >= tail_cutoff]
bins_tail = np.linspace(tail_cutoff, max(tail_mvn.max(), tail_copula.max()), 100)
ax3.hist(tail_mvn, bins=bins_tail, alpha=0.6, label='Method A', color='steelblue', density=True)
ax3.hist(tail_copula, bins=bins_tail, alpha=0.6, label='Method B', color='coral', density=True)
ax3.axvline(var_99_mvn, color='blue', linestyle='--', linewidth=2, label=f'A: 99% VaR=${var_99_mvn:.2f}')
ax3.axvline(var_99_copula, color='red', linestyle='--', linewidth=2, label=f'B: 99% VaR=${var_99_copula:.2f}')
ax3.set_xlabel('Credit Loss ($)', fontsize=11)
ax3.set_ylabel('Probability Density', fontsize=11)
ax3.set_title('Tail Risk (>90th Percentile)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Joint rating distribution for the copula method (discrete outcomes, not Gaussian losses)
ax4 = axes[1, 0]
n_rating_states = len(ratings)
flat_joint_idx = rating_idx1_copula * n_rating_states + rating_idx2_copula
joint_counts = np.bincount(flat_joint_idx, minlength=n_rating_states ** 2)
joint_freq = joint_counts.reshape(n_rating_states, n_rating_states) / n_simulations * 100

im = ax4.imshow(joint_freq, cmap='YlOrRd', aspect='auto')
ax4.set_xlabel('Bond 2 Rating', fontsize=11)
ax4.set_ylabel('Bond 1 Rating', fontsize=11)
ax4.set_title('Joint Rating Distribution (Copula)', fontsize=12, fontweight='bold')
ax4.set_xticks(range(n_rating_states))
ax4.set_xticklabels(ratings, rotation=45, fontsize=8)
ax4.set_yticks(range(n_rating_states))
ax4.set_yticklabels(ratings, fontsize=8)
cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
cbar.set_label('Probability (%)', fontsize=9)

# 5. Exceedance probability (empirical survival function on sorted losses)
ax5 = axes[1, 1]
sorted_mvn = np.sort(credit_losses_mvn)
sorted_copula = np.sort(credit_losses_copula)
rank_fraction = np.arange(1, len(sorted_mvn) + 1) / len(sorted_mvn)
exceedance_mvn = 1 - rank_fraction
exceedance_copula = 1 - rank_fraction
ax5.plot(sorted_mvn, exceedance_mvn, label='Method A', color='steelblue', linewidth=2)
ax5.plot(sorted_copula, exceedance_copula, label='Method B', color='coral', linewidth=2)
ax5.axhline(0.01, color='black', linestyle=':', linewidth=1, label='99% level')
ax5.axvline(var_99_mvn, color='blue', linestyle='--', alpha=0.5)
ax5.axvline(var_99_copula, color='red', linestyle='--', alpha=0.5)
ax5.set_xlabel('Credit Loss ($)', fontsize=11)
ax5.set_ylabel('Exceedance Probability', fontsize=11)
ax5.set_title('Loss Exceedance Curve', fontsize=12, fontweight='bold')
ax5.set_yscale('log')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# 6. Q-Q plot of portfolio loss quantiles
ax6 = axes[1, 2]
quantile_grid = np.linspace(0, 1, 1000)
q_mvn = np.percentile(credit_losses_mvn, quantile_grid * 100)
q_copula = np.percentile(credit_losses_copula, quantile_grid * 100)
ax6.scatter(q_mvn, q_copula, alpha=0.5, s=10, color='purple')
ax6.plot([q_mvn.min(), q_mvn.max()], [q_mvn.min(), q_mvn.max()], 'r--', linewidth=2, label='Perfect match')
ax6.set_xlabel('Method A Quantiles ($)', fontsize=11)
ax6.set_ylabel('Method B Quantiles ($)', fontsize=11)
ax6.set_title('Q-Q Plot: Method A vs Method B', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / '001_credit_risk_comparison_overview.png', dpi=300, bbox_inches='tight')
print('Saved: 001_credit_risk_comparison_overview.png')
plt.show()
