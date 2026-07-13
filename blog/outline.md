# Adaptive outline: Credit losses are discrete

## Archetype decision

- **Problem:** Compare two dependence constructions for the one-year loss of a BBB bond portfolio and explain why matching first and second moments does not preserve tail risk.
- **Options considered:** `risk-model`, `pricing-model`, and `mixed`.
- **Choice:** `risk-model`.
- **Why:** The project starts from rating-migration probabilities, simulates correlated portfolio losses, reports Value-at-Risk (VaR), and studies the loss tail and portfolio-size scaling. It does not price a traded contract.
- **Verify during drafting:** The results must show similar expected loss but materially different high loss quantiles, while preserving the distinction between latent Gaussian correlation and realized loss correlation.

## Section blueprint

1. **A mean can survive a bad model**
   - Two identical BBB bonds, one-year horizon, eight migration states.
   - The practical question: what is lost when a discrete migration distribution is replaced by a normal loss?
2. **The migration table is the marginal model**
   - Transition probabilities, state values, and state losses.
   - Exact single-name mean and variance.
3. **Two meanings of “Gaussian”**
   - Method A: Gaussian approximation in loss space.
   - Method B: Gaussian copula in latent space with discrete inverse-transform sampling.
   - One-factor representation, threshold sign convention, and conditional default probability.
4. **Same centre, different tail**
   - Deterministic Monte Carlo evidence for two bonds.
   - Exact 64-cell integration and nested-prefix convergence check.
   - Explain the zero median, skewness, bounded state support, and quantile jumps.
5. **Dependence does not survive the threshold map unchanged**
   - Contrast the input latent correlation with empirical Pearson and Spearman correlations of realized losses.
6. **What changes as the portfolio grows**
   - 99% VaR and VaR per bond for 1 to 100 equicorrelated names.
7. **What this model can and cannot say**
   - Homogeneous names, static migration matrix, fixed asset correlation, no recovery randomness, Gaussian tail dependence, Monte Carlo sampling error.

## Planned equations

1. State loss: $\ell_k = V_0 - V_k$, where $V_0$ is the initial BBB value and $V_k$ is the value in rating state $k$.
2. Exact marginal moments: $\mu = \sum_k p_k\ell_k$ and $\sigma^2 = \sum_k p_k(\ell_k-\mu)^2$, where $p_k$ is the probability of state $k$.
3. Equicorrelation matrix: $R_{ij}=1$ for $i=j$ and $R_{ij}=\rho$ otherwise; correlated latent variables satisfy $Z=L\varepsilon$ and $LL^\mathsf{T}=R$.
4. One-factor representation: $Z_i=\sqrt{\rho}M+\sqrt{1-\rho}\varepsilon_i$.
5. Gaussian approximation: $\widetilde{\ell}_i=\mu+\sigma Z_i$.
6. Copula map: $U_i=\Phi(Z_i)$ and $K_i=\min\{k:U_i\le C_k\}$, where $C_k=\sum_{j\le k}p_j$.
7. Default threshold and conditional default probability for $D_i=\mathbf{1}\{Z_i>a\}$.
8. Portfolio loss: $L_N=\sum_{i=1}^N\ell_{K_i}$.
9. Matched Gaussian portfolio variance: $\sigma^2[N+\rho N(N-1)]$.
10. Quantile definition: $\operatorname{VaR}_q(L_N)=\inf\{x:\Pr(L_N\le x)\ge q\}$.

## Planned code excerpts

- The five-line Gaussian-copula transformation from correlated normals to rating indices.
- The matched-moment Gaussian loss construction, placed beside the copula excerpt for comparison.

## Planned graphs

1. **Cover image:** An editorial, text-free visualization of a smooth blue Gaussian surface breaking into discrete coral credit-rating tiles, with a darker clustered tail. It should communicate smooth approximation versus discrete migration without pretending to be data.
2. **Loss quantiles and exceedance:** Empirical quantile curves or survival functions for the two-bond portfolio. Takeaway: the Gaussian curve is smooth, while the copula distribution has discrete jumps and a much heavier far tail.
3. **VaR by portfolio size:** Absolute 99% VaR and 99% VaR per bond from 1 to 100 names. Takeaway: diversification lowers per-name risk, but the Gaussian approximation remains below the discrete copula result at the reported sizes.

## Evidence and assumptions

- Use the repository's transition probabilities and state values without external calibration claims.
- Freeze blog-specific results from deterministic simulation seeds under `blog/data/`.
- Report Monte Carlo estimates as estimates, not exact population values.
- Integrate the two-name copula exactly to benchmark Monte Carlo loss and dependence estimates.
- Use the closed-form matched Gaussian quantile in the portfolio-size chart instead of simulating a known normal distribution.
- Do not interpret $\rho=0.5$ as a 0.5 correlation between realized discrete losses; the threshold map changes Pearson and rank correlation.
- Avoid calling the Gaussian copula generally “fat-tailed.” The heavy portfolio-loss tail here comes from the discrete and strongly skewed marginal loss states; the Gaussian copula itself has no asymptotic tail dependence.

## Workspace and publication decision

The canonical package is `credit-copula/blog/`. The normal publish target would be `~/projects/website/content/post/credit-losses-are-discrete/`, but the user explicitly postponed website publication. Therefore this task will not read, copy to, build, commit, or otherwise touch `~/projects/website`. Only the project-local package will be validated, committed, and pushed on the repository's current branch.
