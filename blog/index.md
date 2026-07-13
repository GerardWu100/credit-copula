---
title: "Credit Losses Are Discrete: Gaussian Copula vs Gaussian Approximation"
description: "A two-bond experiment shows why matching average credit loss and variance can still miss the shape of migration risk and its far tail."
date: 2026-07-13
image: images/cover-credit-copula.png
categories: ["Risk Management", "Quantitative Research"]
---

A portfolio model can get expected loss right and still describe the wrong world. Credit migration makes that failure easy to see. Most bonds do nothing dramatic over one year. A few are downgraded, and a very small fraction default. Replacing those outcomes with a smooth normal distribution preserves two moments, but it erases the steps between them.

I tested that trade-off on a deliberately small portfolio: two identical BBB bonds, each worth \$107.55 today, with a one-year rating transition table and a common dependence parameter of 0.5. The comparison is between a Gaussian approximation applied directly to losses and a Gaussian copula applied before mapping each bond into a discrete rating state.

The result is narrower than a blanket verdict against normal models. Both methods estimate roughly \$0.92 of expected portfolio loss, yet their 99.9% Value-at-Risk (VaR) estimates are \$16.88 and \$61.95. Value-at-Risk at confidence level $q$ is the smallest loss threshold exceeded with probability no greater than $1-q$. The difference comes from the loss marginal, not from a claim that a Gaussian copula has fat-tailed dependence.

## The migration table is the marginal model

For each end-of-year rating state $k$, let $p_k$ be its transition probability, $V_k$ its bond value in dollars, and $V_0=\$107.55$ the initial BBB value. The state loss $\ell_k$ is

$$
\ell_k = V_0 - V_k.
$$

A negative loss is a gain. An upgrade from BBB to AAA, for example, produces $107.55-109.37=-\$1.82$. Default produces $107.55-51.13=\$56.42$.

| End-of-year rating | Probability $p_k$ | Value $V_k$ | Loss $\ell_k$ |
|:--|--:|--:|--:|
| AAA | 0.02% | \$109.37 | -\$1.82 |
| AA | 0.33% | \$109.19 | -\$1.64 |
| A | 5.95% | \$108.66 | -\$1.11 |
| BBB | 86.93% | \$107.55 | \$0.00 |
| BB | 5.30% | \$102.02 | \$5.53 |
| B | 1.17% | \$98.10 | \$9.45 |
| CCC | 0.12% | \$83.64 | \$23.91 |
| Default | 0.18% | \$51.13 | \$56.42 |

The exact one-name expected loss $\mu$ is the probability-weighted average of the state losses:

$$
\mu = \sum_{k=1}^{8} p_k\ell_k = \$0.462082.
$$

The exact variance $\sigma^2$ and standard deviation $\sigma$ follow in two steps:

$$
\sigma^2 = \sum_{k=1}^{8} p_k(\ell_k-\mu)^2,
$$

$$
\sigma = \sqrt{\sigma^2} = \$2.991784.
$$

Two identical names therefore have an expected portfolio loss of $2\mu=\$0.924164$, regardless of their dependence. Dependence changes the distribution around that mean.

## Two meanings of Gaussian

Both constructions start with latent standard normal variables. Let $N$ be the number of bonds, let $R$ be their $N\times N$ correlation matrix, and let $\rho=0.5$ be every off-diagonal entry. Thus, $R_{ii}=1$ and $R_{ij}=\rho$ when $i\ne j$. If $L$ is the lower-triangular Cholesky factor satisfying $LL^\mathsf{T}=R$, and $\varepsilon$ is a vector of independent standard normal draws, then

$$
Z=L\varepsilon
$$

has correlation matrix $R$. What happens to $Z$ is where the methods part company.

### Method A: smooth the loss itself

The matched Gaussian approximation turns each latent value $Z_i$ for bond $i$ directly into a continuous loss $\widetilde{\ell}_i$:

$$
\widetilde{\ell}_i = \mu + \sigma Z_i.
$$

The portfolio loss is the sum $\widetilde{L}_N=\sum_{i=1}^{N}\widetilde{\ell}_i$. This construction reproduces the exact one-name mean and variance. It also assigns positive probability to losses between migration states, losses beyond default, and large gains outside the state table. Those are mathematical consequences of the approximation, not possible rating outcomes in this setup.

```python
single_bond_losses = initial_bond_value - values
mean_loss_single = float(np.dot(probabilities, single_bond_losses))
var_loss_single = float(
    np.dot(probabilities, (single_bond_losses - mean_loss_single) ** 2)
)
losses_mvn = mean_loss_single + np.sqrt(var_loss_single) * z_correlated
credit_losses_mvn = losses_mvn.sum(axis=1)
```

### Method B: keep the rating steps

The Gaussian copula uses the same type of latent $Z_i$, but does not treat it as a loss. Let $\Phi$ be the standard normal cumulative distribution function. The transformation

$$
U_i=\Phi(Z_i)
$$

makes $U_i$ uniform on $(0,1)$. Next define the cumulative migration probability $C_k=\sum_{j=1}^{k}p_j$. The simulated state $K_i$ is the first state whose cumulative threshold contains $U_i$:

$$
K_i=\min\{k:U_i\le C_k\}.
$$

Bond $i$ then incurs the table loss $\ell_{K_i}$, and the portfolio loss is $L_N=\sum_{i=1}^{N}\ell_{K_i}$. The dependence lives in the latent Gaussian variables while every name keeps the eight-state marginal distribution.

```python
z_independent = np.random.randn(n_simulations, 2)
z_correlated = z_independent @ L.T
uniforms = norm.cdf(z_correlated)
rating_indices = np.searchsorted(cumulative_probs, uniforms, side="right")
bond_values = values[rating_indices]
credit_losses_copula = (initial_bond_value - bond_values).sum(axis=1)
```

This separation matters. A Gaussian copula has zero asymptotic tail dependence when $|\rho|<1$. The severe simulated losses below come from mapping correlated latent draws into a highly skewed credit-loss table, including a \$56.42 default loss, rather than from fat-tailed Gaussian dependence.

## The centre agrees, the tail does not

I ran 1,000,000 two-bond simulations for each method with a fixed random seed. The Monte Carlo means land within two-tenths of a cent of the exact \$0.924164 portfolio mean.

| Metric | Matched Gaussian loss | Gaussian copula |
|:--|--:|--:|
| Mean loss | \$0.9231 | \$0.9226 |
| Median loss | \$0.9323 | \$0.0000 |
| Standard deviation | \$5.1842 | \$4.6348 |
| Skewness | -0.0022 | 9.8819 |
| 95% VaR | \$9.4416 | \$5.5300 |
| 99% VaR | \$12.9736 | \$14.9800 |
| 99.9% VaR | \$16.8841 | \$61.9500 |

![Two-bond loss distribution and upper quantiles](images/01_loss_tail.png)

The left panel shows why the median copula loss is zero: an unchanged BBB outcome has 86.93% probability for each name, so a large mass sits exactly at zero. The Gaussian approximation spreads that mass across nearby gains and losses. At the 95% quantile this smoothing is conservative, with VaR of \$9.44 against \$5.53. Farther out, the ordering reverses. The discrete model reaches downgrade and default combinations that the thin Gaussian loss tail assigns too little weight.

The jump from \$14.98 at 99% to \$61.95 at 99.9% is not a numerical glitch. A discrete distribution's quantile can stay flat and then jump when the requested confidence level crosses a state combination. This is also why a single quantile gives an incomplete picture of credit risk.

## The input correlation is not the loss correlation

The parameter $\rho=0.5$ describes the correlation of the latent normals. In Method A, an affine transformation maps $Z_i$ to loss, so the simulated Pearson correlation between name losses remains 0.5002. Pearson correlation measures linear co-movement.

Thresholding changes that relationship in Method B. Its realized loss Pearson correlation is 0.2044, and its Spearman rank correlation is 0.2735. Spearman correlation measures whether two variables tend to move in the same rank order. Ties are common because both names repeatedly land in BBB, so neither statistic should be expected to equal the latent input.

That gap is easy to misuse in calibration. Setting latent correlation to a historical correlation of observed losses does not generally reproduce that same observed number after the rating thresholds. The parameter has to be calibrated through the model's measurement layer.

## Scaling from one name to one hundred

The second experiment uses the same homogeneous BBB table and equicorrelation assumption for portfolios from 1 to 100 bonds. Each point is estimated from 200,000 simulations. For size $N$, 99% VaR is

$$
\operatorname{VaR}_{0.99}(L_N)=\inf\{x:\Pr(L_N\le x)\ge 0.99\}.
$$

![99% VaR and VaR per bond by portfolio size](images/02_portfolio_scaling.png)

Absolute VaR grows with the portfolio. VaR per bond falls because idiosyncratic migration risk is diversified, but the common latent factor prevents it from falling toward zero. At 100 names, the estimates are \$541.28 for the matched Gaussian model and \$624.76 for the copula, or \$5.41 and \$6.25 per bond.

The copula line is uneven at small portfolio sizes. That is expected when VaR is taken from a finite set of attainable portfolio losses: adding one bond changes both the state combinations and the location of the 99th percentile. Smooth scaling rules can hide that granularity.

## Where I would not trust this experiment

This controlled comparison stops well short of a production credit portfolio model. Every bond is identical. The migration matrix is static, the horizon is fixed at one year, and the end-of-year value for each rating is deterministic. Recovery uncertainty, spread dynamics within rating buckets, issuer concentration, sector factors, and estimation error are absent.

The dependence assumption is also narrow. One equicorrelation parameter says every pair shares the same latent relationship. A Gaussian copula cannot produce non-zero asymptotic tail dependence unless correlation is perfect. Replacing it with a Student's $t$ copula could add joint tail dependence, but that would answer a different question and introduce a degrees-of-freedom parameter that needs evidence.

The useful conclusion survives those limits. Matching marginal mean and variance is not enough when loss arrives through rare, discrete state changes. Before choosing a smooth approximation, inspect the quantiles on both sides of the target confidence level, verify how latent dependence maps into observed loss dependence, and keep the economic support of the loss distribution in view.

## Reproducing the figures

The chart script under `blog/generate_charts.py` uses the transition table embedded in the project, fixed seeds, 1,000,000 two-bond draws, and 200,000 draws at each portfolio size. It writes the plotted estimates to `blog/data/` before generating the two figures, so every number in the tables can be traced to a frozen comma-separated values file.
