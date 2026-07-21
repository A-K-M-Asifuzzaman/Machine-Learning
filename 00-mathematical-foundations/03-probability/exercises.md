# 00.03 — Exercises: Probability

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Derive $\mathrm{Var}(X) = \mathbb{E}[X^{2}] - (\mathbb{E}[X])^{2}$ from the definition.
Then explain why the right-hand form, though faster, is numerically dangerous, and what Welford's
algorithm does about it.

**D2.** Prove $\mathrm{Var}(X+Y) = \mathrm{Var}(X)+\mathrm{Var}(Y)+2\mathrm{Cov}(X,Y)$.
Then derive the ensemble variance formula

$$\mathrm{Var}\!\left(\tfrac1n\textstyle\sum_i f_i\right) = \frac{\sigma^{2}}{n} + \frac{n-1}{n}\rho\sigma^{2}$$

and state precisely what it implies about the value of adding more trees to a random forest.

**D3.** Construct a random variable pair with zero correlation and total dependence. Then prove
that for *jointly Gaussian* variables, zero correlation does imply independence. Where does the
Gaussian assumption enter?

**D4.** Derive Bayes' theorem from the definition of conditional probability, in one line.

**D5.** Redo the §7.1 disease calculation for prevalence $10^{-4}$ and a 99.9%-accurate test.
Then find the prevalence at which $P(D\mid+) = 0.5$ exactly, as a function of sensitivity and
specificity.

**D6.** Derive the odds form of Bayes and verify it reproduces §7.1's answer. Why is the odds form
easier to compute mentally?

**D7.** Show that a Gaussian prior on the weights yields $L_2$ regularization and a Laplace prior
yields $L_1$. Give $\lambda$ in terms of the prior's scale parameter in each case.

**D8.** Prove that $\mathrm{Beta}(\alpha,\beta)$ is conjugate to the Bernoulli likelihood: show
the posterior is $\mathrm{Beta}(\alpha+s, \beta+f)$. Then interpret $\alpha,\beta$ as pseudo-counts.

**D9.** Show the Gaussian, Bernoulli, and Poisson are all exponential-family members by writing
each in the form $h(x)\exp(\boldsymbol{\eta}^{\top}\mathbf{T}(x)-A(\boldsymbol{\eta}))$. Identify
$\boldsymbol{\eta}$, $\mathbf{T}$, and $A$ in each case.

**D10.** Prove $\nabla_{\boldsymbol{\eta}}A(\boldsymbol{\eta}) = \mathbb{E}[\mathbf{T}(x)]$ for
exponential families. Then verify it for the Bernoulli.

**D11.** Derive the conditional distribution formula for a partitioned multivariate Gaussian.
Then answer: why does the conditional *covariance* not depend on the observed value?

**D12.** Show that for jointly Gaussian $(X,Y)$, $\mathbb{E}[Y\mid X=x]$ is linear in $x$. Explain
what this says about when linear regression is not just convenient but *optimal*.

**D13.** Derive the change-of-variables formula in one dimension. Then explain why normalizing
flows are designed to have triangular Jacobians.

**D14.** Derive Chebyshev's inequality from Markov's. Then derive the sample size needed for a
95% confidence interval of width ±1% on a test error, using both Chebyshev and Hoeffding.
Compare the two answers.

**D15.** State Jensen's inequality and use it to prove $D_{\mathrm{KL}}(p\Vert q) \ge 0$.

**D16.** Derive the Box-Muller transform from the change-of-variables formula in polar
coordinates.

**D17.** *(Berry-Esseen in practice.)* The skewness of $\mathrm{Bernoulli}(p)$ is
$(1-2p)/\sqrt{p(1-p)}$. For $p = 0.001$, how large must $n$ be for the sample mean's skewness to
fall below 0.1? What does this say about Gaussian confidence intervals on rare-event rates?

---

## Tier 2 — Implementation

**I1.** Implement the Gaussian PDF, CDF, and log-PDF from their formulas. Verify against
`scipy.stats.norm`. Then show that `log(pdf(x))` and `logpdf(x)` diverge for $|z| > 40$ and
explain why.

**I2.** Implement Box-Muller. Verify with a Kolmogorov-Smirnov test, and confirm the two outputs
of each pair are uncorrelated.

**I3.** Implement inverse-CDF sampling for the exponential and Laplace distributions using their
closed-form inverses. Verify with KS tests.

**I4.** Implement rejection sampling for a Beta target with a uniform proposal. Measure the
acceptance rate and confirm it equals $1/M$. Then repeat in $d$ dimensions and plot the acceptance
rate against $d$ — this is why MCMC exists.

**I5.** Implement the Cholesky factorization from scratch. Use it to sample a multivariate
Gaussian, and verify the empirical covariance converges to $\boldsymbol{\Sigma}$.

**I6.** Implement `condition()` for a multivariate Gaussian. Verify the formula against an
empirical conditional obtained by filtering a large sample.

**I7.** Reproduce Experiment 1. Add a fourth source distribution with skewness above 10 (e.g.
$\mathrm{Bernoulli}(0.005)$) and find empirically the $n$ at which its sample mean passes a
normality test.

**I8.** Reproduce Experiment 2 and add Markov's inequality to the comparison. At which $t$ does
Markov become vacuous (bound $> 1$)?

**I9.** Implement Beta-Binomial conjugate updating. Plot the posterior after 0, 1, 5, 20, and 100
coin flips from a biased coin, starting from $\mathrm{Beta}(1,1)$ and from
$\mathrm{Beta}(50,50)$. Explain how the strong prior's influence decays.

**I10.** *(Loss ⇔ likelihood.)* Generate data with 5% outliers. Fit a constant by minimizing MSE,
MAE, and Huber loss. Separately, fit by maximizing Gaussian, Laplace, and Student-$t$ likelihood.
Verify each loss/likelihood pair gives the identical answer.

**I11.** *(Naive Bayes and the independence assumption.)* Implement Gaussian naive Bayes. Then
build a dataset with two strongly correlated informative features and measure how its accuracy
degrades relative to LDA (which models the full covariance) as the correlation rises.

**I12.** Implement the reparameterization trick: sample $z\sim\mathcal{N}(\mu,\sigma^{2})$ two
ways — directly, and as $\mu+\sigma\varepsilon$ — and show only the second admits a gradient
with respect to $\mu$ and $\sigma$.

---

## Tier 3 — Interview

**Q1.** What is the difference between a probability and a probability density?

**Q2.** A test for a disease affecting 1 in 1,000 people is 99% accurate both ways. You test
positive. What is the chance you have it? Explain the answer in counts, not formulas.

**Q3.** Your fraud model is 99.9% accurate. Is it good? What do you ask next?

**Q4.** Does zero correlation imply independence? Give a counterexample.

**Q5.** Why is MSE sensitive to outliers? Answer in terms of the noise model, not the algebra.

**Q6.** What distributional assumption are you making when you use cross-entropy loss?

**Q7.** What is the relationship between MAP estimation and regularization?

**Q8.** Explain the Central Limit Theorem. What are its two separate claims, and when does it
fail?

**Q9.** How much data do you need to halve a confidence interval? Why?

**Q10.** What does the naive Bayes independence assumption say, is it ever true, and why does the
classifier work anyway?

**Q11.** Why is the Gaussian so ubiquitous in ML? Give three distinct reasons.

**Q12.** What is conjugacy and why did anyone care before we had computers?

**Q13.** Explain the reparameterization trick and what problem it solves.

**Q14.** What is the difference between aleatoric and epistemic uncertainty? Why does it matter
operationally?

**Q15.** Your model outputs a 0.9 probability. What does that number mean, and how would you check
whether it is honest?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Do the base rate calculation in your head using odds
- [ ] Name the noise model implied by any loss function, and vice versa
- [ ] Explain regularization as a prior without hand-waving
- [ ] State the multivariate Gaussian's closure properties and why they matter
- [ ] Explain why marginalization is the hard part of probabilistic ML
- [ ] Say what the CLT does and does not promise, including the skewness caveat
- [ ] Derive a generalization bound from Hoeffding's inequality
