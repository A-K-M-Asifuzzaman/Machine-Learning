# 04.04 — Exercises: Gaussian Mixtures & EM

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Write the GMM generative model and its marginal density $p(\mathbf{x})$. Identify the latent
variable and the parameters.

**D2.** Write the log-likelihood and explain why the "log of a sum" makes direct MLE hard and
non-convex.

**D3.** Derive the E-step: show the responsibility $\gamma_{nk}$ is the posterior
$p(z_n=k\mid\mathbf{x}_n)$ by Bayes' rule.

**D4.** Derive the M-step updates for $\pi_k$, $\boldsymbol\mu_k$, $\boldsymbol\Sigma_k$ by maximizing
the expected complete-data log-likelihood (use a Lagrange multiplier for $\sum_k\pi_k=1$).

**D5.** Derive the ELBO via Jensen's inequality and show the gap to $\ell$ is
$\sum_n \mathrm{KL}(q\,\Vert\,p(z\mid\mathbf{x}_n))$.

**D6.** Show that the E-step makes the ELBO tight and the M-step raises it, and conclude EM never
decreases $\ell$.

**D7.** Take the spherical-equal-covariance GMM and let $\sigma^2\to0$. Show the responsibilities
become hard nearest-center assignments, recovering k-means.

**D8.** Count the free parameters of a GMM for full, diagonal, spherical, and tied covariance, and
write the BIC.

**D9.** Explain the covariance singularity: show that a component centered on one point with
$\boldsymbol\Sigma\to0$ drives the likelihood to $+\infty$. How does `reg_covar` fix it?

**D10.** Explain why the global maximum likelihood of a GMM is degenerate, and why EM's *local*
optimum is what you actually want.

---

## Tier 2 — Implementation

**I1.** Implement the E-step (responsibilities via log-sum-exp) and M-step (weighted updates). Verify
the log-likelihood and labels against `sklearn.mixture.GaussianMixture`.

**I2.** Implement k-means initialization and `reg_covar`. Reproduce Experiment 3: trigger a
singularity without regularization and fix it with the floor.

**I3.** Reproduce Experiment 1: show full-covariance GMM beating k-means on tilted, elongated
clusters.

**I4.** Reproduce Experiment 2: record the log-likelihood per iteration and confirm it never
decreases.

**I5.** Implement all four covariance types. Reproduce Experiment 4: spherical ≈ k-means, full wins.

**I6.** Implement BIC and AIC. Reproduce Experiment 5: select $K$ by minimizing BIC.

**I7.** Reproduce Experiment 6: show responsibilities hardening as $\sigma^2\to0$.

**I8.** Use a fitted GMM as a **density model**: score held-out points and flag low-density ones as
anomalies. Compare to a single Gaussian.

**I9.** Sample from a fitted GMM (pick a component by $\pi$, draw from its Gaussian) and compare the
synthetic data to the real data.

**I10.** *(General EM.)* Implement EM for a mixture of Bernoullis (binary data) and confirm the same
E/M/ELBO structure applies.

---

## Tier 3 — Interview

**Q1.** What is a Gaussian mixture model?

**Q2.** How is a GMM different from k-means?

**Q3.** Explain the EM algorithm.

**Q4.** Why does EM increase the likelihood every iteration?

**Q5.** What is a responsibility?

**Q6.** How do you choose the number of components?

**Q7.** What is the singularity problem, and how do you prevent it?

**Q8.** What covariance type would you use, and why?

**Q9.** Does EM find the global optimum?

**Q10.** How is k-means a special case of a GMM?

**Q11.** What else can a fitted GMM do besides cluster?

**Q12.** Where else does the EM algorithm appear?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive the E-step and M-step from maximum likelihood
- [ ] Explain EM as coordinate ascent on the ELBO
- [ ] Connect k-means to GMM via the zero-variance limit
- [ ] Choose the number of components with BIC
- [ ] Explain and prevent covariance singularities
- [ ] Use a GMM as a density model, not just a clusterer
