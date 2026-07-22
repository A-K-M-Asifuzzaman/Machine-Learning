# 00.04 — Exercises: Statistics and Inference

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Prove $\mathrm{MSE}(\hat\theta) = \mathrm{Bias}(\hat\theta)^{2} + \mathrm{Var}(\hat\theta)$.
Show explicitly why the cross term vanishes.

**D2.** Derive the MLE for the Bernoulli parameter. Then explain what goes wrong when you observe
5 failures and no successes, and how a $\mathrm{Beta}(\alpha,\beta)$ prior fixes it.

**D3.** Derive both Gaussian MLEs ($\hat\mu$ and $\hat\sigma^{2}$) by setting partial derivatives
to zero. Confirm the variance estimate divides by $n$.

**D4.** Prove $\mathbb{E}[\hat\sigma^{2}_{\mathrm{MLE}}] = \frac{n-1}{n}\sigma^{2}$.
*Hint*: start from $\sum_i(x_i-\bar x)^{2} = \sum_i(x_i-\mu)^{2} - n(\bar x-\mu)^{2}$.

**D5.** Show that $s^{2}$ is unbiased for $\sigma^{2}$ but $s$ is **not** unbiased for $\sigma$.
Which inequality from [00.03](../03-probability/) explains this?

**D6.** Derive the Fisher information for the Bernoulli, $I(\pi) = 1/(\pi(1-\pi))$. At which $\pi$
is information minimized, and what does that mean intuitively?

**D7.** Show the sample mean attains the Cramér-Rao bound for a Gaussian with known $\sigma$ —
i.e. it is efficient at every $n$, not just asymptotically.

**D8.** State precisely what a 95% confidence interval means, and write down three common
misreadings with an explanation of why each is wrong.

**D9.** Derive the Wilson score interval by inverting the score test. Then show that the naive
Wald interval has zero width when $\hat p = 0$, and explain why its coverage collapses for small
$p$.

**D10.** Prove that $p$-values are $\mathrm{Uniform}(0,1)$ under the null for a continuous test
statistic. *Hint*: use the probability integral transform ([00.03 §15.1](../03-probability/)).

**D11.** Derive $P(\text{at least one false positive}) = 1-(1-\alpha)^{m}$ for $m$ independent
null tests. Then derive the Bonferroni threshold from the union bound, and explain why Bonferroni
is conservative when the tests are correlated.

**D12.** Explain why the Benjamini-Hochberg procedure controls the FDR rather than the FWER, and
give a concrete situation where FDR control is the correct choice.

**D13.** Explain why McNemar's test uses only the discordant pairs. What information do the
concordant pairs carry about which model is better?

**D14.** Explain why cross-validation fold scores are not independent, and what that does to the
naive standard error $s/\sqrt{k}$.

**D15.** *(Selection bias, quantified.)* You train $m$ models whose true accuracies are all
identical, with test accuracy $\hat{a}_i \sim \mathcal{N}(a, \sigma^{2})$. Show that
$\mathbb{E}[\max_i \hat{a}_i] > a$, and that the gap grows roughly like
$\sigma\sqrt{2\log m}$. This is the "we tried 50 models" bias, with a number attached.

---

## Tier 2 — Implementation

**I1.** Implement both variance estimators. Reproduce Experiment 4: verify the $(n-1)/n$ bias
empirically, **and** verify that the biased estimator has lower MSE. Explain the second result.

**I2.** Implement `measure_estimator`. Use it to verify MSE = bias² + variance exactly. Explain
why you must use `ddof=0` for the identity to hold.

**I3.** Implement z-, t-, Wilson-, and bootstrap intervals. Reproduce Experiment 1's coverage
table. At what $n$ does the z-interval become acceptable for Gaussian data?

**I4.** Reproduce the proportion-coverage table. Then find, by search, the smallest $n$ at which
the naive Wald interval achieves ≥93% coverage for $p = 0.01$.

**I5.** Reproduce Experiment 2: verify $p$-values are uniform under the null. Then make the null
*false* (shift one group) and re-plot — the distribution should concentrate near zero.

**I6.** Implement Bonferroni, Holm, and Benjamini-Hochberg. Verify against
`statsmodels.stats.multitest.multipletests`. Then confirm empirically that Holm rejects a superset
of Bonferroni's rejections, always.

**I7.** Implement a permutation test. Compare its $p$-value to a $t$-test on (a) Gaussian data —
they should agree; (b) heavily skewed data with $n = 10$ — they should not. Which do you trust?

**I8.** Implement McNemar's test. Simulate two classifiers with a known true accuracy gap and
compare McNemar's power against an (incorrect) unpaired two-sample $t$-test at the same $n$.

**I9.** Implement the bootstrap. Reproduce Experiment 5: verify it recovers the true standard
error for the median, a percentile, and F1 — and that it **fails** for the maximum.

**I10.** *(Real error bars.)* Take any classifier and test set. Produce bootstrap 95% CIs for
accuracy, precision, recall, F1, and AUC. Then answer: how many test examples would you need for
the F1 interval to be narrower than ±1%?

**I11.** *(The selection bias you commit weekly.)* Train 50 models that differ only by random
seed. Record the best validation accuracy and its test accuracy. Repeat 100 times. Measure the
average gap between "best validation" and its test score. This is the optimism you are shipping.

**I12.** Implement a power calculation: given effect size, $\sigma$, and $\alpha$, find the $n$
needed for 80% power. Then compute the $n$ required to detect a 1% accuracy difference between two
classifiers.

---

## Tier 3 — Interview

**Q1.** What does a 95% confidence interval mean? What does it not mean?

**Q2.** What is a $p$-value? Name three things it is not.

**Q3.** Your model gets 94.2% and the baseline gets 93.8% on a 1,000-example test set. Is your
model better?

**Q4.** How would you put an error bar on an F1 score?

**Q5.** Why does the sample variance divide by $n-1$?

**Q6.** Is the MLE unbiased? Give a counterexample.

**Q7.** Is an unbiased estimator always preferable? Justify with the MSE decomposition.

**Q8.** You tried 30 hyperparameter configurations and the best one is significantly better than
baseline at $p = 0.03$. What is wrong with this claim, and how would you fix it?

**Q9.** What is the difference between controlling FWER and FDR? When do you want each?

**Q10.** How would you compare two classifiers on the same test set? Why not a two-sample t-test?

**Q11.** When does the bootstrap fail?

**Q12.** What is statistical power, and why are underpowered studies worse than no study?

**Q13.** What is the difference between a confidence interval and a credible interval?

**Q14.** With 10 million examples, every difference is statistically significant. What do you
report instead?

**Q15.** Your A/B test has been running a week and you've been checking daily. The $p$-value just
dipped below 0.05. Ship it?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] State what a CI means without saying "95% probability the parameter is in it"
- [ ] Explain the $n-1$ correction and connect it to training error underestimating test error
- [ ] Explain why the biased variance MLE can still be the better estimator
- [ ] Put a defensible error bar on any metric, using the bootstrap
- [ ] Recognize a multiple-comparisons problem in an ML workflow and correct it
- [ ] Choose the right test for two models on the same test set, and say why
- [ ] Say why a 1% improvement on a 1,000-example test set is not a result
