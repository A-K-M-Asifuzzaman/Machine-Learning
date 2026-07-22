# 03.04 — Exercises: Logistic Regression

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Derive $\sigma'(z) = \sigma(z)(1-\sigma(z))$ from the definition.

**D2.** Show that $\sigma^{-1}(p) = \log\frac{p}{1-p}$, and hence that logistic regression is a
linear model **for the log-odds**.

**D3.** Show the decision boundary $p = 0.5$ is the hyperplane $\mathbf{w}^{\top}\mathbf{x}+b=0$.

**D4.** Derive the negative log-likelihood from the Bernoulli likelihood
$p_i^{y_i}(1-p_i)^{1-y_i}$, and identify it as binary cross-entropy.

**D5.** Derive $\nabla J = \mathbf{X}^{\top}(\mathbf{p}-\mathbf{y})$. Compare its form with linear
regression's gradient and explain why they match.

**D6.** Derive $\mathbf{H} = \mathbf{X}^{\top}\mathbf{S}\mathbf{X}$ with
$\mathbf{S}=\mathrm{diag}(p_i(1-p_i))$. What does the weighting mean about which examples the
model learns from?

**D7.** Prove $J$ is convex. Then state precisely when it is *strictly* convex, and what fails
otherwise.

**D8.** Derive the IRLS update from the Newton step, including the working response
$\mathbf{z}=\mathbf{X}\mathbf{w}+\mathbf{S}^{-1}(\mathbf{y}-\mathbf{p})$. Show it is weighted least
squares.

**D9.** Prove that under perfect separation the likelihood has no finite maximizer. *Hint*: show
scaling $\mathbf{w}\to c\mathbf{w}$ with $c>1$ strictly increases $\ell$.

**D10.** Show that adding $\lambda\Vert\mathbf{w}\Vert_2^{2}$ guarantees a finite unique minimum
even under separation.

**D11.** Derive the stable form $J_i = \max(z_i,0) - z_iy_i + \log(1+e^{-|z_i|})$ from the naive
one, and verify they agree for moderate $z$.

**D12.** *(Calibration.)* Show that at the optimum $\sum_i p_i = \sum_i y_i$ exactly, when an
intercept is fitted. Which component of the gradient gives this?

**D13.** Show the softmax model is over-parameterized: adding a constant vector to every
$\mathbf{w}_k$ leaves all probabilities unchanged. What does that imply about uniqueness?

**D14.** Show that softmax regression with $K=2$ reduces exactly to binary logistic regression.

**D15.** Derive the cost-optimal threshold $p^{*} = C_{FP}/(C_{FP}+C_{FN})$ by minimizing expected
cost.

**D16.** *(Odds vs risk.)* Show that an odds ratio of 2 corresponds to a risk ratio of $2/(1+p)$
where $p$ is the baseline probability. When are they approximately equal?

---

## Tier 2 — Implementation

**I1.** Implement `stable_sigmoid` and `bce_with_logits`. Verify they stay finite at $|z|=800$
and agree with the naive versions for $|z|<10$.

**I2.** Implement logistic regression by gradient descent. Verify against sklearn with
`penalty=None`.

**I3.** Implement Newton/IRLS. Reproduce Experiment 2 and verify quadratic convergence by checking
that $\log\Vert\mathbf{w}_{t+1}-\mathbf{w}^{*}\Vert \approx 2\log\Vert\mathbf{w}_t-\mathbf{w}^{*}\Vert$.

**I4.** Implement the IRLS step **explicitly as a weighted least squares call** (build
$\mathbf{z}$ and $\mathbf{S}$, then call a WLS routine). Verify it matches your Newton step.

**I5.** Work out the correspondence between your $\lambda$ and sklearn's $C$ from the two
objectives, then verify empirically across three values of $C$. (It is easy to get an extra factor
of $n$ wrong here — derive it, do not guess.)

**I6.** Reproduce Experiment 3. Then give the separable problem increasing iteration budgets and
show $\Vert\mathbf{w}\Vert$ grows without settling.

**I7.** Implement $\ell_1$-penalized logistic regression by proximal gradient (ISTA,
[00.02 §15](../../00-mathematical-foundations/02-calculus-and-optimization/)). Verify sparsity
against `sklearn` with `penalty="l1", solver="saga"`.

**I8.** Implement standard errors from the inverse Hessian and verify against `statsmodels.Logit`.
Then explain why they are invalid once you regularize.

**I9.** Implement `odds_ratios()` with confidence intervals. Verify the intervals are built on the
log-odds scale and then exponentiated, and explain why symmetric intervals around $e^{w}$ would be
wrong.

**I10.** Reproduce Experiment 1's non-obvious result: show that adding correctly-classified points
far from the boundary degrades an OLS classifier and not a logistic one.

**I11.** Implement softmax regression. Verify $K=2$ reproduces binary logistic regression, and
compare multinomial against one-vs-rest on a 4-class problem for both accuracy and probability
coherence.

**I12.** Reproduce Experiment 4. Add `sklearn.calibration.CalibratedClassifierCV` around the
random forest and show it closes the calibration gap.

**I13.** *(Threshold tuning.)* On an imbalanced dataset, sweep the threshold and plot precision,
recall, and expected cost for a chosen $C_{FP}/C_{FN}$ ratio. Compare the cost-optimal threshold
against 0.5 and against what resampling would have done.

---

## Tier 3 — Interview

**Q1.** Why not use linear regression for classification? Give three reasons, including one that
is not obvious.

**Q2.** What does logistic regression actually model — the probability or something else?

**Q3.** Interpret a coefficient of 0.7.

**Q4.** Is an odds ratio the same as a risk ratio?

**Q5.** Derive the logistic loss from maximum likelihood.

**Q6.** Is the logistic loss convex? Prove it.

**Q7.** Why is there no closed-form solution?

**Q8.** What is IRLS, and why does it converge in so few iterations?

**Q9.** Why does sklearn default to L-BFGS rather than Newton?

**Q10.** Your coefficients came back in the hundreds with huge standard errors. Diagnose.

**Q11.** Is sklearn's `LogisticRegression` regularized by default? What is `C`?

**Q12.** Why is logistic regression well-calibrated, and what exactly does that mean?

**Q13.** Should you use 0.5 as your threshold?

**Q14.** For imbalanced data, would you resample or move the threshold? Why?

**Q15.** Multinomial or one-vs-rest? What breaks with OvR?

**Q16.** Why does `BCEWithLogitsLoss` exist when `Sigmoid` and `BCELoss` already do?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive the loss, gradient, and Hessian from scratch
- [ ] Prove convexity in one line
- [ ] Explain IRLS as repeated weighted least squares
- [ ] Recognize perfect separation from the symptoms and know three fixes
- [ ] Interpret a coefficient as an odds ratio, and say why that is not a risk ratio
- [ ] Explain why the model is calibrated, from the optimality condition
- [ ] Choose a threshold from costs rather than convention
- [ ] Write a numerically stable implementation without looking it up
