# 03.02 — Exercises: Regularized Linear Models

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Derive the ridge solution $\hat{\mathbf{w}} = (\mathbf{X}^{\top}\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^{\top}\mathbf{y}$.
Then prove it exists and is unique for every $\lambda>0$, including when $d>n$.

**D2.** Using the SVD, derive

$$\hat{\mathbf{w}}_{\text{ridge}} = \sum_i \frac{\sigma_i}{\sigma_i^{2}+\lambda}(\mathbf{u}_i^{\top}\mathbf{y})\mathbf{v}_i$$

and compare term by term with OLS. Which directions are shrunk most, and why is that the
statistically sensible choice?

**D3.** Show $\kappa(\mathbf{X}^{\top}\mathbf{X}+\lambda\mathbf{I}) < \kappa(\mathbf{X}^{\top}\mathbf{X})$
for every $\lambda>0$.

**D4.** Derive $\mathrm{df}(\lambda) = \sum_i \sigma_i^{2}/(\sigma_i^{2}+\lambda)$ as the trace of
the ridge hat matrix. Verify it equals $d$ at $\lambda=0$ and tends to 0 as $\lambda\to\infty$.

**D5.** Derive the bias and variance of the ridge estimator explicitly. Then show there always
exists a $\lambda>0$ with lower MSE than OLS. *(This is Theobald's / Hoerl-Kennard's result — it
is the formal statement of the Gauss-Markov loophole.)*

**D6.** Compute $\partial|w|$ at $w=0$ and use it to derive the condition under which the Lasso
solution has $w_j = 0$ exactly. Express that condition in terms of the correlation between
feature $j$ and the residual.

**D7.** Derive the soft-thresholding operator by solving
$\min_u \left(\lambda|u| + \tfrac12(u-v)^{2}\right)$ by cases on the sign of $u$.

**D8.** Derive the Lasso coordinate-descent update. Show it is soft thresholding applied to the
one-dimensional least-squares solution.

**D9.** Derive the Elastic Net coordinate update and show it reduces to Lasso at $\alpha=1$ and
to a ridge-like update at $\alpha=0$.

**D10.** Show that Ridge is MAP with a Gaussian prior and Lasso is MAP with a Laplace prior. Give
$\lambda$ in terms of the noise variance and prior scale in each case.

**D11.** Explain, using the shape of the Laplace density at zero, why its MAP estimate is sparse.
Then explain why the posterior *mean* under the same prior is not.

**D12.** Prove Lasso selects at most $\min(n,d)$ features. *Hint*: consider the KKT conditions and
the number of active constraints.

**D13.** Show that ridge regression on standardized features is equivalent to OLS on an augmented
dataset with $\sqrt{\lambda}\mathbf{I}$ appended to $\mathbf{X}$ and zeros appended to
$\mathbf{y}$. What does this tell you about implementing ridge with an OLS solver?

**D14.** Show that the penalty is not scale-invariant: if $x_j \to c x_j$, describe what happens
to $\hat{w}_j$ under OLS and under ridge, and explain why only one of them is harmless.

**D15.** *(Grouping effect.)* For two identical features $\mathbf{x}_1 = \mathbf{x}_2$, show that
Elastic Net gives $\hat{w}_1 = \hat{w}_2$ exactly, while Lasso admits any split
$\hat{w}_1 + \hat{w}_2 = c$.

---

## Tier 2 — Implementation

**I1.** Implement Ridge by both the closed form and the SVD. Verify they agree, and verify both
against `sklearn.linear_model.Ridge` to $10^{-9}$.

**I2.** Implement soft thresholding and Lasso by coordinate descent. Verify against
`sklearn.linear_model.Lasso` — coefficients **and** the exact-zero support.

**I3.** Implement Lasso by ISTA as well. Compare the number of iterations each needs for the same
tolerance, and explain the gap.

**I4.** Implement Elastic Net. Verify against sklearn for several $(\alpha, \ell_1\text{-ratio})$
pairs.

**I5.** Reproduce Experiment 1: decompose ridge's coefficient error into bias² and variance across
a $\lambda$ grid, and find the $\lambda$ minimizing MSE. Confirm it is strictly greater than zero.

**I6.** Reproduce Experiment 2. Construct a design matrix with prescribed singular values and
verify the shrinkage factors match $\sigma_i^{2}/(\sigma_i^{2}+\lambda)$ exactly.

**I7.** Implement `effective_dof`. Plot it against $\lambda$ and against the number of nonzero
Lasso coefficients at the same $\lambda$. Are they comparable notions of complexity?

**I8.** Reproduce Experiment 4. Then fix the unstandardized case using
`sklearn.pipeline.Pipeline(StandardScaler(), Ridge())` and confirm it matches the standardized
result.

**I9.** Reproduce Experiment 5's correlation sweep. Find, for your own data-generating process,
the correlation at which Lasso starts dropping group members.

**I10.** Implement the regularization path with warm starts. Time it against fitting each
$\lambda$ independently, and plot the coefficient paths for both Ridge and Lasso. Explain why one
is smooth and the other piecewise linear.

**I11.** Implement $k$-fold CV with the **one-standard-error rule**. Compare $\lambda_{\min}$ and
$\lambda_{1\text{SE}}$ on a real dataset: how much test error do you give up, and how many
features do you save?

**I12.** *(The leak.)* Fit a Lasso two ways — standardizing before splitting, and standardizing
inside each CV fold. Measure the difference in reported CV error. This is the leak of
[02.06](../../02-data/06-data-leakage/) in miniature.

**I13.** Implement ridge via the augmented-data trick of D13 and verify it matches your closed
form.

**I14.** *(Lasso for selection, OLS for estimation.)* Implement the "relaxed Lasso": use Lasso to
select a support, then refit unpenalized OLS on just those features. Compare its coefficients and
test error against plain Lasso. When is each better?

---

## Tier 3 — Interview

**Q1.** What problem does regularization solve? Answer in terms of bias and variance.

**Q2.** How can a biased estimator beat OLS if Gauss-Markov says OLS is best?

**Q3.** Explain what ridge does in terms of the SVD.

**Q4.** Why does Lasso produce exact zeros and ridge not? Give two independent explanations.

**Q5.** When would you choose ridge over Lasso, and vice versa?

**Q6.** What is Elastic Net for? Name the two Lasso failures it fixes.

**Q7.** Do you have to standardize features for ridge? For OLS? Why the difference?

**Q8.** How do you choose $\lambda$? What is the one-standard-error rule and why use it?

**Q9.** What is the Bayesian interpretation of ridge and Lasso?

**Q10.** You have 20,000 features and 200 samples. What do you do?

**Q11.** Your Lasso zeroed a feature you know is important. What happened?

**Q12.** Can you interpret regularized coefficients the way you interpret OLS coefficients?

**Q13.** What is "effective degrees of freedom" and why is it more useful than the feature count?

**Q14.** Why does ridge help with multicollinearity? Answer using the SVD, not hand-waving.

**Q15.** Why should you not penalize the intercept?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Explain the Gauss-Markov loophole and why regularization exploits it
- [ ] Describe ridge as direction-dependent shrinkage, not uniform shrinkage
- [ ] Give three independent reasons $\ell_1$ produces exact zeros
- [ ] Derive and implement the coordinate-descent update from memory
- [ ] Say what $\lambda$ means in Bayesian terms
- [ ] Explain why standardization is mandatory here but not for OLS
- [ ] Choose between ridge, Lasso, and Elastic Net for a described problem, and defend it
- [ ] State the *actual* correlation level at which Lasso's arbitrary selection begins
