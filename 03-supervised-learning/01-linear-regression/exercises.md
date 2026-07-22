# 03.01 — Exercises: Linear Regression

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Derive the OLS estimator three ways — by projection, by calculus, and by maximum
likelihood — and state where each derivation uses which assumption.

**D2.** Show that the Hessian of the least-squares objective is $2\mathbf{X}^{\top}\mathbf{X}$,
hence PSD, hence the problem is convex. When is it *strictly* convex?

**D3.** Prove $\mathbb{E}[\hat{\mathbf{w}}] = \mathbf{w}$. Which assumptions did you use?

**D4.** Derive $\mathrm{Cov}(\hat{\mathbf{w}}) = \sigma^{2}(\mathbf{X}^{\top}\mathbf{X})^{-1}$.
Then read three practical consequences off the formula.

**D5.** Prove the Gauss-Markov theorem. Identify the exact step where unbiasedness is used, and
explain why ridge escapes the conclusion.

**D6.** Show $\mathbb{E}[\mathrm{RSS}] = (n-d)\sigma^{2}$, hence that $\hat\sigma^{2} =
\mathrm{RSS}/(n-d)$ is unbiased. *Hint*: $\mathrm{RSS} = \boldsymbol{\varepsilon}^{\top}(\mathbf{I}-\mathbf{H})\boldsymbol{\varepsilon}$
and $\mathrm{tr}(\mathbf{I}-\mathbf{H}) = n-d$.

**D7.** Show the hat matrix satisfies $\mathbf{H}^{2}=\mathbf{H}=\mathbf{H}^{\top}$ and
$\mathrm{tr}(\mathbf{H}) = d$. Explain why the latter means average leverage is $d/n$.

**D8.** Show $\mathrm{Var}(r_i) = \sigma^{2}(1-h_{ii})$. Explain why this means raw residual plots
systematically under-show high-leverage points.

**D9.** Prove that $R^{2}$ cannot decrease when a feature is added. Then derive adjusted $R^{2}$
and show it can.

**D10.** *(Omitted-variable bias.)* Truth: $y = w_0+w_1x_1+w_2x_2+\varepsilon$. You fit only
$y\sim x_1$. Show

$$\mathbb{E}[\hat{w}_1] = w_1 + w_2\frac{\mathrm{Cov}(x_1,x_2)}{\mathrm{Var}(x_1)}$$

Then use it to explain why omitting $x^{2}$ does **not** bias the slope when $x$ is symmetric
about zero.

**D11.** Derive the VIF formula $\mathrm{VIF}_j = 1/(1-R_j^{2})$ from
$\mathrm{Cov}(\hat{\mathbf{w}}) = \sigma^{2}(\mathbf{X}^{\top}\mathbf{X})^{-1}$.

**D12.** Show that with two perfectly collinear features, $\hat{\mathbf{w}}$ is not unique but
$\hat{\mathbf{y}}$ is. Explain using the column space.

**D13.** Derive Cook's distance from the definition "how much do all fitted values move when point
$i$ is deleted", and show it factors into a residual term and a leverage term.

**D14.** Show that the OLS residuals sum to zero if and only if an intercept is fitted.

**D15.** *(Frisch-Waugh-Lovell.)* Show that $\hat{w}_j$ from a multiple regression equals the
coefficient from regressing the residuals of $y$ on the other features against the residuals of
$x_j$ on the other features. Explain what "holding the others fixed" really means.

---

## Tier 2 — Implementation

**I1.** Implement OLS with all four solvers. Verify they agree on well-conditioned data to $10^{-10}$.

**I2.** Implement standard errors, t-statistics, p-values, and confidence intervals. Verify against
`statsmodels.api.OLS().fit()` — every quantity, to $10^{-9}$.

**I3.** Reproduce Experiment 2. At what $\kappa(\mathbf{X})$ do the normal equations lose 8 digits?
Does it match the $\kappa^{2}$ prediction?

**I4.** Reproduce Experiment 1's Gauss-Markov demonstration. Construct your own unbiased linear
estimator (hint: perturb OLS inside the null space of $\mathbf{X}$) and confirm it has higher
variance.

**I5.** Reproduce Experiment 3. Find the number of noise features at which test $R^{2}$ first goes
negative, for $n = 60$.

**I6.** Implement leverage and Cook's distance. Reproduce Experiment 5 and construct your own
point with high leverage but zero influence.

**I7.** Implement VIF. On a dataset with three correlated features, show that VIF rises as the
correlation does, and confirm the standard errors rise as $\sqrt{\mathrm{VIF}}$.

**I8.** *(Anscombe's quartet.)* Load or construct it. Fit OLS to all four. Verify the coefficients,
$R^{2}$, and standard errors are (nearly) identical, then plot the residuals and explain what each
one actually is.

**I9.** Implement heteroscedasticity-robust (HC3) standard errors:
$\widehat{\mathrm{Cov}} = (\mathbf{X}^{\top}\mathbf{X})^{-1}\mathbf{X}^{\top}\mathbf{\Omega}\mathbf{X}(\mathbf{X}^{\top}\mathbf{X})^{-1}$
with $\Omega_{ii} = r_i^{2}/(1-h_{ii})^{2}$. Reproduce Experiment 4's heteroscedastic row and show
HC3 restores 95% coverage.

**I10.** Implement weighted least squares. On data with known heteroscedasticity, show WLS has
lower variance than OLS — i.e. that OLS stops being BLUE when assumption 3 fails.

**I11.** *(Bootstrap vs analytic.)* Compute coefficient confidence intervals both analytically and
by bootstrapping rows ([00.04 §12](../../00-mathematical-foundations/04-statistics-and-inference/)).
Compare on clean data, then on heteroscedastic data. Which one stays honest?

**I12.** Implement the Frisch-Waugh-Lovell procedure from D15 and verify numerically that it
reproduces the multiple-regression coefficient exactly.

---

## Tier 3 — Interview

**Q1.** Derive the OLS solution. Why is squared error the right loss?

**Q2.** What are the assumptions of linear regression, and which one matters most?

**Q3.** Your coefficients are $+10^{6}$ and $-10^{6}$ on two features. What happened?

**Q4.** Does multicollinearity hurt predictions? Does it hurt interpretation?

**Q5.** What does Gauss-Markov say, and what does it *not* say?

**Q6.** Why is ridge regression allowed to beat OLS?

**Q7.** Your $R^{2}$ is 0.95. Is the model good?

**Q8.** Your $R^{2}$ is 0.04. Is the model useless?

**Q9.** Why does $R^{2}$ always increase when you add a feature?

**Q10.** Explain the difference between leverage and influence.

**Q11.** You see a funnel shape in the residuals-vs-fitted plot. What is wrong, what still works,
and what do you do?

**Q12.** Why does `sklearn` use SVD instead of $(\mathbf{X}^{\top}\mathbf{X})^{-1}\mathbf{X}^{\top}\mathbf{y}$?

**Q13.** You have 1,000 features and 100 rows. What happens, and what do you do?

**Q14.** What does the coefficient $w_j$ mean, precisely?

**Q15.** You ran stepwise selection and three features came out significant at $p < 0.01$. What is
wrong with reporting that?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive OLS from the geometry in one line
- [ ] State what each assumption buys and rank them by how much the violation costs
- [ ] Explain Gauss-Markov *and* why regularization is not a contradiction of it
- [ ] Compute a coefficient's standard error and say what it means
- [ ] Read a residual plot and name the violation
- [ ] Explain why multicollinearity breaks interpretation but not prediction
- [ ] Say why you should never trust a training $R^{2}$
