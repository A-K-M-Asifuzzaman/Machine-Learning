# 03.03 — Exercises: Basis Expansion

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Explain why $y = w_0 + w_1x + w_2x^{2} + w_3\log x$ is a *linear* model. What would make
a model nonlinear in the parameters?

**D2.** Count the free parameters of a cubic piecewise polynomial with $K$ knots under each
continuity condition (none, $C^0$, $C^1$, $C^2$). Show the cubic spline has $K+4$.

**D3.** Prove that $(x-\xi)_+^{3}$ has value, first derivative, and second derivative all zero at
$x=\xi$. Use this to explain why the truncated power basis produces a $C^{2}$ spline.

**D4.** Verify the Cox-de Boor recursion produces a partition of unity: $\sum_i B_{i,d}(x)=1$.

**D5.** Show that a degree-$d$ B-spline is nonzero on exactly $d+2$ knot intervals, and explain
what that implies for the sparsity of the design matrix.

**D6.** Derive the natural cubic spline basis from the truncated power basis by imposing
$f''=0$ outside the boundary knots. Confirm the dimension drops from $K+4$ to $K$.

**D7.** Show that the natural cubic spline is *linear* beyond the boundary knots — i.e. that its
second derivative is identically zero there.

**D8.** State the smoothing-spline theorem: the minimizer of
$\sum(y_i-f(x_i))^{2}+\lambda\int f''^{2}$ over all $C^{2}$ functions is a natural cubic spline
with knots at the unique $x_i$. Why is it remarkable that an infinite-dimensional problem has a
finite-dimensional solution?

**D9.** Show the smoothing spline solution is $\hat{\mathbf{f}} = (\mathbf{I}+\lambda\boldsymbol{\Omega})^{-1}\mathbf{y}$
and identify the structural correspondence with ridge regression
([03.02 §2](../02-regularized-linear-models/)).

**D10.** Show that $\mathrm{df}(\lambda)=\mathrm{tr}(\mathbf{S}_\lambda)$ tends to $n$ as
$\lambda\to0$ and to **2** as $\lambda\to\infty$. What is the null space of $\boldsymbol{\Omega}$,
and why does its dimension determine that limit?

**D11.** Derive the GCV formula and explain why it approximates leave-one-out CV without $n$
refits. Which property of $\mathbf{S}_\lambda$ makes this possible?

**D12.** Show that a GAM $\beta_0+\sum_j f_j(x_j)$ is identified only up to a constant per $f_j$,
and explain why backfitting must centre each function.

**D13.** Prove that $y = x_1x_2$ has no additive decomposition $f_1(x_1)+f_2(x_2)$ when $x_1,x_2$
are independent with mean zero. *Hint*: compute $\mathbb{E}[y\mid x_1]$.

**D14.** Show that raw polynomial and orthogonal polynomial bases of the same degree span the
same function space, so the fitted values are identical. What differs?

**D15.** *(Kernel connection.)* Show that a smoothing spline is a kernel ridge regression, and
identify the kernel. See [03.07](../07-svm/).

---

## Tier 2 — Implementation

**I1.** Implement `polynomial_basis` and `legendre_basis`. Verify they give identical fitted
values, then measure $\kappa$ for each up to degree 20.

**I2.** Implement the truncated power basis and B-splines. Verify they produce the same fit and
compare their condition numbers.

**I3.** Implement B-splines by Cox-de Boor. Verify partition of unity, non-negativity, and compact
support, then check against `scipy.interpolate.BSpline`.

**I4.** Implement the natural cubic basis. Verify numerically that the fitted function's second
derivative is zero beyond the boundary knots.

**I5.** Reproduce Experiment 1. Then repeat with **Chebyshev nodes** instead of equally spaced
points and show the divergence disappears. What does that tell you about whether Runge is about
the polynomial or about the sampling?

**I6.** Reproduce Experiment 3. Add a model that raises an exception outside the training range
and argue for or against that as the default behaviour.

**I7.** Implement a smoothing spline with the true roughness penalty
$\Omega_{jk}=\int N_j''N_k''$. Verify $\mathrm{df}\to2$ as $\lambda\to\infty$. Then implement the
naive second-difference penalty and show its df tends to 4 instead — explain why.

**I8.** Implement GCV. Compare the $\lambda$ it selects against 5-fold CV on the same data. How
close are they, and how much faster is GCV?

**I9.** Implement GAM backfitting. Verify it converges, and plot each $f_j$ against the true
component function.

**I10.** Reproduce Experiment 5. Then add an explicit interaction term $f_{12}(x_1,x_2)$ to your
GAM and show it recovers the interaction case.

**I11.** Compare `sklearn.preprocessing.SplineTransformer` against your B-spline basis. Verify
they agree, and read its source to see how it handles extrapolation.

**I12.** *(Real data.)* Fit a GAM to a dataset with 4-6 features. Plot every partial dependence
function. Write one sentence per feature describing what the model believes — this is the thing a
black-box model cannot give you.

---

## Tier 3 — Interview

**Q1.** Is polynomial regression a linear model? Justify.

**Q2.** Why shouldn't you use a degree-15 polynomial?

**Q3.** What is the Runge phenomenon, and what fixes it?

**Q4.** What is a cubic spline? Why cubic and not quadratic or quartic?

**Q5.** Why are B-splines preferred over the truncated power basis?

**Q6.** What makes a spline "natural", and why would you want that?

**Q7.** What is the difference between a regression spline and a smoothing spline?

**Q8.** What does $\lambda$ control in a smoothing spline, and why do practitioners specify `df`
instead?

**Q9.** What is a GAM? What can it represent that a linear model cannot?

**Q10.** What can a GAM *not* represent?

**Q11.** Where would you place knots, and how many?

**Q12.** Your model predicts nonsense outside the training range. Diagnose and fix.

**Q13.** How do basis expansion, kernel methods, and neural networks relate?

**Q14.** When would you choose a GAM over gradient boosting?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Explain why "linear model" is about parameters, not features
- [ ] Say what goes wrong with high-degree polynomials — three distinct failures
- [ ] Construct a cubic spline basis and count its degrees of freedom
- [ ] Explain what "natural" buys and what it costs
- [ ] Recognize a smoothing spline as ridge regression in a spline basis
- [ ] Fit and *read* a GAM, and state precisely what it cannot capture
- [ ] Place basis expansion, kernels, and networks on one axis: who chooses $\phi$
