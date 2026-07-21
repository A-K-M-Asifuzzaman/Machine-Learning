# 00.02 — Exercises: Calculus and Optimization

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).
Implementation exercises can be checked against `from_scratch.py`.

---

## Tier 1 — Derivation

**D1.** Prove that the gradient is the direction of steepest ascent, using Cauchy-Schwarz.
Then answer: what changes if "steepest" is measured in the $\ell_1$ norm instead of $\ell_2$?
What descent direction do you get?

**D2.** Prove that the gradient is orthogonal to the level set through any point.

**D3.** *(The descent lemma.)* Assume $\nabla f$ is $L$-Lipschitz. Prove

$$f(\mathbf{y}) \le f(\mathbf{x}) + \nabla f(\mathbf{x})^{\top}(\mathbf{y}-\mathbf{x}) + \tfrac{L}{2}\Vert \mathbf{y}-\mathbf{x}\Vert ^{2}$$

Then substitute $\mathbf{y} = \mathbf{x} - \eta\nabla f(\mathbf{x})$ to derive the guaranteed
decrease per step, and show the decrease is positive exactly when $\eta < 2/L$.

**D4.** For the quadratic $f(\boldsymbol{\theta}) = \tfrac12\boldsymbol{\theta}^{\top}\mathbf{H}\boldsymbol{\theta}$,
derive $z_i^{(t)} = (1-\eta\lambda_i)^{t}z_i^{(0)}$ in the eigenbasis. Then derive the optimal
learning rate $\eta^{\star} = 2/(\lambda_{\min}+\lambda_{\max})$ and the resulting convergence
rate $(\kappa-1)/(\kappa+1)$.

**D5.** Prove that for convex $f$, every local minimum is global. Where exactly does the proof
use convexity?

**D6.** Prove that the composition $f(\mathbf{A}\mathbf{x}+\mathbf{b})$ is convex in $\mathbf{x}$
whenever $f$ is convex. Use this to show logistic regression's loss is convex in $\mathbf{w}$.

**D7.** Show that $\mathbf{X}^{\top}\mathbf{X} + \lambda\mathbf{I} \succ 0$ for $\lambda > 0$
regardless of $\mathbf{X}$, and hence that ridge regression is strongly convex. State the strong
convexity constant.

**D8.** Derive Newton's method by minimizing the second-order Taylor model exactly. Then prove
that on a quadratic it converges in exactly one step from any starting point.

**D9.** *(Adam's bias, done properly.)* Unroll $\mathbf{m}_t$ and $\mathbf{v}_t$ from zero
initialization. Show $\mathbb{E}[\mathbf{m}_t] \approx (1-\beta_1^{t})\mathbb{E}[\mathbf{g}]$ and
the analogous result for $\mathbf{v}_t$. Then show the *ratio* $\mathbf{m}_t/\sqrt{\mathbf{v}_t}$
is inflated by $(1-\beta_1^{t})/\sqrt{1-\beta_2^{t}}$ relative to the corrected version.
At what $t$ is this factor largest for $\beta_1 = 0.9, \beta_2 = 0.999$? Does the answer
change your view of why warmup is used?

**D10.** Derive the KKT conditions for
$\min \tfrac12\Vert \mathbf{w}\Vert ^{2}$ s.t. $y_i(\mathbf{w}^{\top}\mathbf{x}_i + b)\ge 1$.
Show that complementary slackness forces $\alpha_i = 0$ for all points strictly outside the
margin — i.e. derive the *existence of support vectors* rather than asserting it.

**D11.** Derive the Lagrange dual of the hard-margin SVM. Show the dual depends on the data only
through inner products $\mathbf{x}_i^{\top}\mathbf{x}_j$, and explain precisely why that makes the
kernel trick possible in the dual but not the primal.

**D12.** Prove weak duality ($q(\boldsymbol{\alpha}) \le p^{\star}$ always). Then state Slater's
condition and what it buys you.

**D13.** Compute the subdifferential of $f(w) = |w|$ at $w = 0$. Then derive the soft-thresholding
operator by solving

$$\mathrm{prox}_{\lambda}(v) = \arg\min_{u}\left(\lambda|u| + \tfrac12(u-v)^{2}\right)$$

**by cases** on the sign of $u$. Show explicitly why the solution is exactly zero for
$|v| \le \lambda$.

**D14.** Derive the proximal operator of $\tfrac{\lambda}{2}\Vert \mathbf{w}\Vert _2^{2}$ (ridge) and show
it is $\mathbf{v}/(1+\lambda)$. Use this to explain in one sentence why ridge never produces
exact zeros.

---

## Tier 2 — Implementation

**I1.** Implement `numerical_gradient` and `check_gradient`. Use it to verify a gradient you
derive by hand for a two-layer neural network. Deliberately introduce a sign error and confirm the
check catches it.

**I2.** Implement gradient descent. On a quadratic with known $\lambda_{\max}$, empirically find
the largest learning rate that converges. Confirm it matches $2/\lambda_{\max}$ to 3 significant
figures.

**I3.** Reproduce Experiment 2: measure iterations-to-converge for GD and Nesterov across
$\kappa \in \{10, 10^2, 10^3, 10^4\}$. Verify the $\kappa$ and $\sqrt{\kappa}$ scalings by
checking that iterations/$\kappa$ and iterations/$\sqrt{\kappa}$ are roughly constant.

**I4.** Implement Adam from the update rule. Then implement it *without* bias correction and
measure the step-size inflation ratio, isolating it as `from_scratch.py` does (same $(m,v)$
state, both rules). Compare against $(1-\beta_1^t)/\sqrt{1-\beta_2^t}$.

**I5.** Implement AdamW and vanilla Adam-with-L2-penalty. Train both on a problem with features
of wildly different scales. Show that the L2-in-the-gradient version regularizes
large-gradient parameters *less*, and explain why using the update rule.

**I6.** Implement backtracking (Armijo) line search, then Wolfe. Run BFGS on Rosenbrock with each.
Explain — using the $\mathbf{s}^{\top}\mathbf{y} > 0$ condition — why Armijo alone can make BFGS
silently degrade to steepest descent.

**I7.** Implement L-BFGS with the two-loop recursion. Verify against `scipy.optimize.minimize`
with `method="L-BFGS-B"`. Then measure memory and time as a function of the history size $m$.

**I8.** Implement ISTA for Lasso, then FISTA (add Nesterov momentum to the proximal step).
Plot objective vs iteration for both and confirm the $O(1/t)$ → $O(1/t^{2})$ improvement.

**I9.** Implement Newton's method for logistic regression (this is IRLS). Compare iteration counts
against gradient descent and L-BFGS on the same data. At what dimension does Newton stop being
worth it on your machine?

**I10.** *(Saddle points, seen.)* Construct $f(x,y) = x^{2} - y^{2}$. Start gradient descent at
$(0.001, 0.001)$ and at $(0, 0.001)$. Explain the different outcomes in terms of the Hessian's
eigenvalues, then show momentum escapes faster.

---

## Tier 3 — Interview

**Q1.** Why does gradient descent work? What exactly is it approximating, and what does the
learning rate represent?

**Q2.** Your training loss goes to `NaN` after a few hundred steps. Walk through your diagnosis.

**Q3.** Why do we standardize features? Give the optimization answer, not the "it's good practice"
answer.

**Q4.** What is the largest learning rate you can use, and what determines it?

**Q5.** Explain the difference between SGD, momentum, and Adam in two sentences each.

**Q6.** Why does momentum help? What convergence rate does it achieve, and is that optimal?

**Q7.** Why is Adam usually paired with warmup?

**Q8.** What is the difference between Adam and AdamW, and why does every modern transformer use
the latter?

**Q9.** Why don't we use Newton's method for deep learning?

**Q10.** What does convexity buy you? Which common ML models are convex?

**Q11.** Are neural networks plagued by bad local minima? Justify your answer with a counting
argument.

**Q12.** What are the KKT conditions, and which one explains why SVMs are sparse?

**Q13.** Why does L1 regularization produce exact zeros but L2 does not? Give both the geometric
and the proximal-operator answer.

**Q14.** Why does SGD need a decaying learning rate to converge? What happens with a constant one?

**Q15.** Your model's training loss is still decreasing but validation loss is rising. Is this an
optimization problem? What would you change?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive gradient descent from a Taylor expansion, and say what $\eta$ *means*
- [ ] Predict the divergence threshold for a learning rate from the Hessian
- [ ] Explain the zig-zag of GD in a narrow valley in terms of eigenvalues
- [ ] State what momentum buys and why it is optimal
- [ ] Write down Adam from memory, including — and justifying — bias correction
- [ ] Derive the SVM dual and explain where support vectors come from
- [ ] Explain L1 sparsity two independent ways
- [ ] Diagnose a failing training run from the loss curve alone
