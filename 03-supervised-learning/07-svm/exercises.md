# 03.07 — Exercises: Support Vector Machines

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Show that the functional margin can be scaled arbitrarily without changing the hyperplane,
and that the geometric margin cannot. Why does this force the normalization
$\min_i y_i(\mathbf{w}^{\top}\mathbf{x}_i+b)=1$?

**D2.** Show the margin width is $2/\Vert\mathbf{w}\Vert$, and hence that maximizing the margin is
minimizing $\Vert\mathbf{w}\Vert$.

**D3.** Write the Lagrangian for the hard-margin primal, take $\partial/\partial\mathbf{w}$ and
$\partial/\partial b$, and derive $\mathbf{w}=\sum_i\alpha_iy_i\mathbf{x}_i$ and
$\sum_i\alpha_iy_i=0$.

**D4.** Substitute those back to derive the dual in full. Verify the $b$ terms cancel and that the
data appears only as $\mathbf{x}_i^{\top}\mathbf{x}_j$.

**D5.** State complementary slackness for this problem and **derive** the existence of support
vectors from it. Why is this a theorem rather than a definition?

**D6.** Derive the soft-margin dual and show the only change is $\alpha_i\ge0 \to 0\le\alpha_i\le C$.

**D7.** Classify the three cases $\alpha_i=0$, $0<\alpha_i<C$, $\alpha_i=C$ by where the point
sits relative to the margin. Which are "free" and which "bounded"?

**D8.** Eliminate the slack variables to derive the hinge-loss form
$\sum_i\max(0,1-y_if_i)+\frac{1}{2C}\Vert\mathbf{w}\Vert^{2}$.

**D9.** Compare the gradients of hinge and log loss at $yf=2$. Explain how the difference produces
sparsity in one and not the other.

**D10.** For $k(\mathbf{x},\mathbf{z})=(\mathbf{x}^{\top}\mathbf{z})^{2}$ in $\mathbb{R}^{2}$,
find $\phi$ explicitly. Then count the dimension of $\phi$ for
$(\mathbf{x}^{\top}\mathbf{z})^{p}$ in $\mathbb{R}^{d}$.

**D11.** Show the RBF kernel corresponds to an infinite-dimensional feature map. *Hint*: expand
$e^{2\gamma\mathbf{x}^{\top}\mathbf{z}}$ as a power series.

**D12.** State Mercer's condition. Prove that if $k_1,k_2$ are valid kernels then so are
$k_1+k_2$, $k_1k_2$, and $ck_1$ for $c>0$.

**D13.** Show that the sigmoid kernel is not PSD for some parameter values, and explain the
consequence for the dual.

**D14.** Explain why SMO optimizes **two** multipliers at a time and not one. What role does
$\sum_i\alpha_iy_i=0$ play?

**D15.** Derive the box bounds $[L,H]$ for $\alpha_j$ in the SMO subproblem, treating the cases
$y_i=y_j$ and $y_i\ne y_j$ separately.

**D16.** Derive the $\epsilon$-insensitive loss's dual for SVR, and show that points strictly
inside the tube have zero dual coefficient.

---

## Tier 2 — Implementation

**I1.** Implement the five kernels. Verify each Gram matrix is symmetric PSD, and find parameter
values making the sigmoid kernel indefinite.

**I2.** Verify the kernel trick numerically: check
$(\mathbf{x}^{\top}\mathbf{z})^{2}=\phi(\mathbf{x})^{\top}\phi(\mathbf{z})$ to machine precision
with your explicit $\phi$ from D10.

**I3.** Implement SMO with **random** pair selection. Then implement Platt's heuristic (maximize
$|E_i-E_j|$ over non-bound points) and compare KKT violations at a fixed iteration budget.

**I4.** Implement `kkt_violations()`. Verify all three conditions hold at the solution for several
$C$, and confirm $\sum_i\alpha_iy_i=0$ to machine precision.

**I5.** Reproduce Experiment 1: refit on support vectors only and confirm the model is unchanged.
Then find the `tol` at which the $C=100$ case becomes exact.

**I6.** Reproduce Experiment 2. Plot margin width and support-vector count against $C$ on log
axes, and locate the test-accuracy peak.

**I7.** Reproduce Experiment 3 on XOR, circles, and moons. Then add a *polynomial* kernel of
degree 2 to XOR and explain why it succeeds where degree 3 may not.

**I8.** Reproduce Experiment 5's $C$-$\gamma$ grid. Then fix $\gamma$ at a bad value, search $C$
alone, and show the conclusion you would have drawn.

**I9.** Implement SVR with the $\epsilon$-tube. Plot the fitted curve, the tube, and the support
vectors. Verify points strictly inside the tube have zero coefficient.

**I10.** Implement one-vs-one multiclass on top of your binary SVC. Compare training time and
accuracy against one-vs-rest on a 4-class problem, and explain why OvO is often faster despite
training more classifiers.

**I11.** *(Platt scaling.)* Fit a logistic regression to your SVM's decision values and compare
the resulting probabilities' calibration against a directly-fitted logistic regression
([05.06](../../05-model-evaluation/06-calibration/)).

**I12.** Compare `LinearSVC` (primal, liblinear) against `SVC(kernel="linear")` (dual, libsvm) on
a text dataset with $d\gg n$ and then with $n\gg d$. Explain the timing difference from §4's
observation about which variable count each solves for.

**I13.** *(The representer theorem, empirically.)* Implement kernel ridge regression and verify it
gives the same predictions as a smoothing spline ([03.03 §8](../03-basis-expansion/)) for a
suitable kernel.

---

## Tier 3 — Interview

**Q1.** What does an SVM optimize, and why that rather than accuracy?

**Q2.** Derive the dual. What two things change relative to the primal?

**Q3.** What is a support vector? Why do they exist — derive it.

**Q4.** If I delete a non-support-vector from the training set and refit, what happens?

**Q5.** What does $C$ control? Which direction is more regularization?

**Q6.** What is the kernel trick, and why is it only possible in the dual?

**Q7.** What makes a function a valid kernel?

**Q8.** Why does the RBF kernel have an infinite-dimensional feature space, and why is that not a
problem?

**Q9.** Do SVMs output probabilities?

**Q10.** How do hinge loss and logistic loss differ, and what does the difference buy each?

**Q11.** How would you tune $C$ and $\gamma$? Why not one at a time?

**Q12.** Why is SMO's subproblem solvable in closed form?

**Q13.** Your SVM has 95% of the training set as support vectors. What does that tell you?

**Q14.** Why do SVMs scale badly to a million samples? What would you use instead?

**Q15.** When would you choose a linear kernel over RBF?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive the dual from the primal without notes
- [ ] Derive support vectors from complementary slackness, not assert them
- [ ] Explain why the kernel trick needs the dual
- [ ] State Mercer's condition and why PSD is the right requirement
- [ ] Explain sparsity from *both* the KKT side and the loss side
- [ ] Say what $C$ and $\gamma$ each do, and why they must be tuned together
- [ ] Explain why SMO uses exactly two multipliers
