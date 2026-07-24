# 05.01 — Exercises: Bias-Variance & Learning Theory

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Derive the bias-variance decomposition
$\mathrm{Err}(\mathbf{x}_0) = \sigma^2 + \mathrm{Bias}^2 + \mathrm{Var}$ from
$\mathbb{E}_{D,\varepsilon}[(y_0 - \hat f_D(\mathbf{x}_0))^2]$. Show explicitly why the two cross
terms vanish.

**D2.** State precisely what expectation bias and variance are taken over. Why is the decomposition a
statement about the *procedure*, not about a single fitted model?

**D3.** For ridge regression, show that increasing the penalty $\lambda$ raises bias and lowers
variance. Sketch both as functions of $\lambda$ and locate the total-error minimum.

**D4.** Explain why bagging reduces variance without increasing bias, and why boosting reduces bias.
Tie each to the decomposition (cross-reference [06.01](../../06-ensembles/01-bagging/) and
[06.03](../../06-ensembles/03-boosting-theory/)).

**D5.** *(Classification.)* Explain, following Domingos (2000), why under 0/1 loss variance can
*reduce* error at a point. On which points does variance help, and on which does it hurt?

**D6.** Prove Hoeffding's inequality's consequence: for a single fixed $h$,
$\mathbb{P}(|R(h)-\hat R(h)|>\epsilon)\le 2e^{-2n\epsilon^2}$. Then apply the union bound to get the
finite-class generalization bound and solve for the sample complexity $n(\epsilon,\delta)$.

**D7.** Why can ERM not use the single-$h$ Hoeffding bound directly? Explain the role of the union
bound and why the chosen hypothesis is exactly the one prone to a large gap.

**D8.** Define shattering and VC dimension. Prove that linear classifiers in $\mathbb{R}^2$ have
$d_{VC}=3$ by showing (a) some 3 points are shattered and (b) no 4 points are.

**D9.** Show that the classifier $\mathrm{sign}(\sin(\theta x))$, with one real parameter $\theta$,
has *infinite* VC dimension. What does this say about equating capacity with parameter count?

**D10.** State the fundamental theorem of statistical learning (PAC-learnable $\iff$ finite VC).
Why is finite VC dimension the exact boundary of learnability?

**D11.** *(Double descent.)* Explain why test error peaks at the interpolation threshold
($n_{\text{params}}\approx n$) and descends again beyond it. What replaces "parameter count" as the
right complexity axis?

---

## Tier 2 — Implementation

**I1.** Implement the Monte-Carlo bias-variance decomposer. Verify numerically that
noise + bias$^2$ + variance equals the measured total error to machine precision.

**I2.** Reproduce Experiment 2: sweep polynomial degree and plot bias$^2$, variance, and total.
Confirm the U and find the minimizing degree. Repeat with ridge and show the U's minimum shifting.

**I3.** Reproduce Experiment 3: show, for 0/1 loss, that variance adds error on unbiased points and
subtracts it on biased ones.

**I4.** Reproduce Experiment 4: draw the three learning-curve shapes with a stable learner (trees)
and read each. Confirm more data closes the variance gap but not the bias gap.

**I5.** Reproduce Experiment 5: measure the max generalization gap over an $M$-hypothesis class and
show it scaling as $\sqrt{\ln M / n}$; overlay the Hoeffding bound.

**I6.** Reproduce Experiment 6: implement the LP that tests linear separability, and count shattered
labellings for 3 and 4 points. Then find a 4-point set your code shatters — and prove you cannot.

**I7.** Reproduce Experiment 7 (double descent): min-norm least squares vs number of features, and
show the peak at $P=n$ and the second descent below the classical minimum.

**I8.** Add ridge regularization to Experiment 7 and show that a well-chosen $\lambda$ *removes* the
double-descent peak — the interpolation catastrophe is a ridgeless phenomenon.

**I9.** Empirically estimate the effective degrees of freedom of ridge regression,
$\mathrm{tr}(S)$ where $S$ is the hat matrix, and show it — not the parameter count — traces the
bias-variance U in Experiment 7's overparametrized regime.

**I10.** Build a validation curve (error vs a hyperparameter) and a learning curve (error vs $n$) for
one model, and explain what each tells you that the other does not.

---

## Tier 3 — Interview

**Q1.** What is the bias-variance decomposition?

**Q2.** Is bias/variance averaged over test points or over training sets?

**Q3.** Your model underfits. Will more data help?

**Q4.** How do you tell overfitting from underfitting using a learning curve?

**Q5.** Why does regularization help? Answer in bias-variance terms.

**Q6.** Does the additive bias$^2$+variance decomposition hold for classification?

**Q7.** Why does fitting a finite sample tell you anything about unseen data?

**Q8.** What is VC dimension, and why does finite VC matter?

**Q9.** Is VC dimension the same as the number of parameters?

**Q10.** A huge over-parametrized network generalizes well. Doesn't that contradict the
bias-variance tradeoff?

**Q11.** What is double descent, and where is the peak?

**Q12.** How do bagging and boosting sit in the bias-variance picture?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive the decomposition and explain what the averaging is over
- [ ] Diagnose bias vs variance from a learning curve and know which data cures
- [ ] Explain why classification's decomposition is not additive
- [ ] State the Hoeffding and VC generalization bounds and what they imply about $n$
- [ ] Compute the VC dimension of linear classifiers and explain shattering
- [ ] Explain double descent and why parameter count is not the right complexity axis
- [ ] See regularization, ensembling, and more data as moves in one bias-variance game
