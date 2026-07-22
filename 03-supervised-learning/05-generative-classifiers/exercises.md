# 03.05 — Exercises: Generative Classifiers

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** State the difference between modelling $p(y\mid\mathbf{x})$ and modelling
$p(\mathbf{x}\mid y)p(y)$. List three things the generative route gives you that the
discriminative one does not.

**D2.** Derive the Bayes optimal decision rule and explain why $p(\mathbf{x})$ drops out.

**D3.** Show that modelling $p(\mathbf{x}\mid y)$ for $d$ binary features needs $2^{d}-1$
parameters per class, and that the naive assumption reduces this to $d$.

**D4.** Derive the naive Bayes decision rule in log space, and explain why the log is not
optional.

**D5.** Show that add-$\alpha$ smoothing is the posterior mean under a $\mathrm{Dirichlet}(\alpha)$
prior. What does $\alpha$ mean in units of data?

**D6.** *(Why NB works.)* Construct a two-feature example where the naive Bayes posterior is badly
wrong but its argmax is correct. Then construct one where the argmax flips. What distinguishes
them?

**D7.** Explain, using the log-odds, why duplicating an informative feature $m$ times inflates
naive Bayes's log-odds by roughly a factor of $m$.

**D8.** Derive the LDA discriminant $\delta_k(\mathbf{x})$. Show explicitly which term cancels
because $\boldsymbol{\Sigma}$ is shared, and hence why the boundary is linear.

**D9.** Derive the QDA discriminant. Show the term that fails to cancel and gives a quadratic
boundary.

**D10.** Count the parameters of Gaussian NB, diagonal LDA, LDA, and QDA. At $d=50$, $K=3$, how
many samples per class would you want for each?

**D11.** Show that Gaussian naive Bayes is exactly QDA with diagonal covariances. Then place all
four models of §9 on a single nesting chain.

**D12.** Show that LDA's two-class log-odds are linear in $\mathbf{x}$, i.e. that LDA and logistic
regression share a hypothesis class. Then list three ways they still differ.

**D13.** Derive Fisher's criterion and show its solution is the top eigenvector of
$\mathbf{S}_W^{-1}\mathbf{S}_B$. Why are there at most $K-1$ useful directions?

**D14.** Explain why LDA's pooled covariance divides by $n-K$ rather than $n$, and what changes if
you use $n$ (as sklearn does).

**D15.** State Ng & Jordan's result precisely. Then explain why it does **not** imply "naive Bayes
always wins at small $n$".

---

## Tier 2 — Implementation

**I1.** Implement `GaussianNB` in log space. Verify log-probabilities against sklearn to $10^{-9}$.

**I2.** Implement `MultinomialNB` with add-$\alpha$ smoothing. Verify against sklearn for
$\alpha \in \{0.1, 1.0\}$, and show that $\alpha \to 0$ produces $-\infty$ log-probabilities.

**I3.** Implement `BernoulliNB`. On short documents, compare it against `MultinomialNB` and explain
the difference in terms of how each treats absent features.

**I4.** Implement LDA. Verify the log-odds are **exactly affine** in $\mathbf{x}$ by regressing
them on $[1, \mathbf{x}]$ and checking the residual is at machine precision.

**I5.** Implement QDA. Verify its log-odds are **not** affine. Compute $\log|\boldsymbol{\Sigma}_k|$
from the Cholesky factor and explain why not from `np.linalg.det`.

**I6.** Implement RDA. Verify $\gamma=0$ reproduces LDA and $\gamma=1$ reproduces QDA, then
cross-validate $\gamma$ on a dataset where neither extreme is best.

**I7.** Reproduce Experiment 1. Find the correlation at which naive Bayes's accuracy gap to LDA
first exceeds 2 percentage points.

**I8.** Reproduce Experiment 2. Then apply `sklearn.calibration.CalibratedClassifierCV` to the
naive Bayes model and show it repairs the calibration without changing accuracy.

**I9.** Reproduce Experiment 3's covariance spectrum. For your own $d$, find empirically the
$n_k$ at which QDA overtakes LDA, and compare it against the $d^{2}/2$ rule of thumb.

**I10.** Reproduce Experiment 4's two-dimensional sweep. Find a $(\rho, n)$ cell where naive Bayes
wins and a nearby one where it loses, and explain the boundary between them.

**I11.** Implement Fisher's LDA projection. Reproduce Experiment 5 by constructing data where the
top PCA direction is orthogonal to the discriminant direction.

**I12.** *(Text baseline.)* Fit `MultinomialNB` and TF-IDF + logistic regression on a text dataset
(e.g. 20 newsgroups). Compare accuracy, training time, and calibration. Then subsample the
training set down to 100 documents and repeat.

**I13.** *(Missing features.)* Take a trained Gaussian NB and predict on inputs with some features
missing, by *marginalizing them out* (simply omitting their log-likelihood terms). Show this is
principled, and that a logistic regression has no equivalent.

---

## Tier 3 — Interview

**Q1.** What is the difference between a generative and a discriminative classifier?

**Q2.** What exactly does naive Bayes assume? Is the assumption ever true?

**Q3.** If the assumption is false, why does it work?

**Q4.** Can you trust naive Bayes's predicted probabilities?

**Q5.** What is Laplace smoothing and why is it not a hack?

**Q6.** What makes LDA's decision boundary linear?

**Q7.** When would you use QDA over LDA, and when would that be a mistake?

**Q8.** How are Gaussian naive Bayes, LDA, and QDA related?

**Q9.** LDA and logistic regression have the same functional form. Why would you pick one over the
other?

**Q10.** What is LDA used for besides classification?

**Q11.** Why can LDA give at most $K-1$ components?

**Q12.** When would PCA before classification hurt you?

**Q13.** Is a discriminative model always better with enough data? What about without enough?

**Q14.** You have 200 labelled documents and 20,000 features. What do you try first?

**Q15.** How would a generative model detect an out-of-distribution input?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] State the generative/discriminative tradeoff in both directions
- [ ] Explain why NB's argmax survives an assumption its probabilities do not
- [ ] Derive both the LDA and QDA discriminants and say which term cancels and why
- [ ] Place NB, LDA, and QDA on one covariance spectrum and pick between them from $n$ and $d$
- [ ] Explain the LDA/logistic-regression relationship — same form, different fit, different failures
- [ ] Say what LDA-as-projection optimizes and how it differs from PCA
- [ ] State Ng & Jordan's result *without* overclaiming it
