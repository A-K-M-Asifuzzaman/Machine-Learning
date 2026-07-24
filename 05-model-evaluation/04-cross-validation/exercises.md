# 05.04 — Exercises: Cross-Validation & Model Selection

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Explain why a single train/test split gives a high-variance estimate of generalization, and
why $K$-fold reduces that variance. What does the per-fold spread tell you?

**D2.** Explain why $K$-fold CV is *pessimistically* biased as an estimate of the final model's
performance. How does the bias depend on $K$?

**D3.** Explain why the variance of the CV *estimate* rises as $K\to n$ (LOO). (Hint: correlation
between the fold models.)

**D4.** Derive the LOOCV closed-form shortcut for a linear smoother:
$\mathrm{LOOCV} = \frac1n\sum_i \left(\frac{y_i - \hat y_i}{1 - H_{ii}}\right)^2$, where $H$ is the
hat matrix. Why does the leverage $H_{ii}$ appear?

**D5.** State the generalized cross-validation (GCV) approximation to LOOCV and explain when it is
used (e.g. tuning ridge).

**D6.** Explain the one-standard-error rule and justify it as a bias-variance argument about the
model-*selection* process.

**D7.** *(Leakage.)* Prove that selecting the $k$ features most correlated with $y$ on the full
dataset, then cross-validating, gives an optimistic estimate even when $X \perp y$. Where exactly
does the fold's test label enter training?

**D8.** *(Nested CV.)* Explain why reporting the best-of-many CV scores after hyperparameter search
is optimistically biased, and how nested CV removes the bias. What does the outer score estimate?

**D9.** Explain why random $K$-fold overestimates performance on autocorrelated time series, and why
forward chaining fixes it. Relate to the notion of exchangeability.

**D10.** Explain why grouped data (multiple rows per entity) needs `GroupKFold`, and what leaks if
you use plain $K$-fold.

---

## Tier 2 — Implementation

**I1.** Implement `KFold`, `StratifiedKFold`, `LeaveOneOut`, and `cross_val_score`. Verify the CV
mean against `sklearn.model_selection.cross_val_score`.

**I2.** Implement the LOOCV closed-form shortcut for OLS/ridge via the hat matrix and verify it
matches brute-force LOOCV to machine precision.

**I3.** Reproduce Experiment 1: show the CV estimate's standard deviation is a fraction of a single
split's, over many datasets.

**I4.** Reproduce Experiment 2: measure bias and variance of the CV estimate vs $K$. Identify the
sweet spot.

**I5.** Reproduce Experiment 3: build pure-noise data, "validate" with feature selection outside the
fold (fake 80–90%), then move selection inside the fold (chance). This is the most important
implementation exercise in the chapter.

**I6.** Reproduce Experiment 4: implement nested CV and measure the optimism gap against naive CV.
Increase the number of hyperparameter candidates and show the gap grow.

**I7.** Reproduce Experiment 5: show plain $K$-fold producing zero-positive folds on imbalanced data
and stratified $K$-fold fixing it.

**I8.** Reproduce Experiment 6: two independent random walks; show random $K$-fold fabricating a
positive $R^2$ and forward chaining exposing the truth.

**I9.** Implement `GroupKFold` and show a synthetic example where plain $K$-fold leaks (same group in
train and test) but GroupKFold does not.

**I10.** Wrap a scaler + feature selector + classifier in a single pipeline and cross-validate the
*pipeline*. Show it gives the same (honest) score as manual inside-fold preprocessing, and a
different (optimistic) score when you preprocess first.

**I11.** *(Repeated CV.)* Implement repeated stratified $K$-fold and show the estimate's standard
error shrinking with the number of repeats.

---

## Tier 3 — Interview

**Q1.** Why not just use a single train/test split?

**Q2.** How do you choose $K$?

**Q3.** When is leave-one-out worth it?

**Q4.** You scaled your features before cross-validating. What's wrong?

**Q5.** Your CV accuracy after grid search is 92%. Is that your expected deployment accuracy?

**Q6.** What is nested cross-validation and when do you need it?

**Q7.** How do you cross-validate a time series?

**Q8.** You have multiple records per patient. How do you split?

**Q9.** Why stratify?

**Q10.** What is the one-standard-error rule?

**Q11.** A feature-selection step made a random dataset look 90% accurate under CV. How?

**Q12.** What is the difference between the validation set and the test set?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Explain the bias-variance-compute tradeoff behind choosing $K$
- [ ] Derive and use the LOOCV hat-matrix shortcut
- [ ] Spot and fix leakage — put every learned step inside the fold
- [ ] Explain and implement nested CV, and why tuning needs its own held-out data
- [ ] Choose the right splitter for time-series and grouped data
- [ ] Read a validation curve with the one-standard-error rule
