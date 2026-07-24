# 05.06 — Exercises: Calibration

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Define calibration formally ($\mathbb{P}(y=1\mid\hat p=p)=p$) and explain why it is independent
of discrimination (AUC). Give a transformation that preserves AUC but destroys calibration.

**D2.** Define the reliability diagram, ECE, and MCE. Why do they depend on the binning?

**D3.** State Murphy's decomposition of the Brier score into reliability − resolution + uncertainty,
and prove it is exact for a forecast taking finitely many distinct values.

**D4.** Explain why logistic regression is calibrated by construction. (Hint: the log-loss stationarity
conditions.)

**D5.** Explain why bagging (random forests) produces *under*confident probabilities and boosting
produces *over*confident ones, in terms of how each combines base learners.

**D6.** Derive the Pool Adjacent Violators (PAV) algorithm as the exact solution to isotonic
regression (non-decreasing least squares). Why does pooling adjacent violators preserve optimality?

**D7.** Show that temperature scaling ($\hat p = \mathrm{softmax}(\mathbf z / T)$) preserves the
argmax, and hence accuracy, for any $T > 0$.

**D8.** Show that temperature scaling is the special case of Platt scaling with a shared slope and
zero intercept across classes.

**D9.** Explain the bias-variance tradeoff between Platt scaling and isotonic regression. When does
each win?

**D10.** Explain why a base-rate predictor is perfectly calibrated yet useless, and why "sharp,
subject to calibration" (proper scoring rules) is the right objective.

---

## Tier 2 — Implementation

**I1.** Implement the reliability diagram, ECE, and MCE. Plot the diagram for a miscalibrated model.

**I2.** Implement Murphy's Brier decomposition and verify (on a binned forecast) that
reliability − resolution + uncertainty equals the Brier score to machine precision.

**I3.** Implement Platt scaling (1-D logistic regression) and isotonic regression via PAV. Verify
isotonic against `sklearn.isotonic.IsotonicRegression`.

**I4.** Reproduce Experiment 1: a high-AUC miscalibrated model; recalibrate and confirm ECE drops
while AUC is unchanged.

**I5.** Reproduce Experiment 2: measure ECE and the confidence profile (fraction extreme vs mid) for
logistic, random forest, and boosting, and read off the under/over-confidence signatures.

**I6.** Reproduce Experiment 3: compare Platt and isotonic across calibration-set sizes for a
sigmoidal and a non-sigmoidal distortion; show each winning in its regime.

**I7.** Implement temperature scaling and reproduce Experiment 4: ECE drops, accuracy and AUC are
bit-for-bit unchanged.

**I8.** Reproduce Experiment 5: calibrate an overfitting model on its training predictions (leak) vs
held-out predictions, and compare test ECE.

**I9.** Reproduce Experiment 6: show a base-rate predictor with ECE ≈ 0, AUC 0.5, and zero resolution.

**I10.** Implement cross-validated calibration (à la `CalibratedClassifierCV`): out-of-fold
predictions to fit the calibrator, so no data is wasted. Compare to a single held-out split.

**I11.** *(Multiclass.)* Extend temperature scaling to multiclass softmax and calibrate a small neural
net's logits; report top-1 ECE before and after.

---

## Tier 3 — Interview

**Q1.** What does it mean for a classifier to be calibrated?

**Q2.** How is calibration different from AUC?

**Q3.** When does calibration matter and when does it not?

**Q4.** How do you measure calibration?

**Q5.** Which common models are miscalibrated, and in which direction?

**Q6.** Platt scaling vs isotonic regression — how do you choose?

**Q7.** What is temperature scaling and why is it popular for deep nets?

**Q8.** Why must the calibrator be fit on held-out data?

**Q9.** Does recalibration change accuracy?

**Q10.** A model is perfectly calibrated. Is it a good model?

**Q11.** Your neural net says 99% confident but is right 90% of the time. What do you do?

**Q12.** Why are log loss and Brier better final metrics than ECE?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Distinguish calibration from discrimination and explain why they are orthogonal
- [ ] Measure calibration with a reliability diagram and ECE
- [ ] Predict a model family's miscalibration direction from its loss
- [ ] Implement and choose between Platt, isotonic, and temperature scaling
- [ ] Fit a calibrator without leakage
- [ ] Explain why calibration needs sharpness, and why proper scoring rules are the target
