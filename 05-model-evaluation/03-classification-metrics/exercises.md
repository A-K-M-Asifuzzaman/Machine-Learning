# 05.03 — Exercises: Classification Metrics

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Write the confusion matrix and define accuracy, precision, recall, specificity, and NPV in
terms of TP/FP/TN/FN. Which cells does each ignore?

**D2.** Show that on 1%-positive data the all-negative classifier has 99% accuracy, 0 recall, and
MCC = 0. Explain why MCC is not fooled.

**D3.** Derive why F1 is the *harmonic* mean of precision and recall, and show it is dominated by the
smaller of the two. Contrast with the arithmetic mean on P=1.0, R=0.01.

**D4.** Derive Fβ and show that $\beta$ is the factor by which recall is weighted over precision.
What are $F_2$ and $F_{0.5}$ for?

**D5.** Prove that AUC equals $P(\text{score of a random positive} > \text{score of a random
negative})$ — the Wilcoxon-Mann-Whitney identity. Start from the area under the ROC curve.

**D6.** Explain why AUC is prevalence-independent (resampling class balance does not change it) but
average precision is prevalence-dependent. Which do you want under heavy imbalance, and why?

**D7.** Show that the PR curve's baseline (random classifier) is the prevalence, while the ROC's
baseline is the diagonal (AUC 0.5).

**D8.** Prove that log loss and Brier are *proper* scoring rules: the expected score is minimized by
reporting the true probability. (Differentiate $\mathbb{E}_{y\sim p}[L(y, q)]$ w.r.t. $q$.)

**D9.** Derive the Bayes-optimal decision threshold $t^\star = c_{FP}/(c_{FP}+c_{FN})$ for a
calibrated classifier by minimizing expected cost per example.

**D10.** Define MCC as a correlation coefficient between the binary prediction and label vectors, and
show it reduces to the $2\times2$ formula. Why is it in $[-1,1]$?

**D11.** Define Cohen's kappa and explain the chance-correction term $p_e$. When is kappa much lower
than accuracy?

**D12.** Explain the difference between macro, micro, and weighted averaging for multiclass F1, and
show micro-F1 equals accuracy for single-label problems.

---

## Tier 2 — Implementation

**I1.** Implement the confusion matrix and all its ratios (accuracy, precision, recall, specificity,
NPV, F1, Fβ, MCC, kappa). Verify against `sklearn.metrics`.

**I2.** Implement the ROC curve and AUC (both by the curve integral and by pair-counting) and the PR
curve with average precision. Verify against sklearn, handling ties correctly.

**I3.** Reproduce Experiment 1: the accuracy paradox — the all-negative classifier on 1%-positive
data scoring 99% accuracy but 0 on recall/F1/MCC.

**I4.** Reproduce Experiment 2: sweep the threshold and trace the precision-recall tradeoff.

**I5.** Reproduce Experiment 3: verify AUC = the ranking probability three ways (curve integral,
exhaustive pair count, Monte-Carlo sampling of pairs).

**I6.** Reproduce Experiment 4: construct imbalanced data where AUC looks great (~0.9) but average
precision is mediocre. Plot both curves.

**I7.** Reproduce Experiment 5: with *calibrated* scores, find the cost-optimal threshold and confirm
it matches $c_{FP}/(c_{FP}+c_{FN})$. Then show it does NOT match for uncalibrated scores.

**I8.** Reproduce Experiment 6: build two models with identical accuracy and AUC but different log
loss / Brier (a monotonic temperature transform of the logits).

**I9.** Implement log loss and Brier, and Murphy's decomposition of Brier into
reliability − resolution + uncertainty. Confirm the three terms sum to the Brier score.

**I10.** Implement multiclass macro/micro/weighted F1 and verify against sklearn on an imbalanced
3-class problem. Show macro < micro when the rare class is hard.

**I11.** *(Threshold selection.)* Given a validation set and a cost matrix, write a function that
returns the expected-cost-minimizing threshold, and apply it end-to-end.

---

## Tier 3 — Interview

**Q1.** Why is accuracy misleading under class imbalance?

**Q2.** Explain precision vs recall. When does each matter?

**Q3.** Why is F1 a harmonic mean?

**Q4.** What does AUC actually measure?

**Q5.** When would you use a PR curve instead of ROC?

**Q6.** Is a high AUC enough under heavy imbalance?

**Q7.** What threshold should you use, and is 0.5 special?

**Q8.** What's a proper scoring rule, and why does it matter?

**Q9.** Your model has AUC 0.99 but terrible probabilities. How is that possible?

**Q10.** What is MCC and why might you prefer it to F1?

**Q11.** Macro vs micro F1 — what's the difference?

**Q12.** A model has 95% accuracy on fraud detection. Are you impressed?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive every metric from the confusion matrix and know which cells each ignores
- [ ] Explain the accuracy paradox and name metrics immune to it
- [ ] State the AUC ranking interpretation and prove it
- [ ] Choose ROC vs PR from the class balance and the question asked
- [ ] Set a threshold from the cost ratio, not habit
- [ ] Explain why a proper scoring rule sees what accuracy and AUC miss
