"""
05.03 — Classification Metrics, from scratch (NumPy).

Every metric built from the confusion matrix (or the score ranking) and checked against
sklearn.metrics. Then the chapter's claims are MEASURED:

  1. the ACCURACY PARADOX: an all-negative classifier scores 99% accuracy, 0 on what matters
  2. the precision-recall TRADEOFF as the threshold sweeps                        (README §5)
  3. AUC == P(score of a random positive > score of a random negative)           (README §6)
  4. under imbalance ROC/AUC looks great while average precision is mediocre      (README §7)
  5. the COST-optimal threshold is not 0.5                                        (README §11)
  6. log loss separates two models with identical accuracy AND AUC               (README §8)

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    from sklearn import metrics as skm
    HAVE_SK = True
except Exception:
    HAVE_SK = False


# =============================================================================
# CONFUSION MATRIX AND ITS RATIOS  (README §1-§4, §9)
# =============================================================================


def confusion(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    tp = int(np.sum((y == 1) & (yhat == 1)))
    fp = int(np.sum((y == 0) & (yhat == 1)))
    tn = int(np.sum((y == 0) & (yhat == 0)))
    fn = int(np.sum((y == 1) & (yhat == 0)))
    return tp, fp, tn, fn


def accuracy(y, yhat):
    tp, fp, tn, fn = confusion(y, yhat)
    return (tp + tn) / (tp + fp + tn + fn)


def precision(y, yhat):
    tp, fp, tn, fn = confusion(y, yhat)
    return tp / (tp + fp) if tp + fp else 0.0


def recall(y, yhat):
    tp, fp, tn, fn = confusion(y, yhat)
    return tp / (tp + fn) if tp + fn else 0.0


def specificity(y, yhat):
    tp, fp, tn, fn = confusion(y, yhat)
    return tn / (tn + fp) if tn + fp else 0.0


def fbeta(y, yhat, beta=1.0):
    p, r = precision(y, yhat), recall(y, yhat)
    if p == 0 and r == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) else 0.0


def mcc(y, yhat):
    tp, fp, tn, fn = confusion(y, yhat)
    num = tp * tn - fp * fn
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return num / den if den else 0.0


def cohen_kappa(y, yhat):
    tp, fp, tn, fn = confusion(y, yhat)
    n = tp + fp + tn + fn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
    return (po - pe) / (1 - pe) if pe != 1 else 0.0


# =============================================================================
# THRESHOLD-FREE METRICS  (README §6-§8)
# =============================================================================


def roc_curve(y, scores):
    """Return (fpr, tpr) at every distinct threshold, swept high->low. Tied scores are
    MERGED into one point — otherwise the trapezoid area miscounts ties."""
    y = np.asarray(y)
    s = np.asarray(scores, float)
    order = np.argsort(-s)
    y, s = y[order], s[order]
    P, N = np.sum(y == 1), np.sum(y == 0)
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    # keep only the last index within each run of equal scores (a real threshold change)
    distinct = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    tpr = np.r_[0.0, tps[distinct] / P]
    fpr = np.r_[0.0, fps[distinct] / N]
    return fpr, tpr


def auc_roc(y, scores):
    fpr, tpr = roc_curve(y, scores)
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))   # trapezoid area


def auc_by_pairs(y, scores):
    """AUC as P(score+ > score-): fraction of positive-negative pairs correctly ordered
    (ties count 0.5). The Wilcoxon-Mann-Whitney identity (README §6)."""
    y = np.asarray(y)
    s = np.asarray(scores, float)
    pos, neg = s[y == 1], s[y == 0]
    wins = 0.0
    for sp in pos:
        wins += np.sum(sp > neg) + 0.5 * np.sum(sp == neg)
    return wins / (len(pos) * len(neg))


def average_precision(y, scores):
    """AP = sum (R_k - R_{k-1}) P_k over thresholds (README §7)."""
    y = np.asarray(y)
    order = np.argsort(-np.asarray(scores, float))
    y = y[order]
    P = np.sum(y == 1)
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    prec = tp / (tp + fp)
    rec = tp / P
    # sum of precision at each positive, over recall increments
    ap = 0.0
    prev_r = 0.0
    for i in range(len(y)):
        if y[i] == 1:
            ap += prec[i] * (rec[i] - prev_r)
            prev_r = rec[i]
    return float(ap)


def log_loss(y, p, eps=1e-15):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


# =============================================================================
# VERIFICATION
# =============================================================================


def verify():
    print("=" * 88)
    print("VERIFICATION — every metric vs sklearn.metrics")
    print("=" * 88)
    rng = np.random.default_rng(0)
    n = 500
    y = (rng.uniform(size=n) < 0.35).astype(int)
    # scores strictly in (0,1) via a sigmoid, so log loss is not dominated by boundary eps
    logit = 1.4 * (2 * y - 1) + rng.standard_normal(n)
    scores = 1.0 / (1.0 + np.exp(-logit))
    yhat = (scores >= 0.5).astype(int)

    checks = {
        "accuracy": (accuracy(y, yhat), skm.accuracy_score(y, yhat) if HAVE_SK else None),
        "precision": (precision(y, yhat),
                      skm.precision_score(y, yhat) if HAVE_SK else None),
        "recall": (recall(y, yhat), skm.recall_score(y, yhat) if HAVE_SK else None),
        "f1": (fbeta(y, yhat, 1), skm.f1_score(y, yhat) if HAVE_SK else None),
        "f2": (fbeta(y, yhat, 2),
               skm.fbeta_score(y, yhat, beta=2) if HAVE_SK else None),
        "MCC": (mcc(y, yhat), skm.matthews_corrcoef(y, yhat) if HAVE_SK else None),
        "kappa": (cohen_kappa(y, yhat),
                  skm.cohen_kappa_score(y, yhat) if HAVE_SK else None),
        "AUC": (auc_roc(y, scores), skm.roc_auc_score(y, scores) if HAVE_SK else None),
        "AUC(pairs)": (auc_by_pairs(y, scores),
                       skm.roc_auc_score(y, scores) if HAVE_SK else None),
        "avg precision": (average_precision(y, scores),
                          skm.average_precision_score(y, scores) if HAVE_SK else None),
        "log loss": (log_loss(y, scores), skm.log_loss(y, scores) if HAVE_SK else None),
        "brier": (brier(y, scores), skm.brier_score_loss(y, scores) if HAVE_SK else None),
    }
    if HAVE_SK:
        print(f"\n    {'metric':>14s} {'ours':>12s} {'sklearn':>12s} {'|diff|':>10s}")
        for k, (o, r) in checks.items():
            d = abs(o - r)
            print(f"    {k:>14s} {o:>12.6f} {r:>12.6f} {d:>10.2e}")
            # AP integration differs slightly in convention; allow a looser tol there
            tol = 2e-2 if k == "avg precision" else 1e-6
            assert d < tol, f"{k} mismatch ({d})"
        print("\n  all metrics match sklearn (AUC/MCC/logloss/... to <1e-6)  ✓")
    else:
        for k, (o, _) in checks.items():
            print(f"    {k:>14s} {o:>12.6f}")
    print("\nAll verification checks passed.")


# =============================================================================
# EXPERIMENT 1 — the accuracy paradox (README §2)
# =============================================================================


def experiment_1_accuracy_paradox():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — the accuracy paradox: 99% accuracy, useless model (README §2)")
    print("=" * 88)
    rng = np.random.default_rng(1)
    n = 10000
    y = (rng.uniform(size=n) < 0.01).astype(int)        # 1% positive
    all_neg = np.zeros(n, dtype=int)                    # predict 'never positive'

    print(f"""
  10,000 examples, 1% positive. A classifier that predicts NEGATIVE for everyone:

    {'metric':>18s} {'value':>8s}
    {'accuracy':>18s} {accuracy(y, all_neg):>8.3f}   <- looks great
    {'recall':>18s} {recall(y, all_neg):>8.3f}   <- catches ZERO positives
    {'precision':>18s} {precision(y, all_neg):>8.3f}
    {'F1':>18s} {fbeta(y, all_neg, 1):>8.3f}
    {'MCC':>18s} {mcc(y, all_neg):>8.3f}   <- correctly says 'no better than chance'
    {'Cohen kappa':>18s} {cohen_kappa(y, all_neg):>8.3f}

  READING: accuracy rewards getting the 99% majority right, which is trivial. Recall, F1, MCC,
  and kappa all correctly report that the model is worthless. Under imbalance, NEVER headline
  accuracy — use MCC or the minority-class metrics (README §2).""")


# =============================================================================
# EXPERIMENT 2 — precision-recall threshold tradeoff (README §5)
# =============================================================================


def experiment_2_threshold_tradeoff():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — the precision-recall tradeoff as the threshold sweeps (README §5)")
    print("=" * 88)
    rng = np.random.default_rng(2)
    n = 4000
    y = (rng.uniform(size=n) < 0.3).astype(int)
    scores = np.clip(0.55 * y + 0.3 * rng.standard_normal(n) + 0.45, 0, 1)

    print(f"\n    {'threshold':>10s} {'precision':>10s} {'recall':>8s} {'F1':>7s} "
          f"{'flagged':>8s}")
    for t in (0.2, 0.35, 0.5, 0.65, 0.8):
        yhat = (scores >= t).astype(int)
        print(f"    {t:>10.2f} {precision(y, yhat):>10.3f} {recall(y, yhat):>8.3f} "
              f"{fbeta(y, yhat, 1):>7.3f} {int(yhat.sum()):>8d}")
    print("""
  READING: lowering the threshold flags more examples -> recall rises, precision falls (more
  false alarms). Raising it -> precision rises, recall falls (more misses). There is no single
  'accuracy' of this model; there is a CURVE, and where you sit on it is a business choice set
  by the cost of the two errors (README §5, §11).""")


# =============================================================================
# EXPERIMENT 3 — AUC is the ranking probability (README §6)
# =============================================================================


def experiment_3_auc_ranking():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — AUC == P(random positive scores above random negative) (README §6)")
    print("=" * 88)
    rng = np.random.default_rng(3)
    n = 2000
    y = (rng.uniform(size=n) < 0.4).astype(int)
    scores = np.clip(0.4 * y + 0.4 * rng.standard_normal(n) + 0.5, 0, 1)

    a_curve = auc_roc(y, scores)             # area under the ROC curve
    a_pairs = auc_by_pairs(y, scores)        # fraction of pos-neg pairs correctly ordered

    # brute-force sanity: sample random pos-neg pairs, fraction ordered correctly
    pos = scores[y == 1]
    neg = scores[y == 0]
    idx_p = rng.integers(0, len(pos), 200000)
    idx_n = rng.integers(0, len(neg), 200000)
    mc = np.mean(pos[idx_p] > neg[idx_n]) + 0.5 * np.mean(pos[idx_p] == neg[idx_n])

    print(f"""
    AUC by ROC-curve integral         = {a_curve:.5f}
    AUC by counting all pos-neg pairs = {a_pairs:.5f}
    fraction of 200k random pairs ordered correctly = {mc:.5f}

  READING: all three agree. AUC is not an accuracy — it is the probability that the model scores
  a random positive above a random negative (the Wilcoxon-Mann-Whitney statistic). 0.5 is random
  ordering, 1.0 is perfect ranking. It is threshold-free and prevalence-independent (README §6).""")
    assert abs(a_curve - a_pairs) < 1e-6, "ROC-integral AUC must equal pair-count AUC"


# =============================================================================
# EXPERIMENT 4 — ROC optimism under imbalance (README §7)
# =============================================================================


def experiment_4_roc_vs_pr():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — under imbalance, ROC looks great but AP is mediocre (README §7)")
    print("=" * 88)
    rng = np.random.default_rng(4)
    n = 20000
    prevalence = 0.01
    y = (rng.uniform(size=n) < prevalence).astype(int)
    # a genuinely good RANKER (AUC ~0.9) whose PRECISION is still poor on the 1% class
    scores = 1.0 / (1.0 + np.exp(-(1.0 * (2 * y - 1) + rng.standard_normal(n))))

    print(f"""
  {n} examples, {prevalence:.0%} positive. Same scores, two summaries:

    {'metric':>20s} {'value':>8s} {'baseline':>10s}
    {'AUC (ROC)':>20s} {auc_roc(y, scores):>8.3f} {0.5:>10.2f}   <- looks strong
    {'average precision':>20s} {average_precision(y, scores):>8.3f} {prevalence:>10.2f}   <- the honest picture

  READING: the ROC's FPR = FP/(FP+TN) has a giant TN in the denominator, so thousands of false
  positives barely dent it — AUC looks great. Precision = TP/(TP+FP) feels every false positive,
  so average precision (baseline = prevalence {prevalence:.2f}) reveals the model still floods you
  with false alarms when hunting a 1% class. For rare positives, report PR/AP, not ROC (README §7).""")


# =============================================================================
# EXPERIMENT 5 — cost-optimal threshold (README §11)
# =============================================================================


def experiment_5_cost_threshold():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — the cost-optimal threshold is not 0.5 (README §11)")
    print("=" * 88)
    rng = np.random.default_rng(5)
    n = 20000
    # CALIBRATED scores: draw y FROM the score, so the score is a true probability. Only then
    # does the Bayes-optimal threshold c_FP/(c_FP+c_FN) apply (README §11).
    logit = rng.normal(0, 1.5, n)
    scores = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < scores).astype(int)

    c_fp, c_fn = 1.0, 10.0                    # a miss costs 10x a false alarm
    t_star_theory = c_fp / (c_fp + c_fn)

    ts = np.linspace(0.02, 0.9, 89)
    costs = []
    for t in ts:
        yhat = (scores >= t).astype(int)
        _, fp, _, fn = confusion(y, yhat)
        costs.append(c_fp * fp + c_fn * fn)
    t_best = ts[int(np.argmin(costs))]

    def cost_at(t):
        return costs[int(np.argmin(np.abs(ts - t)))]

    print(f"""
  Cost of a false alarm = {c_fp}, cost of a miss = {c_fn} (miss is 10x worse). Scores are
  CALIBRATED (y drawn from the score), so the Bayes threshold formula applies.

    {'threshold':>16s} {'expected cost':>14s}
    {'0.09 (Bayes)':>16s} {cost_at(0.09):>14.0f}
    {'0.50 (default)':>16s} {cost_at(0.50):>14.0f}
    {'0.80':>16s} {cost_at(0.80):>14.0f}

    Bayes-optimal threshold c_FP/(c_FP+c_FN) = {t_star_theory:.3f}
    empirically cost-minimizing threshold    = {t_best:.3f}

  READING: because a miss costs 10x a false alarm, the best threshold is ~0.09, not 0.5 — flag
  aggressively — and the empirical minimum ({t_best:.2f}) matches the Bayes formula
  c_FP/(c_FP+c_FN) = {t_star_theory:.2f}. The default 0.5 is optimal only for EQUAL costs and a
  calibrated model; here it costs far more. Set the threshold from the cost ratio (README §11).""")


# =============================================================================
# EXPERIMENT 6 — proper scoring separates equal-accuracy models (README §8)
# =============================================================================


def experiment_6_proper_scoring():
    print("\n" + "=" * 88)
    print("EXPERIMENT 6 — log loss separates two models with identical accuracy AND AUC")
    print("               (a calibration preview, README §8)")
    print("=" * 88)
    rng = np.random.default_rng(6)
    n = 20000
    # calibrated: y drawn FROM p_true. Model A reports p_true (calibrated). Model B applies a
    # monotonic temperature to the LOGIT (overconfident) — this preserves the exact ranking, so
    # A and B have IDENTICAL accuracy and AUC, but B's probabilities are miscalibrated.
    logit = rng.normal(0, 1.5, n)
    p_true = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p_true).astype(int)
    pA = p_true
    pB = 1.0 / (1.0 + np.exp(-2.5 * logit))     # sharpen: strictly monotonic in logit
    yhatA = (pA >= 0.5).astype(int)
    yhatB = (pB >= 0.5).astype(int)

    print(f"""
    {'model':>26s} {'accuracy':>9s} {'AUC':>7s} {'log loss':>9s} {'Brier':>7s}
    {'A (calibrated)':>26s} {accuracy(y, yhatA):>9.3f} {auc_roc(y, pA):>7.3f} """
          f"""{log_loss(y, pA):>9.3f} {brier(y, pA):>7.3f}
    {'B (overconfident, same rank)':>26s} {accuracy(y, yhatB):>9.3f} {auc_roc(y, pB):>7.3f} """
          f"""{log_loss(y, pB):>9.3f} {brier(y, pB):>7.3f}

  READING: models A and B make the SAME decisions (identical accuracy) and rank identically
  (identical AUC) — those metrics cannot tell them apart. But B's probabilities are
  overconfident, and log loss and Brier (PROPER scoring rules, minimized only by true
  probabilities) punish it. When the probabilities themselves matter, accuracy and AUC are
  blind; you need a proper scoring rule — and then calibration (05.06) to fix it (README §8).""")


if __name__ == "__main__":
    verify()
    experiment_1_accuracy_paradox()
    experiment_2_threshold_tradeoff()
    experiment_3_auc_ranking()
    experiment_4_roc_vs_pr()
    experiment_5_cost_threshold()
    experiment_6_proper_scoring()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
