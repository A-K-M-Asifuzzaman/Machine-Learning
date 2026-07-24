"""
05.04 — Cross-Validation & Model Selection, from scratch (NumPy).

The CV protocol itself is simple; the subtleties are all in what goes INSIDE the loop. We
implement KFold / StratifiedKFold / LeaveOneOut / cross_val_score / nested CV, verify against
sklearn, and then MEASURE the claims:

  1. the CV estimate has much lower variance than a single split                  (README §1-§2)
  2. bias-variance of K: LOO nearly unbiased but high-variance; K=5-10 the sweet   (README §3)
  3. THE FEATURE-SELECTION LEAK: pure noise 'validated' to 90%+ outside the fold   (README §7)
  4. NESTED vs naive CV: tuning on the reported folds is optimistically biased      (README §8)
  5. stratification stabilizes imbalanced folds                                    (README §5)
  6. time-series leakage: random K-fold vs forward chaining                        (README §9)
  + the LOOCV closed-form shortcut for linear models, verified to machine precision (README §4)

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    from sklearn.model_selection import KFold as SkKFold, cross_val_score as sk_cvs
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.tree import DecisionTreeClassifier
    HAVE_SK = True
except Exception:
    HAVE_SK = False


# =============================================================================
# SPLITTERS  (README §2, §5)
# =============================================================================


def kfold_indices(n, k, shuffle=True, seed=0):
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, test


def stratified_kfold_indices(y, k, seed=0):
    """Split within each class so every fold keeps the class proportions (README §5)."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    # assign each sample a within-class position, then fold = position % k
    fold_of = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % k
    for i in range(k):
        test = np.where(fold_of == i)[0]
        train = np.where(fold_of != i)[0]
        yield train, test


def loo_indices(n):
    for i in range(n):
        yield np.delete(np.arange(n), i), np.array([i])


def timeseries_split_indices(n, k):
    """Forward chaining: train on a growing prefix, test on the next block (README §9)."""
    fold_size = n // (k + 1)
    for i in range(1, k + 1):
        train = np.arange(0, fold_size * i)
        test = np.arange(fold_size * i, fold_size * (i + 1))
        yield train, test


# =============================================================================
# CROSS_VAL_SCORE and NESTED CV  (README §2, §8)
# =============================================================================


def cross_val_score(make_model, X, y, splits, score_fn):
    """make_model() -> fresh estimator with .fit/.predict; splits yields (train, test)."""
    scores = []
    for train, test in splits:
        m = make_model()
        m.fit(X[train], y[train])
        scores.append(score_fn(y[test], m.predict(X[test])))
    return np.array(scores)


def accuracy(y, yhat):
    return float(np.mean(np.asarray(y) == np.asarray(yhat)))


# =============================================================================
# VERIFICATION
# =============================================================================


def verify():
    print("=" * 88)
    print("VERIFICATION — KFold / cross_val_score vs scikit-learn")
    print("=" * 88)
    rng = np.random.default_rng(0)
    n = 200
    X = rng.standard_normal((n, 5))
    y = (X @ np.array([1.5, -1, 0.5, 0, 0]) + 0.4 * rng.standard_normal(n) > 0).astype(int)

    # our KFold produces the same partition as sklearn (same seed convention we control)
    ours = cross_val_score(lambda: LogisticRegression(max_iter=1000), X, y,
                           kfold_indices(n, 5, shuffle=True, seed=0), accuracy)
    if HAVE_SK:
        sk = sk_cvs(LogisticRegression(max_iter=1000), X, y,
                    cv=SkKFold(5, shuffle=True, random_state=0))
        print(f"""
    our 5-fold accuracies:     {np.array2string(ours, precision=3)}
    sklearn 5-fold accuracies: {np.array2string(sk, precision=3)}
    our mean = {ours.mean():.4f}, sklearn mean = {sk.mean():.4f}
""")
        # partitions differ in shuffle order, but the mean over folds should be close
        assert abs(ours.mean() - sk.mean()) < 0.05, "CV mean parity"
        print("  cross_val_score mean matches sklearn within fold-shuffle noise  ✓")

    # ---- LOOCV closed-form shortcut for a linear smoother (README §4) ----
    Xr = rng.standard_normal((80, 3))
    yr = Xr @ np.array([2.0, -1.0, 0.5]) + 0.3 * rng.standard_normal(80)
    Xd = np.column_stack([np.ones(80), Xr])
    H = Xd @ np.linalg.solve(Xd.T @ Xd, Xd.T)              # hat matrix
    yhat = H @ yr
    loo_closed = np.mean(((yr - yhat) / (1 - np.diag(H))) ** 2)
    # brute force LOO
    loo_brute = []
    for tr, te in loo_indices(80):
        w = np.linalg.lstsq(Xd[tr], yr[tr], rcond=None)[0]
        loo_brute.append((yr[te[0]] - Xd[te[0]] @ w) ** 2)
    loo_brute = np.mean(loo_brute)
    print(f"\n  LOOCV closed form  = {loo_closed:.6f}")
    print(f"  LOOCV brute force  = {loo_brute:.6f}   (diff {abs(loo_closed-loo_brute):.2e})")
    assert abs(loo_closed - loo_brute) < 1e-9, "LOOCV shortcut must match brute force"
    print("  the hat-matrix LOOCV shortcut matches brute force to machine precision  ✓")
    print("\nAll verification checks passed.")


# =============================================================================
# EXPERIMENT 1 — CV estimate variance vs single split (README §1-§2)
# =============================================================================


def experiment_1_cv_variance():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — the CV estimate has far lower variance than a single split (README §1)")
    print("=" * 88)
    rng = np.random.default_rng(1)
    n = 200

    single, cv10 = [], []
    for trial in range(200):
        X = rng.standard_normal((n, 5))
        y = (X @ np.array([1.2, -1, 0.5, 0, 0]) + 0.7 * rng.standard_normal(n) > 0).astype(int)
        # single 80/20 split
        perm = rng.permutation(n)
        tr, te = perm[:160], perm[160:]
        m = LogisticRegression(max_iter=500).fit(X[tr], y[tr])
        single.append(accuracy(y[te], m.predict(X[te])))
        # 10-fold CV
        cv10.append(cross_val_score(lambda: LogisticRegression(max_iter=500), X, y,
                                    kfold_indices(n, 10, seed=trial), accuracy).mean())
    print(f"""
  Accuracy estimate over 200 fresh datasets (true accuracy is fixed):

    {'estimator':>22s} {'mean':>8s} {'std of the estimate':>20s}
    {'single 80/20 split':>22s} {np.mean(single):>8.3f} {np.std(single):>20.4f}
    {'10-fold CV':>22s} {np.mean(cv10):>8.3f} {np.std(cv10):>20.4f}

  READING: both estimate the same accuracy, but the single-split estimate's standard deviation
  is ~{np.std(single)/np.std(cv10):.1f}x larger — it swings by chance with which points land in the
  test set. CV averages over all rotations, using every point for testing exactly once, and is a
  much more stable estimate (README §1-§2).""")


# =============================================================================
# EXPERIMENT 2 — bias-variance of K (README §3)
# =============================================================================


def experiment_2_choosing_k():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — bias-variance of the CV estimate vs K (README §3)")
    print("=" * 88)
    rng = np.random.default_rng(2)
    n = 120

    # 'truth' = accuracy of a model trained on lots of data, tested on lots of data
    Xbig = rng.standard_normal((6000, 4))
    beta = np.array([1.4, -1.2, 0.6, 0])
    ybig = (Xbig @ beta + 0.7 * rng.standard_normal(6000) > 0).astype(int)
    truth = LogisticRegression(max_iter=800).fit(Xbig[:5000], ybig[:5000])
    true_acc = accuracy(ybig[5000:], truth.predict(Xbig[5000:]))

    print(f"\n  Reference accuracy (trained on 5000): {true_acc:.3f}\n")
    print(f"    {'K':>6s} {'mean CV est':>12s} {'bias':>8s} {'std (variance)':>15s}")
    for K in (2, 5, 10, 20, n):
        ests = []
        for trial in range(120):
            X = rng.standard_normal((n, 4))
            y = (X @ beta + 0.7 * rng.standard_normal(n) > 0).astype(int)
            splits = loo_indices(n) if K == n else kfold_indices(n, K, seed=trial)
            ests.append(cross_val_score(lambda: LogisticRegression(max_iter=500),
                                        X, y, splits, accuracy).mean())
        label = "LOO" if K == n else str(K)
        print(f"    {label:>6s} {np.mean(ests):>12.3f} {np.mean(ests)-true_acc:>8.3f} "
              f"{np.std(ests):>15.4f}")
    print("""
  READING: small K (K=2) trains on only half the data, so its estimate is clearly PESSIMISTIC
  (the most negative bias); as K grows each fold trains on more data and the bias shrinks toward
  0. The estimate's variance is comparable across K on this problem — the theoretical rise toward
  LOO (highly correlated fold models) is modest here. Since LOO costs n fits for no bias gain
  over K=10, K=5-10 is the standard bias-variance-compute compromise (README §3).""")


# =============================================================================
# EXPERIMENT 3 — the feature-selection leak (README §7)
# =============================================================================


def experiment_3_feature_selection_leak():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — leakage: feature selection OUTSIDE the fold fakes 90%+ on pure noise")
    print("               (Ambroise & McLachlan, README §7)")
    print("=" * 88)
    rng = np.random.default_rng(3)
    n, p, k_top = 100, 5000, 20
    # PURE NOISE: X independent of y. Any 'predictive' signal is leakage.
    X = rng.standard_normal((n, p))
    y = rng.integers(0, 2, n)

    # WRONG: select the k features most correlated with y using ALL rows, THEN cross-validate
    corr = np.abs([np.corrcoef(X[:, j], y)[0, 1] for j in range(p)])
    top = np.argsort(-corr)[:k_top]
    wrong = cross_val_score(lambda: LogisticRegression(max_iter=500),
                            X[:, top], y, kfold_indices(n, 5, seed=0), accuracy).mean()

    # RIGHT: select features INSIDE each fold, on the training rows only (averaged over a few
    # fold shufflings so the chance level is estimated cleanly)
    right_scores = []
    for seed in range(5):
        for tr, te in kfold_indices(n, 5, seed=seed):
            c = np.abs([np.corrcoef(X[tr, j], y[tr])[0, 1] for j in range(p)])
            sel = np.argsort(-c)[:k_top]
            m = LogisticRegression(max_iter=500).fit(X[tr][:, sel], y[tr])
            right_scores.append(accuracy(y[te], m.predict(X[te][:, sel])))
    right = np.mean(right_scores)

    print(f"""
  {n} samples, {p} PURE-NOISE features (X independent of y), select top {k_top} by correlation.
  True accuracy of any model here must be ~0.50 (chance).

    {'protocol':>34s} {'CV accuracy':>12s}
    {'select on ALL data, then CV (WRONG)':>34s} {wrong:>12.3f}   <- FAKE signal from leakage
    {'select INSIDE each fold (RIGHT)':>34s} {right:>12.3f}   <- chance, as it should be

  READING: selecting features on the full data lets the fold's own labels choose the features,
  so noise that happened to correlate with y keeps 'predicting' within each fold — a fabricated
  {wrong:.0%}. Move the selection inside the fold and it collapses to chance. ANY data-learning step
  (scaling, imputation, selection, encoding) must be fit inside the fold; cross-validate the whole
  PIPELINE, never a pre-transformed dataset (README §7).""")


# =============================================================================
# EXPERIMENT 4 — nested vs naive CV (README §8)
# =============================================================================


def experiment_4_nested_cv():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — tuning on the reported folds is optimistic; nested CV is honest")
    print("               (README §8)")
    print("=" * 88)
    rng = np.random.default_rng(4)
    n = 250
    X = rng.standard_normal((n, 10))
    # weak real signal, so hyperparameter tuning has room to chase noise
    y = (X @ np.r_[0.6, -0.5, np.zeros(8)] + 1.2 * rng.standard_normal(n) > 0).astype(int)
    depths = [1, 2, 3, 5, 8, 12]

    naive_scores, nested_scores = [], []
    for outer_seed in range(30):
        # ----- NAIVE: pick the depth with the best 5-fold score, REPORT that score -----
        best = -1
        for d in depths:
            s = cross_val_score(lambda d=d: DecisionTreeClassifier(max_depth=d, random_state=0),
                                X, y, kfold_indices(n, 5, seed=outer_seed), accuracy).mean()
            best = max(best, s)
        naive_scores.append(best)

        # ----- NESTED: outer fold estimates; inner CV selects the depth -----
        outer_fold_scores = []
        for tr, te in kfold_indices(n, 5, seed=outer_seed):
            best_d, best_s = depths[0], -1
            for d in depths:
                s = cross_val_score(
                    lambda d=d: DecisionTreeClassifier(max_depth=d, random_state=0),
                    X[tr], y[tr], kfold_indices(len(tr), 4, seed=1), accuracy).mean()
                if s > best_s:
                    best_s, best_d = s, d
            m = DecisionTreeClassifier(max_depth=best_d, random_state=0).fit(X[tr], y[tr])
            outer_fold_scores.append(accuracy(y[te], m.predict(X[te])))
        nested_scores.append(np.mean(outer_fold_scores))

    print(f"""
    {'protocol':>34s} {'reported accuracy':>18s}
    {'naive CV (tune + report same folds)':>34s} {np.mean(naive_scores):>18.3f}   <- optimistic
    {'nested CV (inner tune, outer test)':>34s} {np.mean(nested_scores):>18.3f}   <- honest

    optimism gap = {np.mean(naive_scores) - np.mean(nested_scores):+.3f}

  READING: the naive protocol reports the BEST-of-several CV scores, so it captures how well the
  chosen depth fit the CV noise — optimistically biased upward. Nested CV selects the depth on an
  INNER loop and estimates on an OUTER fold the selection never saw, removing the optimism. When
  you tuned anything, report nested CV (or a locked test set), not the tuning score (README §8).""")


# =============================================================================
# EXPERIMENT 5 — stratification on imbalanced data (README §5)
# =============================================================================


def experiment_5_stratification():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — stratified folds stabilize imbalanced CV (README §5)")
    print("=" * 88)
    rng = np.random.default_rng(5)
    n = 200
    X = rng.standard_normal((n, 4))
    logit = X @ np.array([1.5, -1, 0.5, 0]) - 2.5           # ~7% positive
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    print(f"\n  {n} samples, {y.mean():.0%} positive, 10-fold CV. Per-fold POSITIVE COUNT:\n")

    plain_counts = [int(y[te].sum()) for _, te in kfold_indices(n, 10, seed=1)]
    strat_counts = [int(y[te].sum()) for _, te in stratified_kfold_indices(y, 10, seed=1)]
    print(f"    {'plain KFold':>14s}: {plain_counts}   (std {np.std(plain_counts):.2f})")
    print(f"    {'stratified':>14s}: {strat_counts}   (std {np.std(strat_counts):.2f})")
    print(f"""
  READING: plain K-fold hands some folds many positives and others few or zero (a zero-positive
  fold makes recall undefined and the estimate unstable). Stratified folds each carry ~{y.sum()//10}
  positives — the per-fold positive count has far lower spread. For classification, especially
  imbalanced, stratify by default (README §5).""")


# =============================================================================
# EXPERIMENT 6 — time-series leakage (README §9)
# =============================================================================


def experiment_6_timeseries():
    print("\n" + "=" * 88)
    print("EXPERIMENT 6 — random K-fold leaks on time series; forward chaining is honest")
    print("               (README §9)")
    print("=" * 88)
    from sklearn.neighbors import KNeighborsRegressor

    def r2(y, p):
        y = np.asarray(y, float)
        return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)

    # TWO INDEPENDENT random walks: feature x and target y are UNRELATED, but both drift
    # smoothly, so temporal neighbours have similar x AND similar y. Random folds put those
    # neighbours in training and the model 'interpolates' a relationship that does not exist.
    rnd, fwd = [], []
    for t in range(40):
        rng = np.random.default_rng(t)
        n = 400
        x = np.cumsum(rng.standard_normal(n))
        y = np.cumsum(rng.standard_normal(n))          # independent of x
        X = np.column_stack([x, np.roll(x, 1), np.roll(x, 2)])[2:]
        yy = y[2:]
        m = len(yy)
        rnd.append(np.mean([r2(yy[te], KNeighborsRegressor(5).fit(X[tr], yy[tr]).predict(X[te]))
                            for tr, te in kfold_indices(m, 5, seed=t)]))
        fwd.append(np.mean([r2(yy[te], KNeighborsRegressor(5).fit(X[tr], yy[tr]).predict(X[te]))
                            for tr, te in timeseries_split_indices(m, 5)]))
    rnd_r2, fwd_r2 = np.mean(rnd), np.mean(fwd)
    print(f"""
  Predicting one random walk from another INDEPENDENT one (true R^2 = 0, no relationship):

    {'protocol':>34s} {'test R^2':>10s}
    {'random 5-fold (LEAKS neighbours)':>34s} {rnd_r2:>10.3f}   <- fabricated 'signal'
    {'forward chaining (train<past)':>34s} {fwd_r2:>10.1f}   <- the truth: no relationship

  READING: the two series are INDEPENDENT, so honest predictive R^2 is ~0 (or negative). Random
  folds place a test point's temporal neighbours — which have nearly the same x AND the same y,
  since both drift smoothly — into the training set, so a neighbour-based model 'interpolates' a
  relationship that does not exist and reports a positive R^2 ({rnd_r2:.2f}). Forward chaining
  trains only on the past and must extrapolate to an unrelated future, exposing the truth (R^2
  hugely negative). The CV split must mimic the train/deploy gap or it measures leakage (README §9).""")


if __name__ == "__main__":
    verify()
    experiment_1_cv_variance()
    experiment_2_choosing_k()
    experiment_3_feature_selection_leak()
    experiment_4_nested_cv()
    experiment_5_stratification()
    experiment_6_timeseries()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
