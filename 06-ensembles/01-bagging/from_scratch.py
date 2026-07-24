"""
06.01 — Bagging from Scratch
============================

Bootstrap aggregating: train many high-variance models on resamples, average them. The
whole theory is one variance identity, and the experiments measure it.

Implemented here
----------------
    bootstrap_sample        n draws with replacement, returns in-bag and OOB indices
    BaggingRegressor        any base learner, mean aggregation, OOB scoring
    BaggingClassifier       soft or hard voting, OOB scoring

Run it
------
    python from_scratch.py

Verified against sklearn, then four experiments:
  1. The variance floor: total variance -> rho*sigma^2, not zero (README §2-§3)
  2. Out-of-bag error tracks true test error, for free (README §6)
  3. Bagging transforms trees, does nothing for linear regression (README §7)
  4. Decorrelating the base learners lowers the floor (the random-forest preview, §3)

Reference: README.md sections 2-9.
"""

from __future__ import annotations

import numpy as np

# =============================================================================
# THE BOOTSTRAP  (README §4)
# =============================================================================


def bootstrap_sample(n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Return (in_bag_indices, out_of_bag_indices) for one bootstrap draw.

    n draws WITH replacement, so the in-bag set has duplicates and covers about 63.2% of
    the distinct points; the rest are out-of-bag (README §4). The OOB points are the free
    validation set of README §6.
    """
    in_bag = rng.integers(0, n, n)
    oob_mask = np.ones(n, dtype=bool)
    oob_mask[in_bag] = False
    return in_bag, np.where(oob_mask)[0]


# =============================================================================
# BAGGING  (README §5-§9)
# =============================================================================


class _BaseBagging:
    """Shared machinery: fit B base learners on bootstrap samples, track OOB membership.

    `base_estimator_factory` is a zero-argument callable returning a FRESH unfitted model,
    so each of the B learners is independent. The base learner should be HIGH VARIANCE
    (a deep, unpruned tree) — averaging fixes variance and cannot fix bias (README §8), so
    a stable base learner wastes the whole method.
    """

    def __init__(self, base_estimator_factory, n_estimators: int = 50,
                 max_samples: float = 1.0, oob_score: bool = False, random_state: int = 0):
        self.base_estimator_factory = base_estimator_factory
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.oob_score = oob_score
        self.random_state = random_state

    def _fit_estimators(self, X, y):
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        self.oob_indices_ = []      # which points are OOB for each estimator

        for _ in range(self.n_estimators):
            in_bag, oob = bootstrap_sample(n, rng)
            model = self.base_estimator_factory()
            model.fit(X[in_bag], y[in_bag])
            self.estimators_.append(model)
            self.oob_indices_.append(oob)
        return self


class BaggingRegressor(_BaseBagging):
    """Average the base regressors' predictions.  README §5"""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self._fit_estimators(X, y)

        if self.oob_score:
            self.oob_prediction_, self.oob_score_ = self._compute_oob(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.mean([m.predict(X) for m in self.estimators_], axis=0)

    def _compute_oob(self, X, y):
        """Predict each point using ONLY the trees that did not train on it.  README §6

        This is an honest held-out prediction at zero extra cost — the resampling happened
        anyway. Points never left OOB (rare, only at small B) are skipped.
        """
        n = X.shape[0]
        sums = np.zeros(n)
        counts = np.zeros(n)
        for model, oob in zip(self.estimators_, self.oob_indices_):
            if oob.size:
                sums[oob] += model.predict(X[oob])
                counts[oob] += 1
        seen = counts > 0
        preds = np.full(n, np.nan)
        preds[seen] = sums[seen] / counts[seen]
        ss_res = np.sum((y[seen] - preds[seen]) ** 2)
        ss_tot = np.sum((y[seen] - y[seen].mean()) ** 2)
        return preds, float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    def score(self, X, y):
        y = np.asarray(y, dtype=float).ravel()
        pred = self.predict(X)
        return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


class BaggingClassifier(_BaseBagging):
    """Aggregate the base classifiers by soft or hard voting.  README §9

    Soft voting (the default) averages predicted probabilities, using each tree's
    confidence; it is almost always better than hard majority voting and yields usable
    probabilities. Requires the base learner to expose predict_proba.
    """

    def __init__(self, base_estimator_factory, n_estimators=50, voting="soft",
                 max_samples=1.0, oob_score=False, random_state=0):
        super().__init__(base_estimator_factory, n_estimators, max_samples,
                         oob_score, random_state)
        self.voting = voting

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self._fit_estimators(X, y)
        if self.oob_score:
            self.oob_score_ = self._compute_oob(X, y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if self.voting == "soft":
            return np.mean([m.predict_proba(X) for m in self.estimators_], axis=0)
        # Hard voting expressed as a probability: fraction of trees choosing each class.
        votes = np.zeros((X.shape[0], self.classes_.size))
        for m in self.estimators_:
            pred = m.predict(X)
            for k, c in enumerate(self.classes_):
                votes[:, k] += (pred == c)
        return votes / len(self.estimators_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def _compute_oob(self, X, y):
        n = X.shape[0]
        proba = np.zeros((n, self.classes_.size))
        counts = np.zeros(n)
        for model, oob in zip(self.estimators_, self.oob_indices_):
            if oob.size:
                proba[oob] += model.predict_proba(X[oob])
                counts[oob] += 1
        seen = counts > 0
        pred = self.classes_[np.argmax(proba[seen], axis=1)]
        return float(np.mean(pred == y[seen]))

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))


# =============================================================================
# A MINIMAL DECISION TREE  (self-contained, so this file runs standalone)
# =============================================================================


class _Stump:
    """A tiny CART tree, just enough to be a high-variance base learner.

    Deliberately minimal — the real tree is 03.08. What matters for bagging is only that
    it is HIGH VARIANCE (deep, unpruned by default), which is exactly what averaging needs.
    """

    def __init__(self, max_depth=None, min_samples_leaf=1, task="regression",
                 max_features=None, random_state=0):
        self.max_depth = max_depth if max_depth is not None else 2 ** 31
        self.min_samples_leaf = min_samples_leaf
        self.task = task
        self.max_features = max_features        # for the decorrelation experiment
        self.random_state = random_state

    def fit(self, X, y):
        self.X, self.y = np.asarray(X, dtype=float), np.asarray(y)
        if self.task == "classification":
            self.classes_ = np.unique(self.y)
            self._y_idx = np.searchsorted(self.classes_, self.y)
            self._K = self.classes_.size
        self._rng = np.random.default_rng(self.random_state)
        self.root = self._build(np.arange(len(y)), 0)
        return self

    def _leaf(self, idx):
        if self.task == "regression":
            return {"value": float(self.y[idx].mean())}
        counts = np.bincount(self._y_idx[idx], minlength=self._K).astype(float)
        return {"proba": counts / counts.sum()}

    def _best_split(self, idx):
        """Incremental split scan (03.08 §7): sort each feature once, sweep the boundary
        updating counts/sums in O(1). Makes a node O(m log m * d) rather than O(m^2 * d)."""
        d = self.X.shape[1]
        features = (range(d) if self.max_features is None
                    else self._rng.choice(d, min(self.max_features, d), replace=False))
        m = idx.size
        msl = self.min_samples_leaf
        best_gain, best = 0.0, None

        if self.task == "regression":
            yv = self.y[idx]
            # Parent SSE is fixed, so maximizing variance reduction = minimizing the total
            # child SSE. We track the smallest child SSE seen and accept a split only if it
            # beats the no-split baseline (the parent's own SSE).
            parent_sse = float(np.sum((yv - yv.mean()) ** 2))
            best_sse = parent_sse - 1e-12
            for f in features:
                xf = self.X[idx, f]
                order = np.argsort(xf, kind="stable")
                xs, ys = xf[order], yv[order]
                csum = np.cumsum(ys)
                csum_sq = np.cumsum(ys ** 2)
                total, total_sq = csum[-1], csum_sq[-1]
                for i in range(msl - 1, m - msl):
                    if xs[i] == xs[i + 1]:
                        continue
                    nl = i + 1
                    nr = m - nl
                    left_sse = csum_sq[i] - csum[i] ** 2 / nl
                    right_sse = (total_sq - csum_sq[i]) - (total - csum[i]) ** 2 / nr
                    child_sse = left_sse + right_sse
                    if child_sse < best_sse:
                        best_sse = child_sse
                        best = (f, 0.5 * (xs[i] + xs[i + 1]))
            return best
        else:
            yi = self._y_idx[idx]
            parent_counts = np.bincount(yi, minlength=self._K).astype(float)
            parent_gini = (1 - np.sum((parent_counts / m) ** 2)) * m
            for f in features:
                xf = self.X[idx, f]
                order = np.argsort(xf, kind="stable")
                xs, yy = xf[order], yi[order]
                left = np.zeros(self._K)
                right = parent_counts.copy()
                for i in range(m - 1):
                    left[yy[i]] += 1
                    right[yy[i]] -= 1
                    if xs[i] == xs[i + 1] or i + 1 < msl or m - i - 1 < msl:
                        continue
                    nl, nr = i + 1, m - i - 1
                    gl = (1 - np.sum((left / nl) ** 2)) * nl
                    gr = (1 - np.sum((right / nr) ** 2)) * nr
                    gain = parent_gini - gl - gr
                    if best is None or gain > best_gain:
                        best_gain, best = gain, (f, 0.5 * (xs[i] + xs[i + 1]))
            return best if best_gain > 1e-12 else None

    def _build(self, idx, depth):
        node = self._leaf(idx)
        if depth >= self.max_depth or idx.size < 2 * self.min_samples_leaf or \
                np.unique(self.y[idx]).size == 1:
            return node

        best = self._best_split(idx)
        if best is None:
            return node
        f, thr = best
        mask = self.X[idx, f] <= thr
        left, right = idx[mask], idx[~mask]
        if left.size < self.min_samples_leaf or right.size < self.min_samples_leaf:
            return node
        node["feature"], node["threshold"] = f, thr
        node["left"] = self._build(left, depth + 1)
        node["right"] = self._build(right, depth + 1)
        return node

    def _traverse(self, x, node):
        while "feature" in node:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.task == "regression":
            return np.array([self._traverse(x, self.root)["value"] for x in X])
        return self.classes_[np.array([np.argmax(self._traverse(x, self.root)["proba"])
                                       for x in X])]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse(x, self.root)["proba"] for x in X])


# =============================================================================
# VERIFICATION
# =============================================================================


def _report(name, error, threshold):
    status = "PASS" if error < threshold else "FAIL"
    print(f"  [{status}]  {name:<58s}  err = {error:.3e}")
    return error < threshold


def verify():
    ok = True
    rng = np.random.default_rng(0)

    print("=" * 88)
    print("VERIFICATION")
    print("=" * 88)

    print("\nThe bootstrap 63.2% fact (README §4)")
    # Empirically, a bootstrap sample covers ~1 - 1/e of the distinct points.
    coverage = []
    for _ in range(500):
        in_bag, oob = bootstrap_sample(1000, rng)
        coverage.append(1 - oob.size / 1000)
    ok &= _report("bootstrap covers ~63.2% of distinct points",
                  abs(np.mean(coverage) - (1 - 1 / np.e)), 3e-3)

    n = 400
    X = rng.standard_normal((n, 6))
    y_reg = X[:, 0] * 2 + np.sin(2 * X[:, 1]) - X[:, 2] + rng.standard_normal(n) * 0.3
    y_clf = (X[:, 0] + X[:, 1] ** 2 - X[:, 2] > 0.5).astype(int)
    X_te = rng.standard_normal((2000, 6))
    y_te_reg = X_te[:, 0] * 2 + np.sin(2 * X_te[:, 1]) - X_te[:, 2]
    y_te_clf = (X_te[:, 0] + X_te[:, 1] ** 2 - X_te[:, 2] > 0.5).astype(int)

    print("\nBagging beats a single tree (README §2)")
    single = _Stump(task="regression").fit(X, y_reg)
    single_r2 = float(1 - np.sum((y_te_reg - single.predict(X_te)) ** 2)
                      / np.sum((y_te_reg - y_te_reg.mean()) ** 2))
    bag = BaggingRegressor(lambda: _Stump(task="regression"),
                           n_estimators=50).fit(X, y_reg)
    bag_r2 = float(1 - np.sum((y_te_reg - bag.predict(X_te)) ** 2)
                   / np.sum((y_te_reg - y_te_reg.mean()) ** 2))
    print(f"  [{'PASS' if bag_r2 > single_r2 else 'FAIL'}]  "
          f"{'bagged trees beat a single tree (test R^2)':<58s}  "
          f"{single_r2:.4f} -> {bag_r2:.4f}")
    ok &= bag_r2 > single_r2

    print("\nAgainst sklearn (README §5)")
    try:
        from sklearn.ensemble import (BaggingRegressor as SKBagR,
                                      BaggingClassifier as SKBagC)
        from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

        sk = SKBagR(DecisionTreeRegressor(random_state=0), n_estimators=30,
                    random_state=0, oob_score=True).fit(X, y_reg)
        mine = BaggingRegressor(
            lambda: DecisionTreeRegressor(random_state=0), n_estimators=30,
            oob_score=True, random_state=0).fit(X, y_reg)
        # Predictions won't be bitwise identical (different bootstrap RNG), but the test
        # performance and OOB estimates should be very close.
        sk_r2 = sk.score(X_te, y_te_reg)
        mine_r2 = mine.score(X_te, y_te_reg)
        print(f"  [{'PASS' if abs(sk_r2 - mine_r2) < 0.05 else 'FAIL'}]  "
              f"{'test R^2 close to sklearn BaggingRegressor':<58s}  "
              f"{mine_r2:.4f} vs {sk_r2:.4f}")
        ok &= abs(sk_r2 - mine_r2) < 0.05
        print(f"  [{'PASS' if abs(sk.oob_score_ - mine.oob_score_) < 0.1 else 'FAIL'}]  "
              f"{'OOB score close to sklearn':<58s}  "
              f"{mine.oob_score_:.4f} vs {sk.oob_score_:.4f}")
        ok &= abs(sk.oob_score_ - mine.oob_score_) < 0.1

        sk_c = SKBagC(DecisionTreeClassifier(random_state=0), n_estimators=30,
                      random_state=0).fit(X, y_clf)
        mine_c = BaggingClassifier(lambda: DecisionTreeClassifier(random_state=0),
                                   n_estimators=30, random_state=0).fit(X, y_clf)
        print(f"  [{'PASS' if abs(sk_c.score(X_te, y_te_clf) - mine_c.score(X_te, y_te_clf)) < 0.03 else 'FAIL'}]  "
              f"{'classifier accuracy close to sklearn':<58s}  "
              f"{mine_c.score(X_te, y_te_clf):.4f} vs {sk_c.score(X_te, y_te_clf):.4f}")
        ok &= abs(sk_c.score(X_te, y_te_clf) - mine_c.score(X_te, y_te_clf)) < 0.03
    except ImportError:
        print("  [SKIP]  sklearn not installed")

    print("\nOOB tracks test error (README §6)")
    bag = BaggingRegressor(lambda: _Stump(task="regression"),
                           n_estimators=100, oob_score=True).fit(X, y_reg)
    test_r2 = bag.score(X_te, y_te_reg)
    ok &= _report("OOB R^2 is within 0.1 of test R^2",
                  abs(bag.oob_score_ - test_r2), 0.1)
    print(f"  [INFO]  {'OOB R^2 vs test R^2':<58s}  "
          f"{bag.oob_score_:.4f} vs {test_r2:.4f}")

    print("\nSoft voting beats hard voting (README §9)")
    soft = BaggingClassifier(lambda: _Stump(task="classification", max_depth=4),
                             n_estimators=50, voting="soft").fit(X, y_clf)
    hard = BaggingClassifier(lambda: _Stump(task="classification", max_depth=4),
                             n_estimators=50, voting="hard").fit(X, y_clf)
    print(f"  [INFO]  {'soft vs hard voting accuracy':<58s}  "
          f"{soft.score(X_te, y_te_clf):.4f} vs {hard.score(X_te, y_te_clf):.4f}")
    ok &= soft.score(X_te, y_te_clf) >= hard.score(X_te, y_te_clf) - 0.02

    return ok


# =============================================================================
# EXPERIMENTS
# =============================================================================


def experiment_variance_floor():
    """README §2-§3: total variance -> rho*sigma^2, not zero."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — the variance floor  (README §2-§3)")
    print("=" * 88)
    print("""
The theory: bagged variance = rho*sigma^2 + (1-rho)/B * sigma^2. The second term vanishes
with B; the first is a FLOOR that averaging cannot break. We measure the bagged predictor's
variance across independent training sets as B grows, and check it flattens at rho*sigma^2.
""")
    rng = np.random.default_rng(1)
    n = 120
    x_query = np.array([[0.5] * 5])                 # a fixed query point
    n_datasets = 150
    B_max = 60

    def make():
        X = rng.standard_normal((n, 5))
        y = X[:, 0] + X[:, 1] ** 2 - X[:, 2] + rng.standard_normal(n) * 0.5
        return X, y

    # Fit a POOL of B_max trees per dataset ONCE, store each tree's query prediction.
    # The bagged prediction for any B is then the mean of the first B — so every B reuses
    # the same trees instead of refitting, turning O(sum B) trees into O(B_max) per dataset.
    all_tree_preds = np.empty((n_datasets, B_max))
    for j in range(n_datasets):
        X, y = make()
        for b in range(B_max):
            idx = rng.integers(0, n, n)
            all_tree_preds[j, b] = _Stump(task="regression", max_depth=6,
                                          random_state=rng.integers(1 << 30)) \
                .fit(X[idx], y[idx]).predict(x_query)[0]

    # sigma^2: variance of a single tree's prediction across datasets.
    sigma2 = float(np.var(all_tree_preds[:, 0]))
    # rho: the correlation between two trees floor. As B -> inf the bagged variance is
    # rho*sigma^2, so estimate rho from the large-B bagged variance directly.
    bagged_full = all_tree_preds.mean(axis=1)
    var_full = float(np.var(bagged_full))
    rho = float(np.clip((var_full - sigma2 / B_max) / (sigma2 * (1 - 1 / B_max)), 0, 1)) \
        if sigma2 > 0 else 0.0

    print(f"  single-tree variance sigma^2 = {sigma2:.4f}")
    print(f"  estimated correlation rho     ~ {rho:.3f}")
    print(f"  predicted floor rho*sigma^2   = {rho * sigma2:.4f}\n")
    print(f"  {'B (trees)':>10s}  {'bagged variance':>16s}  "
          f"{'predicted rho*s^2 + (1-rho)/B*s^2':>34s}")
    print("  " + "-" * 64)

    for B in (1, 2, 5, 10, 25, 60):
        # Bagged prediction using the first B pooled trees, per dataset.
        measured = float(np.var(all_tree_preds[:, :B].mean(axis=1)))
        predicted = rho * sigma2 + (1 - rho) / B * sigma2
        print(f"  {B:10d}  {measured:16.4f}  {predicted:34.4f}")

    print("""
  The measured bagged variance falls steeply at first, then flattens — it does NOT go to
  zero. It levels off near rho*sigma^2, matching the predicted curve closely.

  This is the single most important fact about bagging (README §3). The (1-rho)/B term is
  what more trees buy, and it is spent quickly: most of the gain is in the first ~25 trees.
  The rho*sigma^2 floor is what more trees CANNOT buy — and lowering it means decorrelating
  the trees, which is exactly what a random forest does (Experiment 4, and all of 06.02).""")


def experiment_oob():
    """README §6: OOB error tracks test error, for free."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — out-of-bag error is a free validation set  (README §6)")
    print("=" * 88)
    print("""
Each point is out-of-bag for ~37% of the trees. Predicting it with only those trees is an
honest held-out estimate, at no extra cost. Comparing OOB error against true test error as
B grows:
""")
    rng = np.random.default_rng(2)
    n = 300
    X = rng.standard_normal((n, 8))
    y = X[:, 0] * 1.5 - X[:, 1] + X[:, 2] * X[:, 3] + rng.standard_normal(n) * 0.4
    X_te = rng.standard_normal((3000, 8))
    y_te = X_te[:, 0] * 1.5 - X_te[:, 1] + X_te[:, 2] * X_te[:, 3]

    print(f"  {'B (trees)':>10s}  {'OOB R^2':>9s}  {'TEST R^2':>9s}  {'gap':>8s}")
    print("  " + "-" * 42)
    for B in (5, 10, 25, 50, 100, 200):
        bag = BaggingRegressor(lambda: _Stump(task="regression", max_depth=8),
                               n_estimators=B, oob_score=True, random_state=0).fit(X, y)
        test_r2 = bag.score(X_te, y_te)
        print(f"  {B:10d}  {bag.oob_score_:9.4f}  {test_r2:9.4f}  "
              f"{abs(bag.oob_score_ - test_r2):8.4f}")

    print("""
  The OOB estimate tracks the true test R^2 closely, and the gap shrinks as B grows — at
  small B some points are OOB for too few trees, making the estimate noisy. By ~100 trees
  the two agree to a couple of percent.

  Practically: for a bagged model you often do not need a separate validation set. The OOB
  estimate is nearly unbiased and free, which matters most on small datasets where holding
  out rows is expensive. sklearn exposes it as oob_score=True.

  The caveat (README §6): OOB is reliable only once B is large enough. Do not trust it at
  B = 10.""")


def experiment_trees_vs_linear():
    """README §7: bagging transforms trees, does nothing for linear regression."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — what bagging helps, and what it does not  (README §7)")
    print("=" * 88)
    print("""
Bagging reduces variance and cannot touch bias. So it should transform a high-variance
model (a deep tree) and do essentially nothing for a low-variance one (linear regression),
whose bootstrapped fits are nearly identical. Measuring the improvement for each:
""")
    rng = np.random.default_rng(3)

    class LinReg:
        def fit(self, X, y):
            A = np.column_stack([np.ones(len(X)), X])
            self.w = np.linalg.lstsq(A, y, rcond=None)[0]
            return self
        def predict(self, X):
            return np.column_stack([np.ones(len(X)), X]) @ self.w

    n = 200
    X = rng.standard_normal((n, 5))
    y = X @ np.array([1.5, -2, 0.5, 1, -0.5]) + rng.standard_normal(n) * 1.0  # linear truth
    X_te = rng.standard_normal((3000, 5))
    y_te = X_te @ np.array([1.5, -2, 0.5, 1, -0.5])

    def r2(model):
        return float(1 - np.sum((y_te - model.predict(X_te)) ** 2)
                     / np.sum((y_te - y_te.mean()) ** 2))

    print(f"  {'base learner':<24s}  {'single R^2':>11s}  {'bagged (50) R^2':>16s}  "
          f"{'improvement':>12s}")
    print("  " + "-" * 68)

    single_tree = _Stump(task="regression", max_depth=None).fit(X, y)
    bag_tree = BaggingRegressor(lambda: _Stump(task="regression", max_depth=None),
                                n_estimators=50).fit(X, y)
    print(f"  {'deep tree':<24s}  {r2(single_tree):11.4f}  {r2(bag_tree):16.4f}  "
          f"{r2(bag_tree) - r2(single_tree):+12.4f}")

    single_lin = LinReg().fit(X, y)
    bag_lin = BaggingRegressor(lambda: LinReg(), n_estimators=50).fit(X, y)
    print(f"  {'linear regression':<24s}  {r2(single_lin):11.4f}  {r2(bag_lin):16.4f}  "
          f"{r2(bag_lin) - r2(single_lin):+12.4f}")

    print("""
  The deep tree improves substantially — averaging crushes its high variance. Linear
  regression barely moves: it is already a low-variance estimator, so every bootstrapped
  fit is nearly the same line (rho ~ 1), and the variance floor rho*sigma^2 is essentially
  the original variance. Averaging near-identical models achieves near-nothing.

  The rule (README §7): bag models that OVERFIT. If your base learner is already stable,
  bagging is wasted compute — the compute would be better spent reducing BIAS, which is
  what boosting does (06.03).""")


def experiment_decorrelation():
    """README §3: decorrelating the base learners lowers the floor."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — decorrelation lowers the floor (the random-forest idea)  (README §3)")
    print("=" * 88)
    print("""
The floor is rho*sigma^2, so lowering rho lowers the floor. A random forest lowers rho by
restricting each split to a random subset of features (max_features), forcing different
trees to use different features. Sweeping max_features from all (plain bagging) down to
few (a random forest) and measuring the ensemble's test error:
""")
    rng = np.random.default_rng(4)
    n, d = 300, 20
    X = rng.standard_normal((n, d))
    # A few strong features plus noise: strong features would dominate every tree's root
    # split under plain bagging, keeping the trees correlated.
    y = (2 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + 0.5 * rng.standard_normal(n) > 0).astype(int)
    X_te = rng.standard_normal((3000, d))
    y_te = (2 * X_te[:, 0] + 1.5 * X_te[:, 1] - X_te[:, 2] > 0).astype(int)

    print(f"  {'max_features':>13s}  {'= what':>22s}  {'tree correlation':>17s}  "
          f"{'ensemble TEST acc':>18s}")
    print("  " + "-" * 76)

    for mf, label in [(d, "all (plain bagging)"), (10, "half"), (5, "sqrt-ish"),
                      (3, "few"), (1, "one (max decorrelation)")]:
        bag = BaggingClassifier(
            lambda mf=mf: _Stump(task="classification", max_depth=8, max_features=mf,
                                 random_state=rng.integers(1 << 30)),
            n_estimators=100, voting="soft", random_state=0).fit(X, y)

        # Estimate correlation between trees via agreement on the test set relative to
        # chance — a proxy for rho.
        preds = np.array([m.predict(X_te) for m in bag.estimators_])
        agreements = []
        for _ in range(200):
            a, b = rng.integers(0, len(preds), 2)
            if a != b:
                agreements.append(np.mean(preds[a] == preds[b]))
        corr_proxy = float(np.mean(agreements))

        acc = bag.score(X_te, y_te)
        print(f"  {mf:13d}  {label:>22s}  {corr_proxy:17.4f}  {acc:18.4f}")

    print("""
  As max_features falls, the trees agree with each other LESS (the correlation proxy drops)
  — they are being forced onto different features — and the ensemble's test accuracy RISES,
  because the variance floor rho*sigma^2 is coming down.

  This is the whole idea of a random forest, previewed: bagging attacks the (1-rho)/B term
  with more trees; feature subsampling attacks the rho term itself. Restricting each split
  to a random feature subset is the single ingredient 06.02 adds, and it is why a random
  forest beats plain bagging with the same number of trees.

  Note the tradeoff: too few features (max_features=1) can start to hurt, because each tree
  becomes too weak (higher bias) to be worth decorrelating. The sweet spot — often sqrt(d)
  for classification — balances lower rho against higher per-tree bias, which is exactly
  the hyperparameter 06.02 tunes.""")


# =============================================================================

if __name__ == "__main__":
    print(__doc__)

    all_passed = verify()

    experiment_variance_floor()
    experiment_oob()
    experiment_trees_vs_linear()
    experiment_decorrelation()

    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 88)
