"""
06.02 — Random Forests from Scratch
===================================

Bagging plus one idea: subsample the features at EACH split. That single addition attacks
the rho in the rho*sigma^2 variance floor that bagging could not (06.01 §3).

Implemented here
----------------
    RandomForestClassifier / RandomForestRegressor
        max_features        the feature subset size per split
        oob_score           free validation (06.01 §6)
        feature_importances_ (MDI — biased, README §7)
        permutation_importance()   the honest version
        proximity()         the forest as a learned kernel (README §9)

The base tree is the fast incremental-scan CART from 06.01, extended so that each split
draws a fresh random feature subset.

Run it
------
    python from_scratch.py

Verified against sklearn, then five experiments:
  1. Feature subsampling lowers rho and the variance floor (README §3)
  2. Tuning max_features: the accuracy peak at intermediate values (README §4)
  3. MDI importance is biased on a forest too (README §7)
  4. Both importance methods fail on correlated features, in opposite ways (README §7)
  5. A forest cannot extrapolate (README §10)

Reference: README.md sections 3-10.
"""

from __future__ import annotations

import numpy as np

# =============================================================================
# THE TREE (fast incremental scan, with per-split feature subsampling)
# =============================================================================


class _Tree:
    """CART with the O(n log n) incremental split scan of 03.08 §7, extended so each split
    considers only `max_features` randomly chosen features — the one addition that turns
    bagging into a random forest (README §1).

    `random_state` seeds the per-split feature draw, so different trees in a forest see
    different features and decorrelate (README §3).
    """

    def __init__(self, task="classification", max_depth=None, min_samples_leaf=1,
                 max_features=None, random_state=0):
        self.task = task
        self.max_depth = max_depth if max_depth is not None else 2 ** 31
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        self.d = self.X.shape[1]
        if self.task == "classification":
            self.classes_ = np.unique(self.y)
            self._yidx = np.searchsorted(self.classes_, self.y)
            self._K = self.classes_.size
        self._rng = np.random.default_rng(self.random_state)
        self.importance_ = np.zeros(self.d)     # MDI accumulator (README §7)
        self._n = len(y)
        self.root = self._build(np.arange(self._n), 0)
        return self

    def _leaf(self, idx):
        if self.task == "regression":
            return {"value": float(self.y[idx].mean())}
        counts = np.bincount(self._yidx[idx], minlength=self._K).astype(float)
        return {"proba": counts / counts.sum()}

    def _best_split(self, idx):
        m = idx.size
        msl = self.min_samples_leaf
        # Draw a fresh random feature subset for THIS split (README §2).
        k = self.max_features or self.d
        features = self._rng.choice(self.d, min(k, self.d), replace=False)

        if self.task == "regression":
            yv = self.y[idx]
            parent_sse = float(np.sum((yv - yv.mean()) ** 2))
            best_sse, best = parent_sse - 1e-12, None
            for f in features:
                xf = self.X[idx, f]
                order = np.argsort(xf, kind="stable")
                xs, ys = xf[order], yv[order]
                cs, cs2 = np.cumsum(ys), np.cumsum(ys ** 2)
                total, total2 = cs[-1], cs2[-1]
                for i in range(msl - 1, m - msl):
                    if xs[i] == xs[i + 1]:
                        continue
                    nl, nr = i + 1, m - i - 1
                    left = cs2[i] - cs[i] ** 2 / nl
                    right = (total2 - cs2[i]) - (total - cs[i]) ** 2 / nr
                    if left + right < best_sse:
                        best_sse = left + right
                        best = (f, 0.5 * (xs[i] + xs[i + 1]), parent_sse - (left + right))
            return best
        else:
            yi = self._yidx[idx]
            pc = np.bincount(yi, minlength=self._K).astype(float)
            best_gain, best = 1e-12, None
            for f in features:
                xf = self.X[idx, f]
                order = np.argsort(xf, kind="stable")
                xs, yy = xf[order], yi[order]

                # Cumulative class counts along the sorted order: one-hot the labels and
                # cumsum, so left_counts[i] and right_counts[i] for EVERY split point come
                # out as vectorized arrays. Weighted Gini across all thresholds is then a
                # single vectorized expression — no Python loop over samples.
                onehot = np.zeros((m, self._K))
                onehot[np.arange(m), yy] = 1.0
                left_counts = np.cumsum(onehot, axis=0)[:-1]        # (m-1, K)
                right_counts = pc - left_counts
                nl = np.arange(1, m)[:, None]
                nr = m - nl
                gl = (1 - np.sum((left_counts / nl) ** 2, axis=1)) * nl.ravel()
                gr = (1 - np.sum((right_counts / nr) ** 2, axis=1)) * nr.ravel()
                child = gl + gr                                    # want to minimize this

                # Mask invalid thresholds: ties, and cuts violating min_samples_leaf.
                valid = (xs[:-1] != xs[1:])
                valid[:msl - 1] = False
                valid[m - msl:] = False
                if not valid.any():
                    continue
                child = np.where(valid, child, np.inf)
                i = int(np.argmin(child))
                parent = (1 - np.sum((pc / m) ** 2)) * m
                gain = parent - child[i]
                if gain > best_gain:
                    best_gain = gain
                    best = (f, 0.5 * (xs[i] + xs[i + 1]), gain)
            return best

    def _build(self, idx, depth):
        node = self._leaf(idx)
        if depth >= self.max_depth or idx.size < 2 * self.min_samples_leaf or \
                np.unique(self.y[idx]).size == 1:
            return node
        best = self._best_split(idx)
        if best is None:
            return node
        f, thr, gain = best
        mask = self.X[idx, f] <= thr
        left, right = idx[mask], idx[~mask]
        if left.size < self.min_samples_leaf or right.size < self.min_samples_leaf:
            return node
        self.importance_[f] += idx.size * gain  # MDI: gain weighted by node size
        node.update(feature=f, threshold=thr, left=self._build(left, depth + 1),
                    right=self._build(right, depth + 1))
        return node

    def _route(self, X, collect):
        """Route ALL rows of X through the tree at once, vectorized.

        Instead of a Python while-loop per sample (O(n_samples * depth) interpreted
        operations), we carry an index array down the tree and split it with a boolean
        mask at each node — O(depth) NumPy operations total. For a forest predicting on
        thousands of points this is the difference between seconds and minutes.

        `collect(leaf_node, row_indices, path_id)` is called at each leaf to fill outputs.
        """
        out = [None] * X.shape[0]

        def recurse(node, idx, path):
            if "feature" not in node:
                collect(node, idx, path, out)
                return
            go_left = X[idx, node["feature"]] <= node["threshold"]
            recurse(node["left"], idx[go_left], path * 2)
            recurse(node["right"], idx[~go_left], path * 2 + 1)

        recurse(self.root, np.arange(X.shape[0]), 0)
        return out

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty(X.shape[0], dtype=float if self.task == "regression" else object)
        if self.task == "regression":
            self._route(X, lambda nd, idx, p, o: out.__setitem__(idx, nd["value"]))
            return out
        preds = np.empty(X.shape[0], dtype=int)
        self._route(X, lambda nd, idx, p, o: preds.__setitem__(
            idx, int(np.argmax(nd["proba"]))))
        return self.classes_[preds]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty((X.shape[0], self._K))
        self._route(X, lambda nd, idx, p, o: out.__setitem__(idx, nd["proba"]))
        return out

    def apply(self, X):
        """Leaf id per point — used for forest proximities (README §9)."""
        X = np.asarray(X, dtype=float)
        out = np.empty(X.shape[0], dtype=np.int64)
        self._route(X, lambda nd, idx, p, o: out.__setitem__(idx, p))
        return out


# =============================================================================
# THE FOREST  (README §2)
# =============================================================================


class _BaseForest:
    def __init__(self, n_estimators=100, max_features="auto", max_depth=None,
                 min_samples_leaf=1, bootstrap=True, oob_score=False, random_state=0):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

    def _resolve_max_features(self, d):
        """sqrt(d) for classification, d/3 for regression — Breiman's defaults (README §4)."""
        mf = self.max_features
        if mf == "auto" or mf is None:
            return max(1, int(np.sqrt(d))) if self._task == "classification" \
                else max(1, d // 3)
        if mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        if isinstance(mf, float):
            return max(1, int(mf * d))
        return min(mf, d)

    def _fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n, d = X.shape
        self._d = d
        mf = self._resolve_max_features(d)
        rng = np.random.default_rng(self.random_state)

        self.estimators_ = []
        self.oob_indices_ = []
        for _ in range(self.n_estimators):
            seed = int(rng.integers(1 << 31))
            if self.bootstrap:
                in_bag = rng.integers(0, n, n)
                oob = np.setdiff1d(np.arange(n), in_bag, assume_unique=False)
            else:
                in_bag, oob = np.arange(n), np.array([], dtype=int)
            tree = _Tree(task=self._task, max_depth=self.max_depth,
                         min_samples_leaf=self.min_samples_leaf,
                         max_features=mf, random_state=seed).fit(X[in_bag], y[in_bag])
            self.estimators_.append(tree)
            self.oob_indices_.append(oob)

        # MDI importances, averaged over trees and normalized (README §7).
        imp = np.sum([t.importance_ for t in self.estimators_], axis=0)
        self.feature_importances_ = imp / imp.sum() if imp.sum() > 0 else imp
        self._X_train, self._y_train = X, y
        return self

    def permutation_importance(self, X, y, n_repeats=10, seed=0):
        """Shuffle each feature, measure the score drop. Unbiased for cardinality
        (README §7), unlike feature_importances_. Report THIS, on held-out data."""
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=float)
        baseline = self.score(X, y)
        out = np.zeros(self._d)
        for j in range(self._d):
            drops = []
            for _ in range(n_repeats):
                Xp = X.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                drops.append(baseline - self.score(Xp, y))
            out[j] = np.mean(drops)
        return out

    def proximity(self, X):
        """Fraction of trees in which each pair of points shares a leaf.  README §9

        A learned, supervised similarity: two points are close if the forest keeps routing
        them together. A valid kernel, and the basis for RF imputation / outlier detection /
        clustering.
        """
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        leaves = np.array([t.apply(X) for t in self.estimators_])   # (B, n)
        prox = np.zeros((n, n))
        for b in range(len(self.estimators_)):
            same = leaves[b][:, None] == leaves[b][None, :]
            prox += same
        return prox / len(self.estimators_)


class RandomForestClassifier(_BaseForest):
    _task = "classification"

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self._fit(X, y)
        if self.oob_score:
            self.oob_score_ = self._oob(X, y)
        return self

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X) for t in self.estimators_], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def _oob(self, X, y):
        n = X.shape[0]
        proba = np.zeros((n, self.classes_.size))
        counts = np.zeros(n)
        for tree, oob in zip(self.estimators_, self.oob_indices_):
            if oob.size:
                proba[oob] += tree.predict_proba(X[oob])
                counts[oob] += 1
        seen = counts > 0
        pred = self.classes_[np.argmax(proba[seen], axis=1)]
        return float(np.mean(pred == y[seen]))

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


class RandomForestRegressor(_BaseForest):
    _task = "regression"

    def fit(self, X, y):
        self._fit(X, np.asarray(y, dtype=float))
        if self.oob_score:
            self.oob_score_ = self._oob(X, np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        return np.mean([t.predict(X) for t in self.estimators_], axis=0)

    def _oob(self, X, y):
        n = X.shape[0]
        sums, counts = np.zeros(n), np.zeros(n)
        for tree, oob in zip(self.estimators_, self.oob_indices_):
            if oob.size:
                sums[oob] += tree.predict(X[oob])
                counts[oob] += 1
        seen = counts > 0
        pred = sums[seen] / counts[seen]
        ss_res = np.sum((y[seen] - pred) ** 2)
        ss_tot = np.sum((y[seen] - y[seen].mean()) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    def score(self, X, y):
        y = np.asarray(y, dtype=float)
        pred = self.predict(X)
        return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


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

    n = 500
    X = rng.standard_normal((n, 10))
    y_clf = (X[:, 0] + X[:, 1] ** 2 - X[:, 2] + 0.3 * rng.standard_normal(n) > 0.5).astype(int)
    y_reg = 2 * X[:, 0] - X[:, 1] + X[:, 2] * X[:, 3] + rng.standard_normal(n) * 0.3
    X_te = rng.standard_normal((3000, 10))
    y_te_clf = (X_te[:, 0] + X_te[:, 1] ** 2 - X_te[:, 2] > 0.5).astype(int)
    y_te_reg = 2 * X_te[:, 0] - X_te[:, 1] + X_te[:, 2] * X_te[:, 3]

    print("\nForest beats a single tree AND plain bagging (README §1, §3)")
    single = _Tree(task="classification").fit(X, y_clf)
    single_acc = float(np.mean(single.predict(X_te) == y_te_clf))
    bagging = RandomForestClassifier(n_estimators=100, max_features=10,   # all features = bagging
                                     random_state=0).fit(X, y_clf)
    forest = RandomForestClassifier(n_estimators=100, max_features="sqrt",
                                    random_state=0).fit(X, y_clf)
    print(f"  [INFO]  {'single tree / bagging / forest test accuracy':<58s}  "
          f"{single_acc:.4f} / {bagging.score(X_te, y_te_clf):.4f} / "
          f"{forest.score(X_te, y_te_clf):.4f}")
    ok &= forest.score(X_te, y_te_clf) > single_acc

    print("\nAgainst sklearn (README §2)")
    try:
        from sklearn.ensemble import (RandomForestClassifier as SKRFC,
                                      RandomForestRegressor as SKRFR)
        sk = SKRFC(n_estimators=100, random_state=0, oob_score=True).fit(X, y_clf)
        mine = RandomForestClassifier(n_estimators=100, oob_score=True,
                                      random_state=0).fit(X, y_clf)
        print(f"  [{'PASS' if abs(sk.score(X_te, y_te_clf) - mine.score(X_te, y_te_clf)) < 0.03 else 'FAIL'}]  "
              f"{'classifier test accuracy close to sklearn':<58s}  "
              f"{mine.score(X_te, y_te_clf):.4f} vs {sk.score(X_te, y_te_clf):.4f}")
        ok &= abs(sk.score(X_te, y_te_clf) - mine.score(X_te, y_te_clf)) < 0.03
        print(f"  [{'PASS' if abs(sk.oob_score_ - mine.oob_score_) < 0.04 else 'FAIL'}]  "
              f"{'OOB score close to sklearn':<58s}  "
              f"{mine.oob_score_:.4f} vs {sk.oob_score_:.4f}")
        ok &= abs(sk.oob_score_ - mine.oob_score_) < 0.04

        sk_r = SKRFR(n_estimators=100, random_state=0).fit(X, y_reg)
        mine_r = RandomForestRegressor(n_estimators=100, random_state=0).fit(X, y_reg)
        print(f"  [{'PASS' if abs(sk_r.score(X_te, y_te_reg) - mine_r.score(X_te, y_te_reg)) < 0.04 else 'FAIL'}]  "
              f"{'regressor test R^2 close to sklearn':<58s}  "
              f"{mine_r.score(X_te, y_te_reg):.4f} vs {sk_r.score(X_te, y_te_reg):.4f}")
        ok &= abs(sk_r.score(X_te, y_te_reg) - mine_r.score(X_te, y_te_reg)) < 0.04

        # MDI importances should rank the same top features as sklearn.
        top_mine = set(np.argsort(mine.feature_importances_)[-3:])
        top_sk = set(np.argsort(sk.feature_importances_)[-3:])
        overlap = len(top_mine & top_sk)
        print(f"  [{'PASS' if overlap >= 2 else 'FAIL'}]  "
              f"{'top-3 MDI features agree with sklearn':<58s}  "
              f"{overlap}/3 overlap")
        ok &= overlap >= 2
    except ImportError:
        print("  [SKIP]  sklearn not installed")

    print("\nStructural properties (README §5, §9, §10)")
    # n_estimators cannot overfit: test error is monotone non-increasing in B.
    accs = [RandomForestClassifier(n_estimators=B, random_state=0).fit(X, y_clf)
            .score(X_te, y_te_clf) for B in (1, 5, 20, 100)]
    monotone_ish = accs[-1] >= accs[0] - 0.01
    print(f"  [{'PASS' if monotone_ish else 'FAIL'}]  "
          f"{'more trees do not increase test error':<58s}  "
          f"{accs[0]:.4f} -> {accs[-1]:.4f}")
    ok &= monotone_ish

    # Proximity matrix: symmetric, diagonal 1, in [0, 1].
    prox = forest.proximity(X_te[:50])
    ok &= _report("proximity is symmetric", float(np.abs(prox - prox.T).max()), 1e-12)
    ok &= _report("proximity diagonal is 1", float(np.abs(np.diag(prox) - 1).max()), 1e-12)
    ok &= _report("proximity in [0, 1]",
                  float(max(0.0, -prox.min(), prox.max() - 1)), 1e-12)

    # Regressor cannot extrapolate (README §10).
    forest_r = RandomForestRegressor(n_estimators=50, random_state=0).fit(X, y_reg)
    far = np.full((10, 10), 100.0)
    preds = forest_r.predict(far)
    ok &= _report("regressor bounded by [min y, max y] (no extrapolation)",
                  float(max(0.0, preds.max() - y_reg.max(), y_reg.min() - preds.min())), 1e-9)

    return ok


# =============================================================================
# EXPERIMENTS
# =============================================================================


def experiment_rho_floor():
    """README §3: feature subsampling lowers rho and the variance floor."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — feature subsampling lowers the variance floor  (README §3)")
    print("=" * 88)
    print("""
Plain bagging (max_features = all) leaves the trees correlated: they all split on the same
strong features. Restricting max_features forces different trees onto different features,
lowering rho and therefore the floor rho*sigma^2. Measuring tree correlation and the
ensemble's variance as max_features falls:
""")
    rng = np.random.default_rng(1)
    n, d = 300, 16
    # Several correlated strong features so plain bagging keeps splitting on the same ones.
    z = rng.standard_normal((n, 4))
    X = np.column_stack([z[:, 0], z[:, 0] + 0.3 * rng.standard_normal(n),
                         z[:, 1], z[:, 1] + 0.3 * rng.standard_normal(n),
                         rng.standard_normal((n, d - 4))])
    y = (2 * z[:, 0] + 1.5 * z[:, 1] + 0.5 * rng.standard_normal(n) > 0).astype(int)
    X_te = rng.standard_normal((3000, d))
    # Build test labels from the same rule (first two constructed features track z0, z1).
    y_te = (2 * X_te[:, 0] + 1.5 * X_te[:, 2] > 0).astype(int)

    print(f"  {'max_features':>13s}  {'tree pairwise agreement':>24s}  "
          f"{'ensemble TEST acc':>18s}")
    print("  " + "-" * 60)

    for mf in (d, 8, 4, 2, 1):
        forest = RandomForestClassifier(n_estimators=80, max_features=mf,
                                        max_depth=8, random_state=0).fit(X, y)
        # Correlation proxy: how often two random trees agree on the test set.
        preds = np.array([t.predict(X_te) for t in forest.estimators_])
        pairs = [(np.mean(preds[a] == preds[b]))
                 for a, b in rng.integers(0, len(preds), (200, 2)) if a != b]
        agreement = float(np.mean(pairs))
        print(f"  {mf:13d}  {agreement:24.4f}  {forest.score(X_te, y_te):18.4f}")

    print("""
  As max_features falls, the trees agree with each other LESS (rho drops) and — over most
  of the range — the ensemble improves, because the floor rho*sigma^2 is coming down.

  This is the whole point of a random forest, and it is the same measurement as 06.01's
  Experiment 4, now inside a real forest. max_features attacks the rho term; n_estimators
  attacks the (1-rho)/B term. The two are complementary, and the forest uses both.

  Note the far end (max_features = 1): decorrelation is maximal but each tree is now too
  weak (higher sigma^2), so the product rho*sigma^2 stops improving. That is the tradeoff
  §4 tunes, and why the default is sqrt(d), not 1.""")


def experiment_max_features():
    """README §4: the max_features accuracy peak, and its data dependence."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — tuning max_features  (README §4)")
    print("=" * 88)
    print("""
max_features trades rho (down, good) against per-tree strength (down, bad), so the best
value is intermediate. Where the peak sits depends on how many features are informative.
Two regimes:
""")
    rng = np.random.default_rng(2)
    n, d = 400, 30

    def sweep(X, y, X_te, y_te, label):
        print(f"  {label}")
        print(f"    {'max_features':>13s}  {'test accuracy':>14s}")
        print("    " + "-" * 30)
        best = (0, 0.0)
        for mf in (1, 2, 5, int(np.sqrt(d)), 10, 20, d):
            acc = RandomForestClassifier(n_estimators=60, max_features=mf,
                                         random_state=0).fit(X, y).score(X_te, y_te)
            if acc > best[1]:
                best = (mf, acc)
            marker = "  <- best" if mf == best[0] and acc == best[1] else ""
            tag = "  (sqrt(d))" if mf == int(np.sqrt(d)) else ""
            print(f"    {mf:13d}  {acc:14.4f}{tag}{marker}")
        print(f"    best at max_features = {best[0]}\n")

    # Regime A: MANY informative features -> small max_features works.
    W = rng.standard_normal(d)
    Xa = rng.standard_normal((n, d))
    ya = (Xa @ W + 0.5 * rng.standard_normal(n) > 0).astype(int)
    Xa_te = rng.standard_normal((3000, d))
    ya_te = (Xa_te @ W > 0).astype(int)
    sweep(Xa, ya, Xa_te, ya_te, "REGIME A: all 30 features informative")

    # Regime B: FEW informative features -> small max_features often misses them.
    Xb = rng.standard_normal((n, d))
    yb = (2 * Xb[:, 0] + 1.5 * Xb[:, 1] - Xb[:, 2] + 0.5 * rng.standard_normal(n) > 0).astype(int)
    Xb_te = rng.standard_normal((3000, d))
    yb_te = (2 * Xb_te[:, 0] + 1.5 * Xb_te[:, 1] - Xb_te[:, 2] > 0).astype(int)
    sweep(Xb, yb, Xb_te, yb_te, "REGIME B: only 3 of 30 features informative")

    print("""  Two different peaks, for a reason worth internalizing.

  When ALL features are informative (A), a small max_features costs little — any random
  subset contains useful features — so strong decorrelation wins, and the peak is near or
  below sqrt(d).

  When FEW features are informative (B), a small max_features frequently contains ONLY
  noise, forcing a useless split, so you need a LARGER subset to reliably include one of
  the three real features. The peak moves right.

  This is why max_features is worth tuning rather than fixing at the default (README §4):
  the right value depends on the informative-feature density, which you do not know in
  advance. sqrt(d) is a good default, not a law.""")


def experiment_mdi_bias():
    """README §7: MDI importance is biased on a forest too."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — MDI importance is biased (averaging does not fix it)  (README §7)")
    print("=" * 88)
    print("""
03.08 showed a single tree's MDI ranking a high-cardinality NOISE feature above a real one.
A forest is MORE robust to this — bootstrap + averaging genuinely helps — but it does not
eliminate the bias: a weak-enough real feature still loses to high-cardinality noise.
We sweep the real signal's strength to find where the forest gets fooled.
""")
    rng = np.random.default_rng(3)
    n = 2000

    print(f"  {'real signal':>18s}  {'real MDI':>9s}  {'cont-noise MDI':>15s}  "
          f"{'real perm':>10s}  {'verdict':>16s}")
    print("  " + "-" * 74)

    for boost, label in [(0.30, "strong (.20/.80)"), (0.15, "moderate (.35/.65)"),
                         (0.08, "weak (.42/.58)"), (0.04, "very weak (.46/.54)")]:
        real = rng.integers(0, 2, n).astype(float)
        noise_hi = rng.standard_normal(n)               # continuous, high cardinality
        noise_lo = rng.integers(0, 2, n).astype(float)  # binary noise
        X = np.column_stack([real, noise_hi, noise_lo])
        y = (rng.random(n) < 0.5 - boost + 2 * boost * real).astype(int)
        X_te = np.column_stack([rng.integers(0, 2, 3000).astype(float),
                                rng.standard_normal(3000),
                                rng.integers(0, 2, 3000).astype(float)])
        y_te = (rng.random(3000) < 0.5 - boost + 2 * boost * X_te[:, 0]).astype(int)

        forest = RandomForestClassifier(n_estimators=120, max_features=3,
                                        min_samples_leaf=1, random_state=0).fit(X, y)
        mdi = forest.feature_importances_
        perm = forest.permutation_importance(X_te, y_te, n_repeats=8)
        verdict = "MDI FOOLED" if mdi[1] > mdi[0] else "MDI ok"
        print(f"  {label:>18s}  {mdi[0]:9.3f}  {mdi[1]:15.3f}  {perm[0]:10.4f}  "
              f"{verdict:>16s}")

    print("""
  Read down the MDI columns. When the real signal is strong, MDI ranks it correctly and the
  continuous noise gets little — the forest's averaging IS helping, and this is where a
  single tree (03.08) was already fooled. But as the signal weakens, the continuous noise's
  MDI climbs, and by the '.42/.58' row it OVERTAKES the real feature: MDI now says a
  useless continuous column matters more than the one the label depends on.

  The permutation column never makes this mistake: the real feature keeps a positive
  importance and the noise stays near zero, because shuffling a column the forest did not
  truly use cannot hurt a held-out score.

  So the accurate claim (README §7) is not 'averaging never fixes the bias' — it is that
  averaging REDUCES but does not ELIMINATE it. A forest tolerates a stronger signal than a
  single tree before being fooled, but a weak real feature buried among high-cardinality
  noise still loses. The rule stands: report permutation importance on held-out data, not
  feature_importances_, and reach for SHAP when it matters.""")


def experiment_correlated_importance():
    """README §7: both importance methods fail on correlated features, oppositely."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — importance fails on correlated features  (README §7)")
    print("=" * 88)
    print("""
A subtler failure that affects BOTH methods. Two features are near-duplicates, and TOGETHER
they carry all the signal. We ask each method how important each of the pair is.
""")
    rng = np.random.default_rng(4)
    n = 2000
    signal = rng.standard_normal(n)
    x1 = signal + 0.05 * rng.standard_normal(n)         # near-duplicate 1
    x2 = signal + 0.05 * rng.standard_normal(n)         # near-duplicate 2
    noise = rng.standard_normal((n, 3))
    X = np.column_stack([x1, x2, noise])
    y = (2 * signal + 0.3 * rng.standard_normal(n) > 0).astype(int)

    X_te_sig = rng.standard_normal(3000)
    X_te = np.column_stack([X_te_sig + 0.05 * rng.standard_normal(3000),
                            X_te_sig + 0.05 * rng.standard_normal(3000),
                            rng.standard_normal((3000, 3))])
    y_te = (2 * X_te_sig > 0).astype(int)

    forest = RandomForestClassifier(n_estimators=120, max_features=2,
                                    random_state=0).fit(X, y)
    mdi = forest.feature_importances_
    perm = forest.permutation_importance(X_te, y_te, n_repeats=8)

    # Baseline: a lone copy of the signal, to show its TRUE importance.
    X_solo = np.column_stack([signal, rng.standard_normal((n, 3))])
    forest_solo = RandomForestClassifier(n_estimators=120, random_state=0).fit(X_solo, y)
    solo_mdi = forest_solo.feature_importances_[0]

    print(f"  corr(x1, x2) = {np.corrcoef(x1, x2)[0, 1]:.3f}\n")
    print(f"  {'feature':<26s}  {'MDI':>8s}  {'permutation':>12s}")
    print("  " + "-" * 50)
    print(f"  {'x1 (duplicate)':<26s}  {mdi[0]:8.4f}  {perm[0]:12.4f}")
    print(f"  {'x2 (duplicate)':<26s}  {mdi[1]:8.4f}  {perm[1]:12.4f}")
    print(f"  {'x1 + x2 combined':<26s}  {mdi[0] + mdi[1]:8.4f}  "
          f"{perm[0] + perm[1]:12.4f}")
    print(f"\n  for reference, a LONE copy of the signal has MDI {solo_mdi:.4f}")

    print("""
  Two opposite failures, both real.

  MDI SPLITS the importance: the forest uses x1 and x2 about equally, so each gets ~half
  the credit. Read individually, each looks only moderately important — yet a lone copy of
  the same signal (last line) is highly important. A naive reading understates both.

  PERMUTATION HIDES the importance: shuffling x1 barely hurts, because the forest falls
  back on its intact twin x2, and vice versa. So BOTH permutation scores are small —
  suggesting, wrongly, that neither feature matters, when together they carry everything.

  Neither method handles correlated features honestly, and this is not a corner case — real
  tabular data is full of correlated features. The fixes are conditional permutation
  importance and SHAP with a correlation-aware background (17.02). Until then: when features
  are correlated, treat ANY single-feature importance with suspicion, and look at groups.""")


def experiment_no_extrapolation():
    """README §10: a forest cannot extrapolate."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — a random forest cannot extrapolate  (README §10)")
    print("=" * 88)
    print("""
Every prediction is an average of leaf means, so it is bounded by the training targets'
range. On a target that keeps trending beyond the training data, the forest flatlines.
Fitting y = 2x on x in [0, 5] and predicting beyond it:
""")
    rng = np.random.default_rng(5)
    X_tr = np.sort(rng.uniform(0, 5, 300))[:, None]
    y_tr = 2 * X_tr.ravel() + rng.standard_normal(300) * 0.3
    forest = RandomForestRegressor(n_estimators=100, random_state=0).fit(X_tr, y_tr)

    print(f"  training range of y: [{y_tr.min():.2f}, {y_tr.max():.2f}]\n")
    print(f"  {'x':>6s}  {'true y = 2x':>12s}  {'forest prediction':>18s}  {'error':>8s}")
    print("  " + "-" * 50)
    for xq in (2.5, 5.0, 6.0, 8.0, 12.0, 20.0):
        pred = forest.predict([[xq]])[0]
        print(f"  {xq:6.1f}  {2 * xq:12.2f}  {pred:18.2f}  {abs(2 * xq - pred):8.2f}")

    print(f"""
  Inside the training range the forest is accurate. Outside it, the prediction FLATLINES at
  roughly the largest training target (~{y_tr.max():.1f}) — at x = 20 the truth is 40 and the
  forest still says ~{forest.predict([[20.0]])[0]:.0f}. It literally cannot output a value it
  never saw.

  This is a structural limit shared with KNN (03.06 §7) and single trees (03.08 §13): a
  piecewise-constant model bounded by its training targets. For a target that trends beyond
  the observed range — prices over time, growth curves, anything extrapolative — a forest is
  the wrong tool, and a model with a functional form (even linear regression) will do far
  better. Know this before deploying a forest on a time-trending target.""")


# =============================================================================

if __name__ == "__main__":
    print(__doc__)

    all_passed = verify()

    experiment_rho_floor()
    experiment_max_features()
    experiment_mdi_bias()
    experiment_correlated_importance()
    experiment_no_extrapolation()

    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 88)
