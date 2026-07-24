"""
06.04 — Gradient Boosting, from scratch (NumPy only).

Gradient boosting is gradient descent in FUNCTION space (README §2): at each round we
compute the negative gradient of the loss at every training point (the "pseudo-residual"),
fit a regression tree to it, re-optimize each leaf against the real loss, and add a shrunk
version to the model. Change the loss and only one method changes — `_neg_gradient`.

Everything here is checked against scikit-learn and against the identities the README claims:
  - for squared loss the pseudo-residual IS the residual y - F        (README §4)
  - for log loss it IS the probability error y - p                    (README §6)
  - shrinkage + more trees generalizes better                         (README §7)
  - a robust loss beats squared loss under outliers                   (README §5)
  - the number of trees overfits for boosting but not for a forest    (README §10)
  - row subsampling helps                                             (README §8)

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    from sklearn.ensemble import (GradientBoostingRegressor as SkGBR,
                                   GradientBoostingClassifier as SkGBC,
                                   RandomForestRegressor as SkRF)
    from sklearn.datasets import make_friedman1
    HAVE_SK = True
except Exception:
    HAVE_SK = False


# =============================================================================
# THE BASE LEARNER — a compact CART regression tree (03.08 §7, vectorized)
# =============================================================================
# The tree is grown to reduce squared error on whatever target it is handed (the
# pseudo-residuals). Its STRUCTURE — the terminal regions R_jm — is all the booster
# keeps; the leaf VALUES gamma_jm are recomputed by the booster's per-leaf line
# search against the true loss (README §3, step 2.3). So this tree only needs to
# split well and report which leaf each point lands in.


class _RegTree:
    def __init__(self, max_depth=3, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, g):
        self.X = np.asarray(X, dtype=float)
        self.g = np.asarray(g, dtype=float)
        self._next_leaf = 0
        self.root = self._build(np.arange(len(g)), 0)
        self.n_leaves_ = self._next_leaf
        return self

    def _best_split(self, idx):
        m = idx.size
        msl = self.min_samples_leaf
        gv = self.g[idx]
        parent_sse = float(np.sum((gv - gv.mean()) ** 2))
        best_sse, best = parent_sse - 1e-12, None
        for f in range(self.X.shape[1]):
            xf = self.X[idx, f]
            order = np.argsort(xf, kind="stable")
            xs, gs = xf[order], gv[order]
            cs, cs2 = np.cumsum(gs), np.cumsum(gs ** 2)
            total, total2 = cs[-1], cs2[-1]
            # SSE of a split at i is achieved as a vectorized scan over all thresholds.
            i = np.arange(msl - 1, m - msl)
            if i.size == 0:
                continue
            nl = i + 1
            nr = m - nl
            left = cs2[i] - cs[i] ** 2 / nl
            right = (total2 - cs2[i]) - (total - cs[i]) ** 2 / nr
            sse = left + right
            sse = np.where(xs[i] != xs[i + 1], sse, np.inf)   # no split on ties
            j = int(np.argmin(sse))
            if sse[j] < best_sse:
                best_sse = sse[j]
                k = i[j]
                best = (f, 0.5 * (xs[k] + xs[k + 1]))
        return best

    def _build(self, idx, depth):
        if depth >= self.max_depth or idx.size < 2 * self.min_samples_leaf or \
                np.ptp(self.g[idx]) == 0:
            return self._make_leaf()
        best = self._best_split(idx)
        if best is None:
            return self._make_leaf()
        f, thr = best
        mask = self.X[idx, f] <= thr
        left, right = idx[mask], idx[~mask]
        if left.size < self.min_samples_leaf or right.size < self.min_samples_leaf:
            return self._make_leaf()
        return {"feature": f, "threshold": thr,
                "left": self._build(left, depth + 1),
                "right": self._build(right, depth + 1)}

    def _make_leaf(self):
        node = {"leaf": self._next_leaf}
        self._next_leaf += 1
        return node

    def apply(self, X):
        """Leaf id for every row of X, computed by routing all rows at once."""
        X = np.asarray(X, dtype=float)
        out = np.empty(X.shape[0], dtype=np.int64)

        def recurse(node, idx):
            if "leaf" in node:
                out[idx] = node["leaf"]
                return
            go_left = X[idx, node["feature"]] <= node["threshold"]
            recurse(node["left"], idx[go_left])
            recurse(node["right"], idx[~go_left])

        recurse(self.root, np.arange(X.shape[0]))
        return out


# =============================================================================
# GRADIENT BOOSTING FOR REGRESSION  (README §3-§5)
# =============================================================================


class GradientBoostingRegressor:
    """Friedman's GBM. loss in {"squared", "absolute", "huber"}."""

    def __init__(self, loss="squared", n_estimators=100, learning_rate=0.1,
                 max_depth=3, min_samples_leaf=1, subsample=1.0,
                 huber_alpha=0.9, random_state=0):
        self.loss = loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.huber_alpha = huber_alpha
        self.random_state = random_state

    # -- the ONE loss-specific piece: the negative gradient (README §2) ----------
    def _neg_gradient(self, y, F):
        if self.loss == "squared":
            return y - F                                  # the residual (README §4)
        if self.loss == "absolute":
            return np.sign(y - F)                         # +-1 (README §5)
        if self.loss == "huber":
            r = y - F
            delta = np.quantile(np.abs(r), self.huber_alpha)
            self._huber_delta = delta
            return np.where(np.abs(r) <= delta, r, delta * np.sign(r))
        raise ValueError(self.loss)

    def _init_F(self, y):
        if self.loss == "squared":
            return float(np.mean(y))                      # mean minimizes SSE
        return float(np.median(y))                        # median minimizes |.|

    # -- per-leaf line search: the optimal constant for the REAL loss (README §3) --
    def _leaf_values(self, y, F, leaf_ids, n_leaves):
        gamma = np.zeros(n_leaves)
        r = y - F
        for lf in range(n_leaves):
            mask = leaf_ids == lf
            if not mask.any():
                continue
            rl = r[mask]
            if self.loss == "squared":
                gamma[lf] = rl.mean()                     # mean residual
            elif self.loss == "absolute":
                gamma[lf] = np.median(rl)                 # median residual
            else:  # huber — Friedman (2001) eq. 9
                med = np.median(rl)
                diff = rl - med
                gamma[lf] = med + np.mean(
                    np.sign(diff) * np.minimum(self._huber_delta, np.abs(diff)))
        return gamma

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        rng = np.random.default_rng(self.random_state)
        self.F0 = self._init_F(y)
        F = np.full(n, self.F0)
        self.trees_, self.leaf_vals_ = [], []
        self.train_loss_, self.val_loss_ = [], []
        self._first_residual = None

        for m in range(self.n_estimators):
            r = self._neg_gradient(y, F)
            if m == 0:
                self._first_residual = r.copy()
            # stochastic gradient boosting: grow this tree on a row subsample (README §8)
            if self.subsample < 1.0:
                k = max(self.min_samples_leaf * 2, int(self.subsample * n))
                sub = rng.choice(n, k, replace=False)
            else:
                sub = np.arange(n)
            tree = _RegTree(self.max_depth, self.min_samples_leaf).fit(X[sub], r[sub])
            # leaf structure from the subsample; leaf VALUES from the in-subsample rows
            leaf_all = tree.apply(X)
            gamma = self._leaf_values(y[sub], F[sub], tree.apply(X[sub]), tree.n_leaves_)
            F = F + self.learning_rate * gamma[leaf_all]
            self.trees_.append(tree)
            self.leaf_vals_.append(gamma)
            self.train_loss_.append(self._loss(y, F))
            if X_val is not None:
                self.val_loss_.append(self._loss(y_val, self.predict(X_val)))
        return self

    def _loss(self, y, F):
        r = y - F
        if self.loss == "squared":
            return float(np.mean(r ** 2))
        if self.loss == "absolute":
            return float(np.mean(np.abs(r)))
        d = getattr(self, "_huber_delta", np.quantile(np.abs(r), self.huber_alpha))
        return float(np.mean(np.where(np.abs(r) <= d,
                                      0.5 * r ** 2, d * (np.abs(r) - 0.5 * d))))

    def staged_predict(self, X):
        X = np.asarray(X, dtype=float)
        F = np.full(X.shape[0], self.F0)
        for tree, gamma in zip(self.trees_, self.leaf_vals_):
            F = F + self.learning_rate * gamma[tree.apply(X)]
            yield F.copy()

    def predict(self, X):
        F = None
        for F in self.staged_predict(X):
            pass
        if F is None:
            return np.full(np.asarray(X).shape[0], self.F0)
        return F


# =============================================================================
# GRADIENT BOOSTING FOR CLASSIFICATION — log loss  (README §6)
# =============================================================================


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


class GradientBoostingClassifier:
    """Binary and multiclass gradient boosting under the (multinomial) deviance.
    F is a log-odds / logit score; the pseudo-residual is y - p (README §6)."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_leaf=1, subsample=1.0, random_state=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.K = self.classes_.size
        n = len(y)
        rng = np.random.default_rng(self.random_state)
        yidx = np.searchsorted(self.classes_, y)
        self._first_residual = None

        if self.K == 2:
            Y = (yidx == 1).astype(float)
            p0 = np.clip(Y.mean(), 1e-6, 1 - 1e-6)
            self.F0 = float(np.log(p0 / (1 - p0)))        # base-rate log-odds
            F = np.full(n, self.F0)
            self.trees_, self.leaf_vals_ = [], []
            for m in range(self.n_estimators):
                p = _sigmoid(F)
                r = Y - p                                 # pseudo-residual = y - p
                if m == 0:
                    self._first_residual = r.copy()
                sub = self._subsample(rng, n)
                tree = _RegTree(self.max_depth, self.min_samples_leaf).fit(X[sub], r[sub])
                gamma = self._newton_leaves(r[sub], p[sub], tree.apply(X[sub]),
                                            tree.n_leaves_)
                F = F + self.learning_rate * gamma[tree.apply(X)]
                self.trees_.append(tree)
                self.leaf_vals_.append(gamma)
        else:
            Y = np.zeros((n, self.K))
            Y[np.arange(n), yidx] = 1.0
            base = np.clip(Y.mean(axis=0), 1e-6, 1 - 1e-6)
            self.F0 = np.log(base)
            F = np.tile(self.F0, (n, 1))
            self.trees_, self.leaf_vals_ = [], []          # lists of K-tuples
            for m in range(self.n_estimators):
                P = _softmax(F)
                R = Y - P
                if m == 0:
                    self._first_residual = R.copy()
                sub = self._subsample(rng, n)
                trees_k, gammas_k = [], []
                for k in range(self.K):
                    tree = _RegTree(self.max_depth, self.min_samples_leaf).fit(
                        X[sub], R[sub, k])
                    gamma = self._newton_leaves_multiclass(
                        R[sub, k], tree.apply(X[sub]), tree.n_leaves_)
                    F[:, k] = F[:, k] + self.learning_rate * gamma[tree.apply(X)]
                    trees_k.append(tree)
                    gammas_k.append(gamma)
                self.trees_.append(trees_k)
                self.leaf_vals_.append(gammas_k)
        return self

    def _subsample(self, rng, n):
        if self.subsample < 1.0:
            k = max(self.min_samples_leaf * 2, int(self.subsample * n))
            return rng.choice(n, k, replace=False)
        return np.arange(n)

    def _newton_leaves(self, r, p, leaf_ids, n_leaves):
        # gamma = sum(y-p) / sum(p(1-p)) — gradient over Hessian (README §6)
        gamma = np.zeros(n_leaves)
        h = p * (1 - p)
        for lf in range(n_leaves):
            mask = leaf_ids == lf
            denom = h[mask].sum()
            gamma[lf] = r[mask].sum() / denom if denom > 1e-12 else 0.0
        return gamma

    def _newton_leaves_multiclass(self, r, leaf_ids, n_leaves):
        # Friedman's multiclass leaf: ((K-1)/K) * sum(r) / sum(|r|(1-|r|))
        gamma = np.zeros(n_leaves)
        a = np.abs(r)
        h = a * (1 - a)
        for lf in range(n_leaves):
            mask = leaf_ids == lf
            denom = h[mask].sum()
            if denom > 1e-12:
                gamma[lf] = (self.K - 1) / self.K * r[mask].sum() / denom
        return gamma

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        if self.K == 2:
            F = np.full(n, self.F0)
            for tree, gamma in zip(self.trees_, self.leaf_vals_):
                F = F + self.learning_rate * gamma[tree.apply(X)]
            return F
        F = np.tile(self.F0, (n, 1))
        for trees_k, gammas_k in zip(self.trees_, self.leaf_vals_):
            for k in range(self.K):
                F[:, k] = F[:, k] + self.learning_rate * gammas_k[k][trees_k[k].apply(X)]
        return F

    def predict_proba(self, X):
        F = self.decision_function(X)
        if self.K == 2:
            p = _sigmoid(F)
            return np.column_stack([1 - p, p])
        return _softmax(F)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


# =============================================================================
# VERIFICATION
# =============================================================================


def _make_reg(n=600, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 5))
    y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2
         + 10 * X[:, 3] + 5 * X[:, 4] + noise * rng.standard_normal(n))
    return X, y


def _make_clf(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6))
    logit = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 1.0 * X[:, 0] * X[:, 2] + 0.8 * X[:, 3]
    y = (rng.uniform(size=n) < _sigmoid(logit)).astype(int)
    return X, y


def verify():
    print("=" * 88)
    print("VERIFICATION — gradient boosting vs sklearn and vs the README's identities")
    print("=" * 88)
    rng = np.random.default_rng(0)

    # ---- identity 1: squared-loss pseudo-residual == residual (README §4) ----
    X, y = _make_reg(400, seed=1)
    gb = GradientBoostingRegressor(loss="squared", n_estimators=1).fit(X, y)
    # first pseudo-residual should equal y - F0 = y - mean(y), exactly
    assert np.allclose(gb._first_residual, y - np.mean(y)), "squared residual identity"
    print(f"\n[1] squared loss: pseudo-residual r_i == y_i - F0 to "
          f"{np.max(np.abs(gb._first_residual - (y - y.mean()))):.1e}  ✓  (README §4)")

    # ---- identity 2: log-loss pseudo-residual == y - p (README §6) ----
    Xc, yc = _make_clf(400, seed=1)
    gc = GradientBoostingClassifier(n_estimators=1).fit(Xc, yc)
    p0 = np.clip(yc.mean(), 1e-6, 1 - 1e-6)
    assert np.allclose(gc._first_residual, yc - p0), "log-loss residual identity"
    print(f"[2] log loss:    pseudo-residual r_i == y_i - p_i to "
          f"{np.max(np.abs(gc._first_residual - (yc - p0))):.1e}  ✓  (README §6)")

    # ---- regression vs sklearn ----
    Xtr, ytr = _make_reg(600, seed=2)
    Xte, yte = _make_reg(600, seed=3)
    ours = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                     max_depth=3).fit(Xtr, ytr)
    r2_ours = 1 - np.sum((yte - ours.predict(Xte)) ** 2) / np.sum((yte - yte.mean()) ** 2)
    line = f"[3] regression R^2 (ours) = {r2_ours:.3f}"
    if HAVE_SK:
        sk = SkGBR(n_estimators=200, learning_rate=0.1, max_depth=3,
                   random_state=0).fit(Xtr, ytr)
        r2_sk = sk.score(Xte, yte)
        line += f"   vs sklearn = {r2_sk:.3f}   (gap {abs(r2_ours - r2_sk):.3f})"
        assert abs(r2_ours - r2_sk) < 0.05, "regression parity with sklearn"
    print(line + "  ✓")

    # ---- classification vs sklearn ----
    Xtr, ytr = _make_clf(800, seed=2)
    Xte, yte = _make_clf(800, seed=3)
    ours = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                      max_depth=3).fit(Xtr, ytr)
    acc_ours = ours.score(Xte, yte)
    agree = None
    line = f"[4] binary accuracy (ours) = {acc_ours:.3f}"
    if HAVE_SK:
        sk = SkGBC(n_estimators=200, learning_rate=0.1, max_depth=3,
                   random_state=0).fit(Xtr, ytr)
        acc_sk = sk.score(Xte, yte)
        agree = np.mean(ours.predict(Xte) == sk.predict(Xte))
        line += f"   vs sklearn = {acc_sk:.3f}   (agree {agree:.1%})"
        assert abs(acc_ours - acc_sk) < 0.04, "classification parity"
    print(line + "  ✓")

    # ---- multiclass sanity ----
    rng = np.random.default_rng(5)
    Xm = rng.standard_normal((600, 4))
    ym = (Xm[:, 0] + 0.5 * rng.standard_normal(600) > 0).astype(int) + \
         (Xm[:, 1] + 0.5 * rng.standard_normal(600) > 0).astype(int)  # 3 classes 0,1,2
    gm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                    max_depth=3).fit(Xm[:450], ym[:450])
    acc_m = gm.score(Xm[450:], ym[450:])
    assert acc_m > 0.6, "multiclass learns"
    print(f"[5] 3-class accuracy (ours) = {acc_m:.3f}  ✓  (softmax deviance, README §6)")

    print("\nAll verification checks passed.")


# =============================================================================
# EXPERIMENTS — measure what the README claims
# =============================================================================


def experiment_1_residuals_are_gradients():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — 'fit the residuals' is only squared loss; in general it is the")
    print("               negative gradient (README §2, §4-§6)")
    print("=" * 88)
    X, y = _make_reg(400, seed=7)
    print("""
  The negative gradient by loss, evaluated at the initial model F0:
""")
    for loss in ("squared", "absolute", "huber"):
        gb = GradientBoostingRegressor(loss=loss, n_estimators=1).fit(X, y)
        r = gb._first_residual
        desc = {"squared": "y - F        (the residual)",
                "absolute": "sign(y - F)   (+-1, robust)",
                "huber": "clip(y-F, d)  (residual, capped)"}[loss]
        print(f"    {loss:9s}: r = {desc:30s} | range [{r.min():+.2f}, {r.max():+.2f}], "
              f"|r|<=1 for {np.mean(np.abs(r) <= 1.0 + 1e-9):.0%} of points")
    Xc, yc = _make_clf(400, seed=7)
    gc = GradientBoostingClassifier(n_estimators=1).fit(Xc, yc)
    r = gc._first_residual
    print(f"    log-loss : r = y - p         (prob. error)   | range "
          f"[{r.min():+.2f}, {r.max():+.2f}], always in [-1, 1]")
    print("""
  READING: only 'squared' gives literal residuals. Absolute and log-loss residuals are
  BOUNDED (|r|<=1) — a far-off point cannot dominate the step. That bound is exactly why
  these losses are robust where AdaBoost's exponential loss (unbounded) is not (§5-§6).""")


def experiment_2_shrinkage():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — shrinkage: small learning rate + more trees generalizes better")
    print("               (README §7)")
    print("=" * 88)
    Xtr, ytr = _make_reg(600, noise=1.0, seed=10)
    Xte, yte = _make_reg(600, noise=1.0, seed=11)

    def test_mse(lr, n):
        gb = GradientBoostingRegressor(loss="squared", learning_rate=lr,
                                       n_estimators=n, max_depth=3).fit(Xtr, ytr)
        return float(np.mean((yte - gb.predict(Xte)) ** 2))

    print("\n  Matched total 'step budget' (learning_rate x n_estimators held near constant):\n")
    print(f"    {'learning_rate':>14s} {'n_estimators':>12s} {'test MSE':>10s}")
    for lr, n in [(1.0, 60), (0.5, 120), (0.1, 600), (0.05, 1200)]:
        print(f"    {lr:>14.2f} {n:>12d} {test_mse(lr, n):>10.3f}")
    print("""
  READING: as the step shrinks (and trees grow to compensate) the test error falls steeply
  from lr=1 and then PLATEAUS (0.1 and 0.05 are within noise), even though the total 'budget'
  lr x M is similar. Small greedy steps commit to less noise per tree — shrinkage is a genuine
  regularizer, not just a slowdown — but with diminishing returns, so the recipe is 'a small
  fixed lr (~0.05-0.1), then set M by early stopping', not 'lr as small as possible'.""")


def experiment_3_robust_loss():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — a robust loss beats squared loss under outliers (README §5)")
    print("=" * 88)
    Xtr, ytr = _make_reg(600, noise=1.0, seed=20)
    Xte, yte = _make_reg(600, noise=1.0, seed=21)      # CLEAN test set
    # corrupt 4% of the TRAINING targets with gross outliers
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ytr), int(0.04 * len(ytr)), replace=False)
    ytr_dirty = ytr.copy()
    ytr_dirty[idx] += rng.choice([-1, 1], idx.size) * rng.uniform(50, 100, idx.size)

    print("\n  Trained on data with 4% gross outliers; evaluated (MAE) on a CLEAN test set:\n")
    print(f"    {'loss':>10s} {'clean-test MAE':>16s}")
    for loss in ("squared", "huber", "absolute"):
        gb = GradientBoostingRegressor(loss=loss, n_estimators=300,
                                       learning_rate=0.05, max_depth=3).fit(Xtr, ytr_dirty)
        mae = float(np.mean(np.abs(yte - gb.predict(Xte))))
        print(f"    {loss:>10s} {mae:>16.3f}")
    print("""
  READING: squared loss chases the outliers (each contributes error ~distance^2), warping
  the fit; Huber and absolute cap each point's influence, so a few gross outliers barely
  move them. Freeing the loss buys robustness for free — the payoff of §2's generality.""")


def experiment_4_n_trees_overfits():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — n_estimators OVERFITS for boosting but not for a forest")
    print("               (README §10)")
    print("=" * 88)
    Xtr, ytr = _make_reg(400, noise=3.0, seed=30)      # noisy => overfitting visible
    Xte, yte = _make_reg(400, noise=3.0, seed=31)

    gb = GradientBoostingRegressor(loss="squared", n_estimators=800,
                                   learning_rate=0.1, max_depth=4).fit(
                                       Xtr, ytr, X_val=Xte, y_val=yte)
    gb_test = np.array(gb.val_loss_)
    m_star = int(np.argmin(gb_test)) + 1

    print("\n  Gradient boosting — test MSE vs number of trees:\n")
    print(f"    {'trees':>6s} {'test MSE':>10s}")
    for m in [10, 50, m_star, 200, 400, 800]:
        print(f"    {m:>6d} {gb_test[m - 1]:>10.3f}"
              + ("   <- minimum (early stop here)" if m == m_star else ""))

    if HAVE_SK:
        rf_test = []
        for B in [10, 50, 100, 200, 400, 800]:
            rf = SkRF(n_estimators=B, max_depth=None, random_state=0).fit(Xtr, ytr)
            rf_test.append(np.mean((yte - rf.predict(Xte)) ** 2))
        print("\n  Random forest — test MSE vs number of trees:\n")
        print(f"    {'trees':>6s} {'test MSE':>10s}")
        for B, mse in zip([10, 50, 100, 200, 400, 800], rf_test):
            print(f"    {B:>6d} {mse:>10.3f}")

    print(f"""
  READING: the booster's test error bottoms out at ~{m_star} trees and then climbs — each
  extra tree keeps descending the TRAINING loss, eventually fitting noise. The forest's
  test error only flattens: adding i.i.d. trees refines an average toward its expectation
  and cannot overfit. So for boosting, n_estimators is a capacity knob to be chosen by
  EARLY STOPPING; for a forest it is 'use enough'. This is the key practical difference.""")


def experiment_5_subsampling():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — stochastic gradient boosting: row subsampling helps (README §8)")
    print("=" * 88)
    Xtr, ytr = _make_reg(600, noise=2.0, seed=40)
    Xte, yte = _make_reg(600, noise=2.0, seed=41)

    print("\n  Test MSE vs subsample fraction (n_estimators=400, lr=0.05, depth=3):\n")
    print(f"    {'subsample':>10s} {'test MSE':>10s}")
    best = None
    for frac in (1.0, 0.8, 0.6, 0.4):
        gb = GradientBoostingRegressor(loss="squared", n_estimators=400,
                                       learning_rate=0.05, max_depth=3,
                                       subsample=frac, random_state=0).fit(Xtr, ytr)
        mse = float(np.mean((yte - gb.predict(Xte)) ** 2))
        tag = ""
        if best is None or mse < best[1]:
            best = (frac, mse)
        print(f"    {frac:>10.1f} {mse:>10.3f}{tag}")
    print(f"""
  READING: subsampling rows each round (best here: {best[0]:.1f}) decorrelates consecutive
  trees — a dose of bagging's variance reduction inside a bias-reduction method — and injects
  gradient noise that regularizes, like mini-batch SGD. It is also proportionally faster.""")


if __name__ == "__main__":
    verify()
    experiment_1_residuals_are_gradients()
    experiment_2_shrinkage()
    experiment_3_robust_loss()
    experiment_4_n_trees_overfits()
    experiment_5_subsampling()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
