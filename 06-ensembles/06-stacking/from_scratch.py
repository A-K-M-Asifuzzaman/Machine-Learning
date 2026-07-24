"""
06.06 — Stacking & Blending, from scratch (NumPy only for the machinery).

Bagging averages, boosting adds; stacking LEARNS how to combine. The novel algorithm here is
the stacking PROTOCOL — generating leakage-free meta-features by out-of-fold cross-validation
(README §3-§4) and training a meta-learner on them — so that is what we build from scratch. The
base learners are the models of earlier chapters (we use scikit-learn's for realism and to
verify against sklearn's own StackingClassifier/Regressor); the meta-learner (ridge / logistic
/ non-negative least squares) is implemented here.

What the experiments demonstrate:
  1. the LEAKAGE TRAP: naive in-sample stacking overweights an overfitting base model and
     collapses on test; out-of-fold meta-features fix it                       (README §2-§3)
  2. DIVERSITY: a stack of different models beats clones and the best single model (README §7)
  3. a SIMPLE meta-learner beats a complex one (overfitting the meta-level)      (README §6)
  4. LEARNED weights vs a plain average                                          (README §8)
  5. STACKING (K-fold) vs BLENDING (single holdout)                              (README §5)

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                                  RandomForestClassifier,
                                  StackingClassifier as SkStackC,
                                  StackingRegressor as SkStackR)
    from sklearn.base import clone
    HAVE_SK = True
except Exception:
    HAVE_SK = False


# =============================================================================
# META-LEARNERS (level-1), from scratch  (README §6)
# =============================================================================


class RidgeMeta:
    """L2-regularized linear meta-learner via the normal equations (03.04)."""

    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, Z, y):
        Z = np.asarray(Z, dtype=float)
        if self.fit_intercept:
            Z = np.column_stack([np.ones(len(Z)), Z])
        d = Z.shape[1]
        A = Z.T @ Z + self.alpha * np.eye(d)
        if self.fit_intercept:
            A[0, 0] -= self.alpha            # do not penalize the intercept
        self.w = np.linalg.solve(A, Z.T @ y)
        return self

    def predict(self, Z):
        Z = np.asarray(Z, dtype=float)
        if self.fit_intercept:
            Z = np.column_stack([np.ones(len(Z)), Z])
        return Z @ self.w

    @property
    def weights_(self):
        return self.w[1:] if self.fit_intercept else self.w


class NNLSMeta:
    """Non-negative, (approximately) sum-to-one linear blend — the classic regression
    stacker (README §6). Projected gradient descent on ||Zw - y||^2 over the simplex."""

    def __init__(self, n_iter=2000, lr=None):
        self.n_iter = n_iter
        self.lr = lr

    def fit(self, Z, y):
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y, dtype=float)
        m = Z.shape[1]
        w = np.full(m, 1.0 / m)
        lr = self.lr or 1.0 / (np.linalg.norm(Z, 2) ** 2 + 1e-9)
        for _ in range(self.n_iter):
            grad = Z.T @ (Z @ w - y)
            w = _project_simplex(w - lr * grad)
        self.w = w
        return self

    def predict(self, Z):
        return np.asarray(Z, dtype=float) @ self.w

    @property
    def weights_(self):
        return self.w


def _project_simplex(v):
    """Euclidean projection onto {w >= 0, sum w = 1} (Duchi et al. 2008)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1
    rho = np.nonzero(u - css / (np.arange(len(v)) + 1) > 0)[0][-1]
    theta = css[rho] / (rho + 1)
    return np.maximum(v - theta, 0)


class LogisticMeta:
    """Multinomial-free binary logistic meta-learner by a few Newton (IRLS) steps."""

    def __init__(self, l2=1.0, n_iter=25):
        self.l2 = l2
        self.n_iter = n_iter

    def fit(self, Z, y):
        Z = np.column_stack([np.ones(len(Z)), np.asarray(Z, dtype=float)])
        y = np.asarray(y, dtype=float)
        d = Z.shape[1]
        w = np.zeros(d)
        for _ in range(self.n_iter):
            p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -35, 35)))
            W = p * (1 - p) + 1e-9
            reg = self.l2 * np.eye(d)
            reg[0, 0] = 0.0
            H = Z.T @ (W[:, None] * Z) + reg
            grad = Z.T @ (p - y) + reg @ w
            w = w - np.linalg.solve(H, grad)
        self.w = w
        return self

    def predict_proba(self, Z):
        Z = np.column_stack([np.ones(len(Z)), np.asarray(Z, dtype=float)])
        return 1.0 / (1.0 + np.exp(-np.clip(Z @ self.w, -35, 35)))

    def predict(self, Z):
        return (self.predict_proba(Z) >= 0.5).astype(int)

    @property
    def weights_(self):
        return self.w[1:]


# =============================================================================
# THE STACKING PROTOCOL  (README §4)
# =============================================================================


def _kfold_indices(n, k, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return [idx[i::k] for i in range(k)]     # k disjoint folds covering all rows


class _BaseStacker:
    def __init__(self, base_models, meta_learner, cv=5, use_proba=True,
                 naive=False, random_state=0):
        self.base_models = base_models       # list of (name, estimator)
        self.meta_learner = meta_learner
        self.cv = cv
        self.use_proba = use_proba
        self.naive = naive                   # if True: in-sample meta-features (the trap)
        self.random_state = random_state

    def _base_predict(self, est, X):
        if self.use_proba and hasattr(est, "predict_proba"):
            return est.predict_proba(X)[:, 1]
        return est.predict(X)

    def _make_meta_features(self, X, y):
        n = len(y)
        Z = np.zeros((n, len(self.base_models)))
        if self.naive:
            # THE LEAKAGE TRAP (README §2): train on all rows, predict the SAME rows.
            for j, (_, est) in enumerate(self.base_models):
                e = clone(est).fit(X, y)
                Z[:, j] = self._base_predict(e, X)
            return Z
        # Out-of-fold: each row predicted by a copy that never trained on it (README §3).
        folds = _kfold_indices(n, self.cv, self.random_state)
        for j, (_, est) in enumerate(self.base_models):
            for k in range(self.cv):
                val = folds[k]
                train = np.concatenate([folds[i] for i in range(self.cv) if i != k])
                e = clone(est).fit(X[train], y[train])
                Z[val, j] = self._base_predict(e, X[val])
        return Z

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        Z = self._make_meta_features(X, y)
        self.meta_learner.fit(Z, y)
        # refit base models on ALL data for test-time prediction (README §4, step 4)
        self.fitted_base_ = [clone(est).fit(X, y) for _, est in self.base_models]
        self.Z_ = Z
        return self

    def _meta_input(self, X):
        return np.column_stack([self._base_predict(e, X) for e in self.fitted_base_])


class StackingRegressor(_BaseStacker):
    def __init__(self, base_models, meta_learner=None, cv=5, naive=False, random_state=0):
        super().__init__(base_models, meta_learner or RidgeMeta(alpha=1.0),
                         cv, use_proba=False, naive=naive, random_state=random_state)

    def predict(self, X):
        return self.meta_learner.predict(self._meta_input(np.asarray(X, dtype=float)))

    def score_r2(self, X, y):
        p = self.predict(X)
        y = np.asarray(y, dtype=float)
        return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


class StackingClassifier(_BaseStacker):
    def __init__(self, base_models, meta_learner=None, cv=5, naive=False, random_state=0):
        super().__init__(base_models, meta_learner or LogisticMeta(l2=1.0),
                         cv, use_proba=True, naive=naive, random_state=random_state)

    def predict(self, X):
        return self.meta_learner.predict(self._meta_input(np.asarray(X, dtype=float)))

    def score_acc(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


# =============================================================================
# DATA
# =============================================================================


def _make_clf(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 8))
    logit = (1.4 * X[:, 0] - 1.8 * X[:, 1] + 1.0 * X[:, 0] * X[:, 2]
             + 0.9 * X[:, 3] - 0.7 * X[:, 4] ** 2)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def _make_heterogeneous_clf(n=1500, seed=0):
    """Three DIFFERENT kinds of structure, each suiting a different model — so no single
    base learner is best everywhere and diverse stacking can win (README §7)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 10))
    s_lin = 1.8 * X[:, 0] + 1.5 * X[:, 1] - 1.2 * X[:, 2]          # smooth => logistic
    s_xor = 2.6 * ((X[:, 3] > 0) ^ (X[:, 4] > 0)) - 1.3           # interaction => trees
    r = np.sqrt(X[:, 5] ** 2 + X[:, 6] ** 2)
    s_loc = 2.2 * (r < 1.0) - 0.7                                  # local pocket => KNN
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(s_lin + s_xor + s_loc)))).astype(int)
    return X, y


def _make_reg(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 8))
    y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2
         + 10 * X[:, 3] + 5 * X[:, 4] + rng.standard_normal(n))
    return X, y


def _diverse_clf():
    return [("gbdt", GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                random_state=0)),
            ("logistic", LogisticRegression(max_iter=1000)),
            ("knn", KNeighborsClassifier(n_neighbors=15))]


def _diverse_reg():
    return [("gbdt", GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                               random_state=0)),
            ("ridge", Ridge(alpha=1.0)),
            ("knn", KNeighborsRegressor(n_neighbors=15))]


# =============================================================================
# VERIFICATION
# =============================================================================


def verify():
    print("=" * 88)
    print("VERIFICATION — stacking protocol vs scikit-learn")
    print("=" * 88)

    # simplex projection sanity
    w = _project_simplex(np.array([0.2, 0.5, -0.3, 0.9]))
    assert np.isclose(w.sum(), 1.0) and (w >= 0).all(), "simplex projection"
    print(f"\n[1] simplex projection: w>=0 and sum(w)=={w.sum():.6f}  ✓ (NNLS meta, README §6)")

    if not HAVE_SK:
        print("\n(scikit-learn unavailable — skipping parity checks)")
        return

    # ---- regression vs sklearn StackingRegressor ----
    Xtr, ytr = _make_reg(900, seed=1)
    Xte, yte = _make_reg(900, seed=2)
    base = _diverse_reg()
    ours = StackingRegressor(base, meta_learner=RidgeMeta(alpha=1.0), cv=5).fit(Xtr, ytr)
    r2_ours = ours.score_r2(Xte, yte)
    sk = SkStackR(estimators=base, final_estimator=Ridge(alpha=1.0), cv=5).fit(Xtr, ytr)
    r2_sk = 1 - np.sum((yte - sk.predict(Xte)) ** 2) / np.sum((yte - yte.mean()) ** 2)
    print(f"[2] regression R^2 (ours) = {r2_ours:.3f}   vs sklearn = {r2_sk:.3f}   "
          f"(gap {abs(r2_ours - r2_sk):.3f})  ✓")
    assert abs(r2_ours - r2_sk) < 0.03, "regression parity"

    # ---- classification vs sklearn StackingClassifier ----
    Xtr, ytr = _make_clf(1000, seed=1)
    Xte, yte = _make_clf(1000, seed=2)
    base = _diverse_clf()
    ours = StackingClassifier(base, meta_learner=LogisticMeta(l2=1.0), cv=5).fit(Xtr, ytr)
    acc_ours = ours.score_acc(Xte, yte)
    sk = SkStackC(estimators=base, final_estimator=LogisticRegression(max_iter=1000),
                  cv=5).fit(Xtr, ytr)
    acc_sk = sk.score(Xte, yte)
    agree = np.mean(ours.predict(Xte) == sk.predict(Xte))
    print(f"[3] binary accuracy (ours) = {acc_ours:.3f}   vs sklearn = {acc_sk:.3f}   "
          f"(agree {agree:.1%})  ✓")
    assert abs(acc_ours - acc_sk) < 0.03, "classification parity"

    print("\nAll verification checks passed.")


# =============================================================================
# EXPERIMENTS
# =============================================================================


def experiment_1_leakage():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — the LEAKAGE TRAP: in-sample vs out-of-fold meta-features")
    print("               (README §2-§3)")
    print("=" * 88)
    Xtr, ytr = _make_clf(1000, seed=10)
    Xte, yte = _make_clf(1000, seed=11)
    # deliberately include a MEMORIZING base model (1-NN) alongside honest ones.
    base = [("gbdt", GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                random_state=0)),
            ("logistic", LogisticRegression(max_iter=1000)),
            ("1nn (overfits)", KNeighborsClassifier(n_neighbors=1))]

    naive = StackingClassifier(base, meta_learner=LogisticMeta(l2=1.0), naive=True).fit(Xtr, ytr)
    oof = StackingClassifier(base, meta_learner=LogisticMeta(l2=1.0), naive=False).fit(Xtr, ytr)

    names = [n for n, _ in base]
    print("\n  Base models (test accuracy alone):")
    for n_, e in base:
        from sklearn.base import clone as _cl
        acc = _cl(e).fit(Xtr, ytr).score(Xte, yte)
        print(f"     {n_:>16s}: {acc:.3f}")

    print(f"\n  Meta-learner weight assigned to each base model:\n")
    print(f"    {'base model':>16s} {'naive (in-sample)':>18s} {'OOF (honest)':>14s}")
    for j, n_ in enumerate(names):
        print(f"    {n_:>16s} {naive.meta_learner.weights_[j]:>18.2f} "
              f"{oof.meta_learner.weights_[j]:>14.2f}")
    print(f"""
  Stacked test accuracy:   naive = {naive.score_acc(Xte, yte):.3f}   OOF = {oof.score_acc(Xte, yte):.3f}

  READING: the 1-NN MEMORIZES the training labels, so its in-sample predictions look like an
  oracle. Naive stacking hands the meta-learner that mirage and it piles weight onto the 1-NN
  — then collapses on test. Out-of-fold meta-features evaluate every base model on data it
  did NOT train on, exposing the 1-NN as mediocre; the meta-learner down-weights it and the
  stack generalizes. Meta-features MUST be out-of-fold (README §3).""")


def experiment_2_diversity():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — diversity is the point: diverse models beat clones (README §7)")
    print("=" * 88)
    # Heterogeneous structure (smooth + interaction + local) with base learners that each
    # have a genuine blind spot: a SHALLOW gbdt (weak on the smooth linear part), logistic
    # (blind to the XOR interaction), KNN (local pocket). No single model is best everywhere.
    Xtr, ytr = _make_heterogeneous_clf(1500, seed=20)
    Xte, yte = _make_heterogeneous_clf(1500, seed=21)

    diverse = [("gbdt (shallow)", GradientBoostingClassifier(n_estimators=60, max_depth=2,
                                                             random_state=0)),
               ("logistic", LogisticRegression(max_iter=1000)),
               ("knn", KNeighborsClassifier(n_neighbors=25))]
    clones = [(f"gbdt{i}", GradientBoostingClassifier(n_estimators=60, max_depth=2,
                                                      random_state=i)) for i in range(3)]

    best_single = max(clone(e).fit(Xtr, ytr).score(Xte, yte) for _, e in diverse)
    st_div = StackingClassifier(diverse, meta_learner=LogisticMeta(l2=1.0)).fit(Xtr, ytr)
    st_clone = StackingClassifier(clones, meta_learner=LogisticMeta(l2=1.0)).fit(Xtr, ytr)

    print(f"""
    {'ensemble':>28s} {'test accuracy':>14s}
    {'best single base model':>28s} {best_single:>14.3f}
    {'stack of 3 CLONES (gbdt x3)':>28s} {st_clone.score_acc(Xte, yte):>14.3f}
    {'stack of 3 DIVERSE models':>28s} {st_div.score_acc(Xte, yte):>14.3f}

  READING: stacking three near-identical GBDTs barely moves past the best single model — the
  meta-learner has nothing to arbitrate. Stacking a GBDT + logistic + KNN, which make
  DIFFERENT errors, lets the meta-learner exploit complementary strengths and clears the best
  single model. Maximize base-model diversity, not individual accuracy (README §7).""")


def experiment_3_meta_complexity():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — a SIMPLE meta-learner beats a complex one (README §6)")
    print("=" * 88)
    Xtr, ytr = _make_reg(900, seed=30)
    Xte, yte = _make_reg(900, seed=31)
    base = _diverse_reg()

    simple = StackingRegressor(base, meta_learner=RidgeMeta(alpha=1.0)).fit(Xtr, ytr)

    class GBDTMeta:   # a deliberately over-powerful meta-learner
        def fit(self, Z, y):
            self.m = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                               random_state=0).fit(Z, y)
            return self
        def predict(self, Z):
            return self.m.predict(Z)
    complex_ = StackingRegressor(base, meta_learner=GBDTMeta()).fit(Xtr, ytr)

    print(f"""
    {'meta-learner':>28s} {'test R^2':>10s}
    {'ridge (simple, regularized)':>28s} {simple.score_r2(Xte, yte):>10.3f}
    {'GBDT (complex)':>28s} {complex_.score_r2(Xte, yte):>10.3f}

  READING: the meta-features are already strong, few, and correlated with the target, so the
  meta-learner's job is gentle — decide relative trust. A GBDT meta-learner finds spurious
  interactions among the base predictions and overfits the meta-level; a regularized linear
  model generalizes better. Keep level-0 rich, level-1 simple (README §6).""")


def experiment_4_learned_vs_average():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — learned weights vs a plain average (README §8)")
    print("=" * 88)
    Xtr, ytr = _make_reg(900, seed=40)
    Xte, yte = _make_reg(900, seed=41)
    # UNEVEN quality: one strong model, two weak ones.
    base = [("gbdt (strong)", GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                        random_state=0)),
            ("ridge (weak)", Ridge(alpha=1.0)),
            ("knn (weak)", KNeighborsRegressor(n_neighbors=40))]

    st = StackingRegressor(base, meta_learner=NNLSMeta()).fit(Xtr, ytr)
    # plain average = fixed equal weights on the same OOF meta-features
    avg_pred = st._meta_input(Xte).mean(axis=1)
    avg_r2 = 1 - np.sum((yte - avg_pred) ** 2) / np.sum((yte - yte.mean()) ** 2)

    print(f"\n  Base-model OOF quality is uneven; non-negative blend weights learned:\n")
    for (n_, _), w in zip(base, st.meta_learner.weights_):
        print(f"     {n_:>16s}: weight {w:.2f}")
    print(f"""
    {'combiner':>24s} {'test R^2':>10s}
    {'plain average (1/3 each)':>24s} {avg_r2:>10.3f}
    {'learned blend (NNLS)':>24s} {st.score_r2(Xte, yte):>10.3f}

  READING: with uneven base quality, the learned blend concentrates weight on the strong GBDT
  and starves the weak models, beating the uniform average that the weak models drag down.
  When base models are of EQUAL quality the two combiners nearly tie — so always compare a
  stack against a plain average and ship the average if the stack is not clearly better
  (README §8).""")


def experiment_5_blending():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — stacking (K-fold OOF) vs blending (single holdout) (README §5)")
    print("=" * 88)
    Xtr, ytr = _make_clf(1000, seed=50)
    Xte, yte = _make_clf(1000, seed=51)
    base = _diverse_clf()

    st = StackingClassifier(base, meta_learner=LogisticMeta(l2=1.0), cv=5).fit(Xtr, ytr)

    # blending: single 20% holdout for meta-features
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ytr))
    hold = perm[:len(ytr) // 5]
    rest = perm[len(ytr) // 5:]
    Zblend = np.column_stack([
        clone(e).fit(Xtr[rest], ytr[rest]).predict_proba(Xtr[hold])[:, 1] for _, e in base])
    meta = LogisticMeta(l2=1.0).fit(Zblend, ytr[hold])
    full_base = [clone(e).fit(Xtr, ytr) for _, e in base]
    Zte = np.column_stack([e.predict_proba(Xte)[:, 1] for e in full_base])
    blend_acc = float(np.mean(meta.predict(Zte) == yte))

    print(f"""
    {'method':>26s} {'meta-train rows':>16s} {'test accuracy':>14s}
    {'stacking (5-fold OOF)':>26s} {len(ytr):>16d} {st.score_acc(Xte, yte):>14.3f}
    {'blending (single 20% hold)':>26s} {len(hold):>16d} {blend_acc:>14.3f}

  READING: here they TIE on accuracy, but the difference that matters is data efficiency —
  stacking's meta-learner trains on OOF predictions for ALL {len(ytr)} rows while blending's sees
  only the {len(hold)} held-out ones. With less data or noisier base models that edge shows up as
  accuracy; the cost is retraining each base model K times. Blending is the simpler shortcut
  when base models are expensive to refit (README §5).""")


if __name__ == "__main__":
    verify()
    experiment_1_leakage()
    experiment_2_diversity()
    experiment_3_meta_complexity()
    experiment_4_learned_vs_average()
    experiment_5_blending()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
