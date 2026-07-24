"""
06.05 — XGBoost / LightGBM / CatBoost, from scratch (NumPy only).

The libraries add three things to gradient boosting (06.04): a SECOND-ORDER step (use the
Hessian, not just the gradient — Newton, not gradient descent), an EXPLICIT regularized
objective (gamma per leaf, L2 lambda on leaf weights, baked into the split criterion), and
systems engineering (histograms, sparsity-aware splits). This file implements the first two
faithfully and checks them against the real `xgboost` library; it also demonstrates the
ideas behind the third and behind LightGBM's leaf-wise growth and CatBoost's ordered target
statistics.

The mathematical core (README §2-§5), verbatim:
    leaf weight        w_j* = -G_j / (H_j + lambda)
    split gain         0.5 [ G_L^2/(H_L+l) + G_R^2/(H_R+l) - G^2/(H+l) ] - gamma
with G_j = sum of gradients, H_j = sum of Hessians in the leaf.

Run:  python3 from_scratch.py
"""

import time
import numpy as np

try:
    import xgboost as xgb
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

try:
    from sklearn.ensemble import GradientBoostingRegressor as SkGBR
    HAVE_SK = True
except Exception:
    HAVE_SK = False


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


# =============================================================================
# LOSSES — each supplies gradient g = dL/dF and Hessian h = d2L/dF2 (README §3)
# =============================================================================


class SquaredLoss:
    def init_F(self, y):
        return float(np.mean(y))            # base_score = mean

    def grad_hess(self, y, F):
        return F - y, np.ones_like(y)       # g = F - y, h = 1

    def eval(self, y, F):
        return float(np.mean((y - F) ** 2))

    def predict(self, F):
        return F


class LogisticLoss:
    def init_F(self, y):
        return 0.0                          # base_score = 0.5  =>  margin 0

    def grad_hess(self, y, F):
        p = _sigmoid(F)
        return p - y, np.maximum(p * (1 - p), 1e-12)   # g = p - y, h = p(1-p)

    def eval(self, y, F):
        p = np.clip(_sigmoid(F), 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def predict(self, F):
        return _sigmoid(F)


# =============================================================================
# ONE SECOND-ORDER TREE  (README §4-§5)
# =============================================================================


class _SecondOrderTree:
    """Grows one tree that minimizes the regularized second-order objective, using the
    exact gain formula of README §5. Supports level-wise (XGBoost default) and leaf-wise
    (LightGBM) growth, and exact or histogram split finding (README §6)."""

    def __init__(self, max_depth=6, reg_lambda=1.0, gamma=0.0, min_child_weight=1.0,
                 grow="levelwise", max_leaves=31, method="exact", bin_edges=None,
                 Xbin=None, n_bins=64):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.grow = grow
        self.max_leaves = max_leaves
        self.method = method
        self.bin_edges = bin_edges
        self.Xbin = Xbin
        self.n_bins = n_bins
        self.n_candidates_ = 0        # thresholds evaluated — for the histogram experiment

    def _leaf_weight(self, G, H):
        return -G / (H + self.reg_lambda)                      # w* = -G/(H+lambda)

    # -- exact split scan: cumulative G,H over sorted values (README §5) ----------
    def _best_split_exact(self, idx):
        g, h, X = self.g, self.h, self.X
        G, H, lam = g[idx].sum(), h[idx].sum(), self.reg_lambda
        best = None
        best_gain = 0.0
        for f in range(X.shape[1]):
            xf = X[idx, f]
            order = np.argsort(xf, kind="stable")
            xs = xf[order]
            cg = np.cumsum(g[idx][order])
            ch = np.cumsum(h[idx][order])
            GL, HL = cg[:-1], ch[:-1]
            GR, HR = G - GL, H - HL
            self.n_candidates_ += xs.size - 1
            gain = 0.5 * (GL ** 2 / (HL + lam) + GR ** 2 / (HR + lam)
                          - G ** 2 / (H + lam)) - self.gamma
            valid = (xs[:-1] != xs[1:]) & (HL >= self.min_child_weight) \
                & (HR >= self.min_child_weight)
            gain = np.where(valid, gain, -np.inf)
            j = int(np.argmax(gain))
            if gain[j] > best_gain:
                best_gain = gain[j]
                best = (f, 0.5 * (xs[j] + xs[j + 1]), gain[j])
        return best

    # -- histogram split scan: accumulate G,H per bin, scan bins (README §6) ------
    def _best_split_hist(self, idx):
        g, h, lam = self.g, self.h, self.reg_lambda
        Xb = self.Xbin
        G, H = g[idx].sum(), h[idx].sum()
        best, best_gain = None, 0.0
        for f in range(Xb.shape[1]):
            b = Xb[idx, f]
            gh = np.bincount(b, weights=g[idx], minlength=self.n_bins)
            hh = np.bincount(b, weights=h[idx], minlength=self.n_bins)
            GL = np.cumsum(gh)[:-1]
            HL = np.cumsum(hh)[:-1]
            GR, HR = G - GL, H - HL
            self.n_candidates_ += self.n_bins - 1
            gain = 0.5 * (GL ** 2 / (HL + lam) + GR ** 2 / (HR + lam)
                          - G ** 2 / (H + lam)) - self.gamma
            valid = (HL >= self.min_child_weight) & (HR >= self.min_child_weight)
            gain = np.where(valid, gain, -np.inf)
            j = int(np.argmax(gain))
            if gain[j] > best_gain:
                best_gain = gain[j]
                thr = self.bin_edges[f][j + 1]     # split at the bin's upper edge
                best = (f, thr, gain[j])
        return best

    def _best_split(self, idx):
        return (self._best_split_hist(idx) if self.method == "hist"
                else self._best_split_exact(idx))

    def fit(self, X, g, h):
        self.X, self.g, self.h = np.asarray(X, dtype=float), g, h
        self.nodes = []
        root = {"idx": np.arange(len(g)), "depth": 0, "is_leaf": True}
        self.nodes.append(root)
        if self.grow == "leafwise":
            self._grow_leafwise(root)
        else:
            self._grow_levelwise(root)
        # freeze leaf weights
        for nd in self.nodes:
            if nd["is_leaf"]:
                G, H = self.g[nd["idx"]].sum(), self.h[nd["idx"]].sum()
                nd["weight"] = self._leaf_weight(G, H)
        self.n_leaves_ = sum(nd["is_leaf"] for nd in self.nodes)
        return self

    def _split_node(self, nd, best):
        f, thr, gain = best
        mask = self.X[nd["idx"], f] <= thr
        left = {"idx": nd["idx"][mask], "depth": nd["depth"] + 1, "is_leaf": True}
        right = {"idx": nd["idx"][~mask], "depth": nd["depth"] + 1, "is_leaf": True}
        nd.update(is_leaf=False, feature=f, threshold=thr, left=left, right=right)
        self.nodes.append(left)
        self.nodes.append(right)
        return left, right

    def _grow_levelwise(self, node):
        if node["depth"] >= self.max_depth or node["idx"].size < 2:
            return
        best = self._best_split(node["idx"])
        if best is None:
            return
        left, right = self._split_node(node, best)
        self._grow_levelwise(left)
        self._grow_levelwise(right)

    def _grow_leafwise(self, root):
        # frontier of (gain, tie, node, split); always split the highest-gain leaf.
        import heapq
        counter = 0
        frontier = []

        def consider(nd):
            nonlocal counter
            if nd["depth"] >= self.max_depth or nd["idx"].size < 2:
                return
            best = self._best_split(nd["idx"])
            if best is not None:
                heapq.heappush(frontier, (-best[2], counter, nd, best))
                counter += 1

        consider(root)
        n_leaves = 1
        while frontier and n_leaves < self.max_leaves:
            _, _, nd, best = heapq.heappop(frontier)
            if not nd["is_leaf"]:
                continue
            left, right = self._split_node(nd, best)
            n_leaves += 1
            consider(left)
            consider(right)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        out = np.zeros(X.shape[0])

        def recurse(nd, rows):
            if nd["is_leaf"]:
                out[rows] = nd["weight"]
                return
            go_left = X[rows, nd["feature"]] <= nd["threshold"]
            recurse(nd["left"], rows[go_left])
            recurse(nd["right"], rows[~go_left])

        recurse(self.nodes[0], np.arange(X.shape[0]))
        return out


# =============================================================================
# THE BOOSTER  (README §2-§5)
# =============================================================================


class XGBoostFromScratch:
    def __init__(self, loss="squared", n_estimators=100, learning_rate=0.3,
                 max_depth=6, reg_lambda=1.0, gamma=0.0, min_child_weight=1.0,
                 subsample=1.0, colsample=1.0, grow="levelwise", max_leaves=31,
                 method="exact", n_bins=64, first_order=False, random_state=0):
        self.loss_name = loss
        self.loss = SquaredLoss() if loss == "squared" else LogisticLoss()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.grow = grow
        self.max_leaves = max_leaves
        self.method = method
        self.n_bins = n_bins
        self.first_order = first_order          # if True, force h=1 (=> gradient boosting)
        self.random_state = random_state

    def _bin(self, X):
        # quantile bin edges per feature; map values to bin indices (README §6)
        edges, Xbin = [], np.empty_like(X, dtype=np.int64)
        for f in range(X.shape[1]):
            qs = np.quantile(X[:, f], np.linspace(0, 1, self.n_bins + 1))
            qs = np.unique(qs)
            e = np.concatenate([[-np.inf], qs[1:-1], [np.inf]]) if qs.size > 2 \
                else np.array([-np.inf, np.inf])
            Xbin[:, f] = np.clip(np.searchsorted(e, X[:, f], side="right") - 1,
                                 0, len(e) - 2)
            # pad edges to n_bins so indices are consistent
            pad = np.full(self.n_bins + 1, e[-1])
            pad[:len(e)] = e
            edges.append(pad)
        return edges, Xbin

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        rng = np.random.default_rng(self.random_state)
        self.F0 = self.loss.init_F(y)
        F = np.full(n, self.F0)
        self.trees_ = []
        self.total_candidates_ = 0
        bin_edges = Xbin = None
        if self.method == "hist":
            bin_edges, Xbin = self._bin(X)

        for m in range(self.n_estimators):
            g, h = self.loss.grad_hess(y, F)
            if self.first_order:
                h = np.ones_like(h)                 # ablate curvature => 1st-order GBM
            if self.subsample < 1.0:
                k = max(2, int(self.subsample * n))
                sub = rng.choice(n, k, replace=False)
                gg, hh, Xs = np.zeros(n), np.zeros(n), X
                gg[sub], hh[sub] = g[sub], h[sub]   # zero-weight the held-out rows
                g, h = gg, hh
            tree = _SecondOrderTree(
                max_depth=self.max_depth, reg_lambda=self.reg_lambda, gamma=self.gamma,
                min_child_weight=self.min_child_weight, grow=self.grow,
                max_leaves=self.max_leaves, method=self.method,
                bin_edges=bin_edges, Xbin=Xbin, n_bins=self.n_bins).fit(X, g, h)
            F = F + self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
            self.total_candidates_ += tree.n_candidates_
        return self

    def decision_function(self, X):
        F = np.full(np.asarray(X).shape[0], self.F0)
        for tree in self.trees_:
            F = F + self.learning_rate * tree.predict(X)
        return F

    def predict(self, X):
        return self.loss.predict(self.decision_function(X))

    def predict_label(self, X):
        return (self.predict(X) >= 0.5).astype(int)

    def score_r2(self, X, y):
        p = self.predict(X)
        return 1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)

    def score_acc(self, X, y):
        return float(np.mean(self.predict_label(X) == np.asarray(y)))


# =============================================================================
# DATA
# =============================================================================


def _make_reg(n=600, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 6))
    y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2
         + 10 * X[:, 3] + 5 * X[:, 4] + noise * rng.standard_normal(n))
    return X, y


def _make_clf(n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6))
    logit = 1.5 * X[:, 0] - 2 * X[:, 1] + X[:, 0] * X[:, 2] + 0.8 * X[:, 3]
    y = (rng.uniform(size=n) < _sigmoid(logit)).astype(int)
    return X, y


# =============================================================================
# VERIFICATION — against the real xgboost library
# =============================================================================


def verify():
    print("=" * 88)
    print("VERIFICATION — second-order booster vs the real xgboost library")
    print("=" * 88)

    # ---- leaf weight and gain formulas, checked by hand on one node ----
    rng = np.random.default_rng(0)
    g = rng.standard_normal(20)
    h = rng.uniform(0.5, 1.5, 20)
    lam = 1.0
    G, H = g.sum(), h.sum()
    w = -G / (H + lam)
    # objective of one leaf: G*w + 0.5*(H+lam)*w^2  == -0.5 G^2/(H+lam)
    obj = G * w + 0.5 * (H + lam) * w ** 2
    assert np.isclose(obj, -0.5 * G ** 2 / (H + lam)), "structure score identity"
    print(f"\n[1] leaf weight w* = -G/(H+lambda) minimizes the leaf objective; "
          f"score = -0.5 G^2/(H+lambda) to {abs(obj + 0.5*G**2/(H+lam)):.1e}  ✓ (README §4)")

    # ---- regression vs xgboost ----
    Xtr, ytr = _make_reg(700, seed=2)
    Xte, yte = _make_reg(700, seed=3)
    params = dict(n_estimators=120, learning_rate=0.3, max_depth=4,
                  reg_lambda=1.0, gamma=0.0, min_child_weight=1.0)
    ours = XGBoostFromScratch(loss="squared", **params).fit(Xtr, ytr)
    r2_ours = ours.score_r2(Xte, yte)
    line = f"[2] regression R^2 (ours) = {r2_ours:.3f}"
    if HAVE_XGB:
        sk = xgb.XGBRegressor(objective="reg:squarederror", tree_method="exact",
                              base_score=float(ytr.mean()), **params).fit(Xtr, ytr)
        r2_x = 1 - np.sum((yte - sk.predict(Xte)) ** 2) / np.sum((yte - yte.mean()) ** 2)
        line += f"   vs xgboost = {r2_x:.3f}   (gap {abs(r2_ours - r2_x):.3f})"
        assert abs(r2_ours - r2_x) < 0.04, "regression parity with xgboost"
    print(line + "  ✓")

    # ---- classification vs xgboost ----
    Xtr, ytr = _make_clf(1000, seed=2)
    Xte, yte = _make_clf(1000, seed=3)
    ours = XGBoostFromScratch(loss="logistic", **params).fit(Xtr, ytr)
    acc_ours = ours.score_acc(Xte, yte)
    line = f"[3] binary accuracy (ours) = {acc_ours:.3f}"
    if HAVE_XGB:
        sk = xgb.XGBClassifier(objective="binary:logistic", tree_method="exact",
                               base_score=0.5, eval_metric="logloss", **params).fit(Xtr, ytr)
        acc_x = sk.score(Xte, yte)
        agree = np.mean(ours.predict_label(Xte) == sk.predict(Xte))
        line += f"   vs xgboost = {acc_x:.3f}   (agree {agree:.1%})"
        assert abs(acc_ours - acc_x) < 0.04, "classification parity with xgboost"
    print(line + "  ✓")

    print("\nAll verification checks passed.")


# =============================================================================
# EXPERIMENTS
# =============================================================================


def experiment_1_newton_vs_gradient():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — second-order (Newton) vs first-order (gradient) step (README §3)")
    print("=" * 88)
    Xtr, ytr = _make_clf(1000, seed=10)
    Xte, yte = _make_clf(1000, seed=11)
    common = dict(loss="logistic", n_estimators=60, learning_rate=0.3, max_depth=3,
                  reg_lambda=1.0)
    second = XGBoostFromScratch(first_order=False, **common).fit(Xtr, ytr)
    first = XGBoostFromScratch(first_order=True, **common).fit(Xtr, ytr)
    print("""
  Same booster, same rounds; the only change is whether each leaf uses the Hessian.
  Leaf weight:  2nd-order  -G/(H+lambda)   vs   1st-order  -G/(count+lambda)
""")
    print(f"    {'round':>6s} {'2nd-order test logloss':>24s} {'1st-order test logloss':>24s}")
    for m in (5, 10, 20, 40, 60):
        F2 = second.loss.eval(yte, np.full(len(yte), second.F0)
                              + second.learning_rate
                              * sum(t.predict(Xte) for t in second.trees_[:m]))
        F1 = first.loss.eval(yte, np.full(len(yte), first.F0)
                             + first.learning_rate
                             * sum(t.predict(Xte) for t in first.trees_[:m]))
        print(f"    {m:>6d} {F2:>24.4f} {F1:>24.4f}")
    print("""
  READING: using the curvature h = p(1-p) rescales each leaf by how confident the current
  model already is, so Newton CONVERGES FASTER — it is well ahead through rounds 5-40 and
  reaches a lower optimum in fewer rounds. Being a stronger step per round, it also passes
  its optimum and starts overfitting sooner (by round 60 the still-descending first-order run
  edges level) — the same early-stopping lesson as 06.04 §10, now per-step-strength. Either
  way the second-order step changes the update itself; it is not a minor optimization.""")


def experiment_2_gamma_lambda():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — gamma prunes splits, lambda shrinks leaves (README §5)")
    print("=" * 88)
    Xtr, ytr = _make_reg(500, noise=3.0, seed=20)   # noisy => regularization matters
    Xte, yte = _make_reg(500, noise=3.0, seed=21)

    print("\n  gamma (min split gain) — larger gamma refuses low-gain splits => smaller trees:\n")
    print(f"    {'gamma':>8s} {'avg leaves/tree':>16s} {'test R^2':>10s}")
    for gm in (0.0, 1.0, 5.0, 20.0):
        b = XGBoostFromScratch(loss="squared", n_estimators=60, learning_rate=0.3,
                               max_depth=5, reg_lambda=1.0, gamma=gm).fit(Xtr, ytr)
        leaves = np.mean([t.n_leaves_ for t in b.trees_])
        print(f"    {gm:>8.1f} {leaves:>16.1f} {b.score_r2(Xte, yte):>10.3f}")

    print("\n  lambda (L2 on leaf weights) — larger lambda shrinks weights toward 0:\n")
    print(f"    {'lambda':>8s} {'mean |leaf weight|':>18s} {'test R^2':>10s}")
    for lam in (0.0, 1.0, 10.0, 100.0):
        b = XGBoostFromScratch(loss="squared", n_estimators=60, learning_rate=0.3,
                               max_depth=4, reg_lambda=lam, gamma=0.0).fit(Xtr, ytr)
        wmean = np.mean([abs(nd["weight"]) for t in b.trees_ for nd in t.nodes
                         if nd["is_leaf"]])
        print(f"    {lam:>8.1f} {wmean:>18.4f} {b.score_r2(Xte, yte):>10.3f}")
    print("""
  READING: gamma is a minimum-gain-to-split threshold read straight off the gain formula
  (built-in pre-pruning): raise it and trees shrink. lambda sits in every denominator
  -G/(H+lambda): raise it and leaf weights shrink toward zero. Both trade fit for
  simplicity from INSIDE the objective, not by pruning afterward.""")


def experiment_3_histogram():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — histogram vs exact split finding (README §6)")
    print("=" * 88)
    Xtr, ytr = _make_reg(2000, seed=30)
    Xte, yte = _make_reg(2000, seed=31)
    common = dict(loss="squared", n_estimators=60, learning_rate=0.3, max_depth=4,
                  reg_lambda=1.0)

    t0 = time.time()
    ex = XGBoostFromScratch(method="exact", **common).fit(Xtr, ytr)
    t_ex = time.time() - t0
    t0 = time.time()
    hi = XGBoostFromScratch(method="hist", n_bins=64, **common).fit(Xtr, ytr)
    t_hi = time.time() - t0

    print(f"""
  n = {len(ytr)} rows, {Xtr.shape[1]} features, {common['n_estimators']} trees, depth {common['max_depth']}:

    {'method':>8s} {'test R^2':>10s} {'candidate splits scanned':>26s} {'fit time (s)':>14s}
    {'exact':>8s} {ex.score_r2(Xte, yte):>10.3f} {ex.total_candidates_:>26,d} {t_ex:>14.2f}
    {'hist(64)':>8s} {hi.score_r2(Xte, yte):>10.3f} {hi.total_candidates_:>26,d} {t_hi:>14.2f}

  READING: the histogram scans ~{ex.total_candidates_ / max(hi.total_candidates_,1):.0f}x fewer
  candidate thresholds (bins, not rows) for essentially the same accuracy. On millions of
  rows this is the difference between minutes and seconds; the small precision loss from
  splitting only at bin edges is negligible and even mildly regularizing.""")


def experiment_4_leafwise():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — leaf-wise (LightGBM) vs level-wise (XGBoost) growth (README §7)")
    print("=" * 88)
    Xtr, ytr = _make_clf(1200, seed=40)
    Xte, yte = _make_clf(1200, seed=41)
    print("""
  Same leaf budget per tree; the only change is WHICH leaf gets split next.
""")
    print(f"    {'growth':>12s} {'max_leaves':>11s} {'train logloss':>14s} {'test logloss':>13s}")
    for leaves in (8, 16, 32):
        lw = XGBoostFromScratch(loss="logistic", n_estimators=80, learning_rate=0.3,
                                max_depth=12, grow="leafwise", max_leaves=leaves,
                                reg_lambda=1.0).fit(Xtr, ytr)
        lv = XGBoostFromScratch(loss="logistic", n_estimators=80, learning_rate=0.3,
                                max_depth=int(np.log2(leaves)), grow="levelwise",
                                reg_lambda=1.0).fit(Xtr, ytr)
        for name, b in (("leaf-wise", lw), ("level-wise", lv)):
            tr = b.loss.eval(ytr, b.decision_function(Xtr))
            te = b.loss.eval(yte, b.decision_function(Xte))
            print(f"    {name:>12s} {leaves:>11d} {tr:>14.4f} {te:>13.4f}")
    print("""
  READING: for the same number of leaves, leaf-wise growth reaches a LOWER TRAINING loss —
  it spends each leaf on the highest-gain split anywhere in the tree, not evenly by level.
  That extra fitting power is why LightGBM is accurate, and also why it overfits more easily
  and must be reined in with num_leaves rather than max_depth (README §7).""")


def experiment_5_ordered_target_stats():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — CatBoost's ordered target statistics vs naive mean encoding")
    print("               (README §8)")
    print("=" * 88)
    # High-cardinality categorical that is PURE NOISE (independent of y). Naive mean
    # encoding will still look predictive on the training set — that is the leakage.
    rng = np.random.default_rng(50)
    n = 2000
    n_cats = 400                       # ~5 rows per category
    cat = rng.integers(0, n_cats, n)
    y = rng.integers(0, 2, n).astype(float)   # label independent of cat
    tr, te = np.arange(n) < n // 2, np.arange(n) >= n // 2

    # --- naive mean encoding: each row's category -> mean label over ALL train rows
    #     of that category (INCLUDING the row itself) ---
    global_mean = y[tr].mean()
    naive = np.full(n, global_mean)
    for c in range(n_cats):
        m = (cat == c) & tr
        if m.any():
            val = y[m].mean()
            naive[(cat == c)] = val

    # --- ordered TS: each row encoded by mean label of PRECEDING rows (same cat) only ---
    perm = rng.permutation(n)
    ordered = np.full(n, global_mean)
    seen_sum = {}
    seen_cnt = {}
    for i in perm:
        c = cat[i]
        s, k = seen_sum.get(c, 0.0), seen_cnt.get(c, 0)
        ordered[i] = (s + global_mean) / (k + 1)     # prior-smoothed, preceding-only
        seen_sum[c] = s + y[i]
        seen_cnt[c] = k + 1

    def leakage(enc):
        # correlation of the encoding with the label, train vs test. For a NOISE feature
        # an honest encoding has ~0 on both; a leaky one is high on train, ~0 on test.
        ctr = abs(np.corrcoef(enc[tr], y[tr])[0, 1])
        cte = abs(np.corrcoef(enc[te], y[te])[0, 1])
        return ctr, cte

    ntr, nte = leakage(naive)
    otr, ote = leakage(ordered)
    print(f"""
  A high-cardinality categorical ({n_cats} levels, ~{n // n_cats} rows each) that is PURE
  NOISE — independent of the label. An honest encoding should correlate ~0 with y on BOTH
  train and test. Correlation |corr(encoding, y)|:

    {'encoding':>16s} {'train':>8s} {'test':>8s}
    {'naive mean':>16s} {ntr:>8.3f} {nte:>8.3f}   <- high on train, ~0 on test = LEAKAGE
    {'ordered TS':>16s} {otr:>8.3f} {ote:>8.3f}   <- ~0 on both = honest

  READING: naive mean encoding lets each row see its own label through its category's mean,
  so a noise feature looks predictive in training and collapses on test — a model trusts it
  and overfits. CatBoost's ordered target statistic encodes each row using only PRECEDING
  rows, so a row never sees its own label; the noise feature reads as noise on both splits.
  Ordered boosting applies the same 'use only the past' fix to the GRADIENTS themselves.""")


if __name__ == "__main__":
    verify()
    experiment_1_newton_vs_gradient()
    experiment_2_gamma_lambda()
    experiment_3_histogram()
    experiment_4_leafwise()
    experiment_5_ordered_target_stats()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
