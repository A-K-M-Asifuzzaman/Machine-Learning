"""
06.03 — AdaBoost from Scratch
=============================

Boosting: combine many WEAK learners into one strong one, sequentially, each fixing the
last's mistakes. The mirror image of bagging (variance) — boosting reduces BIAS.

The chapter's thesis is that AdaBoost is not a bag of heuristics: it is greedy forward-
stagewise minimization of exponential loss (README §5), and every formula follows. The
experiments verify that, and the consequences (exponential error decay, noise fragility,
margins).

Implemented here
----------------
    DecisionStump           a depth-1 weighted weak learner
    AdaBoost                SAMME (discrete) for binary and multiclass
        margins()           the normalized margin distribution (README §9)
        staged_predict()    predictions after each round, for learning curves

Run it
------
    python from_scratch.py

Verified against sklearn, then five experiments:
  1. Weight redistribution: misclassified points get heavier each round (README §4)
  2. Training error decays exponentially (README §6)
  3. AdaBoost overfits noise while a forest plateaus (README §8, §10)
  4. AdaBoost weights ARE the exponential-loss gradient (README §5, §7)
  5. Margins keep improving after training error hits zero (README §9)

Reference: README.md sections 3-10.
"""

from __future__ import annotations

import numpy as np

# =============================================================================
# THE WEAK LEARNER  (README §2)
# =============================================================================


class DecisionStump:
    """A depth-1 decision tree fitted to WEIGHTED data.

    The canonical weak learner (README §2): one threshold on one feature. Weak on purpose
    — a strong base learner would leave nothing for later rounds to correct.

    Fits by scanning every feature and threshold for the one minimizing weighted error,
    trying both polarities (predict +1 above or below the threshold).
    """

    def fit(self, X, y, sample_weight):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.asarray(sample_weight, dtype=float)
        n, d = X.shape

        best_err = np.inf
        for f in range(d):
            xf = X[:, f]
            thresholds = np.unique(xf)
            # Candidate cut points: midpoints between distinct values, plus the extremes.
            if thresholds.size > 1:
                thresholds = np.concatenate([[thresholds[0] - 1],
                                             (thresholds[:-1] + thresholds[1:]) / 2])
            for thr in thresholds:
                for polarity in (1, -1):
                    pred = np.where(polarity * (xf - thr) >= 0, 1.0, -1.0)
                    err = float(np.sum(w[pred != y]))
                    if err < best_err:
                        best_err = err
                        self.feature_, self.threshold_ = f, thr
                        self.polarity_ = polarity
        self.error_ = best_err
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.where(self.polarity_ * (X[:, self.feature_] - self.threshold_) >= 0,
                        1.0, -1.0)


# =============================================================================
# ADABOOST  (README §3, §5)
# =============================================================================


class AdaBoost:
    """AdaBoost, SAMME variant (Zhu et al. 2009), which handles K >= 2 classes.

    For K = 2 this is exactly Freund & Schapire's original AdaBoost (README §3). For K > 2
    the vote gains a +log(K-1) term so a learner need only beat 1/K (random) rather than
    1/2 to contribute.

    Every step is derived from greedy forward-stagewise minimization of exponential loss
    (README §5): the sample weights ARE the current per-point exponential loss, the vote
    alpha is the closed-form loss minimizer, and the reweighting is the loss update.
    """

    def __init__(self, n_estimators=50, base_estimator=None, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator or DecisionStump
        self.learning_rate = learning_rate

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y_raw = np.asarray(y).ravel()
        self.classes_ = np.unique(y_raw)
        self.K_ = self.classes_.size
        n = X.shape[0]

        self.estimators_ = []
        self.alphas_ = []
        self.estimator_errors_ = []
        self.weight_history_ = []               # for Experiment 1

        w = np.full(n, 1.0 / n)

        if self.K_ == 2:
            # Binary: labels to +/-1, the setting the derivation of README §5 assumes.
            y = np.where(y_raw == self.classes_[1], 1.0, -1.0)
            for _ in range(self.n_estimators):
                self.weight_history_.append(w.copy())
                stump = self.base_estimator().fit(X, y, w)
                pred = stump.predict(X)
                err = float(np.sum(w[pred != y]))
                err = np.clip(err, 1e-10, 1 - 1e-10)

                # alpha = 1/2 ln((1-err)/err), the exact exponential-loss minimizer (§5).
                alpha = 0.5 * self.learning_rate * np.log((1 - err) / err)

                # Reweight: w_i *= exp(-alpha y_i h_i). Misclassified -> heavier (§3).
                w = w * np.exp(-alpha * y * pred)
                w /= w.sum()

                self.estimators_.append(stump)
                self.alphas_.append(alpha)
                self.estimator_errors_.append(err)
                if err <= 1e-10:
                    break                       # perfect learner: stop
        else:
            # SAMME multiclass. Weak learners predict class labels; the vote adds
            # log(K-1) so the "better than random" bar is 1/K, not 1/2.
            for _ in range(self.n_estimators):
                self.weight_history_.append(w.copy())
                stump = _MulticlassStump(self.classes_).fit(X, y_raw, w)
                pred = stump.predict(X)
                err = float(np.sum(w[pred != y_raw]))
                err = np.clip(err, 1e-10, 1 - 1e-10)
                alpha = self.learning_rate * (np.log((1 - err) / err) + np.log(self.K_ - 1))
                w = w * np.exp(alpha * (pred != y_raw))
                w /= w.sum()
                self.estimators_.append(stump)
                self.alphas_.append(alpha)
                self.estimator_errors_.append(err)

        return self

    def decision_function(self, X):
        """Weighted sum of votes. For binary, sign(this) is the prediction."""
        X = np.asarray(X, dtype=float)
        if self.K_ == 2:
            return sum(a * s.predict(X) for a, s in zip(self.alphas_, self.estimators_))
        # Multiclass: accumulate votes per class.
        scores = np.zeros((X.shape[0], self.K_))
        for a, s in zip(self.alphas_, self.estimators_):
            pred = s.predict(X)
            for k, c in enumerate(self.classes_):
                scores[:, k] += a * (pred == c)
        return scores

    def predict(self, X):
        if self.K_ == 2:
            return np.where(self.decision_function(X) >= 0, self.classes_[1],
                            self.classes_[0])
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def staged_predict(self, X):
        """Prediction after each round — for learning curves (Experiments 2, 3)."""
        X = np.asarray(X, dtype=float)
        if self.K_ == 2:
            running = np.zeros(X.shape[0])
            for a, s in zip(self.alphas_, self.estimators_):
                running += a * s.predict(X)
                yield np.where(running >= 0, self.classes_[1], self.classes_[0])
        else:
            scores = np.zeros((X.shape[0], self.K_))
            for a, s in zip(self.alphas_, self.estimators_):
                pred = s.predict(X)
                for k, c in enumerate(self.classes_):
                    scores[:, k] += a * (pred == c)
                yield self.classes_[np.argmax(scores, axis=1)]

    def margins(self, X, y):
        """Normalized margin y*F(x)/sum(alpha) in [-1, 1].  README §9

        Positive = correct; larger = more confident. AdaBoost keeps pushing these rightward
        even after training error is zero (Experiment 5).
        """
        X = np.asarray(X, dtype=float)
        y = np.where(np.asarray(y).ravel() == self.classes_[1], 1.0, -1.0)
        total_alpha = sum(self.alphas_)
        return y * self.decision_function(X) / total_alpha

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))


class _MulticlassStump:
    """A depth-1 weighted learner that predicts class labels (for SAMME)."""

    def __init__(self, classes):
        self.classes = classes

    def fit(self, X, y, w):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        best_err = np.inf
        for f in range(d):
            xf = X[:, f]
            thr_vals = np.unique(xf)
            thresholds = ((thr_vals[:-1] + thr_vals[1:]) / 2 if thr_vals.size > 1
                          else thr_vals)
            for thr in thresholds:
                left, right = xf <= thr, xf > thr
                # Majority (weighted) class on each side.
                lc = self._weighted_majority(y[left], w[left])
                rc = self._weighted_majority(y[right], w[right])
                pred = np.where(left, lc, rc)
                err = float(np.sum(w[pred != y]))
                if err < best_err:
                    best_err = err
                    self.feature_, self.threshold_ = f, thr
                    self.left_class_, self.right_class_ = lc, rc
        return self

    def _weighted_majority(self, y_sub, w_sub):
        if y_sub.size == 0:
            return self.classes[0]
        return max(self.classes,
                   key=lambda c: float(np.sum(w_sub[y_sub == c])))

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.where(X[:, self.feature_] <= self.threshold_,
                        self.left_class_, self.right_class_)


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

    n = 400
    X = rng.standard_normal((n, 2))
    # A circular boundary no single stump can capture, but on which each stump has
    # marginal signal (extreme feature values are more likely outside the circle) — so
    # AdaBoost can get started, unlike on symmetric XOR where every stump is exactly at
    # chance and boosting stalls at round 1.
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1.5).astype(int)
    X_te = rng.standard_normal((3000, 2))
    y_te = (X_te[:, 0] ** 2 + X_te[:, 1] ** 2 > 1.5).astype(int)

    print("\nBoosting stumps beats a single stump (README §1-§2)")
    stump = DecisionStump().fit(X, np.where(y == 1, 1.0, -1.0), np.full(n, 1.0 / n))
    stump_acc = float(np.mean((stump.predict(X_te) > 0).astype(int) == y_te))
    boost = AdaBoost(n_estimators=100).fit(X, y)
    print(f"  [INFO]  {'single stump / AdaBoost(100) test accuracy':<58s}  "
          f"{stump_acc:.4f} / {boost.score(X_te, y_te):.4f}")
    ok &= boost.score(X_te, y_te) > stump_acc + 0.1

    print("\nAgainst sklearn (README §3)")
    try:
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        sk = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100,
            algorithm="SAMME", random_state=0).fit(X, y)
        agree = float(np.mean(boost.predict(X_te) == sk.predict(X_te)))
        print(f"  [{'PASS' if agree > 0.95 else 'FAIL'}]  "
              f"{'binary AdaBoost agrees with sklearn SAMME':<58s}  {agree:.4f}")
        ok &= agree > 0.95

        # Multiclass.
        y3 = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)   # 3 classes
        y3_te = (X_te[:, 0] > 0).astype(int) + (X_te[:, 1] > 0).astype(int)
        mine3 = AdaBoost(n_estimators=100).fit(X, y3)
        sk3 = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                                 n_estimators=100, algorithm="SAMME",
                                 random_state=0).fit(X, y3)
        print(f"  [{'PASS' if abs(mine3.score(X_te, y3_te) - sk3.score(X_te, y3_te)) < 0.05 else 'FAIL'}]  "
              f"{'multiclass (SAMME) accuracy close to sklearn':<58s}  "
              f"{mine3.score(X_te, y3_te):.4f} vs {sk3.score(X_te, y3_te):.4f}")
        ok &= abs(mine3.score(X_te, y3_te) - sk3.score(X_te, y3_te)) < 0.05
    except ImportError:
        print("  [SKIP]  sklearn not installed")

    print("\nStructural properties (README §3, §5, §6)")
    boost = AdaBoost(n_estimators=50).fit(X, y)

    # A learner at chance gets alpha = 0; better-than-chance gets alpha > 0.
    ok &= _report("all weak learners beat chance (err < 0.5)",
                  max(0.0, max(boost.estimator_errors_) - 0.5), 1e-9)
    ok &= _report("all votes are positive (alpha > 0)",
                  max(0.0, -min(boost.alphas_)), 1e-9)

    # Weights renormalize to 1 each round.
    ok &= _report("sample weights sum to 1 each round",
                  float(max(abs(wh.sum() - 1) for wh in boost.weight_history_)), 1e-12)

    # The training error bound: actual <= prod 2 sqrt(err(1-err)).
    train_err = 1 - boost.score(X, y)
    bound = np.prod([2 * np.sqrt(e * (1 - e)) for e in boost.estimator_errors_])
    print(f"  [{'PASS' if train_err <= bound + 1e-9 else 'FAIL'}]  "
          f"{'training error <= prod 2 sqrt(err(1-err))':<58s}  "
          f"{train_err:.4f} <= {bound:.4f}")
    ok &= train_err <= bound + 1e-9

    # staged_predict's last stage equals predict.
    last = list(boost.staged_predict(X_te))[-1]
    ok &= _report("staged_predict final == predict",
                  float(np.mean(last != boost.predict(X_te))), 1e-12)

    return ok


# =============================================================================
# EXPERIMENTS
# =============================================================================


def experiment_weights():
    """README §4: misclassified points get heavier each round."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — adaptive reweighting  (README §4)")
    print("=" * 88)
    print("""
Each round boosts the weights of the points the current ensemble gets wrong, forcing the
next weak learner to focus on them. Tracking the total weight on the points that ROUND 1
misclassified, as boosting proceeds:
""")
    rng = np.random.default_rng(1)
    n = 200
    X = rng.standard_normal((n, 2))
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1.5).astype(int)   # circle: stumps have marginal signal

    boost = AdaBoost(n_estimators=20).fit(X, y)

    # Which points did round 1 get wrong?
    y_pm = np.where(y == 1, 1.0, -1.0)
    round1_pred = boost.estimators_[0].predict(X)
    hard = round1_pred != y_pm

    print(f"  round-1 misclassified: {int(hard.sum())} of {n} points\n")
    print(f"  {'round':>7s}  {'weight on hard points':>22s}  {'vs uniform share':>17s}  "
          f"{'concentration':>14s}")
    print("  " + "-" * 66)
    uniform_share = hard.sum() / n
    for m in (0, 1, 2, 4, 8, 15, 19):
        if m < len(boost.weight_history_):
            w = boost.weight_history_[m]
            hard_weight = float(w[hard].sum())
            print(f"  {m + 1:7d}  {hard_weight:22.4f}  {uniform_share:17.4f}  "
                  f"{hard_weight / uniform_share:12.2f}x")

    print("""
  At round 1 the weights are uniform, so the hard points carry their fair share. After the
  first update their combined weight jumps well above the uniform share and stays elevated
  — the ensemble is being pushed, round after round, to attend to exactly the points it
  finds difficult.

  This is the 'adaptive' in AdaBoost (README §3). Where bagging draws every bootstrap from
  the same uniform distribution, boosting reshapes the distribution each round toward the
  current mistakes. It is why the components must be trained SEQUENTIALLY — each one needs
  the weights the previous ones produced — and hence why boosting cannot be parallelized
  across trees the way a random forest can.""")


def experiment_error_decay():
    """README §6: training error decays exponentially."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — exponential training-error decay  (README §6)")
    print("=" * 88)
    print("""
The theorem: if each weak learner is gamma better than chance, training error <=
exp(-2 sum gamma_m^2) — exponential decay. Measuring training error against the round and
against the bound:
""")
    rng = np.random.default_rng(2)
    n = 300
    X = rng.standard_normal((n, 4))
    # A separable-ish problem so stumps stay better than chance for many rounds.
    y = np.where(X[:, 0] + X[:, 1] - X[:, 2] + 0.5 * X[:, 3] > 0, 1, 0)

    boost = AdaBoost(n_estimators=200).fit(X, y)

    print(f"  {'round':>7s}  {'train error':>12s}  {'error bound':>12s}  "
          f"{'weak err_m':>11s}")
    print("  " + "-" * 48)
    cumulative_bound = 1.0
    staged = list(boost.staged_predict(X))
    for m in (1, 2, 5, 10, 25, 50, 100, 200):
        if m <= len(staged):
            train_err = float(np.mean(staged[m - 1] != y))
            bound = np.prod([2 * np.sqrt(e * (1 - e))
                             for e in boost.estimator_errors_[:m]])
            print(f"  {m:7d}  {train_err:12.4f}  {bound:12.4f}  "
                  f"{boost.estimator_errors_[m - 1]:11.4f}")

    zero_round = next((m + 1 for m, p in enumerate(staged)
                       if np.mean(p != y) == 0), None)
    print(f"""
  Training error falls to zero by round {zero_round}, staying under the exp(-2 sum gamma^2)
  bound the whole way (each 'weak err_m' stays below 0.5, so each contributes a factor < 1).

  This is the constructive proof of the weak-learnability theorem (README §2): weak learners,
  each barely better than chance, combine into a classifier with ZERO training error, and
  they do it exponentially fast. The founding result of boosting, measured.

  Note this is TRAINING error. That it hits zero says nothing about generalization — and
  Experiment 3 shows that on noisy data, continuing past this point is where AdaBoost gets
  into trouble.""")


def experiment_noise_overfitting():
    """README §8, §10: AdaBoost overfits noise; a forest does not."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — AdaBoost overfits noise (a forest does not)  (README §8, §10)")
    print("=" * 88)
    print("""
The critical contrast with bagging. On CLEAN data, more AdaBoost rounds keep helping. On
NOISY data (flipped labels), the exponential loss forces AdaBoost to memorize the
mislabelled points, and test error TURNS UP with more rounds — while a random forest, whose
n_estimators only reduces variance, plateaus. Both at increasing label noise:
""")
    rng = np.random.default_rng(3)
    n = 400

    def make(noise):
        X = rng.standard_normal((n, 5))
        y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
        flip = rng.random(n) < noise
        return X, np.where(flip, 1 - y, y)

    try:
        from sklearn.ensemble import RandomForestClassifier
        have_rf = True
    except ImportError:
        have_rf = False

    print(f"  {'label noise':>12s}  {'AdaBoost @10':>13s}  {'AdaBoost @200':>14s}  "
          f"{'change':>8s}  {'forest @10':>11s}  {'forest @200':>12s}")
    print("  " + "-" * 76)

    for noise in (0.0, 0.1, 0.25):
        X, y = make(noise)
        X_te = rng.standard_normal((3000, 5))
        y_te = (X_te[:, 0] + X_te[:, 1] - X_te[:, 2] > 0).astype(int)   # CLEAN test labels

        boost = AdaBoost(n_estimators=200).fit(X, y)
        staged = list(boost.staged_predict(X_te))
        ada10 = float(np.mean(staged[9] == y_te))
        ada200 = float(np.mean(staged[199] == y_te))

        if have_rf:
            rf10 = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y) \
                .score(X_te, y_te)
            rf200 = RandomForestClassifier(n_estimators=200, random_state=0).fit(X, y) \
                .score(X_te, y_te)
            rf_str = f"{rf10:11.4f}  {rf200:12.4f}"
        else:
            rf_str = f"{'n/a':>11s}  {'n/a':>12s}"

        change = ada200 - ada10
        flag = "  <- WORSE" if change < -0.01 else ""
        print(f"  {noise:12.2f}  {ada10:13.4f}  {ada200:14.4f}  {change:+8.4f}{flag}  "
              f"{rf_str}")

    print("""
  Read the 'change' column — AdaBoost's test accuracy from 10 rounds to 200.

  At zero noise, more rounds help (change positive or flat): this is the famous
  'resistance to overfitting' of README §8.

  As noise rises, the change goes NEGATIVE — 200 rounds is WORSE than 10. The exponential
  loss keeps boosting the weight of the flipped-label points (README §10), and the ensemble
  contorts to fit them, hurting the clean test set. More estimators actively harmed the
  model.

  The random forest columns barely move between 10 and 200 trees at any noise level,
  because n_estimators only drives variance to a floor and cannot overfit (06.02 §5).

  This is THE practical difference (README §11): for a random forest, more trees is always
  safe; for AdaBoost, n_estimators is a REGULARIZER that must be tuned with early stopping.
  Confusing the two is a costly and common mistake.""")


def experiment_exp_loss():
    """README §5, §7: AdaBoost weights ARE the exponential-loss gradient."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — AdaBoost = exponential-loss minimization  (README §5, §7)")
    print("=" * 88)
    print("""
The chapter's central claim: AdaBoost is greedy forward-stagewise minimization of
exp(-y F(x)), and the sample weights ARE the per-point exponential loss. We verify this
numerically: at each round, the AdaBoost weight w_i should equal (up to normalization)
exp(-y_i F_{m-1}(x_i)), the current exponential loss.
""")
    rng = np.random.default_rng(4)
    n = 200
    X = rng.standard_normal((n, 3))
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, 0)
    y_pm = np.where(y == 1, 1.0, -1.0)

    boost = AdaBoost(n_estimators=30).fit(X, y)

    print(f"  {'round':>7s}  {'max |w_adaboost - w_exploss|':>30s}  {'total exp loss':>15s}")
    print("  " + "-" * 56)

    max_discrepancy = 0.0
    for m in (1, 2, 5, 10, 20, 30):
        if m <= len(boost.estimators_):
            # F_{m-1}: the ensemble through round m-1.
            F = np.zeros(n)
            for a, s in zip(boost.alphas_[:m - 1], boost.estimators_[:m - 1]):
                F += a * s.predict(X)
            # Exponential-loss weights, normalized.
            w_exploss = np.exp(-y_pm * F)
            w_exploss /= w_exploss.sum()
            w_adaboost = boost.weight_history_[m - 1]
            disc = float(np.abs(w_adaboost - w_exploss).max())
            max_discrepancy = max(max_discrepancy, disc)
            total_loss = float(np.sum(np.exp(-y_pm * F)))
            print(f"  {m:7d}  {disc:30.2e}  {total_loss:15.4f}")

    print(f"""
  The AdaBoost sample weights match the exponential-loss weights to {max_discrepancy:.1e} at
  every round — they are the SAME thing. The weights are not a heuristic; they are the
  current per-point exponential loss, exactly as the forward-stagewise derivation of
  README §5 says.

  And the total exponential loss (last column) falls monotonically — each round is a
  coordinate-descent step on that loss. This is the view that generalizes: keep the
  forward-stagewise framework, swap exp(-yF) for ANY differentiable loss, and you have
  gradient boosting (06.04). AdaBoost is the special case with the exponential loss, and
  its aggression (and noise-fragility, §7) is exactly that loss's steep tail.""")


def experiment_margins():
    """README §9: margins improve after training error hits zero."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — margins keep improving after zero training error  (README §9)")
    print("=" * 88)
    print("""
The resolution of the overfitting-resistance puzzle. Even after training error is zero and
the DECISIONS stop changing, AdaBoost keeps pushing up the WORST-CASE (minimum) margin —
the confidence of its least-confident correct classifications — which is the quantity
Schapire's generalization bound depends on. Tracking the margin distribution past the
zero-training-error point:
""")
    rng = np.random.default_rng(5)
    n = 300
    X = rng.standard_normal((n, 4))
    y = np.where(X[:, 0] + X[:, 1] - X[:, 2] > 0, 1, 0)

    boost = AdaBoost(n_estimators=400).fit(X, y)
    staged = list(boost.staged_predict(X))
    zero_round = next((m + 1 for m, p in enumerate(staged)
                       if np.mean(p != y) == 0), len(staged))

    print(f"  training error first reaches zero at round {zero_round}\n")
    print(f"  {'round':>7s}  {'train err':>10s}  {'MIN margin':>11s}  "
          f"{'5th pctile':>11s}  {'median':>8s}")
    print("  " + "-" * 54)

    for m in sorted({zero_round, zero_round + 30, zero_round + 130, 200, 300, 400}):
        if m <= len(boost.estimators_):
            sub = AdaBoost(n_estimators=m)
            sub.classes_, sub.K_ = boost.classes_, boost.K_
            sub.estimators_ = boost.estimators_[:m]
            sub.alphas_ = boost.alphas_[:m]
            train_err = float(np.mean(staged[m - 1] != y))
            margins = sub.margins(X, y)
            print(f"  {m:7d}  {train_err:10.4f}  {margins.min():11.4f}  "
                  f"{np.percentile(margins, 5):11.4f}  {np.median(margins):8.4f}")

    print(f"""
  Training error is zero from round {zero_round} on and stays there — the decisions never
  change again. But the MINIMUM margin climbs steadily (roughly 0.001 -> 0.05), and so does
  the 5th percentile: the ensemble is classifying the same points, but its LEAST confident
  correct predictions become progressively more confident.

  Note the honest subtlety in the last column: the MEDIAN margin does NOT climb — it drifts
  slightly down. So it is wrong to say 'all the margins keep improving'. What improves is the
  LOWER TAIL — the worst-case margin — which is precisely the quantity Schapire et al.'s
  generalization bound depends on. Pushing up the minimum margin is what the bound rewards,
  and it happens even after the training set is perfectly separated.

  This is the same principle as the SVM's max-margin objective (03.07 §1) — maximize the
  SMALLEST margin — reached from a completely different direction, which is why AdaBoost is
  described as approximately a margin maximizer. The improvement is real but, as this data
  shows, modest and confined to the tail; and the caveat of Experiment 3 still governs
  everything: this benign behaviour is a CLEAN-data phenomenon, and label noise reverses it.""")


# =============================================================================

if __name__ == "__main__":
    print(__doc__)

    all_passed = verify()

    experiment_weights()
    experiment_error_decay()
    experiment_noise_overfitting()
    experiment_exp_loss()
    experiment_margins()

    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 88)
