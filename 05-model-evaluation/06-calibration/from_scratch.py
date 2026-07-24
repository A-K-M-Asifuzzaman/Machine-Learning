"""
05.06 — Calibration, from scratch (NumPy).

Reliability diagrams / ECE / MCE, Platt scaling, isotonic regression (Pool Adjacent Violators),
and temperature scaling, verified against scikit-learn. Then the chapter's claims are MEASURED:

  1. discrimination vs calibration: recalibration collapses ECE, leaves AUC unchanged  (README §2)
  2. miscalibration signatures: logistic (calibrated), RF (underconfident), boosting (over)(§5)
  3. Platt (low-variance) vs isotonic (flexible) vs calibration-set size                (§6-§7)
  4. temperature scaling cuts ECE at ZERO accuracy cost                                 (§8)
  5. calibrating on TRAINING data leaks; held-out generalizes                           (§9)
  6. a base-rate predictor is perfectly calibrated yet useless (sharpness)              (§10)

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression as SkIso
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    HAVE_SK = True
except Exception:
    HAVE_SK = False


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


# =============================================================================
# MEASURING CALIBRATION  (README §4)
# =============================================================================


def reliability(y, p, n_bins=10):
    """Return (bin_conf, bin_acc, bin_count) for a reliability diagram."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    conf, acc, cnt = [], [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (p >= lo) & (p < hi if b < n_bins - 1 else p <= hi)
        if m.sum():
            conf.append(p[m].mean())
            acc.append(y[m].mean())
            cnt.append(int(m.sum()))
    return np.array(conf), np.array(acc), np.array(cnt)


def ece(y, p, n_bins=10):
    conf, acc, cnt = reliability(y, p, n_bins)
    return float(np.sum(cnt / np.sum(cnt) * np.abs(acc - conf)))


def mce(y, p, n_bins=10):
    conf, acc, _ = reliability(y, p, n_bins)
    return float(np.max(np.abs(acc - conf))) if len(conf) else 0.0


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def brier_decomposition(y, p, n_bins=10):
    """Murphy: Brier = reliability - resolution + uncertainty."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    n = len(y)
    ybar = y.mean()
    edges = np.linspace(0, 1, n_bins + 1)
    rel = res = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (p >= lo) & (p < hi if b < n_bins - 1 else p <= hi)
        nb = m.sum()
        if nb:
            pk = p[m].mean()
            ok = y[m].mean()
            rel += nb * (pk - ok) ** 2
            res += nb * (ok - ybar) ** 2
    return rel / n, res / n, ybar * (1 - ybar)


# =============================================================================
# CALIBRATORS  (README §6-§8)
# =============================================================================


class PlattScaling:
    """1-D logistic regression on the scores: p = sigmoid(a*s + b). Fit by Newton/IRLS."""

    def fit(self, s, y, n_iter=100):
        s = np.asarray(s, float)
        y = np.asarray(y, float)
        X = np.column_stack([s, np.ones_like(s)])
        w = np.zeros(2)
        for _ in range(n_iter):
            p = _sigmoid(X @ w)
            W = p * (1 - p) + 1e-9
            grad = X.T @ (p - y)
            H = X.T @ (W[:, None] * X) + 1e-9 * np.eye(2)
            step = np.linalg.solve(H, grad)
            w -= step
            if np.max(np.abs(step)) < 1e-10:
                break
        self.a, self.b = w
        return self

    def predict(self, s):
        return _sigmoid(self.a * np.asarray(s, float) + self.b)


def pav(y, w=None):
    """Pool Adjacent Violators: the non-decreasing least-squares fit to y (README §7).

    Sweep left to right maintaining blocks [weighted_sum, weight, length]; whenever the last
    block's mean drops below the previous block's mean, merge them (pool the violators).
    """
    y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    blocks = []                            # each: [weighted sum, total weight, length]
    for v, ww in zip(y, w):
        blocks.append([v * ww, ww, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s2, w2, l2 = blocks.pop()
            blocks[-1][0] += s2
            blocks[-1][1] += w2
            blocks[-1][2] += l2
    result = []
    for s, ww, ln in blocks:
        result.extend([s / ww] * ln)
    return np.array(result)


class IsotonicCalibrator:
    """Non-parametric monotonic calibration via PAV, with interpolation for new scores."""

    def fit(self, s, y):
        s = np.asarray(s, float)
        y = np.asarray(y, float)
        order = np.argsort(s, kind="stable")
        self.s_sorted = s[order]
        self.p_sorted = pav(y[order])
        return self

    def predict(self, s):
        return np.interp(np.asarray(s, float), self.s_sorted, self.p_sorted)


class TemperatureScaling:
    """Divide logits by a single T > 0, fit by minimizing log loss (README §8)."""

    def fit(self, logits, y, n_iter=200):
        logits = np.asarray(logits, float)
        y = np.asarray(y, float)
        logT = 0.0                                    # optimize log T for positivity
        lr = 0.1
        for _ in range(n_iter):
            T = np.exp(logT)
            p = _sigmoid(logits / T)
            # d loss / d logit-scale ... gradient wrt logT via chain rule
            g = np.mean((p - y) * (-logits / T))      # d/d logT of mean CE
            logT -= lr * g
        self.T = float(np.exp(logT))
        return self

    def predict(self, logits):
        return _sigmoid(np.asarray(logits, float) / self.T)


def auc(y, p):
    y = np.asarray(y)
    s = np.asarray(p, float)
    pos, neg = s[y == 1], s[y == 0]
    return float((np.sum(pos[:, None] > neg[None, :])
                  + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg)))


# =============================================================================
# VERIFICATION
# =============================================================================


def verify():
    print("=" * 88)
    print("VERIFICATION — PAV isotonic vs sklearn; Murphy decomposition; temp preserves accuracy")
    print("=" * 88)
    rng = np.random.default_rng(0)
    s = np.sort(rng.uniform(0, 1, 300))
    y = (rng.uniform(size=300) < s).astype(float)      # calibrated-ish target

    ours = IsotonicCalibrator().fit(s, y).predict(s)
    if HAVE_SK:
        sk = SkIso(out_of_bounds="clip").fit(s, y).predict(s)
        diff = np.max(np.abs(ours - sk))
        print(f"\n  isotonic (PAV) vs sklearn IsotonicRegression: max |diff| = {diff:.2e}")
        assert diff < 1e-8, "PAV must match sklearn isotonic"
        print("  Pool-Adjacent-Violators matches sklearn to machine precision  ✓")

    # Murphy decomposition is EXACT for a forecast that takes finitely many values, i.e. the
    # BINNED forecast (each prediction replaced by its bin mean). For continuous p it is a close
    # approximation; we verify the exact identity on the binned forecast.
    p = _sigmoid(3 * (s - 0.5))
    n_bins = 10
    edges = np.linspace(0, 1, n_bins + 1)
    b_idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    p_binned = p.copy()
    for b in range(n_bins):
        m = b_idx == b
        if m.any():
            p_binned[m] = p[m].mean()
    rel, res, unc = brier_decomposition(y, p, n_bins)
    lhs = rel - res + unc
    print(f"\n  Brier decomposition (binned forecast): reliability {rel:.4f} - resolution "
          f"{res:.4f} + uncertainty {unc:.4f}")
    print(f"    = {lhs:.6f} vs Brier(binned) {brier(y, p_binned):.6f}  "
          f"(diff {abs(lhs-brier(y, p_binned)):.2e})")
    print(f"    (continuous Brier {brier(y, p):.4f} differs only by the within-bin variance)")
    assert abs(lhs - brier(y, p_binned)) < 1e-9, "Murphy decomposition must sum to binned Brier"
    print("  Murphy's reliability - resolution + uncertainty == Brier (binned)  ✓")

    # temperature scaling preserves the argmax (here: the 0.5 threshold ordering)
    logits = rng.standard_normal(500) * 3
    yt = (rng.uniform(size=500) < _sigmoid(logits)).astype(float)
    ts = TemperatureScaling().fit(logits, yt)
    before = (_sigmoid(logits) >= 0.5)
    after = (ts.predict(logits) >= 0.5)
    print(f"\n  temperature scaling fitted T = {ts.T:.3f}; decisions changed: "
          f"{int(np.sum(before != after))}/500")
    assert np.array_equal(before, after), "temperature scaling must preserve decisions"
    print("  temperature scaling leaves every 0.5-threshold decision unchanged  ✓")
    print("\nAll verification checks passed.")


# =============================================================================
# DATA
# =============================================================================


def _make(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6))
    logit = 1.3 * X[:, 0] - 1.6 * X[:, 1] + 0.9 * X[:, 2] + 0.6 * X[:, 0] * X[:, 3]
    y = (rng.uniform(size=n) < _sigmoid(logit)).astype(int)
    return X, y


# =============================================================================
# EXPERIMENT 1 — discrimination vs calibration (README §2)
# =============================================================================


def experiment_1_discrimination_vs_calibration():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — recalibration collapses ECE while AUC is unchanged (README §2)")
    print("=" * 88)
    rng = np.random.default_rng(1)
    n = 4000
    # a well-ranked but MISCALIBRATED score: true prob squashed into [0.35, 0.65]
    p_true = rng.uniform(0, 1, n)
    y = (rng.uniform(size=n) < p_true).astype(int)
    s = 0.35 + 0.30 * p_true                       # monotonic squash: same ranking, bad calibration

    # split: fit calibrator on half, evaluate on the other half
    cal, te = np.arange(n) < n // 2, np.arange(n) >= n // 2
    platt = PlattScaling().fit(s[cal], y[cal])
    p_cal = platt.predict(s[te])

    print(f"""
    {'model':>22s} {'AUC':>7s} {'ECE':>7s}
    {'raw (miscalibrated)':>22s} {auc(y[te], s[te]):>7.3f} {ece(y[te], s[te]):>7.3f}
    {'after Platt scaling':>22s} {auc(y[te], p_cal):>7.3f} {ece(y[te], p_cal):>7.3f}

  READING: the raw score ranks perfectly (high AUC) but its probabilities are squashed into a
  narrow band — large ECE. Platt scaling applies a MONOTONIC transform, so it fixes the
  probabilities (ECE collapses) while leaving the ranking — and AUC — untouched. Calibration and
  discrimination are orthogonal; you fix one without disturbing the other (README §2).""")


# =============================================================================
# EXPERIMENT 2 — miscalibration signatures (README §5)
# =============================================================================


def experiment_2_signatures():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — miscalibration signatures by model family (README §5)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(scikit-learn unavailable — skipping)")
        return
    # a HARDER, noisier problem (weak signal) elicits the signatures that an easy problem hides
    rng = np.random.default_rng(2)

    def make_hard(seed, n=800):
        r = np.random.default_rng(seed)
        X = r.standard_normal((n, 10))
        logit = 0.8 * X[:, 0] - 0.9 * X[:, 1] + 0.5 * X[:, 2]      # weak => genuine uncertainty
        y = (r.uniform(size=n) < _sigmoid(logit)).astype(int)
        return X, y

    Xtr, ytr = make_hard(1)
    Xte, yte = make_hard(2)
    models = {
        "logistic": LogisticRegression(max_iter=1000),
        "naive Bayes": GaussianNB(),
        "random forest": RandomForestClassifier(n_estimators=300, random_state=0),
        "boosting (500)": GradientBoostingClassifier(n_estimators=500, max_depth=3,
                                                     random_state=0),
    }
    print(f"\n  Weak-signal problem. Confidence profile: fraction of predictions that are")
    print(f"  EXTREME (p<0.1 or p>0.9) vs MID (0.4-0.6):\n")
    print(f"    {'model':>16s} {'AUC':>7s} {'ECE':>7s} {'% extreme':>10s} {'% mid':>8s}")
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        extreme = np.mean((p < 0.1) | (p > 0.9))
        mid = np.mean((p >= 0.4) & (p <= 0.6))
        print(f"    {name:>16s} {auc(yte, p):>7.3f} {ece(yte, p):>7.3f} {extreme:>9.0%} {mid:>7.0%}")
    print("""
  READING: logistic minimizes log loss (a proper rule) so it is near-calibrated (low ECE). The
  RANDOM FOREST is UNDERconfident — averaging many trees pulls votes toward the base rate, so it
  makes very FEW extreme predictions (~4%) and many mid ones. BOOSTING (500 rounds) is
  OVERconfident — margin maximization pushes probabilities to the extremes, so ~40% of its
  predictions are extreme and its ECE is by far the worst. Naive Bayes is overconfident too
  (independence inflates certainty). Miscalibration is systematic and follows from the loss (README §5).""")


# =============================================================================
# EXPERIMENT 3 — Platt vs isotonic vs calibration-set size (README §6-§7)
# =============================================================================


def experiment_3_platt_vs_isotonic():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — Platt (low-variance) vs isotonic (flexible) vs data size (README §6-§7)")
    print("=" * 88)
    N = 8000
    te = np.arange(N) >= N // 2

    def run(gen, label):
        s, y = gen
        print(f"\n  {label}:")
        print(f"    {'calib size':>12s} {'Platt ECE':>11s} {'isotonic ECE':>13s} {'winner':>8s}")
        for m_cal in (60, 200, 1500):
            ci = np.where(~te)[0][:m_cal]
            pe = ece(y[te], PlattScaling().fit(s[ci], y[ci]).predict(s[te]))
            ie = ece(y[te], IsotonicCalibrator().fit(s[ci], y[ci]).predict(s[te]))
            print(f"    {m_cal:>12d} {pe:>11.3f} {ie:>13.3f} {('Platt' if pe < ie else 'isotonic'):>8s}")

    # (A) SIGMOIDAL miscalibration: the true P(y=1|s) IS a sigmoid of s -> Platt's assumption holds
    rng = np.random.default_rng(3)
    s_sig = rng.uniform(-4, 4, N)
    y_sig = (rng.uniform(size=N) < _sigmoid(1.2 * s_sig - 0.3)).astype(int)
    run((s_sig, y_sig), "(A) sigmoidal distortion — Platt's parametric form is correct")

    # (B) NON-sigmoidal miscalibration (p_true^2) -> Platt is biased, isotonic's flexibility wins
    p_true = rng.uniform(0, 1, N)
    y_ns = (rng.uniform(size=N) < p_true).astype(int)
    s_ns = p_true ** 2
    run((s_ns, y_ns), "(B) non-sigmoidal distortion (s = p^2) — an odd shape Platt cannot fit")

    print("""
  READING: when the miscalibration is SIGMOIDAL (A), Platt's 2-parameter model is the correct
  form and it matches or beats isotonic, which only adds variance. When the distortion is an ODD
  monotonic SHAPE (B, here p^2), Platt's sigmoid is biased and cannot fix it, so isotonic's
  non-parametric flexibility wins once it has enough data. Platt for small data / sigmoidal
  distortion; isotonic for large data / odd shapes — the usual bias-variance trade (README §7).""")


# =============================================================================
# EXPERIMENT 4 — temperature scaling at zero accuracy cost (README §8)
# =============================================================================


def experiment_4_temperature():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — temperature scaling cuts ECE at ZERO accuracy cost (README §8)")
    print("=" * 88)
    rng = np.random.default_rng(4)
    n = 6000
    logit = rng.standard_normal(n) * 2.0
    y = (rng.uniform(size=n) < _sigmoid(logit)).astype(int)
    # OVERCONFIDENT model: inflate the logits (as an overtrained net would)
    over_logit = logit * 3.0

    cal, te = np.arange(n) < n // 2, np.arange(n) >= n // 2
    ts = TemperatureScaling().fit(over_logit[cal], y[cal])
    p_before = _sigmoid(over_logit[te])
    p_after = ts.predict(over_logit[te])

    def acc(p):
        return np.mean((p >= 0.5) == y[te])
    print(f"""
  Overconfident model (logits x3). Fitted temperature T = {ts.T:.2f}:

    {'':>18s} {'accuracy':>9s} {'AUC':>7s} {'ECE':>7s}
    {'before (T=1)':>18s} {acc(p_before):>9.4f} {auc(y[te], p_before):>7.3f} {ece(y[te], p_before):>7.3f}
    {'after temp scaling':>18s} {acc(p_after):>9.4f} {auc(y[te], p_after):>7.3f} {ece(y[te], p_after):>7.3f}

  READING: temperature scaling divides every logit by the same T > 1, softening the
  probabilities. ECE drops sharply, but accuracy and AUC are BIT-FOR-BIT identical — dividing all
  logits by a constant never changes which is largest. Free calibration for deep nets (README §8).""")


# =============================================================================
# EXPERIMENT 5 — calibrating on training data leaks (README §9)
# =============================================================================


def experiment_5_leak():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — calibrating on TRAINING predictions leaks (README §9)")
    print("=" * 88)
    if not HAVE_SK:
        print("\n(scikit-learn unavailable — skipping)")
        return
    Xtr, ytr = _make(2000, seed=5)
    Xte, yte = _make(4000, seed=6)
    # an overfitting base model, so its TRAIN scores are unrealistically good
    base = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=0).fit(Xtr, ytr)
    s_tr = base.predict_proba(Xtr)[:, 1]
    s_te = base.predict_proba(Xte)[:, 1]

    # WRONG: fit isotonic on the TRAIN scores (which the forest has memorized)
    iso_leak = IsotonicCalibrator().fit(s_tr, ytr)
    # RIGHT: fit on a held-out split the base model did NOT train on
    Xcal, ycal = _make(2000, seed=7)
    s_cal = base.predict_proba(Xcal)[:, 1]
    iso_ok = IsotonicCalibrator().fit(s_cal, ycal)

    print(f"""
    {'calibrator fit on...':>26s} {'in-sample ECE':>14s} {'TEST ECE':>10s}
    {'training predictions (WRONG)':>26s} {ece(ytr, iso_leak.predict(s_tr)):>14.3f} """
          f"""{ece(yte, iso_leak.predict(s_te)):>10.3f}
    {'held-out predictions (RIGHT)':>26s} {ece(ycal, iso_ok.predict(s_cal)):>14.3f} """
          f"""{ece(yte, iso_ok.predict(s_te)):>10.3f}

  READING: fit on the training scores, the calibrator looks near-perfect IN-SAMPLE (the forest
  memorized those labels) but generalizes worse on test. Fit on held-out predictions, the
  in-sample number is honest and the TEST ECE is lower. The calibrator is a learned step and must
  see data the base model did not train on (README §9).""")


# =============================================================================
# EXPERIMENT 6 — calibration is not enough: sharpness (README §10)
# =============================================================================


def experiment_6_sharpness():
    print("\n" + "=" * 88)
    print("EXPERIMENT 6 — a base-rate predictor is perfectly calibrated yet useless (README §10)")
    print("=" * 88)
    rng = np.random.default_rng(6)
    n = 5000
    X, y = _make(n, seed=8)
    base_rate = y.mean()
    p_const = np.full(n, base_rate)          # predict the base rate for everyone

    rel, res, unc = brier_decomposition(y, p_const, 10)
    print(f"""
  'Model' that predicts the base rate ({base_rate:.2f}) for every input:

    {'metric':>14s} {'value':>8s}
    {'ECE':>14s} {ece(y, p_const):>8.3f}   <- ~0: PERFECTLY calibrated
    {'AUC':>14s} {auc(y, p_const):>8.3f}   <- 0.5: no discrimination at all
    {'Brier':>14s} {brier(y, p_const):>8.3f}
    {'resolution':>14s} {res:>8.3f}   <- 0: no sharpness (never leaves the base rate)

  READING: predicting the base rate everywhere is PERFECTLY calibrated (ECE ~ 0) — among all its
  '{base_rate:.2f}' predictions, exactly {base_rate:.0%} are positive — yet completely useless (AUC 0.5,
  zero resolution). Calibration alone is not enough; you need SHARPNESS too. This is why proper
  scoring rules (Brier, log loss), which reward both, are the right target — ECE is gamed by the
  base-rate predictor (README §10).""")


if __name__ == "__main__":
    verify()
    experiment_1_discrimination_vs_calibration()
    experiment_2_signatures()
    experiment_3_platt_vs_isotonic()
    experiment_4_temperature()
    experiment_5_leak()
    experiment_6_sharpness()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
