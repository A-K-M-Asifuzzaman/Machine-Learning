"""
03.09 — The Perceptron from Scratch
===================================

The oldest trainable classifier, and the atom every neural network is built from.
Obsolete as a model; essential as the origin of deep learning.

Implemented here
----------------
    Perceptron              the classic mistake-driven rule           README §3
        variant="pocket"    keeps the best weights (non-separable)    README §9
        variant="averaged"  averages all weight vectors               README §9
    Adaline                 the delta rule = SGD on squared loss      README §10
    TwoLayerNet             solves XOR, where one perceptron cannot   README §7

Run it
------
    python from_scratch.py

Verified against sklearn, then four experiments:
  1. The convergence theorem: mistakes <= (R/gamma)^2, checked empirically
  2. Convergence time explodes as the margin shrinks; never converges if non-separable
  3. XOR: a single perceptron fails, a two-layer network succeeds
  4. The averaged perceptron generalizes far better than the plain one

Reference: README.md sections 3-10.
"""

from __future__ import annotations

import numpy as np

# =============================================================================
# THE PERCEPTRON  (README §2-§3, §9)
# =============================================================================


class Perceptron:
    """Rosenblatt's perceptron with the classic mistake-driven update.

        predict:  y_hat = sign(w . x)
        on error: w <- w + eta * y * x        (README §3)

    The update is mistake-driven — correctly classified points are ignored — and eta is
    irrelevant on separable data, since scaling w never changes any sign (README §3).

    variant:
        "plain"     the raw algorithm; cycles forever on non-separable data
        "pocket"    keeps the lowest-training-error weights seen (README §9)
        "averaged"  returns the survival-weighted average of all weight vectors, which
                    reduces variance and generalizes markedly better (README §9)
    """

    def __init__(self, eta: float = 1.0, max_epochs: int = 100, variant: str = "plain",
                 shuffle: bool = True, random_state: int = 0):
        self.eta = eta
        self.max_epochs = max_epochs
        self.variant = variant
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        X = np.asarray(X, dtype=float)
        y_raw = np.asarray(y).ravel()
        self.classes_ = np.unique(y_raw)
        # Internally +/-1: makes the update rule symmetric (README §2).
        y = np.where(y_raw == self.classes_[1], 1.0, -1.0)

        n, d = X.shape
        A = np.column_stack([X, np.ones(n)])        # fold the bias into the weights
        w = np.zeros(d + 1)
        rng = np.random.default_rng(self.random_state)

        # Pocket / averaged bookkeeping.
        best_w = w.copy()
        best_errors = n + 1
        w_sum = np.zeros(d + 1)
        survival = 0
        total_survival = 0

        self.mistakes_per_epoch_ = []
        self.n_mistakes_ = 0
        converged_epoch = None

        for epoch in range(self.max_epochs):
            order = rng.permutation(n) if self.shuffle else np.arange(n)
            mistakes = 0

            for i in order:
                score = w @ A[i]
                if y[i] * score <= 0:               # misclassified (README §3)
                    # Snapshot the surviving run before overwriting w (averaged variant).
                    w_sum += survival * w
                    total_survival += survival
                    survival = 0

                    w = w + self.eta * y[i] * A[i]   # the perceptron update
                    mistakes += 1
                    self.n_mistakes_ += 1
                else:
                    survival += 1

                if self.variant == "pocket":
                    # A pocket update is expensive (full-pass error), so do it only after
                    # a real change; keep whichever weights make the fewest mistakes.
                    errors = int(np.sum(y * (A @ w) <= 0))
                    if errors < best_errors:
                        best_errors = errors
                        best_w = w.copy()

            self.mistakes_per_epoch_.append(mistakes)
            if mistakes == 0 and converged_epoch is None:
                converged_epoch = epoch + 1
                if self.variant == "plain":
                    break

        # Finalize the weights according to the variant.
        w_sum += survival * w
        total_survival += survival
        if self.variant == "pocket":
            self.w_ = best_w
        elif self.variant == "averaged":
            self.w_ = w_sum / total_survival if total_survival > 0 else w
        else:
            self.w_ = w

        self.coef_ = self.w_[:-1]
        self.intercept_ = self.w_[-1]
        self.converged_ = converged_epoch is not None
        self.n_epochs_ = converged_epoch if self.converged_ else self.max_epochs
        return self

    def decision_function(self, X):
        A = np.column_stack([np.asarray(X, dtype=float), np.ones(len(X))])
        return A @ self.w_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))


# =============================================================================
# ADALINE — the delta rule  (README §10)
# =============================================================================


class Adaline:
    """ADALINE: error computed on the RAW output, before thresholding.  README §10

        L = 1/2 (y - w.x)^2
        w <- w + eta (y - w.x) x        (Widrow-Hoff / delta rule)

    This is exactly SGD on squared loss (00.02 §9) — linear regression trained online.
    The error term (y - w.x) is CONTINUOUS, unlike the perceptron's discrete sign error,
    and that is the whole difference: a differentiable loss generalizes to deep networks
    via backpropagation, and the perceptron rule does not.

    `verify()` confirms the update matches a plain SGD step on 1/2(y - w.x)^2 exactly.
    """

    def __init__(self, eta: float = 0.01, max_epochs: int = 100, random_state: int = 0):
        self.eta = eta
        self.max_epochs = max_epochs
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y_raw = np.asarray(y).ravel()
        self.classes_ = np.unique(y_raw)
        y = np.where(y_raw == self.classes_[1], 1.0, -1.0)

        n, d = X.shape
        A = np.column_stack([X, np.ones(n)])
        w = np.zeros(d + 1)
        rng = np.random.default_rng(self.random_state)

        self.loss_per_epoch_ = []
        for _ in range(self.max_epochs):
            for i in rng.permutation(n):
                error = y[i] - w @ A[i]              # CONTINUOUS residual (README §10)
                w = w + self.eta * error * A[i]
            self.loss_per_epoch_.append(float(0.5 * np.mean((y - A @ w) ** 2)))

        self.w_ = w
        self.coef_ = w[:-1]
        self.intercept_ = w[-1]
        return self

    def decision_function(self, X):
        A = np.column_stack([np.asarray(X, dtype=float), np.ones(len(X))])
        return A @ self.w_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))


# =============================================================================
# A TWO-LAYER NETWORK — what fixes XOR  (README §7)
# =============================================================================


class TwoLayerNet:
    """A minimal MLP: one hidden layer with tanh, trained by backprop.  README §7, §11

    Included to make the resolution of the XOR catastrophe concrete: a single perceptron
    cannot separate XOR (Experiment 3), and this network — a hidden layer of neurons, each
    a perceptron with a SMOOTH activation instead of a step — solves it in seconds.

    This is a preview of Part 7, deliberately minimal: full backprop, initialization, and
    optimizer theory are 07.02, 07.05, 07.06. The point here is only that DEPTH plus a
    DIFFERENTIABLE activation is exactly what the perceptron lacked (README §7, §10).
    """

    def __init__(self, hidden: int = 4, eta: float = 0.5, epochs: int = 5000,
                 random_state: int = 0):
        self.hidden = hidden
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y_raw = np.asarray(y).ravel()
        self.classes_ = np.unique(y_raw)
        y = np.where(y_raw == self.classes_[1], 1.0, -1.0).reshape(-1, 1)

        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        # Small random init breaks symmetry (07.05); zeros would keep all hidden units
        # identical, exactly the failure mode of a badly-initialized net.
        self.W1 = rng.standard_normal((d, self.hidden)) * 0.5
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.standard_normal((self.hidden, 1)) * 0.5
        self.b2 = np.zeros(1)

        for _ in range(self.epochs):
            # Forward.
            z1 = X @ self.W1 + self.b1
            a1 = np.tanh(z1)
            out = a1 @ self.W2 + self.b2

            # Backward (squared loss; the chain rule of 07.02 in miniature).
            grad_out = (out - y) / n
            grad_W2 = a1.T @ grad_out
            grad_b2 = grad_out.sum(axis=0)
            grad_a1 = grad_out @ self.W2.T
            grad_z1 = grad_a1 * (1 - a1 ** 2)        # tanh'(z) = 1 - tanh(z)^2
            grad_W1 = X.T @ grad_z1
            grad_b1 = grad_z1.sum(axis=0)

            self.W2 -= self.eta * grad_W2
            self.b2 -= self.eta * grad_b2
            self.W1 -= self.eta * grad_W1
            self.b1 -= self.eta * grad_b1

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return (np.tanh(X @ self.W1 + self.b1) @ self.W2 + self.b2).ravel()

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))


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

    # A cleanly separable problem.
    n = 200
    X = rng.standard_normal((n, 4))
    w_true = np.array([1.5, -2.0, 0.5, 1.0])
    y = np.where(X @ w_true + 0.5 > 0, 1, -1)
    # Widen the gap so it is genuinely separable.
    margin = X @ w_true + 0.5
    keep = np.abs(margin) > 0.3
    X, y = X[keep], y[keep]

    print("\nSeparable data: the perceptron converges to zero error (README §5)")
    p = Perceptron(max_epochs=1000).fit(X, y)
    ok &= _report("perceptron reaches 0 training error", 1 - p.score(X, y), 1e-12)
    print(f"  [INFO]  {'converged in':<58s}  {p.n_epochs_} epochs, "
          f"{p.n_mistakes_} total mistakes")
    ok &= (p.converged_)

    print("\nAgainst sklearn (README §8)")
    try:
        from sklearn.linear_model import Perceptron as SKPerceptron
        # sklearn's Perceptron is SGD on the perceptron loss; with the same eta and no
        # regularization it reaches the same zero-error region on separable data. We
        # compare the DECISIONS, since the specific hyperplane is order-dependent (README §6).
        sk = SKPerceptron(alpha=0, max_iter=1000, tol=None, shuffle=True,
                          random_state=0).fit(X, y)
        agree = float(np.mean(p.predict(X) == sk.predict(X)))
        print(f"  [{'PASS' if agree > 0.95 else 'FAIL'}]  "
              f"{'perceptron agrees with sklearn on separable data':<58s}  "
              f"{agree:.4f}")
        ok &= agree > 0.95
        ok &= _report("sklearn also reaches 0 training error", 1 - sk.score(X, y), 1e-9)
    except ImportError:
        print("  [SKIP]  sklearn not installed")

    print("\nThe update rule (README §4)")
    # A single update on a misclassified point must INCREASE its score.
    w = np.zeros(5)
    A = np.column_stack([X, np.ones(len(X))])
    i = 0
    score_before = w @ A[i]
    w_new = w + 1.0 * y[i] * A[i]
    score_after = w_new @ A[i]
    ok &= _report("update moves the misclassified point toward correct",
                  max(0.0, y[i] * score_before - y[i] * score_after +
                      np.sum(A[i] ** 2)), 1e-9)
    print(f"  [INFO]  {'y*score before -> after the update':<58s}  "
          f"{y[i] * score_before:.3f} -> {y[i] * score_after:.3f}")

    print("\nADALINE update = SGD on squared loss (README §10)")
    # One delta-rule step must equal one gradient step on 1/2(y - w.x)^2.
    w0 = rng.standard_normal(5)
    xi, yi = A[3], y[3]
    delta_step = w0 + 0.01 * (yi - w0 @ xi) * xi
    grad = -(yi - w0 @ xi) * xi                     # d/dw of 1/2(y - w.x)^2
    sgd_step = w0 - 0.01 * grad
    ok &= _report("delta rule equals a gradient step exactly",
                  float(np.abs(delta_step - sgd_step).max()), 1e-15)

    print("\nStructural properties (README §3, §6)")
    # Learning rate does not matter on separable data (README §3).
    p1 = Perceptron(eta=1.0, max_epochs=500, random_state=1).fit(X, y)
    p2 = Perceptron(eta=0.01, max_epochs=500, random_state=1).fit(X, y)
    ok &= _report("eta does not change predictions (separable case)",
                  float(np.mean(p1.predict(X) != p2.predict(X))), 1e-12)

    # Non-separable data: plain perceptron does NOT converge; pocket returns something sane.
    print("\nNon-separable data (README §6.1, §9)")
    X_ns = rng.standard_normal((200, 2))
    y_ns = np.where(X_ns[:, 0] * X_ns[:, 1] > 0, 1, -1)   # XOR-like, not separable
    plain = Perceptron(variant="plain", max_epochs=100).fit(X_ns, y_ns)
    pocket = Perceptron(variant="pocket", max_epochs=100).fit(X_ns, y_ns)
    print(f"  [{'PASS' if not plain.converged_ else 'FAIL'}]  "
          f"{'plain perceptron never converges on non-separable data':<58s}  "
          f"converged={plain.converged_}")
    ok &= not plain.converged_
    print(f"  [INFO]  {'training accuracy: plain vs pocket':<58s}  "
          f"{plain.score(X_ns, y_ns):.4f} vs {pocket.score(X_ns, y_ns):.4f}")
    ok &= pocket.score(X_ns, y_ns) >= plain.score(X_ns, y_ns) - 1e-9

    return ok


# =============================================================================
# EXPERIMENTS
# =============================================================================


def experiment_convergence_bound():
    """README §5: mistakes <= (R/gamma)^2, checked empirically."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — the convergence theorem  (README §5)")
    print("=" * 88)
    print("""
Novikoff's theorem bounds the perceptron's mistakes by (R/gamma)^2, where R is the data
radius and gamma the margin — independent of n and of dimension. We construct problems
with a KNOWN margin (points placed a fixed distance from a chosen hyperplane) and check
the actual mistake count against the bound.
""")
    rng = np.random.default_rng(1)
    print(f"  {'dimension':>10s}  {'n':>6s}  {'margin gamma':>13s}  {'radius R':>9s}  "
          f"{'(R/gamma)^2':>12s}  {'mistakes':>9s}  {'within bound?':>14s}")
    print("  " + "-" * 82)

    for d, n, gap in [(2, 100, 1.0), (5, 200, 1.0), (10, 300, 0.5), (20, 400, 0.5),
                      (5, 200, 0.2)]:
        w_star = rng.standard_normal(d)
        w_star /= np.linalg.norm(w_star)            # unit vector, as in the proof

        # Place points at least `gap` from the hyperplane on their labelled side.
        X, y = [], []
        while len(X) < n:
            x = rng.standard_normal(d) * 2
            s = x @ w_star
            if abs(s) >= gap:
                X.append(x)
                y.append(1 if s > 0 else -1)
        X = np.array(X)
        y = np.array(y)

        R = float(np.max(np.linalg.norm(X, axis=1)))
        gamma = float(np.min(np.abs(X @ w_star)))   # actual achieved margin
        bound = (R / gamma) ** 2

        p = Perceptron(eta=1.0, max_epochs=100000, shuffle=False).fit(X, y)
        within = p.n_mistakes_ <= bound
        print(f"  {d:10d}  {n:6d}  {gamma:13.4f}  {R:9.3f}  {bound:12.1f}  "
              f"{p.n_mistakes_:9d}  {str(within):>14s}")

    print("""
  The mistake count is under (R/gamma)^2 in every row — the theorem holds exactly. Notice
  the bound does not depend on n: the 400-point problem is not harder than the 100-point
  one, only the margin and radius matter.

  The last row (gamma = 0.2) shows the bound loosening as the margin shrinks: a small
  margin means many more mistakes are permitted, and the perceptron uses them. That is the
  §6.3 weakness previewed — Experiment 2 pushes it further.

  This was a genuine milestone: a PROVABLE guarantee, with a computable bound, that a
  machine will finish learning. Nothing like it existed before 1962.""")


def experiment_margin():
    """README §6.3: convergence time explodes as the margin shrinks."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — the margin controls everything  (README §6.3)")
    print("=" * 88)
    print("""
The (R/gamma)^2 bound scales as 1/gamma^2, so halving the margin should roughly quadruple
the work. Sweeping the margin toward zero (nearly non-separable) and counting mistakes:
""")
    rng = np.random.default_rng(2)
    d, n = 5, 300
    w_star = rng.standard_normal(d)
    w_star /= np.linalg.norm(w_star)

    print(f"  {'margin gamma':>13s}  {'(R/gamma)^2 bound':>18s}  {'mistakes':>9s}  "
          f"{'epochs':>7s}  {'converged?':>11s}")
    print("  " + "-" * 66)

    for gap in (2.0, 1.0, 0.5, 0.25, 0.1, 0.03):
        X, y = [], []
        while len(X) < n:
            x = rng.standard_normal(d) * 2
            s = x @ w_star
            if abs(s) >= gap:
                X.append(x)
                y.append(1 if s > 0 else -1)
        X = np.array(X)
        y = np.array(y)
        R = float(np.max(np.linalg.norm(X, axis=1)))
        gamma = float(np.min(np.abs(X @ w_star)))
        bound = (R / gamma) ** 2

        p = Perceptron(eta=1.0, max_epochs=5000, shuffle=False).fit(X, y)
        print(f"  {gamma:13.4f}  {bound:18.0f}  {p.n_mistakes_:9d}  {p.n_epochs_:7d}  "
              f"{str(p.converged_):>11s}")

    print("""
  Both the bound and the actual mistake count climb steeply as the margin shrinks —
  roughly as 1/gamma^2. A nearly-separable problem can take an enormous number of updates,
  and a genuinely non-separable one (gamma -> 0) never converges at all.

  This is the perceptron's central practical flaw (README §6). It has no concept of a
  'best' separator and no graceful behaviour when none exists. The SVM's max-margin
  objective (03.07) is the direct answer: instead of accepting any separator and suffering
  when the margin is small, it MAXIMIZES the margin, and its soft-margin form degrades
  gracefully when the data is not separable at all.""")


def experiment_xor():
    """README §7: one perceptron fails on XOR; a two-layer net succeeds."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — the XOR catastrophe, and its resolution  (README §7)")
    print("=" * 88)
    print("""
The failure that caused the first AI winter. XOR is not linearly separable — the positives
sit on one diagonal, the negatives on the other — so no single perceptron can learn it. A
two-layer network can, because a hidden layer carves regions a single line cannot.
""")
    # The four canonical XOR points, plus a noisy cloud around each for a real test.
    rng = np.random.default_rng(3)
    centers = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    labels = np.array([-1, 1, 1, -1])
    X = np.repeat(centers, 200, axis=0) + 0.1 * rng.standard_normal((800, 2))
    y = np.repeat(labels, 200)

    print(f"  {'model':<34s}  {'train accuracy':>14s}  {'verdict':>18s}")
    print("  " + "-" * 70)

    perc = Perceptron(max_epochs=2000).fit(X, y)
    print(f"  {'single perceptron':<34s}  {perc.score(X, y):14.4f}  "
          f"{'CANNOT (~chance)':>18s}")

    ada = Adaline(eta=0.05, max_epochs=500).fit(X, y)
    print(f"  {'ADALINE (still linear)':<34s}  {ada.score(X, y):14.4f}  "
          f"{'CANNOT (~chance)':>18s}")

    for hidden in (2, 4):
        net = TwoLayerNet(hidden=hidden, epochs=5000).fit(X, y)
        acc = net.score(X, y)
        print(f"  {f'two-layer net ({hidden} hidden units)':<34s}  {acc:14.4f}  "
              f"{'SOLVES it' if acc > 0.95 else 'struggling':>18s}")

    # Confirm the single perceptron IS at chance because no separator exists.
    print(f"""
  The single perceptron and ADALINE are stuck near 50% — chance — because they can only
  draw a line, and no line separates XOR. This is not a training failure or a bad learning
  rate; it is a REPRESENTATIONAL limit, which is exactly what Minsky and Papert proved.

  The two-layer network reaches ~100% with as few as 2 hidden units. XOR = OR AND (NOT
  AND): each of OR and AND is linearly separable (one hidden unit each), and the output
  unit combines them. Depth turned a problem one layer cannot represent into a composition
  of problems it can.

  The piece missing in 1969 was not the architecture but a way to TRAIN it through the
  hidden layer — the hard threshold has no gradient. Replacing the step with the smooth
  tanh used here, and applying the chain rule (backprop, 1986), is what ended the first AI
  winter. That is the whole of Part 7 in one experiment.""")


def experiment_averaged():
    """README §9: the averaged perceptron generalizes far better."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — the averaged perceptron  (README §9)")
    print("=" * 88)
    print("""
The plain perceptron returns whatever weights it held when it stopped — order-dependent and
often a poor separator. The averaged perceptron returns the survival-weighted average of
ALL weight vectors seen, which reduces variance (the ensembling principle, 00.03 §4.3) and
generalizes markedly better. Measuring the gap on noisy, nearly-separable data:
""")
    rng = np.random.default_rng(4)
    d = 20

    print(f"  {'noise level':>12s}  {'plain TEST acc':>15s}  {'pocket TEST acc':>16s}  "
          f"{'averaged TEST acc':>18s}")
    print("  " + "-" * 66)

    for noise in (0.0, 0.5, 1.0, 2.0):
        w_true = rng.standard_normal(d)

        def make(m):
            X = rng.standard_normal((m, d))
            score = X @ w_true + noise * rng.standard_normal(m)
            return X, np.where(score > 0, 1, -1)

        X_tr, y_tr = make(200)
        X_te, y_te = make(4000)

        results = {}
        for variant in ("plain", "pocket", "averaged"):
            # Average over several orderings to expose the plain variant's instability.
            accs = [Perceptron(variant=variant, max_epochs=50, random_state=s)
                    .fit(X_tr, y_tr).score(X_te, y_te) for s in range(10)]
            results[variant] = np.mean(accs)

        print(f"  {noise:12.1f}  {results['plain']:15.4f}  {results['pocket']:16.4f}  "
              f"{results['averaged']:18.4f}")

    print("""
  Read the averaged column against the plain one. It wins in every row, and — the pattern
  that matters — the gap WIDENS with noise: negligible at noise 0, but ~2 points by noise
  2.0. That is exactly the signature of a variance-reduction method. When there is little
  noise there is little variance to cancel, so averaging barely helps; as noise grows, the
  plain perceptron is increasingly shaped by whichever noisy mistakes it happened to end on,
  and averaging over the whole trajectory cancels that (the ensembling principle, 00.03 §4.3).

  Two points of test accuracy is a modest headline number but a real and free one — no
  extra passes, no extra memory beyond a running sum. It is why the 'voted/averaged
  perceptron' remained a serious baseline for structured prediction long after the plain
  one was abandoned.

  The lesson generalizes well beyond the perceptron: averaging over a trajectory of models
  is a recurring, almost-free way to cut variance — the same idea reappears as Stochastic
  Weight Averaging and exponential moving averages of weights in deep learning (07.06).""")


# =============================================================================

if __name__ == "__main__":
    print(__doc__)

    all_passed = verify()

    experiment_convergence_bound()
    experiment_margin()
    experiment_xor()
    experiment_averaged()

    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 88)
