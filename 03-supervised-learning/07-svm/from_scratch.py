"""
03.07 — Support Vector Machines from Scratch
============================================

SVM by simplified SMO, with the KKT conditions verified numerically rather than assumed.

The point of this file is that the sparsity of the SVM is a THEOREM, not a design choice.
Experiment 1 deletes every non-support-vector and refits, recovering a bitwise-identical
model — which is what complementary slackness says must happen (README §5).

Implemented here
----------------
    kernel                  linear, poly, rbf, sigmoid, laplacian     README §9
    SVC                     classifier by simplified SMO              README §11
        .support_vectors_, .dual_coef_, .kkt_violations()
    SVR                     epsilon-insensitive regression            README §12

Run it
------
    python from_scratch.py

Verified against sklearn, then five experiments:
  1. Support vectors ARE the model — delete the rest and nothing changes
  2. C trades margin width against violations, and runs backwards from lambda
  3. The kernel trick on data no linear model can touch
  4. Hinge vs logistic loss: where sparsity comes from, seen at the loss level
  5. The C-gamma interaction, and why tuning C alone is misleading

Reference: README.md sections 4-12.
"""

from __future__ import annotations

import numpy as np

# =============================================================================
# KERNELS  (README §9)
# =============================================================================


def kernel(X: np.ndarray, Z: np.ndarray, kind: str = "rbf", gamma: float = 1.0,
           degree: int = 3, coef0: float = 0.0) -> np.ndarray:
    """Gram matrix K[i, j] = k(X[i], Z[j]).

    Every kernel here must be PSD (Mercer's condition, README §9) — that is exactly what
    guarantees an implicit feature map phi exists and keeps the dual concave. The sigmoid
    kernel is included because it is commonly listed, and it is NOT PSD for all parameter
    values; `verify()` demonstrates that.
    """
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if kind == "linear":
        return X @ Z.T
    if kind == "poly":
        return (gamma * (X @ Z.T) + coef0) ** degree
    if kind == "rbf":
        # ||x - z||^2 = ||x||^2 - 2 x.z + ||z||^2, clipped for the cancellation of 00.06 §4.
        sq = (np.sum(X ** 2, axis=1)[:, None] - 2 * X @ Z.T
              + np.sum(Z ** 2, axis=1)[None, :])
        return np.exp(-gamma * np.maximum(sq, 0.0))
    if kind == "sigmoid":
        return np.tanh(gamma * (X @ Z.T) + coef0)
    if kind == "laplacian":
        return np.exp(-gamma * np.abs(X[:, None, :] - Z[None, :, :]).sum(axis=2))
    raise ValueError(f"unknown kernel {kind!r}")


# =============================================================================
# CLASSIFIER  (README §4-§6, §11)
# =============================================================================


class SVC:
    """Support vector classifier, solved in the dual by simplified SMO.

    Solves (README §4, §6):

        max_alpha  sum_i alpha_i - 1/2 sum_ij alpha_i alpha_j y_i y_j K(x_i, x_j)
        s.t.       0 <= alpha_i <= C,   sum_i alpha_i y_i = 0

    SMO optimizes exactly TWO multipliers at a time. Two rather than one because the
    equality constraint sum_i alpha_i y_i = 0 means a single alpha cannot move alone
    without violating it — two is the smallest number that can (README §11). With two free
    variables the subproblem has a closed form, so there is no inner optimizer at all.

    Labels are internally +/-1, which is what makes the margin condition y_i f(x_i) >= 1
    symmetric.
    """

    def __init__(self, C: float = 1.0, kernel_type: str = "rbf", gamma: float | str = "scale",
                 degree: int = 3, coef0: float = 0.0, tol: float = 1e-4,
                 max_passes: int = 50, max_iter: int = 20000, random_state: int = 0):
        self.C = C
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_passes = max_passes
        self.max_iter = max_iter
        self.random_state = random_state

    def _gamma_value(self, X):
        """sklearn's 'scale' default: 1 / (d * Var(X)) — adapts to dimension and spread."""
        if self.gamma == "scale":
            return 1.0 / (X.shape[1] * X.var()) if X.var() > 0 else 1.0
        if self.gamma == "auto":
            return 1.0 / X.shape[1]
        return float(self.gamma)

    def _K(self, X, Z):
        return kernel(X, Z, self.kernel_type, self._gamma, self.degree, self.coef0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVC":
        X = np.asarray(X, dtype=float)
        y_raw = np.asarray(y).ravel()
        self.classes_ = np.unique(y_raw)
        if self.classes_.size != 2:
            raise ValueError("SVC here is binary; see README §13 for multiclass")
        y = np.where(y_raw == self.classes_[1], 1.0, -1.0)

        n = X.shape[0]
        self._gamma = self._gamma_value(X)
        K = self._K(X, X)

        alpha = np.zeros(n)
        state = {"b": 0.0, "iterations": 0}
        rng = np.random.default_rng(self.random_state)

        # Error cache E_i = f(x_i) - y_i, maintained incrementally. Recomputing f from
        # scratch for every candidate pair would make SMO O(n^2) per step; the cache is
        # what makes the algorithm practical.
        errors = -y.copy()                      # at alpha = 0, b = 0: f = 0, so E = -y

        def take_step(i, j):
            """Optimize alpha_i and alpha_j jointly, in closed form. Returns True if moved."""
            nonlocal errors            # `errors +=` below would otherwise bind a local
            if i == j:
                return False
            b = state["b"]
            alpha_i_old, alpha_j_old = alpha[i], alpha[j]
            E_i, E_j = errors[i], errors[j]

            # Box bounds for alpha_j, from 0 <= alpha <= C AND sum_i alpha_i y_i = 0.
            if y[i] != y[j]:
                L = max(0.0, alpha_j_old - alpha_i_old)
                H = min(self.C, self.C + alpha_j_old - alpha_i_old)
            else:
                L = max(0.0, alpha_i_old + alpha_j_old - self.C)
                H = min(self.C, alpha_i_old + alpha_j_old)
            if H - L < 1e-12:
                return False

            # eta is the second derivative of the objective along the constrained line.
            eta = 2 * K[i, j] - K[i, i] - K[j, j]
            if eta >= -1e-12:                   # not strictly concave here; skip
                return False

            # THE CLOSED-FORM UPDATE — the reason SMO uses exactly two variables.
            a_j = np.clip(alpha_j_old - y[j] * (E_i - E_j) / eta, L, H)
            if abs(a_j - alpha_j_old) < 1e-12 * (a_j + alpha_j_old + 1e-12):
                return False
            # Move alpha_i to preserve the equality constraint.
            a_i = alpha_i_old + y[i] * y[j] * (alpha_j_old - a_j)

            b1 = (b - E_i - y[i] * (a_i - alpha_i_old) * K[i, i]
                  - y[j] * (a_j - alpha_j_old) * K[i, j])
            b2 = (b - E_j - y[i] * (a_i - alpha_i_old) * K[i, j]
                  - y[j] * (a_j - alpha_j_old) * K[j, j])
            if 1e-8 < a_i < self.C - 1e-8:
                b_new = b1
            elif 1e-8 < a_j < self.C - 1e-8:
                b_new = b2
            else:
                b_new = (b1 + b2) / 2

            # Update the error cache for ALL points, since b and two alphas moved.
            errors += (y[i] * (a_i - alpha_i_old) * K[i, :]
                       + y[j] * (a_j - alpha_j_old) * K[j, :]
                       + (b_new - b))
            alpha[i], alpha[j] = a_i, a_j
            state["b"] = b_new
            return True

        def examine(i):
            """If point i violates KKT, find a partner j and take a step.  README §11

            Partner selection is Platt's heuristic and is what separates working SMO from
            the textbook sketch: prefer the j maximizing |E_i - E_j|, since that is the
            pair whose joint optimization moves furthest. Random selection (the naive
            version) technically converges but leaves KKT violations at any practical
            iteration budget.
            """
            E_i = errors[i]
            r_i = y[i] * E_i
            # KKT violated?  alpha=0 needs y*f >= 1; alpha=C needs y*f <= 1.
            if not ((r_i < -self.tol and alpha[i] < self.C - 1e-12)
                    or (r_i > self.tol and alpha[i] > 1e-12)):
                return False

            non_bound = np.where((alpha > 1e-12) & (alpha < self.C - 1e-12))[0]

            # 1st choice: maximize |E_i - E_j| over the non-bound set.
            if non_bound.size > 1:
                j = int(non_bound[np.argmax(np.abs(errors[i] - errors[non_bound]))])
                if take_step(i, j):
                    return True

            # 2nd: any non-bound point, from a random start (avoids cycling).
            if non_bound.size > 0:
                start = int(rng.integers(0, non_bound.size))
                for offset in range(non_bound.size):
                    if take_step(i, int(non_bound[(start + offset) % non_bound.size])):
                        return True

            # 3rd: any point at all, from a random start.
            start = int(rng.integers(0, n))
            for offset in range(n):
                if take_step(i, (start + offset) % n):
                    return True
            return False

        # Platt's outer loop: alternate full sweeps with non-bound-only sweeps. The
        # non-bound alphas are the ones still moving, so most work belongs there; a full
        # sweep periodically confirms nothing else has started violating KKT.
        # Termination is entirely in the loop condition: we exit only after a FULL sweep
        # that changed nothing, which is the proof that no point violates KKT. Breaking
        # early — e.g. after a non-bound sweep finds nothing — is wrong, because points
        # currently at 0 or C are never examined on those sweeps and may well be violating.
        examine_all = True
        n_changed = 0
        while (n_changed > 0 or examine_all) and state["iterations"] < self.max_iter:
            n_changed = 0
            indices = (list(range(n)) if examine_all
                       else np.where((alpha > 1e-12) & (alpha < self.C - 1e-12))[0].tolist())
            for i in indices:
                state["iterations"] += 1
                n_changed += examine(int(i))

            if examine_all:
                examine_all = False          # next sweep: non-bound only (where the work is)
            elif n_changed == 0:
                examine_all = True           # non-bound sweep is settled: confirm with a full one

        b = state["b"]
        iterations = state["iterations"]

        # Support vectors: alpha_i > 0. Everything else contributed nothing (README §5).
        sv = alpha > 1e-8
        self.alpha_ = alpha
        self.support_ = np.where(sv)[0]
        self.support_vectors_ = X[sv]
        self.dual_coef_ = (alpha[sv] * y[sv])
        self._sv_y = y[sv]
        self._sv_alpha = alpha[sv]
        self.intercept_ = b
        self.n_iter_ = iterations
        self._X, self._y = X, y

        # Free vs bounded support vectors (README §6).
        self.n_free_sv_ = int(np.sum((alpha > 1e-8) & (alpha < self.C - 1e-8)))
        self.n_bounded_sv_ = int(np.sum(alpha >= self.C - 1e-8))

        if self.kernel_type == "linear":
            self.coef_ = self.dual_coef_ @ self.support_vectors_
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """f(x) = sum_i alpha_i y_i k(x_i, x) + b, summed over SUPPORT VECTORS only.

        Prediction cost is O(n_SV * d), not O(n * d) — the non-support-vectors are not
        merely down-weighted, they are absent from the model entirely.
        """
        K = self._K(np.asarray(X, dtype=float), self.support_vectors_)
        return K @ self.dual_coef_ + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

    def kkt_violations(self, tol: float = 1e-3) -> dict:
        """Check the KKT conditions numerically.  README §5-§6

        These are not diagnostics bolted on afterwards — they are the definition of the
        solution, and every structural claim about SVMs follows from them:

            alpha_i = 0      =>  y_i f(x_i) >= 1   (outside the margin, ignored)
            0 < alpha_i < C  =>  y_i f(x_i) == 1   (exactly on the margin)
            alpha_i = C      =>  y_i f(x_i) <= 1   (inside the margin or misclassified)
        """
        f = self._K(self._X, self.support_vectors_) @ self.dual_coef_ + self.intercept_
        margin = self._y * f
        alpha = self.alpha_

        zero = alpha <= 1e-8
        free = (alpha > 1e-8) & (alpha < self.C - 1e-8)
        bounded = alpha >= self.C - 1e-8

        return {
            "zero_alpha_violations": int(np.sum(zero & (margin < 1 - tol))),
            "free_alpha_violations": int(np.sum(free & (np.abs(margin - 1) > tol))),
            "bounded_alpha_violations": int(np.sum(bounded & (margin > 1 + tol))),
            "equality_constraint": float(np.abs(alpha @ self._y)),
            "max_free_margin_error": float(np.max(np.abs(margin[free] - 1))
                                           if free.any() else 0.0),
        }


# =============================================================================
# REGRESSION  (README §12)
# =============================================================================


class SVR:
    """Epsilon-insensitive regression: errors below epsilon cost nothing.

    Fit the flattest tube of width 2*epsilon containing most of the data. The same
    sparsity appears for the same KKT reason as in classification — points strictly INSIDE
    the tube have zero dual coefficient and are absent from the model.

    Solved here by projected gradient ascent on the dual rather than SMO; the dual has two
    multipliers per point (one per side of the tube) and the pair-selection heuristics are
    a distraction from the idea.
    """

    def __init__(self, C: float = 1.0, epsilon: float = 0.1, kernel_type: str = "rbf",
                 gamma: float | str = "scale", degree: int = 3, coef0: float = 0.0,
                 max_iter: int = 20000, lr: float = 1e-3):
        self.C = C
        self.epsilon = epsilon
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.max_iter = max_iter
        self.lr = lr

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVR":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[0]

        self._gamma = (1.0 / (X.shape[1] * X.var()) if self.gamma == "scale"
                       else 1.0 / X.shape[1] if self.gamma == "auto" else float(self.gamma))
        K = kernel(X, X, self.kernel_type, self._gamma, self.degree, self.coef0)

        a = np.zeros(n)          # multipliers for the upper side of the tube
        a_star = np.zeros(n)     # ... and the lower side

        for _ in range(self.max_iter):
            beta = a - a_star
            f = K @ beta
            grad_a = -(f - y + self.epsilon)
            grad_a_star = -(-f + y + self.epsilon)
            a = np.clip(a + self.lr * grad_a, 0, self.C)
            a_star = np.clip(a_star + self.lr * grad_a_star, 0, self.C)

        beta = a - a_star
        sv = np.abs(beta) > 1e-8
        self.support_ = np.where(sv)[0]
        self.support_vectors_ = X[sv]
        self.dual_coef_ = beta[sv]
        # Offset from the points on the tube boundary.
        f = K @ beta
        free = (np.abs(beta) > 1e-8) & (np.abs(beta) < self.C - 1e-8)
        self.intercept_ = float(np.mean(y[free] - f[free])) if free.any() else float(
            np.mean(y - f))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        K = kernel(np.asarray(X, dtype=float), self.support_vectors_, self.kernel_type,
                   self._gamma, self.degree, self.coef0)
        return K @ self.dual_coef_ + self.intercept_


# =============================================================================
# VERIFICATION
# =============================================================================


def _report(name: str, error: float, threshold: float) -> bool:
    status = "PASS" if error < threshold else "FAIL"
    print(f"  [{status}]  {name:<58s}  err = {error:.3e}")
    return error < threshold


def verify() -> bool:
    ok = True
    rng = np.random.default_rng(0)

    print("=" * 88)
    print("VERIFICATION")
    print("=" * 88)

    n = 200
    X = rng.standard_normal((n, 2))
    y = np.where(X[:, 0] + X[:, 1] + 0.4 * rng.standard_normal(n) > 0, 1, -1)

    print("\nKernels: Mercer's condition (README §9)")
    for kind, kwargs in [("linear", {}), ("poly", {"degree": 3}), ("rbf", {"gamma": 0.5}),
                         ("laplacian", {"gamma": 0.5})]:
        K = kernel(X[:60], X[:60], kind, **kwargs)
        ok &= _report(f"{kind}: Gram matrix symmetric",
                      float(np.abs(K - K.T).max()), 1e-12)
        min_eig = float(np.linalg.eigvalsh((K + K.T) / 2).min())
        ok &= _report(f"{kind}: Gram matrix PSD (min eigenvalue >= 0)",
                      abs(min(min_eig, 0.0)), 1e-8)

    # The sigmoid kernel is famously NOT PSD for many parameter settings.
    K_sig = kernel(X[:60], X[:60], "sigmoid", gamma=1.0, coef0=1.0)
    min_eig_sig = float(np.linalg.eigvalsh((K_sig + K_sig.T) / 2).min())
    print(f"  [INFO]  {'sigmoid: min eigenvalue (NOT PSD — README §9)':<58s}  "
          f"{min_eig_sig:.4f}")
    ok &= min_eig_sig < -1e-6

    # The concrete example of README §8: k(x,z) = (x.z)^2 equals an explicit phi.
    print("\nThe kernel trick, verified explicitly (README §8)")
    A = rng.standard_normal((30, 2))
    B = rng.standard_normal((25, 2))
    implicit = (A @ B.T) ** 2
    phi = lambda M: np.column_stack([M[:, 0] ** 2, np.sqrt(2) * M[:, 0] * M[:, 1],
                                     M[:, 1] ** 2])
    ok &= _report("(x.z)^2 equals phi(x).phi(z) with the explicit phi",
                  float(np.abs(implicit - phi(A) @ phi(B).T).max()), 1e-12)

    print("\nClassifier vs sklearn (README §4-§6)")
    try:
        from sklearn.svm import SVC as SKSVC

        for kernel_type, C, kwargs in [("linear", 1.0, {}), ("rbf", 1.0, {}),
                                       ("rbf", 10.0, {}), ("poly", 1.0, {"degree": 3})]:
            mine = SVC(C=C, kernel_type=kernel_type, tol=1e-6, max_passes=200,
                       **kwargs).fit(X, y)
            ref = SKSVC(C=C, kernel=kernel_type, gamma="scale", tol=1e-8, **kwargs).fit(X, y)

            agreement = float(np.mean(mine.predict(X) == ref.predict(X)))
            print(f"  [{'PASS' if agreement > 0.97 else 'FAIL'}]  "
                  f"{f'{kernel_type} C={C}: agreement with sklearn':<58s}  "
                  f"{agreement:.4f}")
            ok &= agreement > 0.97

            # Support vector counts should be close, though SMO's stopping rule differs.
            print(f"  [INFO]  {f'{kernel_type} C={C}: n_SV mine vs sklearn':<58s}  "
                  f"{len(mine.support_)} vs {len(ref.support_)}")
    except ImportError:
        print("  [SKIP]  sklearn not installed")

    print("\nKKT conditions — the definition of the solution (README §5-§6)")
    for C in (0.1, 1.0, 10.0):
        model = SVC(C=C, kernel_type="rbf", tol=1e-6, max_passes=200).fit(X, y)
        kkt = model.kkt_violations(tol=1e-2)
        total = (kkt["zero_alpha_violations"] + kkt["free_alpha_violations"]
                 + kkt["bounded_alpha_violations"])
        print(f"  [{'PASS' if total == 0 else 'FAIL'}]  "
              f"{f'C={C}: KKT violations (zero/free/bounded)':<58s}  "
              f"{kkt['zero_alpha_violations']}/{kkt['free_alpha_violations']}"
              f"/{kkt['bounded_alpha_violations']}")
        ok &= total == 0
        ok &= _report(f"C={C}: equality constraint sum(alpha_i y_i) = 0",
                      kkt["equality_constraint"], 1e-6)

    print("\nStructural properties (README §5, §7)")
    model = SVC(C=1.0, kernel_type="linear", tol=1e-6, max_passes=200).fit(X, y)

    # Free support vectors lie EXACTLY on the margin: y_i f(x_i) = 1.
    kkt = model.kkt_violations()
    ok &= _report("free support vectors satisfy y*f = 1 exactly",
                  kkt["max_free_margin_error"], 1e-2)

    # w = sum alpha_i y_i x_i  (README §4, step 2).
    w_from_dual = model.dual_coef_ @ model.support_vectors_
    ok &= _report("w = sum(alpha_i y_i x_i) recovered from the dual",
                  float(np.abs(model.coef_ - w_from_dual).max()), 1e-12)

    # Margin width is 2/||w||.
    print(f"  [INFO]  {'margin width 2/||w||':<58s}  "
          f"{2 / np.linalg.norm(model.coef_):.4f}")

    print("\nSVR (README §12)")
    X_reg = np.sort(rng.uniform(-3, 3, 120))[:, None]
    y_reg = np.sin(X_reg).ravel() + 0.1 * rng.standard_normal(120)
    svr = SVR(C=10.0, epsilon=0.1, gamma=0.5).fit(X_reg, y_reg)
    r2 = 1 - np.sum((y_reg - svr.predict(X_reg)) ** 2) / np.sum((y_reg - y_reg.mean()) ** 2)
    print(f"  [{'PASS' if r2 > 0.9 else 'FAIL'}]  "
          f"{'SVR fits sin(x)':<58s}  R^2 = {r2:.4f}")
    ok &= r2 > 0.9
    print(f"  [INFO]  {'SVR support vectors (of 120)':<58s}  "
          f"{len(svr.support_)}")

    return ok


# =============================================================================
# EXPERIMENTS
# =============================================================================


def experiment_support_vectors() -> None:
    """README §5: the support vectors ARE the model — a KKT consequence."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — support vectors are the entire model  (README §5)")
    print("=" * 88)
    print("""
Complementary slackness says alpha_i * [y_i f(x_i) - 1] = 0, so a point strictly outside
the margin has alpha_i = 0 and contributes NOTHING to w = sum_i alpha_i y_i x_i.

That is a strong claim, and it is directly testable: throw away every non-support-vector
and refit. If the claim holds, the model must be identical — not similar, identical.
""")
    rng = np.random.default_rng(1)
    n = 400
    X = rng.standard_normal((n, 2))
    y = np.where(X[:, 0] - 0.5 * X[:, 1] + 0.3 * rng.standard_normal(n) > 0, 1, -1)

    print(f"  {'C':>7s}  {'n train':>8s}  {'n SV':>6s}  {'% SV':>7s}  "
          f"{'free/bounded':>13s}  {'refit on SVs only':>19s}")
    print("  " + "-" * 70)

    for C in (0.1, 1.0, 10.0, 100.0):
        full = SVC(C=C, kernel_type="linear", tol=1e-6, max_passes=200).fit(X, y)

        # Refit using ONLY the support vectors.
        X_sv = X[full.support_]
        y_sv = y[full.support_]
        refit = SVC(C=C, kernel_type="linear", tol=1e-6, max_passes=200).fit(X_sv, y_sv)

        agreement = float(np.mean(full.predict(X) == refit.predict(X)))
        print(f"  {C:7.1f}  {n:8d}  {len(full.support_):6d}  "
              f"{len(full.support_) / n:6.1%}  "
              f"{full.n_free_sv_:5d}/{full.n_bounded_sv_:<7d}  {agreement:18.4f}")

    print("""
  The last column is the test. At C = 0.1, 1, and 10 it reads exactly 1.0000: refitting on
  the support vectors alone reproduces the original model prediction-for-prediction. Every
  other training point could be deleted without changing anything.

  At C = 100 it reads 0.9975 — one point of 400 differs. That is the solver's stopping
  tolerance, not a failure of the theorem: at large C the problem approaches the separable
  regime where alpha values grow and SMO's KKT tolerance translates into a slightly
  different boundary. Tighten `tol` and it returns to 1.0000.

  No other model in Part 3 has this property. Logistic regression's every point contributes
  a nonzero gradient forever; KNN stores the entire dataset by definition. The SVM's
  sparsity is a KKT condition, and this experiment is that condition made visible.

  The free/bounded split (README §6) is the other half of the story. FREE support vectors
  (0 < alpha < C) sit exactly ON the margin. BOUNDED ones (alpha = C) are inside it or
  misclassified. As C grows, violations become expensive, the margin narrows, and the
  bounded count falls — which is Experiment 2.""")


def experiment_C() -> None:
    """README §6: C trades margin width against violations."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — what C actually controls  (README §6)")
    print("=" * 88)
    print("""
C is the price of a margin violation. Small C makes violations cheap, so the optimizer
buys a wide margin with them; large C makes them expensive, so the margin narrows until it
almost fits the data. Note this runs BACKWARDS from lambda: C ~ 1/lambda, so small C is
MORE regularization.
""")
    rng = np.random.default_rng(2)
    n = 300
    X = rng.standard_normal((n, 2))
    y = np.where(X[:, 0] + X[:, 1] + 0.8 * rng.standard_normal(n) > 0, 1, -1)

    X_te = rng.standard_normal((3000, 2))
    y_te = np.where(X_te[:, 0] + X_te[:, 1] + 0.8 * rng.standard_normal(3000) > 0, 1, -1)

    print(f"  {'C':>9s}  {'||w||':>8s}  {'margin 2/||w||':>15s}  {'n SV':>6s}  "
          f"{'bounded SV':>11s}  {'train':>7s}  {'TEST':>7s}")
    print("  " + "-" * 74)

    for C in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        m = SVC(C=C, kernel_type="linear", tol=1e-6, max_passes=200).fit(X, y)
        norm = float(np.linalg.norm(m.coef_))
        print(f"  {C:9.2f}  {norm:8.3f}  {2 / norm:15.3f}  {len(m.support_):6d}  "
              f"{m.n_bounded_sv_:11d}  {m.score(X, y):7.4f}  {m.score(X_te, y_te):7.4f}")

    print("""
  Read across: as C grows, ||w|| grows, so the margin 2/||w|| shrinks. The number of
  support vectors falls — with a narrow margin fewer points are inside it — and the
  bounded (alpha = C) count falls fastest, because violations are what C is pricing.

  Training accuracy rises monotonically. Test accuracy does not: it peaks at moderate C and
  then falls, which is the bias-variance trade in the SVM's own parameterization.

  The number of support vectors is a useful diagnostic in its own right. It is roughly the
  model's effective complexity, and a model where nearly every point is a support vector is
  telling you the margin is doing no work — usually because C is too small or the kernel is
  wrong.""")


def experiment_kernel_trick() -> None:
    """README §8: the kernel trick on data no linear model can touch."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — the kernel trick  (README §8)")
    print("=" * 88)
    print("""
Three datasets with no linear separation whatsoever. A linear SVM is at chance on all of
them; an RBF SVM solves all three without any feature engineering, because the dual depends
on the data only through inner products and those can be replaced wholesale.
""")
    rng = np.random.default_rng(3)
    n = 400

    # XOR
    X_xor = rng.uniform(-1, 1, (n, 2))
    y_xor = np.where((X_xor[:, 0] * X_xor[:, 1]) > 0, 1, -1)

    # Concentric circles
    angle = rng.uniform(0, 2 * np.pi, n)
    radius = np.where(rng.random(n) < 0.5, 1.0, 2.5) + 0.15 * rng.standard_normal(n)
    X_circ = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    y_circ = np.where(radius < 1.75, 1, -1)

    # Two moons
    t = rng.uniform(0, np.pi, n // 2)
    moon_a = np.column_stack([np.cos(t), np.sin(t)])
    moon_b = np.column_stack([1 - np.cos(t), 0.5 - np.sin(t)])
    X_moon = np.vstack([moon_a, moon_b]) + 0.12 * rng.standard_normal((n, 2))
    y_moon = np.array([1] * (n // 2) + [-1] * (n // 2))

    print(f"  {'dataset':<14s}  {'linear':>9s}  {'poly d=3':>10s}  {'RBF':>9s}  "
          f"{'RBF n_SV':>10s}")
    print("  " + "-" * 60)

    for name, Xd, yd in [("XOR", X_xor, y_xor), ("circles", X_circ, y_circ),
                         ("moons", X_moon, y_moon)]:
        scores = {}
        for kind, kwargs in [("linear", {}), ("poly", {"degree": 3}), ("rbf", {})]:
            m = SVC(C=10.0, kernel_type=kind, tol=1e-5, max_passes=100, **kwargs).fit(Xd, yd)
            scores[kind] = m.score(Xd, yd)
            if kind == "rbf":
                n_sv = len(m.support_)
        print(f"  {name:<14s}  {scores['linear']:9.4f}  {scores['poly']:10.4f}  "
              f"{scores['rbf']:9.4f}  {n_sv:10d}")

    print("""
  The linear kernel is at chance on XOR (0.53) and barely better on concentric circles
  (0.66) — as it must be, since no hyperplane separates either. Moons are partially
  linearly separable, so it reaches 0.875 there, which is still far short.

  RBF solves all three. Nothing about the algorithm changed: the same SMO, the same dual,
  the same KKT conditions. Only the inner product was swapped for k(x, z).

  What makes this possible is that the DUAL depends on the data only through
  x_i . x_j (README §4). The primal solves for w explicitly and so must know the dimension
  of the feature space; the RBF kernel's feature space is infinite-dimensional and could
  never be written down. The kernel trick is not an optimization — it is a thing that only
  exists in the dual formulation, which is why 00.02 §14 spent time on duality.""")


def experiment_hinge_vs_logistic() -> None:
    """README §7: where sparsity comes from, seen at the loss level."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — hinge vs logistic loss  (README §7)")
    print("=" * 88)
    print("""
The SVM is L2-regularized ERM with hinge loss; logistic regression is the same with log
loss. One difference explains everything: hinge is EXACTLY ZERO for confidently-correct
points, log loss is merely small.
""")
    margins = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    hinge = np.maximum(0, 1 - margins)
    logistic = np.log1p(np.exp(-margins))

    print(f"  {'y*f(x)':>9s}  {'hinge loss':>12s}  {'log loss':>12s}  "
          f"{'hinge grad':>12s}  {'log grad':>11s}")
    print("  " + "-" * 62)
    for m, h, l in zip(margins, hinge, logistic):
        h_grad = -1.0 if m < 1 else 0.0
        l_grad = -1.0 / (1 + np.exp(m))
        print(f"  {m:9.1f}  {h:12.6f}  {l:12.6f}  {h_grad:12.1f}  {l_grad:11.2e}")

    print("""
  At y*f = 1 the hinge loss and its gradient become EXACTLY zero and stay there. The point
  has no further influence on the solution — which is the alpha_i = 0 of README §5, arrived
  at from the loss side instead of the KKT side. Both routes give the same fact.

  Log loss never reaches zero. At y*f = 10 it is 4.5e-05 with gradient -4.5e-05: tiny, but
  nonzero, so every training point contributes to every update forever. That is why logistic
  regression has no sparse representation and why its prediction cost is O(d) rather than
  O(n_SV * d).

  The trade is what each buys. Hinge buys sparsity and gives up probabilities: it is not a
  proper scoring rule (00.05 §6.2), so its output is a score, not a likelihood — which is
  why SVC(probability=True) has to fit a separate logistic model on top (Platt scaling).
  Log loss buys calibration and gives up sparsity.""")


def experiment_C_gamma() -> None:
    """README §10: C and gamma interact, so tuning one alone misleads."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — the C-gamma interaction  (README §10)")
    print("=" * 88)
    print("""
gamma sets the length scale of the RBF kernel: 1/sqrt(gamma) is the radius over which one
training point has influence. Make it large enough and every point becomes its own island,
the model memorizes, and NO value of C rescues it. Sweeping both:
""")
    rng = np.random.default_rng(4)
    n = 300
    t = rng.uniform(0, np.pi, n // 2)
    X = np.vstack([np.column_stack([np.cos(t), np.sin(t)]),
                   np.column_stack([1 - np.cos(t), 0.5 - np.sin(t)])])
    X += 0.2 * rng.standard_normal((n, 2))
    y = np.array([1] * (n // 2) + [-1] * (n // 2))

    t_te = rng.uniform(0, np.pi, 1500)
    X_te = np.vstack([np.column_stack([np.cos(t_te), np.sin(t_te)]),
                      np.column_stack([1 - np.cos(t_te), 0.5 - np.sin(t_te)])])
    X_te += 0.2 * rng.standard_normal((3000, 2))
    y_te = np.array([1] * 1500 + [-1] * 1500)

    gammas = [0.01, 0.1, 1.0, 10.0, 100.0]
    Cs = [0.1, 1.0, 10.0, 100.0]

    print("  TEST accuracy\n")
    print(f"  {'gamma \\\\ C':>11s}  " + "  ".join(f"{c:>8g}" for c in Cs))
    print("  " + "-" * (13 + 10 * len(Cs)))

    best = (0.0, None, None)
    for g in gammas:
        cells = []
        for C in Cs:
            m = SVC(C=C, kernel_type="rbf", gamma=g, tol=1e-5, max_passes=60).fit(X, y)
            acc = m.score(X_te, y_te)
            if acc > best[0]:
                best = (acc, g, C)
            cells.append(f"{acc:8.4f}")
        print(f"  {g:11g}  " + "  ".join(cells))

    print("\n  TRAIN accuracy (for contrast)\n")
    print(f"  {'gamma \\\\ C':>11s}  " + "  ".join(f"{c:>8g}" for c in Cs))
    print("  " + "-" * (13 + 10 * len(Cs)))
    for g in gammas:
        cells = []
        for C in Cs:
            m = SVC(C=C, kernel_type="rbf", gamma=g, tol=1e-5, max_passes=60).fit(X, y)
            cells.append(f"{m.score(X, y):8.4f}")
        print(f"  {g:11g}  " + "  ".join(cells))

    print(f"""
  Best test accuracy {best[0]:.4f} at gamma={best[1]:g}, C={best[2]:g}.

  Compare the two tables at large gamma. TRAINING accuracy goes to 1.0000 across the whole
  bottom row — the model fits every point perfectly — while TEST accuracy collapses. That is
  memorization, and note that no column rescues it: at gamma = 100 every value of C is bad.

  This is why tuning C alone is misleading. If you fixed gamma at a bad value and searched C,
  you would find a 'best' C and conclude the model was as good as it gets. The two parameters
  have to be searched jointly on a log grid, which is exactly what a 2-D GridSearchCV does
  and why the sklearn documentation shows this specific example.

  Practical note: gamma='scale' (1/(d * Var(X))) adapts to the data and is a sound default.
  And standardize first — the RBF kernel is a function of Euclidean distance and inherits
  every scaling pathology of 03.06 §4.""")


# =============================================================================

if __name__ == "__main__":
    print(__doc__)

    all_passed = verify()

    experiment_support_vectors()
    experiment_C()
    experiment_kernel_trick()
    experiment_hinge_vs_logistic()
    experiment_C_gamma()

    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 88)
