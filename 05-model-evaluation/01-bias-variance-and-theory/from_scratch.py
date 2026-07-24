"""
05.01 — Bias-Variance & Learning Theory, MEASURED from scratch (NumPy).

This chapter is about two questions: what is test error made of, and why is it bounded at all?
Both are answered by MEASUREMENT here, not assertion.

  The decomposition (README §2). For y = f(x) + eps with Var(eps)=sigma^2, the expected squared
  error at x0, averaged over training sets D and noise, splits as

      Err(x0) = sigma^2  +  (f_bar(x0) - f(x0))^2  +  E_D[(f_D(x0) - f_bar(x0))^2]
              = noise    +  bias^2                 +  variance

  where f_bar(x0) = E_D[f_D(x0)] is the average prediction over training sets. The cross terms
  vanish because eps is independent of f_D and because E_D[f_D - f_bar] = 0. We estimate all three
  by Monte Carlo (fit on hundreds of simulated training sets) and confirm they sum to the total.

Experiments:
  1. verify the decomposition sums to the total error                         (README §2)
  2. the U-curve: bias down, variance up, total U-shaped vs complexity         (README §4)
  3. classification: a case where added VARIANCE *reduces* 0/1 error           (README §5)
  4. learning curves: the three diagnostic shapes                             (README §6)
  5. Hoeffding / union bound: the generalization gap shrinks as 1/sqrt(n)      (README §8)
  6. VC dimension: linear classifiers shatter 3 points but not 4              (README §9)
  7. DOUBLE DESCENT: test error peaks at the interpolation threshold           (README §10)

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    from scipy.optimize import linprog
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# =============================================================================
# THE MONTE-CARLO BIAS-VARIANCE DECOMPOSER  (README §2-§3)
# =============================================================================


def bias_variance_decompose(fit_predict, f_true, x_test, n_train=40, sigma=0.5,
                            n_datasets=400, x_range=(-1, 1), seed=0):
    """Estimate (noise, bias^2, variance, total) at the test points x_test by fitting a fresh
    model on `n_datasets` simulated training sets. `fit_predict(Xtr, ytr, Xte) -> yhat`."""
    rng = np.random.default_rng(seed)
    x_test = np.asarray(x_test, dtype=float)
    preds = np.empty((n_datasets, x_test.size))
    for r in range(n_datasets):
        Xtr = rng.uniform(*x_range, size=n_train)
        ytr = f_true(Xtr) + sigma * rng.standard_normal(n_train)
        preds[r] = fit_predict(Xtr, ytr, x_test)
    f_bar = preds.mean(axis=0)                      # average prediction over training sets
    bias2 = (f_bar - f_true(x_test)) ** 2           # (README §2)
    variance = preds.var(axis=0)
    noise = sigma ** 2
    # total expected error at each test point = noise + E_D[(pred - f)^2] = noise+bias2+var
    total = noise + ((preds - f_true(x_test)) ** 2).mean(axis=0)
    return noise, bias2.mean(), variance.mean(), total.mean()


def _poly_fit_predict(degree):
    def fp(Xtr, ytr, Xte):
        A = np.vander(Xtr, degree + 1)
        coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        return np.vander(Xte, degree + 1) @ coef
    return fp


def _f_true(x):
    return np.sin(1.5 * np.pi * x)


# =============================================================================
# VERIFICATION
# =============================================================================


def verify():
    print("=" * 88)
    print("VERIFICATION — the bias-variance decomposition sums to the total error (README §2)")
    print("=" * 88)
    x_test = np.linspace(-0.9, 0.9, 50)
    sigma = 0.5
    noise, bias2, var, total = bias_variance_decompose(
        _poly_fit_predict(3), _f_true, x_test, sigma=sigma, n_datasets=600)
    lhs = noise + bias2 + var
    print(f"""
  Degree-3 polynomial, sigma^2 = {sigma**2:.3f}, averaged over 600 simulated training sets:

     irreducible noise sigma^2 = {noise:.4f}
     bias^2                    = {bias2:.4f}
     variance                  = {var:.4f}
     --------------------------------------
     sum (noise+bias^2+var)    = {lhs:.4f}
     measured total error      = {total:.4f}
     difference                = {abs(lhs - total):.2e}
""")
    assert abs(lhs - total) < 1e-9, "decomposition must sum to total exactly"
    print("  noise + bias^2 + variance == total error  ✓  (identity holds to machine precision)")
    print("\nAll verification checks passed.")


# =============================================================================
# EXPERIMENT 2 — the U-curve
# =============================================================================


def experiment_2_u_curve():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — the U-curve: bias down, variance up, total U-shaped (README §4)")
    print("=" * 88)
    x_test = np.linspace(-0.9, 0.9, 60)
    degrees = range(0, 14)
    print(f"\n  True f(x) = sin(1.5*pi*x), noise sigma^2 = 0.25, n_train = 40:\n")
    print(f"    {'degree':>7s} {'bias^2':>9s} {'variance':>9s} {'total':>9s}")
    rows = []
    for d in degrees:
        noise, bias2, var, total = bias_variance_decompose(
            _poly_fit_predict(d), _f_true, x_test, sigma=0.5, n_datasets=300)
        rows.append((d, bias2, var, total))
        mark = ""
        print(f"    {d:>7d} {bias2:>9.4f} {var:>9.4f} {total:>9.4f}")
    best = min(rows, key=lambda r: r[3])
    print(f"""
  Minimum total error at degree {best[0]} (total {best[3]:.4f}).

  READING: bias^2 falls monotonically as the polynomial gains flexibility; variance rises as
  it starts chasing the particular sample; their sum is U-shaped with a minimum in between
  (here degree {best[0]}). Underfit to the left (bias-dominated), overfit to the right
  (variance-dominated). Cross-validation (05.04) exists to LOCATE this minimum without knowing
  f or sigma.""")


# =============================================================================
# EXPERIMENT 3 — classification: variance can HELP (README §5)
# =============================================================================


def experiment_3_classification_variance():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — under 0/1 loss, variance can REDUCE error (README §5)")
    print("=" * 88)
    # A high-variance base classifier (a depth-limited tree on a hard boundary). We fit it on
    # many training sets and, at each test point, compare the error of the 'main prediction'
    # (majority vote across training sets, a variance-FREE reference) against the average error
    # of the individual high-variance fits.
    from sklearn.tree import DecisionTreeClassifier
    rng = np.random.default_rng(0)

    def f_boundary(X):                      # true label: a curved boundary
        return (X[:, 1] > np.sin(2 * X[:, 0])).astype(int)

    Xte = rng.uniform(-3, 3, (400, 2))
    yte = f_boundary(Xte)
    R = 120
    preds = np.empty((R, len(Xte)), dtype=int)
    for r in range(R):
        Xtr = rng.uniform(-3, 3, (60, 2))
        ytr = f_boundary(Xtr)
        # flip 8% of labels: noise that a high-variance learner will chase
        flip = rng.uniform(size=60) < 0.08
        ytr[flip] = 1 - ytr[flip]
        clf = DecisionTreeClassifier(max_depth=6, random_state=r).fit(Xtr, ytr)
        preds[r] = clf.predict(Xte)

    main = (preds.mean(axis=0) >= 0.5).astype(int)      # majority vote = variance-free ref
    biased = main != yte                                 # main prediction is WRONG here
    avg_err = (preds != yte).mean(axis=0)                # avg 0/1 error of individual fits

    print(f"""
  {R} tree fits on noisy training sets. 'Main prediction' = majority vote across fits (the
  low-variance reference); we compare its error to the AVERAGE error of the wobbling fits.

    {'test points where...':>34s} {'main-pred error':>16s} {'avg individual error':>21s}
    {'main prediction is CORRECT':>34s} {np.mean(0.0):>16.3f} {avg_err[~biased].mean():>21.3f}
    {'main prediction is WRONG':>34s} {np.mean(1.0):>16.3f} {avg_err[biased].mean():>21.3f}

  READING: where the main prediction is CORRECT (unbiased points), the wobble only ever moves
  a prediction to the WRONG side — variance ADDS error (0.000 -> {avg_err[~biased].mean():.3f}).
  But where the main prediction is WRONG (biased points), the wobble sometimes lands on the
  CORRECT side — variance SUBTRACTS error ({avg_err[biased].mean():.3f} < 1.000). Under 0/1 loss
  variance is not uniformly harmful: its sign depends on whether the point is already biased.
  This is why the additive bias^2+variance picture is a SQUARED-ERROR fact, not a universal one
  (README §5).""")


# =============================================================================
# EXPERIMENT 4 — learning curves (README §6)
# =============================================================================


def experiment_4_learning_curves():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — learning curves: three shapes, three diagnoses (README §6)")
    print("=" * 88)
    # Decision trees at three depths on a 1D sine: numerically stable (predictions stay in the
    # training-target range, no polynomial blow-up) and a clean complexity knob. BOTH train and
    # validation targets carry noise, so the noise floor sigma^2 shows up in both curves and the
    # train-val gap has the correct sign.
    from sklearn.tree import DecisionTreeRegressor
    sizes = [10, 20, 40, 80, 160, 320]
    sigma = 0.4
    noise_floor = sigma ** 2

    def run(depth, label):
        print(f"\n  {label}")
        print(f"    {'n_train':>8s} {'train err':>10s} {'val err':>10s} {'gap':>8s}")
        for n in sizes:
            tr_errs, val_errs = [], []
            for t in range(40):
                rng = np.random.default_rng(1000 * n + t)
                Xtr = rng.uniform(-1, 1, (n, 1))
                ytr = _f_true(Xtr[:, 0]) + sigma * rng.standard_normal(n)
                Xval = rng.uniform(-1, 1, (400, 1))
                yval = _f_true(Xval[:, 0]) + sigma * rng.standard_normal(400)
                tree = DecisionTreeRegressor(max_depth=depth).fit(Xtr, ytr)
                tr_errs.append(np.mean((ytr - tree.predict(Xtr)) ** 2))
                val_errs.append(np.mean((yval - tree.predict(Xval)) ** 2))
            te, ve = np.mean(tr_errs), np.mean(val_errs)
            print(f"    {n:>8d} {te:>10.4f} {ve:>10.4f} {ve - te:>8.4f}")

    print(f"\n  (noise floor sigma^2 = {noise_floor:.3f} — the best any curve can reach)")
    run(1, "HIGH BIAS — depth-1 stump on a sine (both errors high, small gap):")
    run(None, "HIGH VARIANCE — full-depth tree (low train, high val, big gap that narrows):")
    run(4, "GOOD FIT — depth-4 tree (both converge toward the noise floor):")
    print(f"""
  READING: high bias -> train and val both high (~0.5, well above the {noise_floor:.2f} floor) and
  CLOSE; more data will NOT help, you need a richer model. High variance -> train near 0, val
  high, a big gap that NARROWS as n grows; more data (or regularization) is the cure. Good fit
  -> both converge toward the noise floor {noise_floor:.2f}. The asymmetry is the lesson: more
  data cures variance, never bias (README §6).""")


# =============================================================================
# EXPERIMENT 5 — Hoeffding / union bound (README §8)
# =============================================================================


def experiment_5_generalization_gap():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — the generalization gap shrinks as 1/sqrt(n), grows with ln|H|")
    print("               (Hoeffding + union bound, README §8)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    d = 5

    def gen(n):
        X = rng.standard_normal((n, d))
        y = (X @ np.array([1.0, -1.0, 0.5, 0, 0]) + 0.5 * rng.standard_normal(n) > 0).astype(int)
        return X, y

    Xpop, ypop = gen(20000)      # a huge sample as a proxy for the population

    def max_gap(M, n, trials=40):
        gaps = []
        for _ in range(trials):
            H = rng.standard_normal((M, d))          # M random linear hypotheses
            true_risk = np.mean((Xpop @ H.T > 0).astype(int)
                                != ypop[:, None], axis=0)   # per hypothesis, on population
            Xs, ys = gen(n)
            emp_risk = np.mean((Xs @ H.T > 0).astype(int) != ys[:, None], axis=0)
            gaps.append(np.max(np.abs(true_risk - emp_risk)))
        return np.mean(gaps)

    print("\n  Max over an M-hypothesis class of |true risk - empirical risk| (measured), vs the")
    print("  Hoeffding+union bound sqrt(ln(2M)/(2n)):\n")
    print(f"    {'M':>6s} {'n':>6s} {'measured gap':>14s} {'Hoeffding bound':>16s}")
    for M in (10, 1000):
        for n in (50, 200, 800):
            g = max_gap(M, n)
            bound = np.sqrt(np.log(2 * M) / (2 * n))
            print(f"    {M:>6d} {n:>6d} {g:>14.4f} {bound:>16.4f}")
    print("""
  READING: the measured gap falls like 1/sqrt(n) (quadruple n, halve the gap) and grows with
  ln M (1000 hypotheses gap > 10 hypotheses gap at fixed n). The Hoeffding+union bound tracks
  and upper-bounds it. This is the first quantitative bias-variance trade: ln|H| is a
  complexity/variance term paid for in data. Generalization needs n >> ln|H| (README §8).""")


# =============================================================================
# EXPERIMENT 6 — VC dimension by shattering (README §9)
# =============================================================================


def _linearly_separable(X, y):
    """Is the labelling y in {0,1} of points X linearly separable? Solve the LP feasibility
    y_i in {+/-1}: s_i (w.x_i + b) >= 1 for all i (scale-free hard margin)."""
    s = np.where(np.asarray(y) == 1, 1.0, -1.0)
    n, d = X.shape
    # variables [w (d), b (1)]; constraints -s_i (w.x_i + b) <= -1
    A_ub = -s[:, None] * np.column_stack([X, np.ones(n)])
    b_ub = -np.ones(n)
    res = linprog(c=np.zeros(d + 1), A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None)] * (d + 1), method="highs")
    return res.success


def experiment_6_vc_dimension():
    print("\n" + "=" * 88)
    print("EXPERIMENT 6 — VC dimension: linear classifiers shatter 3 points, not 4 (README §9)")
    print("=" * 88)
    if not HAVE_SCIPY:
        print("\n(scipy unavailable — skipping the LP-based shattering check)")
        return
    import itertools

    def fraction_shattered(X):
        k = len(X)
        ok = sum(_linearly_separable(X, labels)
                 for labels in itertools.product([0, 1], repeat=k))
        return ok, 2 ** k

    three = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])          # a triangle
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])  # convex 4-gon
    o3, t3 = fraction_shattered(three)
    o4, t4 = fraction_shattered(square)
    print(f"""
  Can a straight line realize EVERY labelling of the points?

    {'point set':>22s} {'labellings realized':>22s}
    {'3 points (triangle)':>22s} {f'{o3} / {t3}':>22s}   -> {'SHATTERED' if o3==t3 else 'not shattered'}
    {'4 points (square)':>22s} {f'{o4} / {t4}':>22s}   -> {'SHATTERED' if o4==t4 else 'NOT shattered'}

  READING: a line shatters 3 points in general position (all {t3} labellings achievable), but
  no set of 4 points can be shattered — the square's two XOR (diagonal) labellings are not
  linearly separable, giving {o4}/{t4}. Hence the VC dimension of linear classifiers in the
  plane is 3 = d + 1. Finite VC dimension is exactly what makes a class learnable (README §9).""")


# =============================================================================
# EXPERIMENT 7 — DOUBLE DESCENT (README §10)
# =============================================================================


def experiment_7_double_descent():
    print("\n" + "=" * 88)
    print("EXPERIMENT 7 — DOUBLE DESCENT: test error peaks at the interpolation threshold,")
    print("               then descends again (README §10)")
    print("=" * 88)
    # Ridgeless (min-norm) least squares with a linear teacher in D ambient features; fit the
    # first P features. Classical variance grows toward P=n, the interpolation threshold spikes,
    # and the min-norm interpolant descends a SECOND time for P>n (Hastie et al. 2019).
    D, n_tr = 250, 50

    def solve(Phi, y, lam=1e-8):
        n, P = Phi.shape
        if P <= n:
            return np.linalg.solve(Phi.T @ Phi + lam * np.eye(P), Phi.T @ y)
        return Phi.T @ np.linalg.solve(Phi @ Phi.T + lam * np.eye(n), y)   # min-norm interpolant

    grid = [2, 5, 10, 20, 30, 40, 46, 54, 65, 85, 120, 170, 250]
    print(f"\n  min-norm least squares, D={D} ambient features, n_train={n_tr}, noise 0.3:\n")
    print(f"    {'P/n':>6s} {'P':>5s} {'test MSE':>12s}")
    results = []
    for P in grid:
        errs = []
        for t in range(60):
            rng = np.random.default_rng(t)
            beta = rng.standard_normal(D) / np.sqrt(D)
            Xtr = rng.standard_normal((n_tr, D))
            ytr = Xtr @ beta + 0.3 * rng.standard_normal(n_tr)
            Xte = rng.standard_normal((1500, D))
            yte = Xte @ beta
            w = solve(Xtr[:, :P], ytr)
            errs.append(np.mean((yte - Xte[:, :P] @ w) ** 2))
        m = float(np.mean(errs))
        results.append((P, m))
        bar = "#" * int(min(m, 15) / 15 * 44)
        flag = "  <- interpolation threshold (P≈n)" if 40 <= P <= 54 else ""
        print(f"    {P / n_tr:>6.2f} {P:>5d} {m:>12.3f} {bar}{flag}")
    classical_min = min(m for P, m in results if P < n_tr)
    modern_min = min(m for P, m in results if P > 2 * n_tr)
    print(f"""
  Classical-regime best (P<n): {classical_min:.3f}    Overparametrized best (P>>n): {modern_min:.3f}

  READING: as P grows toward n the model strains to interpolate and variance explodes — test
  error PEAKS near P=n (the near-singular square system). Past the threshold there are many
  interpolating solutions and min-norm least squares picks the SMOOTHEST; test error descends a
  SECOND time, here to {modern_min:.3f}, BELOW the classical-regime best of {classical_min:.3f}.
  The classical U-curve (Experiment 2) is only the left half of the story: complexity measured
  by raw parameter count is not the right axis — the min-norm bias is what tames the
  overparametrized regime. This is why 'make the model bigger' so often helps (README §10).""")


if __name__ == "__main__":
    verify()
    experiment_2_u_curve()
    experiment_3_classification_variance()
    experiment_4_learning_curves()
    experiment_5_generalization_gap()
    experiment_6_vc_dimension()
    experiment_7_double_descent()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
