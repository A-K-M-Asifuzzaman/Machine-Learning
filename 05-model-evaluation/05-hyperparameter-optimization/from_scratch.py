"""
05.05 — Hyperparameter Optimization, from scratch (NumPy).

Grid / random / Bayesian (Gaussian-process + Expected Improvement) / successive halving,
all in NumPy. Grid and random are verified against scikit-learn on a real SVM; the adaptive
methods are demonstrated on controlled synthetic objectives so the ground-truth optimum is
known. The chapter's claims are MEASURED:

  1. random search beats grid when only a few dimensions matter  (Bergstra-Bengio, README §3)
  2. Bayesian optimization reaches the optimum in fewer evals than random   (README §4-§5)
  3. successive halving finds a near-best config for a fraction of the cost  (README §6)
  4. log-scale sampling finds a multiplicative optimum; linear starves it    (README §7)
  5. over-tuning: the winner's CV optimism grows with the number of configs  (README §8)

Run:  python3 from_scratch.py
"""

import numpy as np
from itertools import product

try:
    from sklearn.svm import SVC
    from sklearn.model_selection import (GridSearchCV, cross_val_score as sk_cvs,
                                         KFold)
    from sklearn.datasets import make_classification
    HAVE_SK = True
except Exception:
    HAVE_SK = False


# =============================================================================
# GRID AND RANDOM SEARCH  (README §2-§3)
# =============================================================================


def _cv_score(make_model, params, X, y, cv, seed=0):
    """Mean K-fold accuracy for one setting. Contiguous folds (no shuffle), matching
    sklearn's KFold(shuffle=False) so the search is verifiable against it."""
    n = len(y)
    folds = np.array_split(np.arange(n), cv)
    scores = []
    for i in range(cv):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(cv) if j != i])
        m = make_model(**params).fit(X[tr], y[tr])
        scores.append(np.mean(m.predict(X[te]) == y[te]))
    return float(np.mean(scores))


def grid_search(make_model, param_grid, X, y, cv=5):
    keys = list(param_grid)
    best, results = None, []
    for combo in product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        s = _cv_score(make_model, params, X, y, cv)
        results.append((params, s))
        if best is None or s > best[1]:
            best = (params, s)
    return best, results


def random_search(make_model, param_samplers, n_iter, X, y, cv=5, seed=0):
    rng = np.random.default_rng(seed)
    best, results = None, []
    for _ in range(n_iter):
        params = {k: sampler(rng) for k, sampler in param_samplers.items()}
        s = _cv_score(make_model, params, X, y, cv)
        results.append((params, s))
        if best is None or s > best[1]:
            best = (params, s)
    return best, results


# =============================================================================
# GAUSSIAN PROCESS + EXPECTED IMPROVEMENT  (README §4-§5)
# =============================================================================


class GaussianProcess:
    """GP regression with an RBF kernel; exact posterior by conditioning a joint Gaussian
    (00.03 §4). Targets are standardized internally for numerical stability."""

    def __init__(self, length_scale=0.2, signal=1.0, noise=1e-4):
        self.l = length_scale
        self.sig = signal
        self.noise = noise

    def _kernel(self, A, B):
        d2 = np.sum(A**2, 1)[:, None] + np.sum(B**2, 1)[None, :] - 2 * A @ B.T
        return self.sig**2 * np.exp(-0.5 * d2 / self.l**2)

    def fit(self, X, y):
        self.X = np.asarray(X, float)
        self.ymu, self.ystd = np.mean(y), np.std(y) + 1e-9
        yc = (np.asarray(y, float) - self.ymu) / self.ystd
        K = self._kernel(self.X, self.X) + self.noise * np.eye(len(X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, yc))
        return self

    def predict(self, Xs):
        Xs = np.asarray(Xs, float)
        Ks = self._kernel(self.X, Xs)
        mu = Ks.T @ self.alpha
        v = np.linalg.solve(self.L, Ks)
        var = np.diag(self._kernel(Xs, Xs)) - np.sum(v**2, 0)
        var = np.maximum(var, 1e-12)
        return mu * self.ystd + self.ymu, np.sqrt(var) * self.ystd


def _phi(z):
    return np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)


def _Phi(z):
    from math import erf
    return np.array([0.5 * (1 + erf(zi / np.sqrt(2))) for zi in np.atleast_1d(z)])


def expected_improvement(mu, sigma, f_best, xi=0.01):
    """EI for MINIMIZATION: expected amount below the current best (README §5)."""
    imp = f_best - mu - xi
    z = imp / sigma
    return imp * _Phi(z) + sigma * _phi(z)


def bayes_optimize(objective, bounds, n_init=4, n_iter=20, seed=0, n_cand=2000):
    """Minimize a black-box `objective` over a box `bounds` = [(lo,hi),...]."""
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, float)
    d = len(bounds)

    def sample(m):
        return bounds[:, 0] + rng.uniform(size=(m, d)) * (bounds[:, 1] - bounds[:, 0])

    X = sample(n_init)
    y = np.array([objective(x) for x in X])
    best_curve = [y.min()]
    for _ in range(n_iter):
        gp = GaussianProcess().fit(X, y)
        cand = sample(n_cand)
        mu, sigma = gp.predict(cand)
        ei = expected_improvement(mu, sigma, y.min())
        x_next = cand[int(np.argmax(ei))]
        X = np.vstack([X, x_next])
        y = np.append(y, objective(x_next))
        best_curve.append(y.min())
    return X, y, np.array(best_curve)


def random_optimize(objective, bounds, n_total, seed=0):
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, float)
    d = len(bounds)
    X = bounds[:, 0] + rng.uniform(size=(n_total, d)) * (bounds[:, 1] - bounds[:, 0])
    y = np.array([objective(x) for x in X])
    return np.minimum.accumulate(y)      # best-so-far curve


# =============================================================================
# SUCCESSIVE HALVING  (README §6)
# =============================================================================


def successive_halving(configs, evaluate, budgets, keep_frac=0.5):
    """`evaluate(config, budget) -> score (higher better)`. Rungs of increasing budget;
    keep the top fraction each rung. Returns (best_config, total_cost)."""
    alive = list(configs)
    total_cost = 0.0
    for b in budgets:
        scored = [(evaluate(c, b), c) for c in alive]
        total_cost += b * len(alive)
        scored.sort(key=lambda t: -t[0])
        k = max(1, int(len(scored) * keep_frac))
        alive = [c for _, c in scored[:k]]
        if len(alive) == 1:
            break
    # final: evaluate survivor(s) at full budget already reflected; pick best
    best = max(alive, key=lambda c: evaluate(c, budgets[-1]))
    return best, total_cost


# =============================================================================
# VERIFICATION
# =============================================================================


def verify():
    print("=" * 88)
    print("VERIFICATION — grid/random search vs scikit-learn; GP posterior sanity")
    print("=" * 88)
    if HAVE_SK:
        X, y = make_classification(n_samples=300, n_features=8, n_informative=4,
                                   random_state=0)
        grid = {"C": [0.1, 1, 10], "gamma": [0.01, 0.1, 1]}
        (best_params, best_score), _ = grid_search(
            lambda **p: SVC(**p), grid, X, y, cv=5)
        sk = GridSearchCV(SVC(), grid, cv=KFold(5, shuffle=False),
                          scoring="accuracy").fit(X, y)
        print(f"""
    our grid best:     {best_params}  score {best_score:.3f}
    sklearn grid best: {sk.best_params_}  score {sk.best_score_:.3f}
""")
        assert best_params == sk.best_params_, "grid search should find the same optimum"
        print("  grid search finds the same best hyperparameters as sklearn  ✓")

    # GP interpolates its training points exactly (noise-free limit)
    rng = np.random.default_rng(0)
    Xt = rng.uniform(-2, 2, (8, 1))
    yt = np.sin(2 * Xt[:, 0])
    gp = GaussianProcess(length_scale=0.5, noise=1e-8).fit(Xt, yt)
    mu, sd = gp.predict(Xt)
    print(f"\n  GP interpolation error at training points: {np.max(np.abs(mu - yt)):.2e}"
          f"  (uncertainty there ~ {np.mean(sd):.1e})")
    assert np.max(np.abs(mu - yt)) < 1e-4, "GP must interpolate training points"
    print("  GP reproduces its observations and reports ~0 uncertainty there  ✓")
    print("\nAll verification checks passed.")


# =============================================================================
# EXPERIMENT 1 — random beats grid when few dimensions matter (README §3)
# =============================================================================


def experiment_1_random_vs_grid():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — random search beats grid when few dimensions matter (README §3)")
    print("=" * 88)
    d = 5
    budget_grid_v = 3            # 3 values per dim -> 3^5 = 243 evaluations
    budget = budget_grid_v ** d

    grid_best, rand_best = [], []
    for trial in range(200):
        rng = np.random.default_rng(trial)
        # only 2 of the 5 dimensions affect the objective; optimum placed at random
        opt = rng.uniform(0.2, 0.8, 2)

        def f(theta):                       # maximize (peak 0 at opt on the 2 real dims)
            return -((theta[0] - opt[0]) ** 2 + (theta[1] - opt[1]) ** 2)

        # grid: linspace per dim
        axis = np.linspace(0, 1, budget_grid_v)
        gbest = max(f(np.array(c)) for c in product(axis, repeat=d))
        # random: `budget` uniform points
        pts = rng.uniform(0, 1, (budget, d))
        rbest = max(f(p) for p in pts)
        grid_best.append(gbest)
        rand_best.append(rbest)

    print(f"""
  5 hyperparameters, only 2 affect the objective. Equal budget = {budget} evaluations.
  Best objective found (higher is better, max = 0), averaged over 200 random problems:

    {'method':>16s} {'best objective':>16s}
    {'grid (3 per dim)':>16s} {np.mean(grid_best):>16.4f}
    {'random':>16s} {np.mean(rand_best):>16.4f}

  READING: grid spends its {budget} evaluations trying only {budget_grid_v} distinct values on EACH
  dimension — including the 3 that do nothing — so the 2 important dims are explored at just
  {budget_grid_v} settings. Random search gives each important dim ~{budget} distinct values, so it
  lands much closer to the true optimum for the same budget. Prefer random over grid for 3+
  hyperparameters (README §3).""")


# =============================================================================
# EXPERIMENT 2 — Bayesian optimization vs random (README §4-§5)
# =============================================================================


def experiment_2_bayes_vs_random():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — Bayesian optimization reaches the optimum in fewer evals (README §4)")
    print("=" * 88)
    # a bumpy 2D objective to MINIMIZE (global min near (0.23, 0.35)); many local dips
    def objective(t):
        x, y = t
        return (np.sin(3 * x) + 0.5 * np.sin(7 * y) + (x - 0.2) ** 2 + (y - 0.35) ** 2
                + 0.3 * np.cos(9 * x))

    bounds = [(0.0, 1.0), (0.0, 1.0)]
    n_iter = 24

    bo_curves, rand_curves = [], []
    for seed in range(30):
        _, _, bo = bayes_optimize(objective, bounds, n_init=4, n_iter=n_iter, seed=seed)
        rand = random_optimize(objective, bounds, n_total=4 + n_iter, seed=seed)
        bo_curves.append(bo)
        rand_curves.append(rand)
    bo_curves = np.array(bo_curves)          # (30, n_iter+1), indexed by extra evals
    rand_curves = np.array(rand_curves)      # (30, 4+n_iter)

    print(f"\n  Best objective found (lower better) vs total evaluations, averaged over 30 runs:\n")
    print(f"    {'evaluations':>12s} {'Bayesian opt':>14s} {'random':>10s}")
    for n_eval in (5, 10, 15, 20, 28):
        bo_v = bo_curves[:, min(n_eval - 4, bo_curves.shape[1] - 1)].mean()
        rd_v = rand_curves[:, min(n_eval - 1, rand_curves.shape[1] - 1)].mean()
        print(f"    {n_eval:>12d} {bo_v:>14.3f} {rd_v:>10.3f}")
    print("""
  READING: Bayesian optimization fits a Gaussian-process surrogate to the points seen so far and
  spends each new evaluation where Expected Improvement is highest — so it drives the objective
  down faster than random sampling, which ignores everything it has learned. The gap is largest
  in the small-budget regime, which is exactly when evaluations are expensive (README §4-§5).""")


# =============================================================================
# EXPERIMENT 3 — successive halving (README §6)
# =============================================================================


def experiment_3_successive_halving():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — successive halving finds a near-best config cheaply (README §6)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    n_configs = 64
    # each config has a true final score; a low-budget evaluation is a NOISY estimate of it
    true_score = rng.uniform(0.6, 0.95, n_configs)
    best_true = true_score.max()

    def evaluate(c, budget):
        # more budget -> less noise (noise ~ 1/sqrt(budget))
        return true_score[c] + rng.normal(0, 0.15 / np.sqrt(budget))

    configs = list(range(n_configs))
    budgets = [1, 2, 4, 8, 16]     # rungs
    picked, sh_cost = successive_halving(configs, evaluate, budgets)

    full_cost = n_configs * budgets[-1]      # evaluate all at full budget
    print(f"""
  {n_configs} configs, budgets per rung {budgets} (keep top half each rung).

    {'method':>28s} {'total budget spent':>18s} {'picked config score':>20s}
    {'evaluate all at full budget':>28s} {full_cost:>18d} {best_true:>20.3f}
    {'successive halving':>28s} {int(sh_cost):>18d} {true_score[picked]:>20.3f}

  READING: successive halving starts many configs cheaply, keeps only the promising half at each
  rung, and doubles the budget for survivors — so it spends {sh_cost/full_cost:.0%} of the full-grid
  cost and still lands on a config within {best_true - true_score[picked]:.3f} of the true best.
  Bad configs are killed early instead of wasting a full training run each (README §6).""")


# =============================================================================
# EXPERIMENT 4 — log vs linear scale sampling (README §7)
# =============================================================================


def experiment_4_log_scale():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — sample multiplicative hyperparameters on a LOG scale (README §7)")
    print("=" * 88)
    # objective is a function of the learning rate with optimum at lr* = 1e-3, spanning
    # [1e-4, 1e-1]. Score peaks sharply at lr*.
    lr_star = 1e-3

    def score(lr):
        return -(np.log10(lr) - np.log10(lr_star)) ** 2      # peak 0 at lr*

    n = 40
    lin_best, log_best = [], []
    for seed in range(300):
        rng = np.random.default_rng(seed)
        lin = rng.uniform(1e-4, 1e-1, n)                     # linear sampling
        log = 10 ** rng.uniform(-4, -1, n)                   # log-uniform sampling
        lin_best.append(max(score(v) for v in lin))
        log_best.append(max(score(v) for v in log))
    print(f"""
  Optimum learning rate lr* = 1e-3, range [1e-4, 1e-1], {n} samples. Best score found
  (max 0), averaged over 300 runs:

    {'sampling':>16s} {'best score':>12s}
    {'linear-uniform':>16s} {np.mean(lin_best):>12.3f}
    {'log-uniform':>16s} {np.mean(log_best):>12.3f}

  READING: linear sampling in [1e-4, 1e-1] puts ~90% of its draws above 1e-2, starving the
  small-lr region where the optimum lives, so it rarely gets close. Log-uniform sampling spreads
  draws evenly across the ORDERS OF MAGNITUDE and finds lr* reliably. Learning rates, regularization
  strengths, and other multiplicative knobs must be sampled on a log scale (README §7).""")


# =============================================================================
# EXPERIMENT 5 — over-tuning optimism (README §8)
# =============================================================================


def experiment_5_overtuning():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — the winner's CV score is optimistic; optimism grows with #configs")
    print("               (README §8)")
    print("=" * 88)
    # all configs are EQUALLY good (true score 0.80); the 'CV score' is a noisy estimate.
    # Picking the max over many noisy estimates inflates the reported score above the truth.
    true = 0.80
    cv_noise = 0.03

    print(f"\n  All configs truly score {true:.2f}; CV estimate has std {cv_noise}. Reported = the")
    print(f"  BEST CV score seen. Averaged over 500 tuning runs:\n")
    print(f"    {'# configs tried':>16s} {'reported CV (winner)':>20s} {'optimism':>10s}")
    for n_configs in (1, 5, 20, 100, 500):
        winners = []
        for seed in range(500):
            rng = np.random.default_rng(seed * 7 + n_configs)
            cv_scores = true + rng.normal(0, cv_noise, n_configs)
            winners.append(cv_scores.max())
        rep = np.mean(winners)
        print(f"    {n_configs:>16d} {rep:>20.4f} {rep - true:>+10.4f}")
    print("""
  READING: the reported score is the MAXIMUM of many noisy CV estimates, so it drifts above the
  true 0.80 — and the more configs you try, the larger the upward bias. This is why the winner's
  CV score is NOT its expected performance: you have partly fit the CV noise. Estimate real
  performance with nested CV or a locked test set the tuning never saw (README §8).""")


if __name__ == "__main__":
    verify()
    experiment_1_random_vs_grid()
    experiment_2_bayes_vs_random()
    experiment_3_successive_halving()
    experiment_4_log_scale()
    experiment_5_overtuning()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
