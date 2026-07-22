"""
00.04 — Statistics and Inference from Scratch
=============================================

Estimators, intervals, and tests implemented from their definitions — and then, more
importantly, *measured*. Almost every claim in this chapter is checkable by simulation:
you can literally count how often a "95% confidence interval" contains the truth.

Implemented here
----------------
    Estimators and MLE           README §3-§5
        sample_variance_mle / sample_variance_unbiased
        mle_bernoulli / mle_gaussian
        fisher_information_bernoulli / _gaussian_mean
        measure_estimator          empirical bias / variance / MSE

    Intervals                    README §8
        z_interval, t_interval    the standard constructions
        wilson_interval           correct for proportions near 0 or 1
        bootstrap_ci              works for ANY statistic

    Tests                        README §9, §13
        two_sample_t_test, welch_t_test
        permutation_test          assumption-free
        mcnemar_test              the right test for two classifiers

    Multiple comparisons         README §11
        bonferroni, holm, benjamini_hochberg

Run it
------
    python from_scratch.py

Verifies everything against scipy, then runs five experiments:
  1. Do 95% confidence intervals actually cover 95% of the time? (and when do they not)
  2. p-values are Uniform(0,1) under the null — the fact that makes alpha mean anything
  3. The multiple-comparisons explosion, and what each correction recovers
  4. The (n-1)/n bias of the variance MLE, measured
  5. Bootstrap vs analytic intervals for a statistic that has no formula

Reference: README.md sections 3-15.
"""

from __future__ import annotations

import math

import numpy as np

# =============================================================================
# ESTIMATORS AND MLE  (README §3-§5)
# =============================================================================


def sample_variance_mle(x: np.ndarray) -> float:
    """MLE of the variance: divide by n.  README §4.2

    Biased LOW by exactly (n-1)/n, because xbar was fitted to this same data and
    therefore sits closer to it than the true mu does. Experiment 4 measures it.
    """
    x = np.asarray(x, dtype=float)
    return float(np.mean((x - x.mean()) ** 2))


def sample_variance_unbiased(x: np.ndarray) -> float:
    """Bessel's correction: divide by n-1.  README §5

    One degree of freedom was consumed estimating the mean, so n-1 remain.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    return float(np.sum((x - x.mean()) ** 2) / (n - 1))


def mle_bernoulli(x: np.ndarray) -> float:
    """MLE for Bernoulli: the sample proportion.  README §4.1

    Note it returns exactly 0 after a run of failures — asserting the event is
    IMPOSSIBLE on the strength of a handful of observations. That is MLE overfitting a
    small sample, and it is why naive Bayes needs Laplace smoothing (a Beta prior).
    """
    return float(np.mean(np.asarray(x, dtype=float)))


def mle_gaussian(x: np.ndarray) -> tuple[float, float]:
    """MLE for a Gaussian: (xbar, biased variance).  README §4.2"""
    x = np.asarray(x, dtype=float)
    return float(x.mean()), sample_variance_mle(x)


def fisher_information_bernoulli(pi: float) -> float:
    """I(pi) = 1 / (pi (1-pi))  — per observation.  README §6

    Derivation: log p(x|pi) = x log pi + (1-x) log(1-pi), so
        d2/dpi2 = -x/pi^2 - (1-x)/(1-pi)^2
        I(pi) = -E[...] = 1/pi + 1/(1-pi) = 1/(pi(1-pi))

    Information is MINIMIZED at pi = 0.5 and blows up near 0 or 1 — which says a
    near-certain coin is easy to pin down, and a fair one is hardest. By Cramer-Rao,
    Var(pi_hat) >= pi(1-pi)/n, which the sample proportion attains exactly: the MLE is
    efficient here, not merely asymptotically.
    """
    return 1.0 / (pi * (1 - pi))


def fisher_information_gaussian_mean(sigma: float) -> float:
    """I(mu) = 1/sigma^2 per observation, so Var(mu_hat) >= sigma^2/n.

    The sample mean attains this exactly — it is efficient at every n.
    """
    return 1.0 / sigma ** 2


def measure_estimator(estimator, sampler, true_value: float,
                      n_trials: int = 20_000, rng=None) -> dict:
    """Empirically measure an estimator's bias, variance, and MSE.  README §3

    This is the sampling distribution (README §2.1) made concrete: draw many samples,
    apply the estimator to each, and look at the spread of answers. Also verifies the
    decomposition MSE = bias^2 + variance holds numerically.
    """
    rng = rng or np.random.default_rng(0)
    estimates = np.array([estimator(sampler(rng)) for _ in range(n_trials)])

    bias = float(estimates.mean() - true_value)
    mse = float(np.mean((estimates - true_value) ** 2))

    # ddof=0 here, deliberately. The identity MSE = bias^2 + Var is a statement about
    # *population* moments, and over these n_trials draws the empirical population
    # moments are the ddof=0 ones. Mixing ddof=1 variance with a ddof=0 MSE breaks the
    # identity by a term of order Var/n_trials — small, but it would mean the check below
    # is passing on a tolerance rather than on the mathematics.
    variance = float(estimates.var(ddof=0))

    return {
        "mean": float(estimates.mean()),
        "bias": bias,
        "variance": variance,
        "variance_unbiased": float(estimates.var(ddof=1)),   # the better estimate of Var
        "mse": mse,
        "bias2_plus_var": bias ** 2 + variance,
        "estimates": estimates,
    }


# =============================================================================
# CONFIDENCE INTERVALS  (README §8)
# =============================================================================


def _normal_ppf(q: float) -> float:
    """Inverse standard normal CDF, by bisection on erf. Avoids a scipy dependency."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def z_interval(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """xbar +/- z * s/sqrt(n).  Assumes the CLT has kicked in.  README §8.2"""
    x = np.asarray(x, dtype=float)
    n = x.size
    se = math.sqrt(sample_variance_unbiased(x) / n)
    z = _normal_ppf(1 - alpha / 2)
    return float(x.mean() - z * se), float(x.mean() + z * se)


def t_interval(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """xbar +/- t_{n-1} * s/sqrt(n).  README §8.2

    The t distribution's heavier tails pay for the fact that we estimated sigma rather
    than knowing it. The difference matters below n ~ 30 and vanishes above it.
    """
    from scipy import stats                       # only for the t quantile
    x = np.asarray(x, dtype=float)
    n = x.size
    se = math.sqrt(sample_variance_unbiased(x) / n)
    t = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return float(x.mean() - t * se), float(x.mean() + t * se)


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a proportion.  README §8.3

    The naive interval phat +/- z sqrt(phat(1-phat)/n) is badly behaved near 0 and 1: it
    can extend below 0 or above 1, and its true coverage is far from nominal. The Wilson
    interval inverts the score test instead, which keeps it inside [0,1] and gives much
    better coverage at small n and extreme p.

    Use this, not the naive interval, for any rare-event rate.
    """
    z = _normal_ppf(1 - alpha / 2)
    p_hat = successes / n
    denominator = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denominator
    half_width = z * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def naive_proportion_interval(successes: int, n: int,
                              alpha: float = 0.05) -> tuple[float, float]:
    """The textbook Wald interval — included so Experiment 1 can show it failing."""
    z = _normal_ppf(1 - alpha / 2)
    p_hat = successes / n
    half = z * math.sqrt(p_hat * (1 - p_hat) / n)
    return p_hat - half, p_hat + half


def bootstrap_ci(data: np.ndarray, statistic, n_boot: int = 10_000,
                 alpha: float = 0.05, rng=None) -> dict:
    """Percentile bootstrap confidence interval.  README §12

    Resample n points WITH REPLACEMENT, recompute the statistic, repeat. The spread of
    the replicates estimates the sampling distribution — with no analytic derivation of
    any kind.

    This is the practical tool for error bars on F1, AUC, a median, a ratio of metrics,
    or anything else with no closed-form standard error.

    Fails for extreme order statistics (resampling cannot exceed the observed range),
    tiny n, and dependent data (README §12.2).
    """
    rng = rng or np.random.default_rng(0)
    data = np.asarray(data)
    n = len(data)

    idx = rng.integers(0, n, size=(n_boot, n))
    replicates = np.array([statistic(data[i]) for i in idx])

    return {
        "estimate": float(statistic(data)),
        "se": float(replicates.std(ddof=1)),
        "ci": (float(np.percentile(replicates, 100 * alpha / 2)),
               float(np.percentile(replicates, 100 * (1 - alpha / 2)))),
        "replicates": replicates,
    }


# =============================================================================
# HYPOTHESIS TESTS  (README §9, §13)
# =============================================================================


def two_sample_t_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Student's t-test, assuming equal variances. Returns (t, p)."""
    from scipy import stats
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = a.size, b.size
    pooled_var = ((na - 1) * sample_variance_unbiased(a)
                  + (nb - 1) * sample_variance_unbiased(b)) / (na + nb - 2)
    t = (a.mean() - b.mean()) / math.sqrt(pooled_var * (1 / na + 1 / nb))
    p = 2 * (1 - stats.t.cdf(abs(t), df=na + nb - 2))
    return float(t), float(p)


def welch_t_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch's t-test — does NOT assume equal variances.

    Should be the default over Student's: it costs almost nothing in power when variances
    are equal, and is far more reliable when they are not.
    """
    from scipy import stats
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    va, vb = sample_variance_unbiased(a) / a.size, sample_variance_unbiased(b) / b.size
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
    p = 2 * (1 - stats.t.cdf(abs(t), df=df))
    return float(t), float(p)


def permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = 20_000,
                     statistic=None, rng=None) -> dict:
    """Assumption-free two-group test.  README §13

    Logic: if the null is true, the group labels carry no information, so any reassignment
    of labels is exactly as probable as the one observed. Build the null distribution by
    doing precisely that, many times.

    No normality assumption, no equal-variance assumption, no large-sample assumption.
    Works for any statistic you can compute.

    The +1 in the p-value is not a fudge: including the observed arrangement among the
    permutations is what keeps the test exact (Phipson & Smyth 2010). Without it a p-value
    of exactly 0 is possible, which is never a correct statement.
    """
    rng = rng or np.random.default_rng(0)
    statistic = statistic or (lambda x, y: x.mean() - y.mean())

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    observed = statistic(a, b)
    pooled = np.concatenate([a, b])
    na = a.size

    count = 0
    null_distribution = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(pooled)
        value = statistic(shuffled[:na], shuffled[na:])
        null_distribution[i] = value
        if abs(value) >= abs(observed):
            count += 1

    return {
        "observed": float(observed),
        "p_value": (count + 1) / (n_perm + 1),
        "null_distribution": null_distribution,
    }


def mcnemar_test(model_a_correct: np.ndarray, model_b_correct: np.ndarray,
                 exact: bool = True) -> dict:
    """McNemar's test — the right test for two classifiers on the SAME test set.

    Build the 2x2 disagreement table:
        n01 = A wrong, B right
        n10 = A right, B wrong

    Only these two counts matter. Examples both models get right (or both wrong) carry no
    information about which is better, and including them — as an unpaired t-test would —
    just adds noise. Under the null the two models are equally good, so each disagreement
    is a fair coin flip, and n01 ~ Binomial(n01 + n10, 0.5).

    `exact=True` uses the binomial test directly, which is correct for any counts. The
    chi-squared approximation needs n01 + n10 >= 25.
    """
    from scipy import stats
    a = np.asarray(model_a_correct, dtype=bool)
    b = np.asarray(model_b_correct, dtype=bool)

    n01 = int(np.sum(~a & b))                    # A wrong, B right
    n10 = int(np.sum(a & ~b))                    # A right, B wrong
    n_discordant = n01 + n10

    if n_discordant == 0:
        return {"n01": 0, "n10": 0, "statistic": 0.0, "p_value": 1.0}

    if exact:
        p = float(stats.binomtest(n01, n_discordant, 0.5).pvalue)
        statistic = float(n01)
    else:
        # With continuity correction.
        statistic = (abs(n01 - n10) - 1) ** 2 / n_discordant
        p = float(1 - stats.chi2.cdf(statistic, df=1))

    return {"n01": n01, "n10": n10, "n_discordant": n_discordant,
            "statistic": statistic, "p_value": p}


# =============================================================================
# MULTIPLE COMPARISONS  (README §11)
# =============================================================================


def bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Reject where p <= alpha/m. Controls FWER. Simple and very conservative."""
    p = np.asarray(p_values, dtype=float)
    return p <= alpha / p.size


def holm(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Holm-Bonferroni step-down. Controls FWER, uniformly more powerful than Bonferroni.

    Sort p ascending; compare p_(i) against alpha/(m - i); stop at the first failure and
    reject nothing after it. Strictly better than Bonferroni at no cost in assumptions —
    there is no reason to prefer plain Bonferroni.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    reject = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        if p[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break                                 # step-down: stop at first failure
    return reject


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg. Controls the FALSE DISCOVERY RATE, not the FWER.

    Sort p ascending, find the largest i with p_(i) <= (i/m) alpha, reject everything up
    to it.

    FWER asks "what is the chance of ANY false positive?"; FDR asks "what fraction of my
    rejections are false?". When screening thousands of hypotheses — genomics, feature
    selection, hyperparameter sweeps — FWER control rejects essentially nothing, and FDR
    is the sane target.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    sorted_p = p[order]

    thresholds = (np.arange(1, m + 1) / m) * alpha
    passing = np.where(sorted_p <= thresholds)[0]

    reject = np.zeros(m, dtype=bool)
    if passing.size:
        cutoff = passing.max()                    # largest index that passes
        reject[order[:cutoff + 1]] = True
    return reject


# =============================================================================
# VERIFICATION
# =============================================================================


def _report(name: str, error: float, threshold: float) -> bool:
    status = "PASS" if error < threshold else "FAIL"
    print(f"  [{status}]  {name:<52s}  err = {error:.3e}")
    return error < threshold


def verify() -> bool:
    ok = True
    rng = np.random.default_rng(0)

    print("=" * 84)
    print("VERIFICATION")
    print("=" * 84)

    try:
        from scipy import stats
    except ImportError:
        print("\n  scipy not installed — skipping comparisons.")
        return True

    x = rng.normal(5.0, 2.0, 200)
    y = rng.normal(5.6, 3.0, 180)

    # --- estimators -------------------------------------------------------
    print("\nEstimators (README §3-§5)")
    ok &= _report("sample_variance_unbiased vs np.var(ddof=1)",
                  abs(sample_variance_unbiased(x) - np.var(x, ddof=1)), 1e-12)
    ok &= _report("sample_variance_mle vs np.var(ddof=0)",
                  abs(sample_variance_mle(x) - np.var(x, ddof=0)), 1e-12)
    ok &= _report("MLE variance = (n-1)/n x unbiased",
                  abs(sample_variance_mle(x)
                      - (x.size - 1) / x.size * sample_variance_unbiased(x)), 1e-12)

    # MSE decomposition must hold exactly.
    res = measure_estimator(sample_variance_mle,
                            lambda r: r.normal(0, 2.0, 10), 4.0, 20_000, rng)
    ok &= _report("MSE = bias^2 + variance", abs(res["mse"] - res["bias2_plus_var"]), 1e-9)

    # --- intervals --------------------------------------------------------
    print("\nConfidence intervals (README §8)")
    lo, hi = t_interval(x)
    ref = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
    ok &= _report("t_interval vs scipy.stats.t.interval",
                  max(abs(lo - ref[0]), abs(hi - ref[1])), 1e-9)

    ok &= _report("_normal_ppf(0.975) = 1.959963985",
                  abs(_normal_ppf(0.975) - 1.959963984540054), 1e-9)

    # Wilson must stay inside [0,1] where the naive interval does not.
    w_lo, w_hi = wilson_interval(2, 100)
    n_lo, n_hi = naive_proportion_interval(2, 100)
    print(f"  [INFO]  {'2 successes in 100: naive interval':<52s}  "
          f"[{n_lo:.4f}, {n_hi:.4f}]  <- negative!")
    print(f"  [INFO]  {'2 successes in 100: Wilson interval':<52s}  "
          f"[{w_lo:.4f}, {w_hi:.4f}]")
    ok &= (w_lo >= 0.0 and w_hi <= 1.0 and n_lo < 0.0)

    # --- tests ------------------------------------------------------------
    print("\nHypothesis tests (README §9, §13)")
    t_mine, p_mine = two_sample_t_test(x, y)
    t_ref, p_ref = stats.ttest_ind(x, y, equal_var=True)
    ok &= _report("two_sample_t_test vs scipy",
                  max(abs(t_mine - t_ref), abs(p_mine - p_ref)), 1e-9)

    t_mine, p_mine = welch_t_test(x, y)
    t_ref, p_ref = stats.ttest_ind(x, y, equal_var=False)
    ok &= _report("welch_t_test vs scipy", max(abs(t_mine - t_ref), abs(p_mine - p_ref)), 1e-9)

    # The experiments use a batched Welch implementation for speed; it must agree
    # exactly with the scalar one it replaces, or the simulations are measuring
    # something other than what verify() checked.
    batch_a = rng.normal(0, 1, size=(50, 30))
    batch_b = rng.normal(0.3, 1.2, size=(50, 30))
    batch_p = _welch_p_batch(batch_a, batch_b)
    scalar_p = np.array([welch_t_test(batch_a[i], batch_b[i])[1] for i in range(50)])
    ok &= _report("_welch_p_batch agrees with welch_t_test",
                  float(np.abs(batch_p - scalar_p).max()), 1e-12)

    perm = permutation_test(x, y, n_perm=20_000, rng=rng)
    ok &= _report("permutation p ~ Welch p (both valid here)",
                  abs(perm["p_value"] - p_mine), 0.02)

    a_correct = rng.random(500) < 0.85
    b_correct = rng.random(500) < 0.88
    mc = mcnemar_test(a_correct, b_correct)
    ref_p = stats.binomtest(mc["n01"], mc["n_discordant"], 0.5).pvalue
    ok &= _report("mcnemar_test vs scipy binomtest", abs(mc["p_value"] - ref_p), 1e-12)

    # --- multiple comparisons ---------------------------------------------
    print("\nMultiple-comparison corrections (README §11)")
    p_vals = np.array([0.001, 0.008, 0.012, 0.03, 0.04, 0.2, 0.5, 0.7, 0.9, 0.95])

    try:
        from statsmodels.stats.multitest import multipletests
        for name, mine, method in [
            ("bonferroni", bonferroni(p_vals), "bonferroni"),
            ("holm", holm(p_vals), "holm"),
            ("benjamini_hochberg", benjamini_hochberg(p_vals), "fdr_bh"),
        ]:
            ref = multipletests(p_vals, alpha=0.05, method=method)[0]
            match = bool(np.array_equal(mine, ref))
            print(f"  [{'PASS' if match else 'FAIL'}]  "
                  f"{name + ' vs statsmodels':<52s}  {mine.sum()} rejected")
            ok &= match
    except ImportError:
        # Fall back to checking the defining property of each procedure.
        print("  [SKIP]  statsmodels not installed — checking definitions directly")
        ok &= _report("bonferroni rejects exactly p <= alpha/m",
                      float(np.sum(bonferroni(p_vals) != (p_vals <= 0.05 / p_vals.size))), 0.5)
        # Every procedure must be at least as powerful as Bonferroni.
        ok &= bool(np.all(holm(p_vals) >= bonferroni(p_vals)))
        ok &= bool(np.all(benjamini_hochberg(p_vals) >= holm(p_vals)))
        print(f"  [PASS]  {'power ordering: BH >= Holm >= Bonferroni':<52s}  "
              f"{benjamini_hochberg(p_vals).sum()} >= {holm(p_vals).sum()} "
              f">= {bonferroni(p_vals).sum()}")

    # --- bootstrap --------------------------------------------------------
    print("\nBootstrap (README §12)")
    boot = bootstrap_ci(x, np.mean, n_boot=10_000, rng=rng)
    analytic_se = math.sqrt(sample_variance_unbiased(x) / x.size)
    ok &= _report("bootstrap SE of the mean ~ analytic SE",
                  abs(boot["se"] - analytic_se), 0.02 * analytic_se)

    t_lo, t_hi = t_interval(x)
    ok &= _report("bootstrap CI ~ t CI for a mean",
                  max(abs(boot["ci"][0] - t_lo), abs(boot["ci"][1] - t_hi)), 0.05)

    return ok


# =============================================================================
# EXPERIMENTS
#
# The simulations below run many thousands of trials, so the inner loops are
# vectorized over trials rather than written as Python loops. The formulas are
# identical to the single-sample functions above — `verify()` checks them against
# scipy, and the batch versions are checked against the scalar ones.
# =============================================================================


def _welch_p_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Welch p-values for a whole batch at once. a, b have shape (..., n).

    Same formula as `welch_t_test`, evaluated over the leading axes simultaneously.
    A single vectorized scipy call replaces hundreds of thousands of scalar ones.
    """
    from scipy import stats
    na, nb = a.shape[-1], b.shape[-1]
    va = a.var(axis=-1, ddof=1) / na
    vb = b.var(axis=-1, ddof=1) / nb
    t = (a.mean(axis=-1) - b.mean(axis=-1)) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    return 2 * stats.t.sf(np.abs(t), df)


def experiment_ci_coverage() -> None:
    """README §8: does a 95% CI actually contain the truth 95% of the time?"""
    from scipy import stats

    print("\n" + "=" * 84)
    print("EXPERIMENT 1 — do 95% confidence intervals actually cover 95%?  (README §8)")
    print("=" * 84)
    print("""
"95% confidence" is a claim about a procedure (README §8.1), and a claim about a procedure
can be tested by running it 20,000 times and counting. Anything far from 95% means the
interval's assumptions are violated — and the interval is lying to you.
""")
    rng = np.random.default_rng(1)
    n_trials = 20_000
    z = _normal_ppf(0.975)

    print("  Gaussian data — the case the theory was built for")
    print(f"    {'n':>5s}  {'z-interval':>12s}  {'t-interval':>12s}")
    print("    " + "-" * 32)
    for n in (5, 10, 30, 100):
        samples = rng.normal(10.0, 3.0, size=(n_trials, n))
        se = np.sqrt(samples.var(axis=1, ddof=1) / n)
        centre = samples.mean(axis=1)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        z_cover = np.mean(np.abs(centre - 10.0) <= z * se)
        t_cover = np.mean(np.abs(centre - 10.0) <= t_crit * se)
        print(f"    {n:5d}  {z_cover:11.1%}  {t_cover:11.1%}")

    print("""
    At n = 5 the z-interval covers only ~88%, not 95% — it ignores the extra uncertainty
    from having estimated sigma. The t-interval, whose heavier tails pay for exactly that,
    is at 95% throughout. This is what the t distribution is FOR.
""")

    print("  Skewed data (Exponential, true mean 1.0) — the CLT has not arrived yet")
    print(f"    {'n':>5s}  {'t-interval':>12s}  {'bootstrap':>12s}")
    print("    " + "-" * 32)
    trials, n_boot = 4000, 400
    for n in (5, 10, 30, 100):
        samples = rng.exponential(1.0, size=(trials, n))
        se = np.sqrt(samples.var(axis=1, ddof=1) / n)
        centre = samples.mean(axis=1)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        t_cover = np.mean(np.abs(centre - 1.0) <= t_crit * se)

        # Bootstrap all trials at once, in chunks to bound peak memory.
        hits = 0
        for start in range(0, trials, 250):
            block = samples[start:start + 250]                       # (B, n)
            idx = rng.integers(0, n, size=(block.shape[0], n_boot, n))
            means = np.take_along_axis(
                block[:, None, :], idx, axis=2).mean(axis=2)         # (B, n_boot)
            lo = np.percentile(means, 2.5, axis=1)
            hi = np.percentile(means, 97.5, axis=1)
            hits += int(np.sum((lo <= 1.0) & (1.0 <= hi)))
        print(f"    {n:5d}  {t_cover:11.1%}  {hits / trials:11.1%}")

    print("""
    Both undercover badly at small n on skewed data — the t-interval is symmetric, and the
    true sampling distribution of the mean is not. Neither method rescues you from a sample
    too small for its own skewness (00.03 §12.1).
""")

    print("  Proportions near zero — where the textbook interval breaks outright")
    print(f"    {'true p':>8s}  {'n':>6s}  {'naive (Wald)':>14s}  {'Wilson':>10s}")
    print("    " + "-" * 44)
    for p_true in (0.01, 0.05, 0.2):
        for n in (30, 100):
            s = rng.binomial(n, p_true, size=n_trials)
            p_hat = s / n

            half = z * np.sqrt(p_hat * (1 - p_hat) / n)
            naive_cover = np.mean((p_hat - half <= p_true) & (p_true <= p_hat + half))

            denom = 1 + z ** 2 / n
            centre = (p_hat + z ** 2 / (2 * n)) / denom
            half_w = z * np.sqrt(p_hat * (1 - p_hat) / n
                                 + z ** 2 / (4 * n ** 2)) / denom
            wilson_cover = np.mean((centre - half_w <= p_true) & (p_true <= centre + half_w))

            print(f"    {p_true:8.2f}  {n:6d}  {naive_cover:13.1%}  {wilson_cover:9.1%}")

    print("""
    The naive interval's coverage collapses for small p — at p = 0.01, n = 30 it covers
    only a fraction of the time, because the most likely outcome is 0 successes, giving
    phat = 0, SE = 0, and an interval of zero width that cannot contain anything.

    Use the Wilson interval for any rate near 0 or 1: click-through, conversion, fraud,
    rare-disease incidence. The textbook formula is not merely imprecise there, it is
    wrong.""")


def experiment_p_value_distribution() -> None:
    """README §9-10: p-values are Uniform(0,1) under the null."""
    print("\n" + "=" * 84)
    print("EXPERIMENT 2 — p-values are uniform under the null  (README §9)")
    print("=" * 84)
    print("""
This is the fact that makes alpha mean anything at all. If p ~ Uniform(0,1) whenever the
null is true, then P(p < 0.05) = 0.05 exactly — the Type I error rate is what it claims.
Testing two identical groups, 40,000 times:
""")
    rng = np.random.default_rng(2)
    n_trials = 40_000

    a = rng.normal(0, 1, size=(n_trials, 30))
    b = rng.normal(0, 1, size=(n_trials, 30))      # SAME distribution: null is true
    p_values = _welch_p_batch(a, b)

    print(f"  {'bin':>12s}  {'observed':>10s}  {'expected':>10s}")
    print("  " + "-" * 36)
    for lo in np.arange(0, 1.0, 0.1):
        frac = float(np.mean((p_values >= lo) & (p_values < lo + 0.1)))
        print(f"  [{lo:.1f}, {lo + 0.1:.1f})  {frac:10.4f}  {0.1:10.4f}")

    print(f"\n  P(p < 0.05) = {np.mean(p_values < 0.05):.4f}   (should be 0.05)")
    print(f"  P(p < 0.01) = {np.mean(p_values < 0.01):.4f}   (should be 0.01)")

    print("""
  Flat, as promised. Two consequences worth holding onto:

  1. A single p < 0.05 is unremarkable on its own — under the null it happens 1 time in 20
     BY CONSTRUCTION. That is the design, not a flaw.

  2. Because the distribution is uniform and not concentrated near 1, running many tests
     is guaranteed to produce small p-values eventually. Which is Experiment 3.""")


def experiment_multiple_comparisons() -> None:
    """README §11: the multiple comparisons explosion, and what corrections recover."""
    print("\n" + "=" * 84)
    print("EXPERIMENT 3 — the multiple comparisons explosion  (README §11)")
    print("=" * 84)
    print("""
Suppose you test m hypotheses that are ALL null — no effect anywhere. How often does at
least one come out "significant" at alpha = 0.05?
""")
    rng = np.random.default_rng(3)
    n_trials = 4000

    print(f"  {'m tests':>9s}  {'P(>=1 false positive)':>23s}  {'predicted 1-0.95^m':>20s}")
    print("  " + "-" * 56)
    for m in (1, 5, 10, 20, 50, 100):
        a = rng.normal(0, 1, size=(n_trials, m, 30))
        b = rng.normal(0, 1, size=(n_trials, m, 30))
        p_values = _welch_p_batch(a, b)                       # (n_trials, m)
        any_significant = np.mean(np.any(p_values < 0.05, axis=1))
        print(f"  {m:9d}  {any_significant:22.1%}  {1 - 0.95 ** m:19.1%}")

    print("""
  Observed matches 1 - 0.95^m exactly. At 20 tests you find "significance" 64% of the time
  with nothing real present at all.

  Now the corrections. Here 10 of 100 hypotheses are genuinely non-null, and we count what
  each procedure recovers:
""")
    n_runs = 2000
    m_total, m_true = 100, 10
    effect = 0.8

    results = {name: {"tp": 0, "fp": 0, "any_fp": 0}
               for name in ("uncorrected", "bonferroni", "holm", "benjamini_hochberg")}

    is_real = np.zeros(m_total, dtype=bool)
    is_real[:m_true] = True
    shift = np.where(is_real, effect, 0.0)[None, :, None]     # broadcast over trials

    all_p = np.empty((n_runs, m_total))
    for start in range(0, n_runs, 250):                       # chunked to bound memory
        block = min(250, n_runs - start)
        a = rng.normal(0, 1, size=(block, m_total, 30))
        b = rng.normal(0, 1, size=(block, m_total, 30)) + shift
        all_p[start:start + block] = _welch_p_batch(a, b)

    for p_values in all_p:
        for name, reject in [("uncorrected", p_values < 0.05),
                             ("bonferroni", bonferroni(p_values)),
                             ("holm", holm(p_values)),
                             ("benjamini_hochberg", benjamini_hochberg(p_values))]:
            results[name]["tp"] += int(np.sum(reject & is_real))
            results[name]["fp"] += int(np.sum(reject & ~is_real))
            results[name]["any_fp"] += int(np.any(reject & ~is_real))

    print(f"  {'method':<20s}  {'true found':>11s}  {'false found':>12s}  "
          f"{'FDR':>7s}  {'P(any FP)':>10s}")
    print("  " + "-" * 68)
    for name, r in results.items():
        tp = r["tp"] / n_runs
        fp = r["fp"] / n_runs
        fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
        print(f"  {name:<20s}  {tp:10.2f}/{m_true}  {fp:12.2f}  {fdr:7.1%}  "
              f"{r['any_fp'] / n_runs:9.1%}")

    print("""
  Read the last two columns together.

  UNCORRECTED finds the most real effects but ~4.5 false ones per run, and produces at
  least one false positive in almost every run. Its "discoveries" are roughly a third junk.

  BONFERRONI and HOLM hold P(any false positive) near 5% — that is family-wise error
  control, and it is strict. Holm dominates Bonferroni: same guarantee, strictly more
  power, so there is no reason to use plain Bonferroni.

  BENJAMINI-HOCHBERG controls the FDR near 5% instead — it tolerates the occasional false
  positive in exchange for finding substantially more of the real effects. When you are
  screening many hypotheses and will follow up on the hits anyway, this is the right
  trade.

  The ML translation: training 100 model variants and reporting the best test score is 100
  uncorrected comparisons. The winner's score is inflated by selection, exactly as the
  maximum of 100 noisy draws exceeds their common mean. Select on validation; report on a
  test set you touch once.""")


def experiment_mle_bias() -> None:
    """README §5: the variance MLE is biased low by exactly (n-1)/n."""
    print("\n" + "=" * 84)
    print("EXPERIMENT 4 — the MLE of variance is biased  (README §5)")
    print("=" * 84)
    print("""
Theory says E[sigma^2_MLE] = (n-1)/n sigma^2 — biased LOW, because xbar was fitted to the
same data whose spread we are measuring. Drawing 40,000 samples from N(0, 4) at each n:
""")
    rng = np.random.default_rng(4)
    true_var = 4.0
    n_trials = 40_000

    print(f"  {'n':>5s}  {'E[MLE]':>10s}  {'predicted':>11s}  {'E[unbiased]':>13s}  "
          f"{'MLE MSE':>10s}  {'unbiased MSE':>13s}")
    print("  " + "-" * 68)

    for n in (2, 3, 5, 10, 30, 100):
        mle = np.empty(n_trials)
        unb = np.empty(n_trials)
        for i in range(n_trials):
            s = rng.normal(0, 2.0, n)
            mle[i] = sample_variance_mle(s)
            unb[i] = sample_variance_unbiased(s)
        predicted = (n - 1) / n * true_var
        print(f"  {n:5d}  {mle.mean():10.4f}  {predicted:11.4f}  {unb.mean():13.4f}  "
              f"{np.mean((mle - true_var) ** 2):10.4f}  "
              f"{np.mean((unb - true_var) ** 2):13.4f}")

    print("""
  Columns 2 and 3 agree throughout: the (n-1)/n bias is exact, not approximate. At n = 2
  the MLE underestimates the variance by half.

  But look at the last two columns — and this is the part that surprises people. The
  BIASED estimator has LOWER MSE at every n. Bessel's correction removes the bias by
  scaling up, which also scales up the variance, and the net effect on MSE is negative.

  So "unbiased" is not the same as "better" (README §3). Which one you want depends on
  whether you care about being centred or being close, and that is exactly the trade
  regularization makes on purpose: ridge accepts bias to buy a larger reduction in
  variance.""")


def experiment_bootstrap_anything() -> None:
    """README §12: the bootstrap works where no formula exists."""
    print("\n" + "=" * 84)
    print("EXPERIMENT 5 — bootstrapping a statistic with no formula  (README §12)")
    print("=" * 84)
    print("""
For a mean, the standard error s/sqrt(n) is known. For the median, a correlation, an F1
score, or a 95th percentile, deriving one ranges from painful to impossible. The bootstrap
does all of them the same way. Checking it against truth by simulating the real sampling
distribution — which we can only do here because we know the population:
""")
    rng = np.random.default_rng(5)
    n = 200
    n_sim = 4000

    def f1_score(pairs):
        y_true, y_pred = pairs[:, 0], pairs[:, 1]
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    print(f"  {'statistic':<26s}  {'true SE':>10s}  {'bootstrap SE':>14s}  {'ratio':>7s}")
    print("  " + "-" * 62)

    # --- median ---
    truth = np.array([np.median(rng.exponential(1.0, n)) for _ in range(n_sim)]).std(ddof=1)
    sample = rng.exponential(1.0, n)
    boot = bootstrap_ci(sample, np.median, n_boot=4000, rng=rng)
    print(f"  {'median (Exponential)':<26s}  {truth:10.5f}  {boot['se']:14.5f}  "
          f"{boot['se'] / truth:7.3f}")

    # --- 90th percentile ---
    truth = np.array([np.percentile(rng.exponential(1.0, n), 90)
                      for _ in range(n_sim)]).std(ddof=1)
    boot = bootstrap_ci(sample, lambda a: np.percentile(a, 90), n_boot=4000, rng=rng)
    print(f"  {'90th percentile':<26s}  {truth:10.5f}  {boot['se']:14.5f}  "
          f"{boot['se'] / truth:7.3f}")

    # --- F1 score ---
    def make_pairs(r):
        y_true = (r.random(n) < 0.3).astype(int)
        y_pred = np.where(r.random(n) < 0.85, y_true, 1 - y_true)
        return np.column_stack([y_true, y_pred])

    truth = np.array([f1_score(make_pairs(rng)) for _ in range(n_sim)]).std(ddof=1)
    pairs = make_pairs(rng)
    boot = bootstrap_ci(pairs, f1_score, n_boot=4000, rng=rng)
    print(f"  {'F1 score':<26s}  {truth:10.5f}  {boot['se']:14.5f}  "
          f"{boot['se'] / truth:7.3f}")
    print(f"\n  F1 = {boot['estimate']:.4f}, 95% CI "
          f"[{boot['ci'][0]:.4f}, {boot['ci'][1]:.4f}]")

    # --- where it fails ---
    truth = np.array([np.max(rng.exponential(1.0, n)) for _ in range(n_sim)]).std(ddof=1)
    boot = bootstrap_ci(sample, np.max, n_boot=4000, rng=rng)
    print(f"\n  {'maximum  <- FAILS':<26s}  {truth:10.5f}  {boot['se']:14.5f}  "
          f"{boot['se'] / truth:7.3f}")

    print("""
  The ratio sits near 1.0 for the median, the 90th percentile, and F1 — the bootstrap
  recovers the true standard error for statistics that have no usable formula. That last
  line is a real F1 confidence interval, obtained without deriving anything.

  The maximum is the documented failure case (README §12.2). Resampling with replacement
  can never produce a value larger than the largest one observed, so the bootstrap
  systematically understates the variability of extreme order statistics. Know the
  failure mode before you rely on the tool.""")


# =============================================================================

if __name__ == "__main__":
    print(__doc__)

    all_passed = verify()

    experiment_ci_coverage()
    experiment_p_value_distribution()
    experiment_multiple_comparisons()
    experiment_mle_bias()
    experiment_bootstrap_anything()

    print("\n" + "=" * 84)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 84)
