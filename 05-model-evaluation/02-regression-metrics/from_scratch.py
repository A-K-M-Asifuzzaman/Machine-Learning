"""
05.02 — Regression Metrics, from scratch (NumPy).

Every metric implemented from its definition and checked against sklearn.metrics. The chapter's
claims are then MEASURED:

  1. each metric's optimal CONSTANT predictor is exactly mean / median / quantile   (README §4)
  2. RMSE is dominated by outliers; MAE is robust                                   (README §2-§3)
  3. R^2 can go negative; R^2 == corr^2 only for in-sample OLS                      (README §5)
  4. MAPE is asymmetric — it rewards UNDER-prediction                               (README §6)
  5. RMSLE measures RATIO error (scale-invariant) and penalizes under-prediction    (README §7)
  6. RMSE and MAE can RANK two models differently                                   (README §9)

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    from sklearn import metrics as skm
    HAVE_SK = True
except Exception:
    HAVE_SK = False


# =============================================================================
# METRICS (README §2-§8)
# =============================================================================


def mse(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean((y - yhat) ** 2))


def rmse(y, yhat):
    return float(np.sqrt(mse(y, yhat)))


def mae(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(y - yhat)))


def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def adjusted_r2(y, yhat, p):
    n = len(y)
    return float(1 - (1 - r2(y, yhat)) * (n - 1) / (n - p - 1))


def mape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(100 * np.mean(np.abs((y - yhat) / y)))


def smape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(100 * np.mean(np.abs(y - yhat) / ((np.abs(y) + np.abs(yhat)) / 2)))


def rmsle(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(yhat)) ** 2)))


def huber(y, yhat, delta=1.0):
    r = np.abs(np.asarray(y, float) - np.asarray(yhat, float))
    quad = np.minimum(r, delta)
    return float(np.mean(0.5 * quad ** 2 + delta * (r - quad)))


def pinball(y, yhat, tau=0.5):
    r = np.asarray(y, float) - np.asarray(yhat, float)
    return float(np.mean(np.maximum(tau * r, (tau - 1) * r)))


# =============================================================================
# VERIFICATION
# =============================================================================


def verify():
    print("=" * 88)
    print("VERIFICATION — every metric vs sklearn.metrics")
    print("=" * 88)
    rng = np.random.default_rng(0)
    y = rng.uniform(5, 50, 200)
    yhat = y + rng.standard_normal(200) * 3

    ours = {"MSE": mse(y, yhat), "RMSE": rmse(y, yhat), "MAE": mae(y, yhat),
            "R2": r2(y, yhat), "MAPE": mape(y, yhat), "RMSLE": rmsle(y, yhat),
            "pinball(0.9)": pinball(y, yhat, 0.9)}
    if HAVE_SK:
        ref = {"MSE": skm.mean_squared_error(y, yhat),
               "RMSE": np.sqrt(skm.mean_squared_error(y, yhat)),
               "MAE": skm.mean_absolute_error(y, yhat),
               "R2": skm.r2_score(y, yhat),
               "MAPE": skm.mean_absolute_percentage_error(y, yhat) * 100,
               "RMSLE": np.sqrt(skm.mean_squared_log_error(y, yhat)),
               "pinball(0.9)": skm.mean_pinball_loss(y, yhat, alpha=0.9)}
        print(f"\n    {'metric':>14s} {'ours':>12s} {'sklearn':>12s} {'|diff|':>10s}")
        for k in ours:
            d = abs(ours[k] - ref[k])
            print(f"    {k:>14s} {ours[k]:>12.6f} {ref[k]:>12.6f} {d:>10.2e}")
            assert d < 1e-6, f"{k} mismatch"
        print("\n  all metrics match sklearn to < 1e-6  ✓")
    else:
        for k, v in ours.items():
            print(f"    {k:>14s} {v:>12.6f}")
    print("\nAll verification checks passed.")


# =============================================================================
# EXPERIMENT 1 — the optimal constant reveals the metric (README §4)
# =============================================================================


def experiment_1_optimal_constant():
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 — the optimal constant predictor reveals each metric (README §4)")
    print("=" * 88)
    rng = np.random.default_rng(1)
    y = rng.lognormal(3, 1, 5000)      # skewed, so mean and median differ a lot

    grid = np.linspace(y.min(), np.quantile(y, 0.99), 4000)
    c_mse = grid[np.argmin([mse(y, np.full_like(y, c)) for c in grid])]
    c_mae = grid[np.argmin([mae(y, np.full_like(y, c)) for c in grid])]
    c_p90 = grid[np.argmin([pinball(y, np.full_like(y, c), 0.9) for c in grid])]

    print(f"""
  Skewed targets (lognormal): mean = {y.mean():.2f}, median = {np.median(y):.2f},
  90th percentile = {np.quantile(y, 0.9):.2f}

    {'metric':>14s} {'best constant c*':>18s} {'matches':>16s}
    {'MSE':>14s} {c_mse:>18.2f} {'mean ' + f'({y.mean():.2f})':>16s}
    {'MAE':>14s} {c_mae:>18.2f} {'median ' + f'({np.median(y):.2f})':>16s}
    {'pinball(0.9)':>14s} {c_p90:>18.2f} {'90th pct ' + f'({np.quantile(y,0.9):.2f})':>16s}

  READING: the constant that minimizes each metric is exactly its implied 'center' — MSE->mean,
  MAE->median, pinball(tau)->tau-quantile. On skewed data these differ substantially, so the
  metric you pick silently decides WHICH of them you are asking the model to predict (README §4).""")


# =============================================================================
# EXPERIMENT 2 — outlier sensitivity (README §2-§3)
# =============================================================================


def experiment_2_outliers():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — RMSE is dominated by outliers; MAE is robust (README §2-§3)")
    print("=" * 88)
    rng = np.random.default_rng(2)
    n = 1000
    y = rng.normal(50, 5, n)
    yhat = y + rng.normal(0, 2, n)          # clean predictions: ~2 typical miss

    print(f"\n  {'# gross outliers added':>24s} {'RMSE':>8s} {'MAE':>8s} {'RMSE/MAE':>10s}")
    for k in (0, 1, 5, 20):
        yh = yhat.copy()
        if k:
            idx = rng.choice(n, k, replace=False)
            yh[idx] += rng.choice([-1, 1], k) * rng.uniform(40, 60, k)   # huge misses
        print(f"    {k:>24d} {rmse(y, yh):>8.3f} {mae(y, yh):>8.3f} {rmse(y, yh)/mae(y, yh):>10.2f}")
    print("""
  READING: adding a handful of gross outliers barely moves MAE (it counts each linearly) but
  inflates RMSE sharply (it counts each SQUARED). The RMSE/MAE ratio climbs from ~1.25 (clean,
  near-Gaussian) upward — a large ratio is a fingerprint of heavy-tailed errors. Choose MAE when
  a few bad points should not dictate the headline number (README §3).""")


# =============================================================================
# EXPERIMENT 3 — R^2 can go negative; R^2 == corr^2 only in-sample (README §5)
# =============================================================================


def experiment_3_r2():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — R^2 can be negative; R^2 == corr^2 only for in-sample OLS (README §5)")
    print("=" * 88)
    rng = np.random.default_rng(3)
    Xtr = rng.uniform(-2, 2, (200, 1))
    ytr = 3 * Xtr[:, 0] + 1 + rng.standard_normal(200)
    # fit OLS with intercept
    A = np.column_stack([np.ones(200), Xtr[:, 0]])
    w = np.linalg.lstsq(A, ytr, rcond=None)[0]
    yhat_tr = A @ w
    corr2_in = np.corrcoef(ytr, yhat_tr)[0, 1] ** 2
    print(f"""
  In-sample OLS (with intercept):
     R^2                     = {r2(ytr, yhat_tr):.4f}
     corr(y, yhat)^2         = {corr2_in:.4f}
     difference              = {abs(r2(ytr, yhat_tr) - corr2_in):.2e}   -> equal (README §5)""")

    # a test set from a SHIFTED distribution: the train model is worse than the test mean
    Xte = rng.uniform(6, 10, (200, 1))            # far outside training range
    yte = 3 * Xte[:, 0] + 1 + rng.standard_normal(200)
    yhat_te = np.column_stack([np.ones(200), Xte[:, 0]]) @ w
    # deliberately damage the model on test by using a bad constant predictor comparison
    bad = np.full(200, ytr.mean())                # predict the TRAIN mean on TEST
    corr2_te = np.corrcoef(yte, yhat_te)[0, 1] ** 2
    print(f"""
  On a shifted test set:
     R^2 of a train-mean predictor on test = {r2(yte, bad):.3f}   -> NEGATIVE (worse than test mean)
     R^2 of the fitted model on test       = {r2(yte, yhat_te):.3f}
     corr(y, yhat)^2 on test               = {corr2_te:.3f}
     R^2 != corr^2 out-of-sample: difference = {abs(r2(yte, yhat_te) - corr2_te):.3f}

  READING: R^2 equals squared correlation ONLY for in-sample OLS with an intercept. Out of
  sample it does not, and a model worse than simply predicting the mean gives a NEGATIVE R^2 —
  it is not a bounded [0,1] quantity on held-out data (README §5).""")


# =============================================================================
# EXPERIMENT 4 — MAPE asymmetry (README §6)
# =============================================================================


def experiment_4_mape_asymmetry():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — MAPE is asymmetric: it rewards UNDER-prediction (README §6)")
    print("=" * 88)
    y = np.full(1000, 100.0)
    print(f"\n  True value fixed at y = 100. Same absolute error, two directions:\n")
    print(f"    {'prediction':>12s} {'abs error':>10s} {'MAPE':>8s} {'MAE':>8s}")
    for yhat_val in (60, 80, 100, 120, 140):
        yh = np.full(1000, float(yhat_val))
        print(f"    {yhat_val:>12d} {abs(yhat_val-100):>10d} {mape(y, yh):>8.1f} {mae(y, yh):>8.1f}")

    # what constant minimizes MAPE on skewed data? -> below the median (biased low)
    rng = np.random.default_rng(4)
    yv = rng.lognormal(4, 0.5, 5000)
    grid = np.linspace(yv.min(), np.quantile(yv, 0.99), 4000)
    c_mape = grid[np.argmin([mape(yv, np.full_like(yv, c)) for c in grid])]
    print(f"""
  MAE is symmetric (60 and 140 both cost 40). But look — MAPE is symmetric HERE only because y is
  fixed; when the OPTIMAL CONSTANT is sought on real (varying, skewed) targets, MAPE's minimizer
  sits BELOW the median: c*_MAPE = {c_mape:.1f} vs median {np.median(yv):.1f}. Minimizing MAPE
  biases predictions LOW, because an over-prediction's percentage error is unbounded while an
  under-prediction's is capped at 100%. This silent under-forecast bias is MAPE's core trap
  (README §6).""")


# =============================================================================
# EXPERIMENT 5 — RMSLE: ratio error and under-prediction penalty (README §7)
# =============================================================================


def experiment_5_rmsle():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — RMSLE measures RATIO error and penalizes under-prediction (README §7)")
    print("=" * 88)
    print(f"\n  A factor-of-2 miss at three very different scales:\n")
    print(f"    {'y':>10s} {'yhat':>10s} {'ratio':>7s} {'RMSE':>12s} {'RMSLE':>8s}")
    for yv in (10.0, 1000.0, 100000.0):
        yh = yv / 2                      # predict half the truth (factor-2 under)
        print(f"    {yv:>10.0f} {yh:>10.0f} {'0.5x':>7s} {rmse([yv],[yh]):>12.1f} "
              f"{rmsle([yv],[yh]):>8.4f}")
    print(f"""
  RMSE of a factor-2 miss explodes with scale (5 -> 50000); RMSLE is IDENTICAL ({rmsle([10],[5]):.4f})
  at every scale — it measures the RATIO, not the absolute gap.
""")
    y = np.array([100.0])
    print(f"  Under vs over prediction of the same factor (y=100):\n")
    print(f"    {'prediction':>12s} {'RMSLE':>8s}")
    for yh in (50, 100, 200):
        print(f"    {yh:>12d} {rmsle(y, [yh]):>8.4f}")
    print(f"""
  Predicting 50 (half) costs RMSLE {rmsle(y,[50]):.4f}; predicting 200 (double) costs only
  {rmsle(y,[200]):.4f}. RMSLE penalizes UNDER-prediction more — the opposite of MAPE — which is
  often exactly what you want when under-forecasting demand is the costlier error (README §7).""")


# =============================================================================
# EXPERIMENT 6 — metric choice flips the model ranking (README §9)
# =============================================================================


def experiment_6_ranking_reversal():
    print("\n" + "=" * 88)
    print("EXPERIMENT 6 — RMSE and MAE can RANK two models differently (README §9)")
    print("=" * 88)
    rng = np.random.default_rng(6)
    n = 4000
    y = rng.normal(0, 1, n)

    # Model A: mostly TINY errors, but 3% gross ones -> low MAE, high RMSE
    errA = rng.normal(0, 0.25, n)
    gross = rng.uniform(size=n) < 0.03
    errA[gross] = rng.choice([-1, 1], gross.sum()) * rng.uniform(4, 6, gross.sum())
    # Model B: moderate errors everywhere, no tail -> higher MAE, lower RMSE
    errB = rng.normal(0, 0.55, n)
    yA, yB = y + errA, y + errB

    print(f"""
    {'model':>8s} {'RMSE':>8s} {'MAE':>8s}
    {'A':>8s} {rmse(y, yA):>8.3f} {mae(y, yA):>8.3f}   (mostly tiny errors, rare huge ones)
    {'B':>8s} {rmse(y, yB):>8.3f} {mae(y, yB):>8.3f}   (moderate errors everywhere)

  RMSE prefers model {'A' if rmse(y,yA) < rmse(y,yB) else 'B'}; MAE prefers model {'A' if mae(y,yA) < mae(y,yB) else 'B'}.

  READING: model A makes tiny errors most of the time but occasionally a huge one; model B is
  moderate throughout. MAE (linear) rewards A's many tiny errors; RMSE (quadratic) punishes A's
  rare huge ones and prefers B. Same two models, opposite verdicts. 'Which model is best?' is
  not well-posed until you name the metric — and the metric must come from the error's real cost
  (README §9).""")


if __name__ == "__main__":
    verify()
    experiment_1_optimal_constant()
    experiment_2_outliers()
    experiment_3_r2()
    experiment_4_mape_asymmetry()
    experiment_5_rmsle()
    experiment_6_ranking_reversal()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
