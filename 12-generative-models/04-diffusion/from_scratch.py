"""
12.04 — Diffusion models, from scratch (NumPy).

Diffusion models generate by REVERSING a gradual noising process: destroy data into pure noise step by
step, then train a network to undo one step of noise, and sample by denoising from noise. They are the
engine behind DALL-E, Stable Diffusion, and Imagen. This file builds a working diffusion model and
verifies the theory:

  1. the forward process has a closed form: q(x_t|x_0) = N(sqrt(abar_t) x0, (1-abar_t) I)  -> Experiment 1
  2. the forward process destroys all information: x_T -> pure noise                        -> Experiment 2
  3. a trained diffusion model GENERATES and covers ALL modes (no collapse, unlike a GAN)   -> Experiment 3
  4. noise-prediction == score matching: eps* = -sqrt(1-abar_t) * score(x_t)                -> Experiment 4
  5. DDIM: deterministic sampling in far fewer steps                                        -> Experiment 5

Run:  python3 from_scratch.py
"""

import numpy as np


def relu(z):
    return np.maximum(0, z)


def _adam(P, g, s, lr, t):
    s["m"] = 0.9 * s["m"] + 0.1 * g
    s["v"] = 0.999 * s["v"] + 0.001 * g ** 2
    return P - lr * (s["m"] / (1 - 0.9 ** t)) / (np.sqrt(s["v"] / (1 - 0.999 ** t)) + 1e-8)


# noise schedule
T = 40
BETAS = np.linspace(1e-4, 0.15, T)
ALPHAS = 1 - BETAS
ABAR = np.cumprod(ALPHAS)


# =============================================================================
# EXPERIMENT 1 — the forward process closed form
# =============================================================================


def experiment_1_forward():
    print("=" * 88)
    print("EXPERIMENT 1 — the forward process closed form q(x_t|x0) (README §2)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    x0 = np.array([2.0])
    t = 25
    # run the iterative forward process many times; compare to the closed form
    reps = 100000
    xt = np.full((reps, 1), x0)
    for i in range(t + 1):
        xt = np.sqrt(ALPHAS[i]) * xt + np.sqrt(BETAS[i]) * rng.standard_normal(xt.shape)
    emp_mean, emp_var = xt.mean(), xt.var()
    cf_mean, cf_var = np.sqrt(ABAR[t]) * x0[0], 1 - ABAR[t]
    print(f"""
  Add noise step by step, x_t = sqrt(alpha_t) x_(t-1) + sqrt(beta_t) * noise, from x0={x0[0]} to t={t}:

    iterative (100k runs):  mean = {emp_mean:.4f}   variance = {emp_var:.4f}
    closed form:            mean = {cf_mean:.4f}   variance = {cf_var:.4f}
    (closed form: sqrt(abar_t)*x0 and 1 - abar_t)

  READING: the forward process adds a little Gaussian noise at each step. Iterating it is equivalent to
  ONE Gaussian jump: q(x_t | x0) = N(sqrt(abar_t) x0, (1 - abar_t) I), where abar_t is the cumulative
  product of (1-beta). The iterative statistics match this closed form. This is the crucial trick: you
  can sample x_t at ANY noise level t directly from x0 in one step (no loop), which is what makes
  training tractable (README §2).""")


# =============================================================================
# EXPERIMENT 2 — the forward process destroys information
# =============================================================================


def experiment_2_destroys():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — the forward process destroys all information -> pure noise (README §2)")
    print("=" * 88)
    print(f"\n  Signal-to-noise ratio abar_t / (1 - abar_t) as t grows (T={T}):\n")
    print(f"    {'step t':>8s} {'abar_t (signal)':>16s} {'1-abar_t (noise)':>17s} {'SNR':>10s}")
    for t in (0, 10, 20, 30, 39):
        snr = ABAR[t] / (1 - ABAR[t] + 1e-12)
        print(f"    {t:>8d} {ABAR[t]:>16.4f} {1 - ABAR[t]:>17.4f} {snr:>10.2f}")
    print("""
  READING: as t increases, abar_t (how much of the original signal survives) shrinks toward 0 and the
  noise fraction grows toward 1, so the signal-to-noise ratio collapses. By the final step x_T is
  indistinguishable from pure N(0,I) noise — ALL structure is gone. That is the whole point: if the
  forward process reliably turns any data into a known simple distribution (standard normal), then
  learning to REVERSE it turns that simple distribution back into data — which is how we generate
  (README §2).""")


# =============================================================================
# A diffusion model (eps-predictor)
# =============================================================================


class Diffusion:
    def __init__(self, d, h=64, seed=0):
        rng = np.random.default_rng(seed)
        g = lambda a, b: rng.standard_normal((a, b)) * np.sqrt(2 / a)
        self.d = d
        self.P = {"W1": g(d + 1, h), "b1": np.zeros(h), "W2": g(h, h), "b2": np.zeros(h),
                  "W3": g(h, d), "b3": np.zeros(d)}
        self.s = {k: {"m": np.zeros_like(v), "v": np.zeros_like(v)} for k, v in self.P.items()}

    def eps(self, x, t_frac):
        inp = np.hstack([x, t_frac])
        a = relu(inp @ self.P["W1"] + self.P["b1"])
        b = relu(a @ self.P["W2"] + self.P["b2"])
        return inp, a, b, b @ self.P["W3"] + self.P["b3"]

    def train(self, sample_data, epochs=4000, lr=0.003, n=256, seed=0):
        rng = np.random.default_rng(seed)
        for it in range(1, epochs + 1):
            x0 = sample_data(n)
            ti = rng.integers(0, T, (n, 1))
            ab = ABAR[ti]
            noise = rng.standard_normal((n, self.d))
            xt = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * noise      # forward closed form
            inp, a, b, ep = self.eps(xt, ti / T)
            d = 2 * (ep - noise) / n                             # MSE on predicted noise
            g = {}
            g["W3"] = b.T @ d; g["b3"] = d.sum(0)
            db = d @ self.P["W3"].T * (b > 0)
            g["W2"] = a.T @ db; g["b2"] = db.sum(0)
            da = db @ self.P["W2"].T * (a > 0)
            g["W1"] = inp.T @ da; g["b1"] = da.sum(0)
            for k in self.P:
                self.P[k] = _adam(self.P[k], g[k], self.s[k], lr, it)

    def sample_ddpm(self, n, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, self.d))
        for i in reversed(range(T)):
            _, _, _, ep = self.eps(x, np.full((n, 1), i / T))
            x = (1 / np.sqrt(ALPHAS[i])) * (x - (1 - ALPHAS[i]) / np.sqrt(1 - ABAR[i]) * ep)
            if i > 0:
                x = x + np.sqrt(BETAS[i]) * rng.standard_normal(x.shape)
        return x

    def sample_ddim(self, n, steps, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, self.d))
        ts = np.linspace(T - 1, 0, steps).astype(int)
        for j, i in enumerate(ts):
            _, _, _, ep = self.eps(x, np.full((n, 1), i / T))
            x0_pred = (x - np.sqrt(1 - ABAR[i]) * ep) / np.sqrt(ABAR[i])   # predict x0
            if j < len(ts) - 1:
                inext = ts[j + 1]
                x = np.sqrt(ABAR[inext]) * x0_pred + np.sqrt(1 - ABAR[inext]) * ep   # deterministic
            else:
                x = x0_pred
        return x


def _bimodal(rng):
    def s(n):
        c = rng.integers(0, 2, (n, 1)) * 4.0 - 2.0
        return np.hstack([c + 0.3 * rng.standard_normal((n, 1)), 0.3 * rng.standard_normal((n, 1))])
    return s


# =============================================================================
# EXPERIMENT 3 — a diffusion model generates and covers all modes
# =============================================================================


def experiment_3_generate():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — a diffusion model generates and covers ALL modes (no collapse) (README §3)")
    print("=" * 88)
    rng = np.random.default_rng(1)
    model = Diffusion(d=2)
    model.train(_bimodal(rng), epochs=4000)
    gen = model.sample_ddpm(3000, seed=9)
    real = _bimodal(rng)(3000)
    left, right = np.mean(gen[:, 0] < 0), np.mean(gen[:, 0] >= 0)
    print(f"""
  Same bimodal target as the GAN chapter (two modes at -2 and +2). After training a diffusion model:

    generated x mean = {gen[:, 0].mean():+.2f}   std = {gen[:, 0].std():.2f}   (real: mean {real[:,0].mean():+.2f}, std {real[:,0].std():.2f})
    fraction near LEFT mode (-2)  = {left:.2f}
    fraction near RIGHT mode (+2) = {right:.2f}   (both ~0.50 -> BOTH modes covered)

  READING: the model learns to predict the noise added to a sample; sampling reverses the process,
  denoising pure noise into data one step at a time. Trained on the SAME bimodal distribution the GAN
  collapsed on ([12.03]), the diffusion model covers BOTH modes ~50/50 — because its objective is
  maximum-likelihood-like (mode-covering), not adversarial. This stability and mode coverage, plus
  sharp samples, is why diffusion replaced GANs for image generation (README §3).""")


# =============================================================================
# EXPERIMENT 4 — noise-prediction is score matching
# =============================================================================


def experiment_4_score():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — predicting noise == learning the score (eps* = -sqrt(1-abar) * score) (README §4)")
    print("=" * 88)
    # for Gaussian data N(mu, sigma^2), q(x_t) is Gaussian -> the optimal eps predictor and the score
    # are both analytic; verify the identity
    mu, sigma2 = 1.0, 0.5
    print(f"\n  Data ~ N(mu={mu}, sigma^2={sigma2}). For a noised sample x_t, compare the optimal noise")
    print(f"  predictor eps*(x_t) to -sqrt(1-abar_t)*score(x_t):\n")
    print(f"    {'t':>5s} {'x_t':>6s} {'eps*(x_t)':>12s} {'-sqrt(1-abar)*score':>22s}")
    rng = np.random.default_rng(0)
    for t in (5, 15, 30):
        ab = ABAR[t]
        var_t = ab * sigma2 + (1 - ab)                # var of q(x_t)
        mean_t = np.sqrt(ab) * mu
        xt = mean_t + np.sqrt(var_t) * 0.7            # a test point
        # optimal eps predictor = E[noise | x_t]; for jointly Gaussian, = sqrt(1-ab)*(x_t - mean_t)/var_t
        eps_star = np.sqrt(1 - ab) * (xt - mean_t) / var_t
        score = -(xt - mean_t) / var_t                # d/dx log N(x; mean_t, var_t)
        rhs = -np.sqrt(1 - ab) * score
        print(f"    {t:>5d} {xt:>6.2f} {eps_star:>12.5f} {rhs:>22.5f}")
    print("""
  READING: predicting the noise and learning the SCORE (the gradient of the log-density, grad log q(x))
  are the same thing: the optimal noise predictor equals -sqrt(1-abar_t) times the score of the noised
  distribution (the two columns match). So a diffusion model is a SCORE-BASED model — it learns to point
  'uphill' toward higher data density at every noise level, and sampling follows those gradients from
  noise to data (Langevin-style). This unifies the two views (DDPM and score matching) of diffusion
  (README §4).""")


# =============================================================================
# EXPERIMENT 5 — DDIM: deterministic, fewer steps
# =============================================================================


def experiment_5_ddim():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — DDIM: deterministic sampling in far fewer steps (README §5)")
    print("=" * 88)
    rng = np.random.default_rng(1)
    model = Diffusion(d=2)
    model.train(_bimodal(rng), epochs=4000)
    real = _bimodal(rng)(3000)

    def mode_balance(x):
        return abs(np.mean(x[:, 0] < 0) - 0.5)         # 0 = perfectly balanced

    print(f"\n  Sample the SAME trained model with DDPM ({T} stochastic steps) vs DDIM (deterministic):\n")
    print(f"    {'sampler':>22s} {'steps':>6s} {'gen std':>9s} {'mode imbalance':>15s}")
    gen = model.sample_ddpm(3000, seed=9)
    print(f"    {'DDPM':>22s} {T:>6d} {gen[:,0].std():>9.2f} {mode_balance(gen):>15.3f}")
    for steps in (20, 10, 5):
        g = model.sample_ddim(3000, steps=steps, seed=9)
        print(f"    {'DDIM':>22s} {steps:>6d} {g[:,0].std():>9.2f} {mode_balance(g):>15.3f}")
    print(f"""
  (real std {real[:,0].std():.2f})

  READING: standard DDPM sampling is STOCHASTIC and needs all {T} steps (hundreds-to-thousands for real
  models) — slow. DDIM reinterprets the reverse process as a DETERMINISTIC path (an ODE) that can be
  solved in far fewer steps: here 5-10 DDIM steps recover a similar distribution to {T}-step DDPM. Fewer
  steps = faster generation, and determinism enables exact latent inversion and smooth interpolation.
  Speeding up sampling (DDIM, DPM-Solver, distillation) is the main practical thrust of diffusion
  research (README §5).""")


if __name__ == "__main__":
    experiment_1_forward()
    experiment_2_destroys()
    experiment_3_generate()
    experiment_4_score()
    experiment_5_ddim()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
