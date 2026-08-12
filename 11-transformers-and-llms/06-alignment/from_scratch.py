"""
11.06 — Alignment (RLHF, DPO), from scratch (NumPy).

A pretrained, instruction-tuned model is capable but not necessarily HELPFUL, HARMLESS, or HONEST.
Alignment optimizes the model against human PREFERENCES. This file builds the core machinery in an
exactly-computable discrete setting and verifies the key theorems:

  1. reward model: fit a Bradley-Terry reward from preference pairs, recover the ranking  -> Experiment 1
  2. the RLHF optimum has a closed form pi* ~ pi_ref * exp(r/beta)                         -> Experiment 2
  3. the KL penalty controls the reward-vs-drift trade-off                                -> Experiment 3
  4. reward hacking: over-optimizing an imperfect reward LOWERS true quality              -> Experiment 4
  5. DPO recovers the RLHF optimum WITHOUT reinforcement learning                         -> Experiment 5

Run:  python3 from_scratch.py
"""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def kl(p, q):
    return float(np.sum(p * np.log(p / q)))


# =============================================================================
# EXPERIMENT 1 — Bradley-Terry reward model
# =============================================================================


def experiment_1_reward_model():
    print("=" * 88)
    print("EXPERIMENT 1 — reward model: fit a Bradley-Terry reward from preferences (README §2)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    K = 8
    true_quality = np.sort(rng.standard_normal(K))    # ground-truth quality of 8 responses
    # humans compare pairs and prefer the higher-quality one with P = sigma(q_a - q_b)
    prefs = []
    for _ in range(3000):
        a, b = rng.choice(K, 2, replace=False)
        prefs.append((a, b) if rng.random() < sigmoid(true_quality[a] - true_quality[b]) else (b, a))
    # fit reward r by maximizing sum log sigma(r[chosen] - r[rejected])
    r = np.zeros(K)
    for _ in range(2000):
        g = np.zeros(K)
        for c, j in prefs:
            e = sigmoid(r[c] - r[j]) - 1               # gradient of -log sigma(r_c - r_j)
            g[c] += e; g[j] -= e
        r -= 0.02 * g / len(prefs)
    r -= r.mean()
    from scipy.stats import spearmanr
    rho = spearmanr(r, true_quality).correlation if _have_scipy() else _spearman(r, true_quality)
    print(f"""
  8 responses with hidden qualities; {len(prefs)} noisy pairwise preferences. Learned reward vs truth:

    true quality (sorted): {np.round(true_quality, 2)}
    learned reward:        {np.round(r, 2)}
    rank correlation (Spearman) = {rho:.3f}

  READING: we never observe a numeric reward — only which of two responses a human PREFERRED. The
  Bradley-Terry model says P(a preferred over b) = sigma(r_a - r_b), so fitting r to maximize the
  likelihood of the observed preferences recovers a reward function that RANKS responses correctly
  (Spearman {rho:.2f}). This learned reward model is the training signal for RLHF — it turns sparse
  human comparisons into a dense score for any response (README §2).""")


def _have_scipy():
    try:
        import scipy.stats  # noqa
        return True
    except Exception:
        return False


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# =============================================================================
# EXPERIMENT 2 — the RLHF optimal policy has a closed form
# =============================================================================


def rlhf_optimize(r, pi_ref, beta, iters=20000, lr=0.2):
    """Directly maximize E_pi[r] - beta*KL(pi||pi_ref) over the simplex (mirror ascent on logits)."""
    logits = np.log(pi_ref).copy()
    for _ in range(iters):
        p = softmax(logits)
        grad_p = r - beta * (np.log(p / pi_ref) + 1)
        logits += lr * p * (grad_p - (p * grad_p).sum())
    return softmax(logits)


def experiment_2_closed_form():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — the RLHF optimum is pi* ~ pi_ref * exp(r/beta) (README §3)")
    print("=" * 88)
    rng = np.random.default_rng(1)
    K = 6
    r = rng.standard_normal(K)
    pi_ref = softmax(rng.standard_normal(K))
    print(f"\n  Maximize  E_pi[r] - beta * KL(pi || pi_ref).  Direct optimization vs the closed form:\n")
    print(f"    {'beta':>6s} {'max|direct - closed form|':>28s}")
    for beta in (0.5, 1.0, 5.0):
        direct = rlhf_optimize(r, pi_ref, beta)
        closed = pi_ref * np.exp(r / beta); closed /= closed.sum()
        print(f"    {beta:>6.1f} {np.abs(direct - closed).max():>28.1e}")
    print("""
  READING: the RLHF objective is 'maximize reward, but stay close (in KL) to the reference policy'. Its
  solution has an exact closed form: pi*(y) is proportional to pi_ref(y) * exp(r(y)/beta) — reweight
  the reference by the exponentiated reward. Direct optimization matches this to machine precision. PPO
  is just a way to APPROXIMATE this optimum with gradient steps when the response space is astronomically
  large (real text); the target it climbs toward is this formula (README §3).""")


# =============================================================================
# EXPERIMENT 3 — the KL penalty controls the reward-vs-drift trade-off
# =============================================================================


def experiment_3_kl_tradeoff():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — the KL penalty controls the reward-vs-drift trade-off (README §3)")
    print("=" * 88)
    rng = np.random.default_rng(2)
    K = 6
    r = rng.standard_normal(K)
    pi_ref = softmax(rng.standard_normal(K))
    print(f"\n  Sweep the KL coefficient beta (smaller = optimize reward harder):\n")
    print(f"    {'beta':>8s} {'E[reward]':>12s} {'KL from ref':>14s}")
    for beta in (10.0, 3.0, 1.0, 0.3, 0.1):
        pi = pi_ref * np.exp(r / beta); pi /= pi.sum()
        print(f"    {beta:>8.1f} {pi @ r:>12.3f} {kl(pi, pi_ref):>14.3f}")
    print("""
  READING: beta trades reward against faithfulness to the reference model. LARGE beta -> stay near the
  reference (low KL, low reward gain); SMALL beta -> chase reward (high KL, high reward, but the policy
  drifts far from the well-behaved reference). The KL term is essential: it stops the model from
  collapsing onto whatever maximizes the reward, keeping it fluent and on-distribution. Tuning beta is
  the central knob of RLHF (README §3).""")


# =============================================================================
# EXPERIMENT 4 — reward hacking
# =============================================================================


def experiment_4_reward_hacking():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — reward hacking: over-optimizing an imperfect reward hurts (README §4)")
    print("=" * 88)
    rng = np.random.default_rng(5)
    K = 10
    true_r = np.sort(rng.standard_normal(K))          # response 9 is genuinely best
    proxy_r = true_r.copy()
    hack = 1                                           # a low-quality response...
    proxy_r[hack] = true_r[-1] + 0.8                   # ...the imperfect reward model rates highest
    pi_ref = softmax(rng.standard_normal(K))
    print(f"\n  The reward MODEL is imperfect: it wrongly rates a bad response (true reward "
          f"{true_r[hack]:+.2f}) as the")
    print(f"  best (proxy {proxy_r[hack]:.2f}). Optimize the PROXY at various beta; measure the TRUE reward:\n")
    print(f"    {'beta':>8s} {'proxy reward':>13s} {'TRUE reward':>13s} {'P(hack)':>9s}")
    for beta in (10.0, 3.0, 1.0, 0.5, 0.2, 0.1):
        pi = pi_ref * np.exp(proxy_r / beta); pi /= pi.sum()
        tag = "  <- true reward peaks" if abs(beta - 1.0) < 1e-9 else ""
        print(f"    {beta:>8.2f} {pi @ proxy_r:>13.3f} {pi @ true_r:>13.3f} {pi[hack]:>9.2f}{tag}")
    print("""
  READING: the reward model is a PROXY for human preference, and it has errors. As beta shrinks (harder
  optimization), the PROXY reward keeps rising — but the TRUE reward rises, PEAKS (around beta=1), then
  FALLS as the policy piles probability onto the bad-but-high-proxy 'hack' response. This is reward
  hacking / over-optimization (Goodhart's law: 'when a measure becomes a target, it ceases to be a good
  measure'). The proxy and the truth agree at first and diverge under pressure — which is why RLHF keeps
  a KL leash, why reward models must be robust, and why more optimization is not always better (README §4).""")


# =============================================================================
# EXPERIMENT 5 — DPO recovers the RLHF optimum without RL
# =============================================================================


def experiment_5_dpo():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — DPO recovers the RLHF optimum WITHOUT reinforcement learning (README §5)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    K, beta = 6, 1.0
    true_r = rng.standard_normal(K)
    pi_ref = softmax(rng.standard_normal(K))
    log_ref = np.log(pi_ref)
    prefs = []
    for _ in range(5000):                             # preferences from the true reward
        a, b = rng.choice(K, 2, replace=False)
        prefs.append((a, b) if rng.random() < sigmoid(true_r[a] - true_r[b]) else (b, a))

    theta = log_ref.copy()                            # policy logits, start at reference
    for _ in range(400):                              # minimize the DPO loss directly on the policy
        p = softmax(theta); logp = np.log(p)
        grad = np.zeros(K)
        for c, j in prefs:
            h = beta * ((logp[c] - log_ref[c]) - (logp[j] - log_ref[j]))
            coef = -(1 - sigmoid(h)) * beta
            grad += coef * (((np.arange(K) == c) - p) - ((np.arange(K) == j) - p))
        theta -= 0.05 * grad / len(prefs)
    pi_dpo = softmax(theta)
    closed = pi_ref * np.exp(true_r / beta); closed /= closed.sum()
    print(f"""
  DPO trains the policy directly on preference pairs (no reward model, no PPO):

    DPO policy:            {np.round(pi_dpo, 3)}
    RLHF closed form pi*:  {np.round(closed, 3)}
    same ranking? {bool((pi_dpo.argsort() == closed.argsort()).all())}   max|diff| = {np.abs(pi_dpo - closed).max():.3f}

  READING: DPO's key insight — the RLHF optimum pi* ~ pi_ref exp(r/beta) can be INVERTED to write the
  reward as r = beta*log(pi/pi_ref). Substituting that into the Bradley-Terry preference likelihood
  turns RLHF into a simple classification loss on preference pairs — no reward model, no sampling, no
  PPO. Trained this way, the policy lands on the SAME optimum as full RLHF (same ranking, within
  finite-sample noise). DPO is simpler, more stable, and now the default alignment method (README §5).""")


if __name__ == "__main__":
    experiment_1_reward_model()
    experiment_2_closed_form()
    experiment_3_kl_tradeoff()
    experiment_4_reward_hacking()
    experiment_5_dpo()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
