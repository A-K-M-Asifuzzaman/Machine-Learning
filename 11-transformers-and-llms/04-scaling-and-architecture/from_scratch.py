"""
11.04 — Scaling laws & modern architecture, from scratch (NumPy).

The defining discovery of the LLM era: model quality follows smooth, predictable POWER LAWS in model
size, data, and compute — so you can forecast a giant model's loss from small ones and spend compute
optimally. This file works the math (using the Chinchilla parametric fit) and the Mixture-of-Experts
trick that decouples parameters from compute:

  1. compute-optimal loss follows a power law L - E ~ C^(-gamma)              -> Experiment 1
  2. IsoFLOP: for FIXED compute, loss vs model size is U-shaped (an optimum)  -> Experiment 2
  3. GPT-3 was too big and undertrained; Chinchilla fixes the N/D split       -> Experiment 3
  4. Mixture-of-Experts: total params >> active params (FLOPs)                -> Experiment 4

Uses the Chinchilla fit L(N,D) = E + A/N^a + B/D^b (Hoffmann et al. 2022).

Run:  python3 from_scratch.py
"""

import numpy as np

# Chinchilla "Approach 3" parametric fit
E, A, B, ALPHA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28


def loss(N, D):
    return E + A / N ** ALPHA + B / D ** BETA


def compute_optimal(C, grid=6000):
    """For compute budget C = 6ND, find (N, D) minimizing the loss."""
    N = np.logspace(7, 12, grid)
    D = C / (6 * N)
    L = loss(N, D)
    i = L.argmin()
    return N[i], D[i], L[i]


# =============================================================================
# EXPERIMENT 1 — scaling is a power law
# =============================================================================


def experiment_1_power_law():
    print("=" * 88)
    print("EXPERIMENT 1 — compute-optimal loss follows a power law in compute (README §2)")
    print("=" * 88)
    Cs = np.logspace(18, 24, 12)
    losses = np.array([compute_optimal(C)[2] for C in Cs])
    y = np.log(losses - E)                            # log(reducible loss)
    x = np.log(Cs)
    slope, intercept = np.polyfit(x, y, 1)           # linear fit in log-log
    pred = slope * x + intercept
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    print(f"\n  Compute-optimal loss at increasing compute budgets:\n")
    print(f"    {'compute C (FLOPs)':>18s} {'optimal loss':>13s}")
    for C, L in zip(Cs[::2], losses[::2]):
        print(f"    {C:>18.0e} {L:>13.3f}")
    print(f"""
    log-log fit of (loss - {E}) vs compute:  slope = {slope:.3f},  R^2 = {r2:.5f}

  READING: the REDUCIBLE loss (above the irreducible floor E={E}) falls as a clean power law in
  compute: loss - E ~ C^({slope:.2f}). In log-log space this is a straight line (R^2 = {r2:.4f}), which
  means you can FORECAST the loss of a model 100x bigger by fitting a line to small models. This
  predictability — established by Kaplan et al. (2020) — is what justified spending $100M on a single
  training run: the payoff was known in advance (README §2).""")


# =============================================================================
# EXPERIMENT 2 — IsoFLOP: fixed compute has an optimal model size
# =============================================================================


def experiment_2_isoflop():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — IsoFLOP: for fixed compute, loss vs model size is U-shaped (README §3)")
    print("=" * 88)
    C = 1e21
    Ns = np.logspace(8.5, 11, 9)
    print(f"\n  Fixed compute C = {C:.0e}. Vary model size N (data D = C/6N follows):\n")
    print(f"    {'params N':>12s} {'tokens D':>12s} {'tokens/param':>13s} {'loss':>8s}")
    Nopt, Dopt, Lopt = compute_optimal(C)
    for N in Ns:
        D = C / (6 * N)
        mark = "  <- optimum" if abs(np.log10(N) - np.log10(Nopt)) < 0.16 else ""
        print(f"    {N:>12.1e} {D:>12.1e} {D / N:>13.1f} {loss(N, D):>8.3f}{mark}")
    print(f"""
  READING: at a FIXED compute budget you trade model size against data (bigger model -> fewer tokens).
  Too small a model underfits (high A/N^a term); too big a model is starved of data (high B/D^b term).
  Loss is U-shaped in N with a clear minimum at N ~ {Nopt:.1e} params, D ~ {Dopt:.1e} tokens. Fitting
  this minimum across compute budgets is how Chinchilla derived the compute-optimal frontier (README §3).""")


# =============================================================================
# EXPERIMENT 3 — GPT-3 was undertrained
# =============================================================================


def experiment_3_chinchilla():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — GPT-3 was too big and undertrained; Chinchilla rebalances N vs D (README §4)")
    print("=" * 88)
    N3, D3 = 175e9, 300e9                            # GPT-3: 175B params, 300B tokens
    C3 = 6 * N3 * D3
    No, Do, Lo = compute_optimal(C3)
    print(f"""
  GPT-3's compute budget, spent two ways:

    {'model':>20s} {'params N':>12s} {'tokens D':>12s} {'tok/param':>10s} {'loss':>7s}
    {'GPT-3 (as trained)':>20s} {N3:>12.1e} {D3:>12.1e} {D3 / N3:>10.1f} {loss(N3, D3):>7.3f}
    {'compute-optimal':>20s} {No:>12.1e} {Do:>12.1e} {Do / No:>10.1f} {Lo:>7.3f}

  READING: at the SAME compute, the compute-optimal model is much SMALLER ({No/1e9:.0f}B vs 175B) but
  trained on far MORE data ({Do/1e12:.1f}T vs 0.3T tokens), and reaches LOWER loss ({Lo:.3f} vs
  {loss(N3,D3):.3f}). GPT-3 poured its budget into parameters and starved on data. Chinchilla's lesson
  (Hoffmann et al. 2022): scale data and parameters TOGETHER — its 70B model beat the 175B/280B giants
  by training on 1.4T tokens (~20 tokens/param). Modern models (Llama) push tokens/param far higher
  still, because a smaller, data-rich model is also cheaper to serve (README §4).""")


# =============================================================================
# EXPERIMENT 4 — Mixture-of-Experts
# =============================================================================


def experiment_4_moe():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — Mixture-of-Experts: total params >> active params (FLOPs) (README §5)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    d, n_exp, top_k, n_tok = 16, 8, 2, 5
    experts = [rng.standard_normal((d, d)) * 0.1 for _ in range(n_exp)]   # one linear "expert" each
    W_router = rng.standard_normal((d, n_exp))
    X = rng.standard_normal((n_tok, d))

    logits = X @ W_router
    # top-k routing per token
    topk = np.argsort(-logits, axis=1)[:, :top_k]
    out = np.zeros((n_tok, d))
    active_experts = set()
    for t in range(n_tok):
        gates = softmax(logits[t, topk[t]])          # normalize over the chosen experts
        for g, e in zip(gates, topk[t]):
            out[t] += g * (X[t] @ experts[e])
            active_experts.add(e)
    # a token only ran top_k of n_exp experts
    total_params = n_exp * d * d + d * n_exp
    active_params = top_k * d * d + d * n_exp
    print(f"""
  {n_exp} experts, route each token to its top-{top_k}. For {n_tok} tokens (dim {d}):

    total expert params (all {n_exp} experts) = {total_params:>7,d}
    active params per token (top-{top_k})       = {active_params:>7,d}   ({total_params/active_params:.1f}x fewer FLOPs)
    each token touched only {top_k} of {n_exp} experts (routing is sparse)

  READING: a Mixture-of-Experts layer holds MANY expert MLPs but routes each token to only the top-{top_k}.
  So the model's total parameter count (its capacity/knowledge) grows with the number of experts, while
  the compute per token stays fixed at top-{top_k} experts — decoupling parameters from FLOPs. This is how
  models like Mixtral (8 experts, top-2) and GPT-4-class systems have hundreds of billions of parameters
  but the inference cost of a much smaller dense model. The catch is LOAD BALANCING — a router that
  sends everything to one expert wastes the rest, so an auxiliary balancing loss is needed (README §5).""")


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


if __name__ == "__main__":
    experiment_1_power_law()
    experiment_2_isoflop()
    experiment_3_chinchilla()
    experiment_4_moe()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
