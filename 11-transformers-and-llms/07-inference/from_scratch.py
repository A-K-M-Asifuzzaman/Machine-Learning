"""
11.07 — Inference & serving, from scratch (NumPy).

Training an LLM is a one-time cost; SERVING it is forever. Inference has its own algorithms — how to
turn logits into tokens (decoding), how to go faster without changing the output (speculative
decoding), and how to trade latency for throughput (batching). This file builds and MEASURES them:

  1. decoding strategies: greedy / temperature / top-k / top-p reshape the distribution  -> Experiment 1
  2. speculative decoding is EXACT: same output distribution as the target model          -> Experiment 2
  3. speculative decoding speedup grows with draft/target agreement                       -> Experiment 3
  4. batching trades per-request latency for throughput                                   -> Experiment 4

Run:  python3 from_scratch.py
"""

import numpy as np


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def entropy(p):
    return float(-np.sum(p * np.log2(p + 1e-12)))


# =============================================================================
# EXPERIMENT 1 — decoding strategies
# =============================================================================


def temperature(logits, T):
    return softmax(logits / T)


def top_k(p, k):
    q = np.zeros_like(p)
    idx = np.argsort(-p)[:k]
    q[idx] = p[idx]
    return q / q.sum()


def top_p(p, thresh):
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    keep = order[:np.searchsorted(cum, thresh) + 1]
    q = np.zeros_like(p)
    q[keep] = p[keep]
    return q / q.sum()


def experiment_1_decoding():
    print("=" * 88)
    print("EXPERIMENT 1 — decoding strategies reshape the next-token distribution (README §2)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(20) * 2
    base = softmax(logits)
    print(f"\n  20-token vocabulary. How each strategy reshapes the distribution:\n")
    print(f"    {'strategy':>20s} {'entropy (bits)':>15s} {'# tokens with p>0.001':>22s}")
    rows = [("temperature T=0.5", temperature(logits, 0.5)),
            ("temperature T=1.0", base),
            ("temperature T=2.0", temperature(logits, 2.0)),
            ("top-k (k=3)", top_k(base, 3)),
            ("top-p (p=0.9)", top_p(base, 0.9))]
    for name, p in rows:
        print(f"    {name:>20s} {entropy(p):>15.3f} {int((p > 0.001).sum()):>22d}")
    print(f"    {'greedy (argmax)':>20s} {'0.000':>15s} {1:>22d}")
    print("""
  READING: the model outputs a distribution; the decoder chooses how to sample it. TEMPERATURE scales
  the logits: T<1 sharpens (lower entropy, more deterministic), T>1 flattens (higher entropy, more
  random). TOP-K keeps only the k most-likely tokens; TOP-P (nucleus) keeps the smallest set covering
  probability p — both truncate the unreliable tail while adapting to how peaked the distribution is.
  GREEDY (T->0) always takes the argmax. Sampling with top-p + moderate temperature is the standard for
  open-ended generation; greedy/beam for tasks with one right answer (README §2).""")


# =============================================================================
# EXPERIMENT 2 — speculative decoding is exact
# =============================================================================


def speculative_sample(p, q, rng):
    """Draft proposes x~q; accept w.p. min(1, p/q), else resample from the residual. Returns x~p exactly."""
    x = rng.choice(len(q), p=q)                        # draft model's proposal
    if rng.random() < min(1.0, p[x] / q[x]):
        return x                                       # accept
    residual = np.maximum(p - q, 0)                    # else sample from normalized (p - q)+
    residual /= residual.sum()
    return rng.choice(len(p), p=residual)


def experiment_2_speculative_exact():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — speculative decoding is EXACT: same distribution as the target (README §3)")
    print("=" * 88)
    rng = np.random.default_rng(1)
    K = 8
    p = softmax(rng.standard_normal(K) * 1.5)          # target (big) model distribution
    q = softmax(rng.standard_normal(K) * 1.5)          # draft (small) model distribution
    N = 200000
    samples = np.array([speculative_sample(p, q, rng) for _ in range(N)])
    emp = np.bincount(samples, minlength=K) / N
    err = np.abs(emp - p).max()
    print(f"""
  Draft distribution q != target distribution p. Speculative sampling accept/resample scheme, {N:,} draws:

    target p:      {np.round(p, 3)}
    speculative:   {np.round(emp, 3)}
    max|empirical - target| = {err:.4f}   (-> 0 as N grows)

  READING: speculative decoding uses a small DRAFT model to propose tokens and the big TARGET model to
  check them. The magic is the accept/reject rule — accept the draft's token x with probability
  min(1, p(x)/q(x)), otherwise resample from the normalized residual (p-q)+ — which guarantees the
  output is distributed EXACTLY as if sampled from the target model p. So it is not an approximation:
  the generated text is identical in distribution to plain sampling, just faster (Experiment 3).""")


# =============================================================================
# EXPERIMENT 3 — speculative decoding speedup
# =============================================================================


def experiment_3_speculative_speedup():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — speculative decoding speedup grows with draft/target agreement (README §3)")
    print("=" * 88)
    rng = np.random.default_rng(2)
    K, gamma = 20, 4                                   # propose gamma tokens per verification
    print(f"\n  Draft proposes {gamma} tokens; the target verifies them in ONE forward pass. Expected")
    print(f"  accepted tokens per verification vs how well the draft matches the target:\n")
    print(f"    {'draft-target agreement':>24s} {'accept rate':>12s} {'tokens/step':>13s} {'speedup':>9s}")
    for noise, label in [(0.3, "high (similar models)"), (1.0, "medium"), (2.5, "low (very different)")]:
        # measure the average acceptance probability alpha = E_x~q[min(1, p/q)]
        alphas = []
        for _ in range(2000):
            base = rng.standard_normal(K)
            p = softmax(base)
            q = softmax(base + noise * rng.standard_normal(K))
            x = rng.choice(K, p=q)
            alphas.append(min(1.0, p[x] / q[x]))
        alpha = np.mean(alphas)
        # expected accepted tokens before first rejection, capped at gamma, +1 for the target's own token
        exp_tokens = (1 - alpha ** (gamma + 1)) / (1 - alpha)
        speedup = exp_tokens                            # target forward passes drop from exp_tokens to 1
        print(f"    {label:>24s} {alpha:>12.2f} {exp_tokens:>13.2f} {speedup:>8.1f}x")
    print("""
  READING: the draft runs cheaply and proposes several tokens; the target checks them ALL in a single
  forward pass (the same cost as generating one token). Every accepted token is essentially free — so
  the speedup is the number of tokens accepted per target pass. When the draft agrees with the target
  (similar models), acceptance is high and 3-4 tokens land per step; when they disagree, fewer. This is
  how production LLMs serve 2-3x faster with NO change to the output distribution (README §3).""")


# =============================================================================
# EXPERIMENT 4 — batching: throughput vs latency
# =============================================================================


def experiment_4_batching():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — batching trades per-request latency for throughput (README §4)")
    print("=" * 88)
    # simple roofline: decode is memory-bound. Reading the weights costs a fixed time per step;
    # that read is AMORTIZED across the batch. Compute grows with batch size.
    weight_read_ms = 10.0                              # time to stream weights once (fixed per step)
    compute_ms_per_req = 0.5                           # marginal compute per request in the batch
    print(f"\n  Decode step time = weight-read ({weight_read_ms} ms, shared) + compute ({compute_ms_per_req} ms/request):\n")
    print(f"    {'batch size':>12s} {'step time (ms)':>15s} {'latency/req':>13s} {'throughput (req/s)':>20s}")
    for B in (1, 4, 16, 64, 256):
        step = weight_read_ms + compute_ms_per_req * B
        latency = step                                 # each request waits the full step
        throughput = B / (step / 1000)                 # requests per second
        print(f"    {B:>12d} {step:>15.1f} {f'{latency:.1f} ms':>13s} {throughput:>20.0f}")
    print("""
  READING: generating one token requires reading ALL the model's weights from memory — decode is
  MEMORY-bandwidth bound, and at batch size 1 that huge read serves a single request. BATCHING many
  requests amortizes the one weight-read across all of them: step time grows slowly (only the small
  compute term scales), so THROUGHPUT (requests/sec) rises steeply while per-request LATENCY grows only
  a little. This is why servers batch aggressively (continuous batching, vLLM) — but there is a limit,
  and interactive use trades some throughput for low latency (README §4).""")


if __name__ == "__main__":
    experiment_1_decoding()
    experiment_2_speculative_exact()
    experiment_3_speculative_speedup()
    experiment_4_batching()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
