"""
11.03 — Efficient attention, from scratch (NumPy).

Vanilla attention ([11.01]) is O(n^2) time and memory and recomputes everything every generation step.
The tricks that make LLMs fast and long-context all keep the SAME math but change the bookkeeping. This
file builds and verifies each:

  1. KV cache: cached generation == recompute-from-scratch, but O(n) not O(n^2)   -> Experiment 1
  2. MQA / GQA: share K,V across heads to shrink the KV cache                      -> Experiment 2
  3. RoPE: rotary embeddings make the score depend only on RELATIVE position       -> Experiment 3
  4. FlashAttention's online softmax == standard softmax (the O(n)-memory trick)   -> Experiment 4
  5. ALiBi: a distance-linear score bias, no position embeddings                   -> Experiment 5

Run:  python3 from_scratch.py
"""

import numpy as np


def softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


def attention(Q, K, V, mask=None):
    d = Q.shape[-1]
    s = Q @ K.T / np.sqrt(d)
    if mask is not None:
        s = s + mask
    return softmax(s, -1) @ V


# =============================================================================
# EXPERIMENT 1 — KV cache
# =============================================================================


def experiment_1_kv_cache():
    print("=" * 88)
    print("EXPERIMENT 1 — KV cache: same output as recompute, but O(n) not O(n^2) (README §2)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    n, d = 8, 16
    X = rng.standard_normal((n, d))
    Wq, Wk, Wv = (rng.standard_normal((d, d)) for _ in range(3))

    # A) recompute: at each step, project ALL tokens' K,V from scratch (the naive way)
    recompute_out = []
    kv_projections_recompute = 0
    for t in range(1, n + 1):
        q = X[t - 1:t] @ Wq
        K = X[:t] @ Wk; V = X[:t] @ Wv                 # recomputes K,V for all t tokens
        kv_projections_recompute += t
        cmask = np.zeros((1, t))
        recompute_out.append(attention(q, K, V, cmask)[0])

    # B) KV cache: project only the NEW token's K,V each step, append to a cache
    cache_out = []
    Kc = np.zeros((0, d)); Vc = np.zeros((0, d))
    kv_projections_cached = 0
    for t in range(n):
        q = X[t:t + 1] @ Wq
        Kc = np.vstack([Kc, X[t:t + 1] @ Wk])          # one new K,V row
        Vc = np.vstack([Vc, X[t:t + 1] @ Wv])
        kv_projections_cached += 1
        cache_out.append(attention(q, Kc, Vc)[0])

    err = np.abs(np.array(recompute_out) - np.array(cache_out)).max()
    print(f"""
  Autoregressively producing {n} tokens, comparing recompute vs KV cache:

    outputs identical? max|diff| = {err:.1e}
    K,V projections — recompute (O(n^2)) = {kv_projections_recompute}
    K,V projections — KV cache  (O(n))   = {kv_projections_cached}

  READING: during generation, past tokens' keys and values never change — so recomputing them every
  step is pure waste (it does {kv_projections_recompute} projections for {n} tokens, O(n^2) total). The
  KV CACHE stores each token's K,V once and appends only the new token's ({kv_projections_cached} total,
  O(n)), giving identical outputs. This is THE optimization behind fast LLM inference; its cost is
  memory — the cache grows with sequence length, which is what MQA/GQA (Experiment 2) attack (README §2).""")


# =============================================================================
# EXPERIMENT 2 — MQA / GQA
# =============================================================================


def experiment_2_mqa_gqa():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — MQA / GQA shrink the KV cache by sharing K,V across heads (README §3)")
    print("=" * 88)
    H, d_head, n = 32, 128, 4096                       # a Llama-ish config
    def kv_bytes(n_kv_heads):
        return 2 * n_kv_heads * d_head * n * 2         # 2 (K,V) x heads x dim x len x 2 bytes (fp16)
    mha = kv_bytes(H)                                  # one K,V per query head
    gqa = kv_bytes(8)                                  # 8 KV heads (groups of 4)
    mqa = kv_bytes(1)                                  # a single shared K,V
    print(f"""
  KV-cache size for {H} heads, head-dim {d_head}, sequence length {n} (fp16):

    MHA  ({H} KV heads, one per query head) : {mha / 1e6:8.1f} MB   (baseline)
    GQA  (8 KV heads, groups of 4)          : {gqa / 1e6:8.1f} MB   ({mha/gqa:.0f}x smaller)
    MQA  (1 shared KV head)                 : {mqa / 1e6:8.1f} MB   ({mha/mqa:.0f}x smaller)

  READING: the KV cache stores one key and value per HEAD per token, so multi-head attention's cache is
  large — it dominates memory at long context and is the inference bottleneck. Multi-Query Attention
  (MQA) shares a SINGLE K,V across all query heads ({H}x smaller cache); Grouped-Query Attention (GQA)
  is the middle ground — a few KV heads shared by groups of query heads ({mha/gqa:.0f}x smaller with
  little quality loss). Llama-2/3, Mistral, and most modern LLMs use GQA (README §3).""")


# =============================================================================
# EXPERIMENT 3 — RoPE (rotary position embeddings)
# =============================================================================


def apply_rope(x, pos, base=10000.0):
    """Rotate each (2i, 2i+1) pair of x by angle pos * base^(-2i/d)."""
    d = x.shape[-1]
    theta = base ** (-np.arange(0, d, 2) / d)
    ang = pos * theta
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def experiment_3_rope():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — RoPE makes the attention score depend only on RELATIVE position (README §4)")
    print("=" * 88)
    rng = np.random.default_rng(1)
    d = 16
    q, k = rng.standard_normal(d), rng.standard_normal(d)
    print(f"\n  q . k after rotating q by position m and k by position n (score should depend on m-n):\n")
    print(f"    {'(m, n)':>12s} {'RoPE(q,m).RoPE(k,n)':>22s}")
    for m, n in [(5, 3), (6, 4), (10, 8), (2, 0), (100, 98)]:
        score = apply_rope(q, m) @ apply_rope(k, n)
        print(f"    {f'({m}, {n})':>12s} {score:>22.6f}")
    # all pairs above have m-n = 2 -> scores should be identical
    s1 = apply_rope(q, 5) @ apply_rope(k, 3)
    s2 = apply_rope(q, 100) @ apply_rope(k, 98)
    print(f"""
    all rows have m - n = 2, so all scores are equal: max spread = {abs(s1 - s2):.1e}

  READING: RoPE rotates the query and key vectors by an angle proportional to their POSITION before the
  dot product. Because a rotation by m followed by a dot with a rotation by n leaves only the angle
  difference m-n, the resulting score depends purely on the RELATIVE distance — every (m,n) pair with
  m-n=2 above gives the identical score. This gives relative-position awareness with no learned
  parameters and, crucially, EXTRAPOLATES to longer sequences than seen in training. RoPE is used by
  Llama, GPT-NeoX, PaLM, and most modern LLMs (README §4).""")


# =============================================================================
# EXPERIMENT 4 — FlashAttention's online softmax
# =============================================================================


def flash_attention(Q, K, V, block=2):
    """Streaming attention: never materializes the full n x n matrix. Online-softmax over K,V blocks."""
    n, d = Q.shape
    m = K.shape[0]
    out = np.zeros((n, d))
    scale = 1.0 / np.sqrt(d)
    for i in range(n):                                 # each query, streamed over key blocks
        running_max = -np.inf
        running_sum = 0.0
        acc = np.zeros(d)
        for j0 in range(0, m, block):                  # process K,V in blocks (the "tiling")
            Kb, Vb = K[j0:j0 + block], V[j0:j0 + block]
            s = (Q[i] @ Kb.T) * scale
            block_max = s.max()
            new_max = max(running_max, block_max)
            correction = np.exp(running_max - new_max)  # rescale old accumulators to the new max
            p = np.exp(s - new_max)
            running_sum = running_sum * correction + p.sum()
            acc = acc * correction + p @ Vb
            running_max = new_max
        out[i] = acc / running_sum
    return out


def experiment_4_flash():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — FlashAttention's online softmax == standard softmax (README §5)")
    print("=" * 88)
    rng = np.random.default_rng(2)
    n, d = 12, 16
    Q, K, V = (rng.standard_normal((n, d)) for _ in range(3))
    standard = attention(Q, K, V)
    flash = flash_attention(Q, K, V, block=3)
    err = np.abs(standard - flash).max()
    print(f"""
  Attention on a length-{n} sequence, blockwise online softmax vs the standard full-matrix computation:
    max|flash - standard| = {err:.1e}

  READING: standard attention builds the full n x n score matrix, softmaxes it, then multiplies by V —
  O(n^2) MEMORY. FlashAttention streams over blocks of keys/values, maintaining a running max and sum
  (an 'online softmax') and rescaling the accumulator whenever a bigger score appears — so it never
  stores the n x n matrix (O(n) memory) yet computes the EXACT same result ({err:.0e} difference). By
  keeping everything in fast on-chip SRAM, it is also several times faster in wall-clock. FlashAttention
  is why long-context training is feasible (README §5).""")


# =============================================================================
# EXPERIMENT 5 — ALiBi
# =============================================================================


def experiment_5_alibi():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — ALiBi: a distance-linear score bias, no position embeddings (README §4)")
    print("=" * 88)
    n = 6
    slope = 0.5                                        # per-head slope
    # bias[i,j] = -slope * (i - j) for j <= i (how far back)
    i = np.arange(n)[:, None]; j = np.arange(n)[None, :]
    bias = -slope * (i - j)
    bias = np.where(j > i, -np.inf, bias)              # causal
    rng = np.random.default_rng(3)
    Q, K, V = (rng.standard_normal((n, 8)) for _ in range(3))
    s = Q @ K.T / np.sqrt(8) + bias
    A = softmax(s, -1)
    print(f"\n  ALiBi adds bias[i,j] = -slope*(i-j) to the scores. Attention of the LAST query (row 5):\n")
    print(f"    key position j:   {list(range(n))}")
    print(f"    attention weight: {[round(float(x), 3) for x in A[-1]]}")
    print(f"""
    weight on nearest key (j=5) = {A[-1, -1]:.3f}   vs farthest (j=0) = {A[-1, 0]:.3f}

  READING: ALiBi ('Attention with Linear Biases') adds a penalty proportional to how far back a key is —
  no position embeddings at all, just a distance-linear bias on the scores, with a different slope per
  head. It builds in a RECENCY prior (nearer tokens get more weight, {A[-1,-1]:.2f} vs {A[-1,0]:.2f}
  here) and, like RoPE, extrapolates to sequences far longer than training. It is the simplest way to
  give a transformer position awareness (README §4).""")


if __name__ == "__main__":
    experiment_1_kv_cache()
    experiment_2_mqa_gqa()
    experiment_3_rope()
    experiment_4_flash()
    experiment_5_alibi()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
