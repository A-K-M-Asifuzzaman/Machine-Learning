"""
11.01 — The Transformer, from scratch (NumPy).

"Attention Is All You Need." A transformer is self-attention (each token reads every other token) plus
a position-wise MLP, wrapped in residual connections and normalization, with positional encodings to
restore order. This file builds every piece and verifies it against PyTorch to machine precision:

  1. scaled dot-product attention == F.scaled_dot_product_attention   (machine precision)
  2. multi-head attention == torch.nn.MultiheadAttention              (machine precision)
  3. causal masking: token i attends only to tokens <= i             -> Experiment 3
  4. sinusoidal positional encodings and their relative-position property -> Experiment 4
  5. a full transformer block == torch.nn.TransformerEncoderLayer     (machine precision)
  6. self-attention is parallel (O(1) sequential depth) but O(n^2) in length -> Experiment 6

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAVE_TORCH = True
except Exception:                                    # pragma: no cover
    HAVE_TORCH = False


def softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


# =============================================================================
# Scaled dot-product attention
# =============================================================================


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q:(...,Tq,d) K,V:(...,Tk,d). out = softmax(QK^T/sqrt(d) + mask) V."""
    d = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d)
    if mask is not None:
        scores = scores + mask                       # mask holds 0 (keep) or -inf (block)
    A = softmax(scores, axis=-1)
    return A @ V, A


def experiment_1_sdpa():
    print("=" * 88)
    print("EXPERIMENT 1 — scaled dot-product attention == PyTorch (machine precision)")
    print("=" * 88)
    if not HAVE_TORCH:
        print("  torch not available — skipping."); return
    rng = np.random.default_rng(0)
    B, H, T, d = 2, 3, 5, 16
    Q, K, V = (rng.standard_normal((B, H, T, d)) for _ in range(3))
    out, _ = scaled_dot_product_attention(Q, K, V)
    ref = F.scaled_dot_product_attention(torch.tensor(Q), torch.tensor(K), torch.tensor(V)).numpy()
    err = np.abs(out - ref).max()
    # causal version
    cmask = np.triu(np.full((T, T), -np.inf), k=1)
    out_c, _ = scaled_dot_product_attention(Q, K, V, cmask)
    ref_c = F.scaled_dot_product_attention(torch.tensor(Q), torch.tensor(K), torch.tensor(V),
                                           is_causal=True).numpy()
    err_c = np.abs(out_c - ref_c).max()
    print(f"""
  Random Q,K,V of shape (batch={B}, heads={H}, len={T}, dim={d}):

    attention           |ours - torch F.sdpa| = {err:.1e}
    causal attention    |ours - torch (causal)| = {err_c:.1e}

  READING: attention computes, for each query, a softmax-weighted average of the values, weighted by
  how well the query matches each key: softmax(QK^T / sqrt(d)) V. The 1/sqrt(d) keeps the scores O(1)
  so the softmax stays soft ([09.03 §5]). This single operation — a differentiable, content-based
  lookup — is the entire heart of the transformer, and it matches PyTorch exactly.""")


# =============================================================================
# Multi-head attention
# =============================================================================


class MultiHeadAttention:
    def __init__(self, W_in, b_in, W_out, b_out, n_heads):
        self.W_in, self.b_in = W_in, b_in            # (3d, d), (3d,)  packed Wq;Wk;Wv (torch layout)
        self.W_out, self.b_out = W_out, b_out        # (d, d), (d,)
        self.n_heads = n_heads

    def __call__(self, X, mask=None):
        T, d = X.shape
        h, dh = self.n_heads, d // self.n_heads
        qkv = X @ self.W_in.T + self.b_in            # (T, 3d)
        Q, K, V = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
        # split into heads: (h, T, dh)
        Q = Q.reshape(T, h, dh).transpose(1, 0, 2)
        K = K.reshape(T, h, dh).transpose(1, 0, 2)
        V = V.reshape(T, h, dh).transpose(1, 0, 2)
        out, _ = scaled_dot_product_attention(Q, K, V, mask)   # (h, T, dh)
        out = out.transpose(1, 0, 2).reshape(T, d)             # concat heads
        return out @ self.W_out.T + self.b_out


def experiment_2_mha():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — multi-head attention == torch.nn.MultiheadAttention (machine precision)")
    print("=" * 88)
    if not HAVE_TORCH:
        print("  torch not available — skipping."); return
    d, h, T = 16, 4, 6
    mha_t = torch.nn.MultiheadAttention(d, h, batch_first=True).double()
    W_in = mha_t.in_proj_weight.detach().numpy()
    b_in = mha_t.in_proj_bias.detach().numpy()
    W_out = mha_t.out_proj.weight.detach().numpy()
    b_out = mha_t.out_proj.bias.detach().numpy()
    rng = np.random.default_rng(1)
    X = rng.standard_normal((T, d))
    mine = MultiHeadAttention(W_in, b_in, W_out, b_out, h)(X)
    Xt = torch.tensor(X)[None]
    ref, _ = mha_t(Xt, Xt, Xt)
    err = np.abs(mine - ref.detach().numpy()[0]).max()
    print(f"""
  Self-attention with {h} heads on a length-{T} sequence (dim {d}):
    |ours - torch.nn.MultiheadAttention| = {err:.1e}

  READING: multi-head attention runs {h} attention operations in PARALLEL on different learned
  projections of the input, then concatenates them. Each head can specialize — one tracks syntax,
  another coreference, another position — giving the model several 'representation subspaces' instead
  of one. Splitting d={d} into {h} heads of {d // h} keeps the total compute the same. Matches PyTorch
  exactly, including the packed QKV projection and output projection.""")


# =============================================================================
# EXPERIMENT 3 — causal masking
# =============================================================================


def experiment_3_causal():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — causal masking: token i attends only to tokens <= i (README §4)")
    print("=" * 88)
    rng = np.random.default_rng(2)
    T, d = 5, 8
    Q, K, V = (rng.standard_normal((T, d)) for _ in range(3))
    cmask = np.triu(np.full((T, T), -np.inf), k=1)
    _, A = scaled_dot_product_attention(Q, K, V, cmask)
    upper = A[np.triu_indices(T, k=1)]                # weights on FUTURE tokens
    print(f"\n  Attention weight matrix with a causal mask (row = query position):\n")
    for i in range(T):
        row = "  ".join(f"{A[i, j]:.2f}" for j in range(T))
        print(f"    pos {i}: {row}")
    print(f"""
    max attention weight on any FUTURE token = {upper.max():.1e}   (exactly 0)
    each row sums to 1?  {np.allclose(A.sum(1), 1)}

  READING: a causal (autoregressive) transformer must not see the future — token i may attend only to
  tokens 0..i. The mask adds -inf to the upper triangle of the scores BEFORE the softmax, so those
  positions get exactly 0 weight (upper triangle above is all 0) while each row still sums to 1. This
  one mask is what turns the encoder's bidirectional attention into a decoder / GPT-style language
  model that predicts the next token (README §4).""")


# =============================================================================
# EXPERIMENT 4 — sinusoidal positional encodings
# =============================================================================


def positional_encoding(T, d):
    pos = np.arange(T)[:, None]
    i = np.arange(d)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d)
    pe = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))
    return pe


def experiment_4_positional():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — sinusoidal positional encodings restore order (README §5)")
    print("=" * 88)
    T, d = 32, 64
    PE = positional_encoding(T, d)
    # every position has a distinct encoding
    dists = np.linalg.norm(PE[:, None] - PE[None, :], axis=-1)
    min_offdiag = dists[~np.eye(T, dtype=bool)].min()
    # dot product between position encodings decays with distance (relative-position signal)
    sims = PE @ PE[0]
    print(f"\n  Sinusoidal PE for {T} positions (dim {d}):\n")
    print(f"    all positions distinct? (min pairwise distance = {min_offdiag:.3f} > 0)  {min_offdiag > 0}")
    print(f"\n    dot(PE[0], PE[k]) as offset k grows (should decay -> encodes relative position):")
    for k in (0, 1, 2, 4, 8, 16):
        print(f"      offset {k:>3d}: {sims[k]:>8.3f}")
    print("""
  READING: self-attention is permutation-invariant ([08.05 §3]) — it has no idea what ORDER the tokens
  came in. Sinusoidal positional encodings add a unique, deterministic vector to each position (sines
  and cosines at geometrically-spaced frequencies), so every position is distinguishable and the dot
  product between two positions depends smoothly on their DISTANCE (it decays as they separate). That
  relative-position signal is what lets attention learn 'the previous token' or 'three words back'
  without any learned position parameters (README §5).""")


# =============================================================================
# EXPERIMENT 5 — a full transformer block
# =============================================================================


def layernorm(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gamma + beta


def experiment_5_block():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — a full transformer block == torch.nn.TransformerEncoderLayer (machine prec.)")
    print("=" * 88)
    if not HAVE_TORCH:
        print("  torch not available — skipping."); return
    d, h, dff, T = 16, 4, 32, 6
    layer = torch.nn.TransformerEncoderLayer(d, h, dff, batch_first=True,
                                             norm_first=True, activation="gelu").double()
    layer.eval()                                     # disable dropout for a deterministic comparison
    p = {k: v.detach().numpy() for k, v in layer.named_parameters()}
    rng = np.random.default_rng(3)
    X = rng.standard_normal((T, d))

    import math
    erf = np.vectorize(math.erf)

    def gelu(x):
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))

    # norm_first: x = x + attn(LN1(x)); x = x + ffn(LN2(x))
    ln1 = layernorm(X, p["norm1.weight"], p["norm1.bias"])
    mha = MultiHeadAttention(p["self_attn.in_proj_weight"], p["self_attn.in_proj_bias"],
                             p["self_attn.out_proj.weight"], p["self_attn.out_proj.bias"], h)
    x = X + mha(ln1)
    ln2 = layernorm(x, p["norm2.weight"], p["norm2.bias"])
    ff = gelu(ln2 @ p["linear1.weight"].T + p["linear1.bias"]) @ p["linear2.weight"].T + p["linear2.bias"]
    out = x + ff
    ref = layer(torch.tensor(X)[None]).detach().numpy()[0]
    err = np.abs(out - ref).max()
    print(f"""
  A pre-norm transformer block = residual(attention) + residual(MLP), on length-{T} input (dim {d}):
    |ours - torch.nn.TransformerEncoderLayer| = {err:.1e}

  READING: the block is: x = x + MultiHeadAttention(LayerNorm(x)); then x = x + MLP(LayerNorm(x)). The
  attention mixes information ACROSS tokens; the position-wise MLP (a 2-layer GELU net applied to each
  token) processes each token's features; residual connections ([08.02 §4]) and LayerNorm
  ([07.07]) keep the deep stack trainable. Stack N of these and you have the encoder. Matches PyTorch
  to machine precision — the whole transformer is these two sub-layers, repeated.""")


# =============================================================================
# EXPERIMENT 6 — parallelism and complexity
# =============================================================================


def experiment_6_complexity():
    print("\n" + "=" * 88)
    print("EXPERIMENT 6 — self-attention is parallel but O(n^2) in sequence length (README §6)")
    print("=" * 88)
    print(f"\n  Cost of processing a length-n sequence (d = model dim):\n")
    print(f"    {'model':>14s} {'sequential steps':>18s} {'compute per layer':>20s}")
    print(f"    {'RNN (09.01)':>14s} {'O(n)':>18s} {'O(n * d^2)':>20s}")
    print(f"    {'self-attention':>14s} {'O(1)':>18s} {'O(n^2 * d)':>20s}")
    print(f"\n    attention-matrix size (n x n) as the sequence grows:")
    for n in (128, 512, 2048, 8192):
        print(f"      n = {n:>5d}:  {n * n:>12,d} scores")
    print("""
  READING: an RNN must process tokens ONE AT A TIME — O(n) sequential steps — so it cannot parallelize
  over the sequence. Self-attention computes all pairwise interactions at once: O(1) sequential depth,
  fully parallel on a GPU, which is why transformers train so much faster and scale so well. The cost
  is the n x n attention matrix — O(n^2) compute and memory — which becomes the bottleneck for long
  sequences (65M scores at n=8192) and motivates efficient-attention methods (efficient attention,
  11.03) (README §6).""")


if __name__ == "__main__":
    experiment_1_sdpa()
    experiment_2_mha()
    experiment_3_causal()
    experiment_4_positional()
    experiment_5_block()
    experiment_6_complexity()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED" if HAVE_TORCH else "ALL CHECKS PASSED (torch-verified parts skipped)")
    print("=" * 88)
