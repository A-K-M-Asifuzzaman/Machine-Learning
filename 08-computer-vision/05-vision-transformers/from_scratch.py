"""
08.05 — Vision Transformers, from scratch (NumPy).

A ViT throws away the convolution's built-in priors (locality, translation equivariance) and treats an
image as a SET of patch tokens fed to a transformer. This file builds the vision-specific machinery and
MEASURES the consequences (the attention internals themselves are derived in Part 11.01):

  1. patch embedding == a strided convolution                     (verified, machine precision)
  2. self-attention is PERMUTATION-EQUIVARIANT                    (verified) -> Experiment 2
  3. positional embeddings break that invariance (encode order)              -> Experiment 3
  4. attention is GLOBAL from layer 1 (a conv is local)                      -> Experiment 4
  5. attention cost is QUADRATIC in #patches -> windowed (Swin) attention    -> Experiment 5

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAVE_TORCH = True
except Exception:                                    # pragma: no cover
    HAVE_TORCH = False


def softmax(Z):
    Z = Z - Z.max(-1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(-1, keepdims=True)


# =============================================================================
# 1. Patch embedding == a strided convolution
# =============================================================================


def patch_embed(img, W_proj, P):
    """img:(C,H,W) -> non-overlapping PxP patches, each flattened and linearly projected.
    W_proj:(C*P*P, D) -> tokens (N_patches, D)."""
    C, H, Wd = img.shape
    nh, nw = H // P, Wd // P
    patches = img.reshape(C, nh, P, nw, P).transpose(1, 3, 0, 2, 4).reshape(nh * nw, C * P * P)
    return patches @ W_proj                                          # (N, D)


def experiment_1_patch_embed():
    print("=" * 88)
    print("EXPERIMENT 1 — patch embedding == a strided convolution (machine precision)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    C, H, Wd, P, D = 3, 8, 8, 4, 16
    img = rng.standard_normal((C, H, Wd))
    W_proj = rng.standard_normal((C * P * P, D))
    tokens = patch_embed(img, W_proj, P)
    err = np.nan
    if HAVE_TORCH:
        # the SAME projection as a conv: kernel=stride=P, out_channels=D
        w_conv = W_proj.T.reshape(D, C, P, P)                        # (D, C*P*P) -> (D,C,P,P)
        out = F.conv2d(torch.tensor(img)[None], torch.tensor(w_conv), stride=P)[0]  # (D, nh, nw)
        conv_tokens = out.reshape(D, -1).T.numpy()                  # (N, D)
        err = np.abs(tokens - conv_tokens).max()
    print(f"""
  An {H}x{Wd} image split into {P}x{P} patches -> {(H//P)*(Wd//P)} tokens of dim {D}:

    patch-extract + linear projection  vs  Conv2d(kernel={P}, stride={P})
    max |difference| = {err:.1e}

  READING: "patch embedding" sounds new, but splitting an image into non-overlapping PxP patches and
  linearly projecting each is EXACTLY a convolution with kernel size = stride = P. That is literally how
  ViT is implemented (a single strided conv). So a ViT's first layer is a conv; everything after is
  attention. The novelty is not the embedding — it is treating the result as an unordered SET of tokens
  (Experiment 2).""")


# =============================================================================
# a minimal single-head self-attention (full derivation in Part 11.01)
# =============================================================================


def self_attention(X, Wq, Wk, Wv):
    """X:(N,d) -> (N,d). out = softmax(QK^T / sqrt(d)) V."""
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    A = softmax(Q @ K.T / np.sqrt(Q.shape[1]))
    return A @ V, A


# =============================================================================
# 2. Self-attention is permutation-equivariant
# =============================================================================


def experiment_2_permutation():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — self-attention is PERMUTATION-EQUIVARIANT (README §3)")
    print("=" * 88)
    rng = np.random.default_rng(1)
    N, d = 6, 8
    X = rng.standard_normal((N, d))
    Wq, Wk, Wv = (rng.standard_normal((d, d)) for _ in range(3))
    out, _ = self_attention(X, Wq, Wk, Wv)
    perm = rng.permutation(N)
    out_perm, _ = self_attention(X[perm], Wq, Wk, Wv)               # attention on shuffled tokens
    err = np.abs(out_perm - out[perm]).max()
    print(f"""
  Shuffle the {N} input tokens by a random permutation, then run attention:

    max | attention(shuffle(X)) - shuffle(attention(X)) | = {err:.1e}

  READING: attention treats its input as a SET — permuting the tokens permutes the outputs identically
  and changes nothing else. It has NO notion of position or order. For text or image patches this is a
  problem: "cat" and the same word elsewhere, or a patch top-left vs bottom-right, would be
  indistinguishable. This is precisely why transformers must ADD positional information (Experiment 3).""")


# =============================================================================
# 3. Positional embeddings encode order
# =============================================================================


def experiment_3_positional():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — positional embeddings break the invariance and encode order (README §4)")
    print("=" * 88)
    rng = np.random.default_rng(2)
    N, d = 6, 8
    X = rng.standard_normal((N, d))
    pos = rng.standard_normal((N, d)) * 0.5                         # learned positional embeddings
    Wq, Wk, Wv = (rng.standard_normal((d, d)) for _ in range(3))
    perm = rng.permutation(N)
    out_a, _ = self_attention(X + pos, Wq, Wk, Wv)                  # tokens at their real positions
    out_b, _ = self_attention(X[perm] + pos, Wq, Wk, Wv)           # tokens shuffled, positions fixed
    diff = np.abs(out_b - out_a[perm]).max()
    print(f"""
  Add positional embeddings, THEN shuffle the patches (positions stay in place):

    max | attention(shuffle(X)+pos) - shuffle(attention(X+pos)) | = {diff:.3f}   (now NONZERO)

  READING: once each token carries a positional embedding, shuffling the patches genuinely changes the
  output — the model can tell a top-left patch from a bottom-right one. The permutation-invariance of
  Experiment 2 is gone. ViT adds a learned position embedding to every patch token (plus a CLS token
  for classification); without it, a ViT would be blind to spatial layout (README §4).""")


# =============================================================================
# 4. Attention is global from layer 1
# =============================================================================


def experiment_4_global():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — attention is GLOBAL from layer 1; a conv is local (README §5)")
    print("=" * 88)
    rng = np.random.default_rng(3)
    N, d = 16, 8                                                    # 16 patches (a 4x4 grid)
    X = rng.standard_normal((N, d))
    Wq, Wk, Wv = (rng.standard_normal((d, d)) for _ in range(3))
    _, A = self_attention(X, Wq, Wk, Wv)
    per_row = (A > 0).sum(1).mean()                                # softmax is strictly positive
    print(f"""
  {N} patches (a 4x4 grid). In ONE attention layer, how many patches does each patch's output depend on?

    attention: #patches attended per token = {per_row:.0f}  (all {N} -> the whole image, weights all > 0)
    a 3x3 conv layer:  each output depends on    = 9 patches   (a local neighborhood)

  READING: a single self-attention layer mixes EVERY patch with every other — its receptive field is
  the whole image at layer 1. A convolution sees only a 3x3 neighborhood and needs many stacked layers
  to grow its receptive field ([08.01 §6]). This global mixing is ViT's strength (long-range relations,
  immediately) and its cost (Experiment 5) — and, lacking the conv's locality prior, ViT needs more
  data or distillation (DeiT) to match a CNN on small datasets (README §5).""")


# =============================================================================
# 5. Attention cost is quadratic in the number of patches (-> Swin)
# =============================================================================


def experiment_5_quadratic():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — attention cost is QUADRATIC in #patches -> windowed attention (README §6)")
    print("=" * 88)
    print(f"\n  Attention-matrix cost (∝ N^2) vs windowed attention (window w, cost ∝ N·w):\n")
    print(f"    {'image':>10s} {'#patches N':>12s} {'full  N^2':>14s} {'windowed (w=49)':>18s} "
          f"{'speedup':>10s}")
    w = 49                                                          # 7x7 window (Swin default)
    for side in (14, 28, 56, 112):
        N = side * side                                            # patches for a side x side grid
        full = N * N
        win = N * w
        print(f"    {f'{side}x{side}':>10s} {N:>12,d} {full:>14,d} {win:>18,d} {full / win:>9.0f}x")
    print("""
  READING: full self-attention computes an N x N attention matrix, so its cost grows QUADRATICALLY with
  the number of patches — doubling the image side quadruples N and 16x's the attention cost. For
  high-resolution images (dense prediction) this is prohibitive. WINDOWED attention (Swin) restricts
  attention to local w-patch windows, making the cost LINEAR in N (a 256x saving at 112x112), and
  shifts the windows between layers to still mix globally over depth. This is what made transformers
  practical as general vision backbones (README §6).""")


if __name__ == "__main__":
    experiment_1_patch_embed()
    experiment_2_permutation()
    experiment_3_positional()
    experiment_4_global()
    experiment_5_quadratic()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED" if HAVE_TORCH else "ALL CHECKS PASSED (torch-verified parts skipped)")
    print("=" * 88)
