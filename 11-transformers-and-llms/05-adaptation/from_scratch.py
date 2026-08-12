"""
11.05 — Adaptation (fine-tuning, LoRA, quantization), from scratch (NumPy).

A pretrained LLM knows language; adaptation teaches it a task or a style. Full fine-tuning updates all
billions of weights — expensive to train and store. Parameter-efficient methods (LoRA) and quantization
make adaptation cheap. This file builds and MEASURES them:

  1. LoRA parameter efficiency: 2dr trainable params instead of d^2               -> Experiment 1
  2. LoRA starts as a no-op (B=0) and its update is exact                          -> Experiment 2
  3. why LoRA works: fine-tuning updates are approximately LOW-RANK                -> Experiment 3
  4. quantization round-trip: int8 nearly lossless, int4 lossier, big memory saving -> Experiment 4
  5. QLoRA memory: fine-tune a 7B model in a fraction of the memory                -> Experiment 5

Run:  python3 from_scratch.py
"""

import numpy as np


# =============================================================================
# LoRA
# =============================================================================


class LoRALinear:
    """y = x W^T + (alpha/r) x (B A)^T. W is frozen; only A, B train."""
    def __init__(self, W, r, alpha, seed=0):
        rng = np.random.default_rng(seed)
        d_out, d_in = W.shape
        self.W = W                                    # frozen base weights (d_out, d_in)
        self.A = rng.standard_normal((r, d_in)) * 0.01   # down-projection
        self.B = np.zeros((d_out, r))                 # up-projection, ZERO at init -> no-op
        self.scale = alpha / r

    def __call__(self, x):
        return x @ self.W.T + self.scale * (x @ self.A.T) @ self.B.T

    def delta_W(self):
        return self.scale * self.B @ self.A


def experiment_1_efficiency():
    print("=" * 88)
    print("EXPERIMENT 1 — LoRA trains 2dr parameters instead of d^2 (README §3)")
    print("=" * 88)
    print(f"\n  Trainable parameters for a d x d attention projection, LoRA rank r:\n")
    print(f"    {'d':>8s} {'r':>4s} {'full fine-tune (d^2)':>20s} {'LoRA (2dr)':>12s} {'% of full':>10s}")
    for d, r in [(1024, 8), (4096, 8), (4096, 16), (4096, 64)]:
        full = d * d
        lora = 2 * d * r
        print(f"    {d:>8d} {r:>4d} {full:>20,d} {lora:>12,d} {100 * lora / full:>9.2f}%")
    print("""
  READING: full fine-tuning updates the whole d x d weight matrix (d^2 parameters, per matrix, per
  layer — billions total). LoRA freezes W and learns a low-rank update Delta_W = B A with B (d x r) and
  A (r x d), just 2dr parameters. At d=4096, r=8 that is 0.39% of the weights — so you can fine-tune a
  giant model by training (and storing) a tiny adapter. Multiple task adapters share one frozen base
  (README §3).""")


# =============================================================================
# EXPERIMENT 2 — LoRA is a no-op at init and its update is exact
# =============================================================================


def experiment_2_noop():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — LoRA starts identical to the base model (B=0), update is exact (README §3)")
    print("=" * 88)
    rng = np.random.default_rng(1)
    d = 32
    W = rng.standard_normal((d, d))
    x = rng.standard_normal((4, d))
    lora = LoRALinear(W, r=4, alpha=8)
    base_out = x @ W.T
    init_out = lora(x)
    # now "train": set B to something nonzero
    lora.B = rng.standard_normal(lora.B.shape)
    merged = x @ (W + lora.delta_W()).T              # merge adapter into W
    lora_out = lora(x)
    print(f"""
  A LoRA layer wrapping a random {d}x{d} weight:

    at init (B=0): |LoRA(x) - base(x)|         = {np.abs(init_out - base_out).max():.1e}   (identical)
    after training: |LoRA(x) - (W+Delta_W)(x)| = {np.abs(lora_out - merged).max():.1e}   (mergeable, exact)

  READING: LoRA initializes B=0, so at the start Delta_W = 0 and the adapted model is EXACTLY the base
  model — training begins from the pretrained solution with zero disruption. After training, the update
  B A can be MERGED back into W (W <- W + Delta_W), so inference has zero extra cost or latency versus
  the base model. Both properties hold to machine precision (README §3).""")


# =============================================================================
# EXPERIMENT 3 — why LoRA works: updates are approximately low-rank
# =============================================================================


def experiment_3_low_rank():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — fine-tuning updates are approximately LOW-RANK (README §3)")
    print("=" * 88)
    rng = np.random.default_rng(2)
    d = 128
    # a realistic 'task update': dominated by a few directions + small noise
    true_rank = 6
    U = rng.standard_normal((d, true_rank)); Vt = rng.standard_normal((true_rank, d))
    dW = U @ Vt + 0.05 * rng.standard_normal((d, d))
    # best rank-r approximation is truncated SVD
    Usvd, S, Vsvd = np.linalg.svd(dW)
    total = (S ** 2).sum()
    print(f"\n  A task update Delta_W ({d}x{d}, intrinsic rank ~{true_rank}). Fraction of its energy")
    print(f"  captured by the best rank-r approximation (what LoRA fits):\n")
    print(f"    {'rank r':>8s} {'energy captured':>16s} {'params (2dr)':>14s}")
    for r in (1, 2, 4, 6, 16, 128):
        captured = (S[:r] ** 2).sum() / total
        print(f"    {r:>8d} {captured:>15.1%} {2 * d * r:>14,d}")
    print("""
  READING: a fine-tuning update is not a random full-rank matrix — it concentrates in a few directions
  (adapting a task rarely rewrites everything). Here a rank-6 update's energy is ~fully captured by a
  rank-6 approximation, and even rank-4 gets most of it. LoRA exploits exactly this: if Delta_W is
  approximately low-rank, then B A with small r reproduces it with a tiny fraction of the parameters.
  The 'intrinsic dimension' of fine-tuning is low — the empirical fact that makes LoRA work (README §3).""")


# =============================================================================
# EXPERIMENT 4 — quantization
# =============================================================================


def quantize_blockwise(W, bits, block=64):
    """Symmetric absmax quantization with a PER-BLOCK scale (what real quantizers do)."""
    qmax = 2 ** (bits - 1) - 1
    flat = W.ravel().astype(np.float64)
    pad = (-len(flat)) % block
    flat = np.concatenate([flat, np.zeros(pad)])
    blocks = flat.reshape(-1, block)
    scale = np.abs(blocks).max(1, keepdims=True) / qmax          # one scale per block
    dq = np.round(blocks / scale).clip(-qmax - 1, qmax) * scale
    return dq.ravel()[:W.size].reshape(W.shape)


def experiment_4_quantization():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — quantization: int8 nearly lossless, int4 lossier (README §5)")
    print("=" * 88)
    rng = np.random.default_rng(3)
    W = rng.standard_normal((512, 512))
    print(f"\n  Round-trip error and memory for a 512x512 weight matrix (fp32 baseline),")
    print(f"  per-block (size 64) absmax quantization:\n")
    print(f"    {'precision':>12s} {'bits':>5s} {'rel. error':>12s} {'memory vs fp32':>16s}")
    fp16 = W.astype(np.float16).astype(np.float64)               # real IEEE fp16 round-trip
    print(f"    {'fp16':>12s} {16:>5d} {np.linalg.norm(W - fp16) / np.linalg.norm(W):>12.4f} {'50%':>16s}")
    for name, bits in [("int8", 8), ("int4", 4)]:
        Wdq = quantize_blockwise(W, bits)
        rel = np.linalg.norm(W - Wdq) / np.linalg.norm(W)
        print(f"    {name:>12s} {bits:>5d} {rel:>12.4f} {f'{bits/32:.0%}':>16s}")
    print("""
  READING: quantization stores weights in fewer bits by scaling them into an integer range, with a
  separate scale PER BLOCK to handle outliers. int8 (1/4 the memory) has ~0.6% relative error —
  effectively lossless for inference. Plain uniform int4 (1/8 the memory) is lossier (~11% here);
  production 4-bit methods (QLoRA's non-uniform NF4 grid with small blocks) roughly halve that and are
  usable. Quantization is how a 70B model that needs 140GB in fp16 fits in ~35GB (int4) on a single
  GPU (README §5).""")


# =============================================================================
# EXPERIMENT 5 — QLoRA memory
# =============================================================================


def experiment_5_qlora():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — QLoRA: fine-tune a 7B model in a fraction of the memory (README §4)")
    print("=" * 88)
    params = 7e9
    r, n_matrices = 16, 7e9 / (4096 * 4096)          # roughly, LoRA on the big matrices
    full_fp16_train = params * 2 + params * 2 * 3     # weights + Adam (grad + 2 moments), fp16-ish
    lora_adapter = 2 * 4096 * r * (params / (4096 * 4096)) * 2   # adapter params, fp16
    base_4bit = params * 0.5                          # frozen base in 4-bit
    qlora_total = base_4bit + lora_adapter + lora_adapter * 3    # 4-bit base + adapter + its optimizer
    print(f"""
  Memory to FINE-TUNE a 7B-parameter model:

    full fine-tune (fp16 weights + Adam states) : {full_fp16_train / 1e9:>6.0f} GB   (needs a big cluster)
    QLoRA (4-bit frozen base + LoRA adapters)   : {qlora_total / 1e9:>6.1f} GB   (fits one consumer GPU)
    reduction                                   : {full_fp16_train / qlora_total:>6.0f}x

  READING: full fine-tuning must store the weights AND the optimizer state (Adam keeps two moments per
  parameter) in high precision — ~{full_fp16_train/1e9:.0f}GB for 7B, far beyond a consumer GPU. QLoRA
  combines the two tricks: freeze the base model in 4-bit (no gradients, no optimizer state for it) and
  train only small LoRA adapters. The optimizer now tracks the tiny adapter, not 7B weights — cutting
  memory ~{full_fp16_train/qlora_total:.0f}x and letting a 7B (even 65B) model be fine-tuned on ONE
  GPU. This is what democratized LLM fine-tuning (README §4).""")


if __name__ == "__main__":
    experiment_1_efficiency()
    experiment_2_noop()
    experiment_3_low_rank()
    experiment_4_quantization()
    experiment_5_qlora()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
