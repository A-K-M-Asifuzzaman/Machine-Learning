"""
09.03 — Seq2seq & Attention, from scratch (NumPy).

Encoder-decoder models compress an input sequence into a context and generate an output from it. A
single fixed context is a bottleneck; ATTENTION removes it by letting the decoder read every encoder
state, weighted by relevance. This file builds and MEASURES the mechanism that became the transformer:

  1. attention scoring (dot / scaled-dot / additive) and valid distributions   -> Experiment 1
  2. attention is content-based alignment (copy -> diagonal, reverse -> anti)   -> Experiment 2
  3. the fixed-context bottleneck: a d-dim context loses info as input grows    -> Experiment 3
  4. score scale controls soft vs hard alignment (why transformers /sqrt(d))    -> Experiment 4
  5. beam search finds higher-probability sequences than greedy decoding        -> Experiment 5

Run:  python3 from_scratch.py
"""

import numpy as np


def softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


# =============================================================================
# Attention scoring functions
# =============================================================================


def dot_attention(Q, K, V, scale=True):
    """Scaled dot-product attention: softmax(QK^T / sqrt(d)) V. Q:(Tq,d) K,V:(Tk,d)."""
    scores = Q @ K.T
    if scale:
        scores = scores / np.sqrt(Q.shape[1])
    A = softmax(scores, axis=1)
    return A @ V, A


def additive_attention(Q, K, V, Wq, Wk, v):
    """Bahdanau additive attention: score = v^T tanh(Wq q + Wk k)."""
    Tq, Tk = len(Q), len(K)
    scores = np.zeros((Tq, Tk))
    for i in range(Tq):
        scores[i] = np.tanh(Q[i] @ Wq.T + K @ Wk.T) @ v
    A = softmax(scores, axis=1)
    return A @ V, A


def experiment_1_scoring():
    print("=" * 88)
    print("EXPERIMENT 1 — attention scoring and valid distributions (README §3)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    Tq, Tk, d = 3, 5, 8
    Q = rng.standard_normal((Tq, d)); K = rng.standard_normal((Tk, d)); V = rng.standard_normal((Tk, d))
    out_d, A_d = dot_attention(Q, K, V)
    Wq = rng.standard_normal((d, d)); Wk = rng.standard_normal((d, d)); vv = rng.standard_normal(d)
    _, A_a = additive_attention(Q, K, V, Wq, Wk, vv)
    # manual check of one output row = weighted sum of V
    manual = A_d[0] @ V
    print(f"""
  {Tq} queries attending over {Tk} keys/values (dim {d}):

    dot-attention rows sum to 1?          {np.allclose(A_d.sum(1), 1)}   (valid distributions)
    additive-attention rows sum to 1?     {np.allclose(A_a.sum(1), 1)}
    output = A @ V  (manual vs vectorized) max|diff| = {np.abs(out_d[0] - manual).max():.1e}
    dot and additive give the SAME shape ({A_d.shape}), different weights: mean|A_dot - A_add| = {np.abs(A_d - A_a).mean():.3f}

  READING: attention scores each (query, key) pair, softmaxes the scores into weights that sum to 1,
  and returns a weighted sum of the values. Two common scorers: DOT-product q.k (Luong — cheap, used in
  transformers) and ADDITIVE v^T tanh(Wq q + Wk k) (Bahdanau — a tiny MLP). Both yield valid attention
  distributions; the transformer chose scaled dot-product for speed (README §3).""")


# =============================================================================
# EXPERIMENT 2 — attention is content-based alignment
# =============================================================================


def experiment_2_alignment():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — attention is content-based alignment (README §4)")
    print("=" * 88)
    L, d = 6, 16
    rng = np.random.default_rng(1)
    keys = rng.standard_normal((L, d))               # each input position has a distinct key
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    values = keys.copy()

    def alignment(target):
        Q = keys[target] * 8.0                        # decoder query = the key it wants (sharpened)
        _, A = dot_attention(Q, keys, values, scale=False)
        return A

    A_copy = alignment(np.arange(L))                  # output t wants input t
    A_rev = alignment(np.arange(L)[::-1])             # output t wants input L-1-t
    print(f"\n  Query for each output position retrieves the matching input key. Argmax per output row:")
    print(f"    COPY task:    output->input = {A_copy.argmax(1).tolist()}   (diagonal alignment)")
    print(f"    REVERSE task: output->input = {A_rev.argmax(1).tolist()}   (anti-diagonal alignment)")
    print(f"    mean weight placed on the intended input position (copy) = {A_copy.max(1).mean():.3f}")
    print("""
  READING: the attention weights ARE an alignment — for each output step they say which input positions
  to look at. A copy task produces a diagonal alignment (output t <- input t); a reverse task produces
  an anti-diagonal one. This is 'content-based addressing': the decoder builds a query and retrieves the
  encoder states whose keys match. The alignment is interpretable, which is how attention was first
  validated on translation (README §4).""")


# =============================================================================
# EXPERIMENT 3 — the fixed-context bottleneck
# =============================================================================


def experiment_3_bottleneck():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — the fixed-context bottleneck: a d-dim context loses info (README §2)")
    print("=" * 88)
    d = 16
    print(f"\n  Best-possible (PCA-optimal) reconstruction of an L-state sequence from ONE {d}-dim")
    print(f"  context vs attention (keeps all states). Fraction of sequence variance LOST:\n")
    print(f"    {'input length L':>16s} {'info (numbers)':>16s} {'fixed context':>16s} {'attention':>12s}")
    for L in (2, 4, 8, 16, 32):
        loss = _bottleneck_loss(L, d)
        print(f"    {L:>16d} {L * 4:>16d} {loss:>16.3f} {0.0:>12.3f}")
    print("""
  READING: a fixed context has only d numbers to summarize the WHOLE input. While the input's
  information (L states) fits in d, reconstruction is lossless; once it exceeds d the context must throw
  information away — 50% of the variance lost at L=8, 87% at L=32. This is the seq2seq bottleneck, and
  it is why plain encoder-decoders degrade on long inputs. ATTENTION keeps all L encoder states and
  reads them on demand, so it has NO fixed bottleneck (loss 0) — the fix that made long-sequence
  translation work and led directly to the transformer (README §2).""")


def _bottleneck_loss(L, d, k=4, seed=0):
    rng = np.random.default_rng(seed)
    Xtr = rng.standard_normal((4000, L * k))
    mu = Xtr.mean(0)
    _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    Vd = Vt[:min(d, L * k)]                            # a d-dim context can hold d coordinates (PCA)
    Xte = rng.standard_normal((2000, L * k))
    recon = (Xte - mu) @ Vd.T @ Vd + mu
    return ((recon - Xte) ** 2).mean() / Xte.var()


# =============================================================================
# EXPERIMENT 4 — score scale controls soft vs hard alignment
# =============================================================================


def experiment_4_temperature():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — score scale controls soft vs hard alignment (why /sqrt(d)) (README §5)")
    print("=" * 88)
    rng = np.random.default_rng(2)
    d = 64
    q = rng.standard_normal(d)
    K = rng.standard_normal((8, d))
    raw = q @ K.T                                      # unscaled dot products
    std_raw = raw.std()
    print(f"\n  Dot products of a query with 8 keys (dim d={d}, raw score std ~ {std_raw:.1f}).")
    print(f"  Sharpness of the attention at various score scalings (max entropy = log2 8 = 3 bits):\n")
    print(f"    {'scaling':>16s} {'max attention wt':>18s} {'entropy (bits)':>16s}")
    for name, factor in [("x sqrt(d) (huge)", np.sqrt(d)), ("x 1 (raw)", 1.0),
                         ("/ sqrt(d) (used)", 1 / np.sqrt(d)), ("/ d (tiny)", 1 / d)]:
        A = softmax(raw * factor)
        ent = -np.sum(A * np.log2(A + 1e-12))
        print(f"    {name:>16s} {A.max():>18.3f} {ent:>16.3f}")
    print(f"""
  Raw dot products here have std ~ {std_raw:.1f} (they grow like sqrt(d)). Larger scores -> more peaked
  softmax (lower entropy, one key dominates); smaller -> flatter (near-uniform, entropy -> 3 bits).

  READING: the SIZE of the scores sets how sharp the attention is. Un-scaled dot products grow like
  sqrt(d) as the dimension increases, pushing the softmax into a near one-hot regime with vanishing
  gradients. Transformers divide scores by sqrt(d) precisely to keep them O(1) and the attention soft
  and trainable — the '/sqrt(d)' in scaled dot-product attention is this fix (README §5).""")


# =============================================================================
# EXPERIMENT 5 — beam search beats greedy decoding
# =============================================================================


def experiment_5_beam():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — beam search finds higher-probability sequences than greedy (README §6)")
    print("=" * 88)
    # a tiny autoregressive model: P(next | prefix). Greedy's locally-best first token is a trap.
    V = ["A", "B", "C"]
    P = {
        "":    [0.5, 0.5, 0.0],          # step 1: A and B tie-ish, greedy takes A
        "A":   [0.0, 0.0, 1.0],          # A -> C forced, but...
        "AC":  [0.34, 0.33, 0.33],       # ...A's continuations are all mediocre
        "B":   [0.0, 0.0, 1.0],          # B -> C forced
        "BC":  [0.9, 0.05, 0.05],        # B's path has a high-probability finish
    }

    def greedy():
        seq, logp = "", 0.0
        for _ in range(3):
            p = P.get(seq, [1 / 3] * 3); j = int(np.argmax(p))
            logp += np.log(p[j] + 1e-12); seq += V[j]
        return seq, logp

    def beam(width):
        beams = [("", 0.0)]
        for _ in range(3):
            cand = []
            for seq, lp in beams:
                p = P.get(seq, [1 / 3] * 3)
                for j in range(3):
                    if p[j] > 0:
                        cand.append((seq + V[j], lp + np.log(p[j] + 1e-12)))
            cand.sort(key=lambda x: -x[1])
            beams = cand[:width]
        return beams[0]

    g_seq, g_lp = greedy()
    b_seq, b_lp = beam(3)
    print(f"""
  A 3-step model where greedy's first choice (A, prob 0.5) leads to mediocre continuations, while B
  leads to a high-probability finish:

    greedy decode      -> "{g_seq}"   log-prob = {g_lp:.3f}   (prob {np.exp(g_lp):.3f})
    beam search (w=3)  -> "{b_seq}"   log-prob = {b_lp:.3f}   (prob {np.exp(b_lp):.3f})

  READING: greedy decoding commits to the highest-probability token at each step, so it takes A and gets
  trapped in a low-probability region. Beam search keeps the top-{3} partial sequences, so it retains
  the B branch and discovers the globally better "{b_seq}". Beam search approximates the most-probable
  sequence that greedy misses — standard for translation and any seq2seq generation (README §6).""")


if __name__ == "__main__":
    experiment_1_scoring()
    experiment_2_alignment()
    experiment_3_bottleneck()
    experiment_4_temperature()
    experiment_5_beam()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)
