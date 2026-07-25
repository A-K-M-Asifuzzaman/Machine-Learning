# 11.01 — Exercises: The Transformer

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Write scaled dot-product attention and explain each factor: $QK^\top$, $1/\sqrt{d_k}$, softmax,
and the multiply by $V$.

**D2.** Show that the entries of $QK^\top$ have variance $\propto d_k$ for unit-variance $q, k$, and
hence justify the $1/\sqrt{d_k}$ scaling.

**D3.** Derive the gradient of the attention output w.r.t. $Q$, $K$, and $V$.

**D4.** Show self-attention is permutation-equivariant, and explain why positional encodings are
therefore required.

**D5.** Derive the multi-head formulation and show that splitting $d$ into $h$ heads of $d/h$ keeps the
total compute the same as single-head attention over $d$.

**D6.** Show that a causal mask (adding $-\infty$ to the upper triangle before softmax) makes token $i$
depend only on tokens $\le i$, and why each row still sums to 1.

**D7.** Prove the relative-position property of sinusoidal encodings: $\text{PE}_{pos+k}$ is a linear
function of $\text{PE}_{pos}$.

**D8.** Derive the per-layer time and memory complexity of self-attention ($O(n^2 d)$) and contrast with
an RNN ($O(n d^2)$, $O(n)$ sequential).

**D9.** Explain pre-norm vs post-norm and why pre-norm is more stable for deep stacks (gradient path
argument).

**D10.** Explain the role of the position-wise MLP and why alternating it with attention (mix-across
vs process-within) is the core design.

---

## Tier 2 — Implementation

**I1.** Implement scaled dot-product attention (with optional mask); verify against
`F.scaled_dot_product_attention` (Experiment 1).

**I2.** Implement multi-head attention with packed QKV; verify against `torch.nn.MultiheadAttention`
(Experiment 2).

**I3.** Implement causal masking and reproduce Experiment 3 (0 weight on the future).

**I4.** Implement sinusoidal positional encoding and reproduce Experiment 4 (distinct positions, decaying
dot product).

**I5.** Implement a full pre-norm transformer block and verify against `nn.TransformerEncoderLayer`
(Experiment 5).

**I6.** Stack $N$ blocks into an encoder and a (causally masked) decoder; run a forward pass.

**I7.** Add learned positional embeddings and compare to sinusoidal on a toy task.

**I8.** Train a tiny character-level GPT (decoder-only) on a small corpus and sample from it.

**I9.** Visualize attention maps of a trained model and interpret what different heads attend to.

**I10.** *(Complexity.)* Measure wall-clock time vs sequence length and confirm the $O(n^2)$ scaling.

---

## Tier 3 — Interview

**Q1.** What is self-attention and how does it differ from seq2seq attention?

**Q2.** Walk through scaled dot-product attention. Why divide by $\sqrt{d_k}$?

**Q3.** What is multi-head attention and why use multiple heads?

**Q4.** What is the difference between an encoder and a decoder transformer?

**Q5.** Why do transformers need positional encodings?

**Q6.** What is a causal mask and what does it enable?

**Q7.** What does a transformer block consist of?

**Q8.** Why are residual connections and LayerNorm needed?

**Q9.** What is the computational complexity of attention, and why does it matter?

**Q10.** Why did transformers replace RNNs?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive scaled dot-product attention and the $\sqrt{d_k}$ scaling
- [ ] Implement multi-head attention verified against PyTorch
- [ ] Explain and implement causal masking
- [ ] Justify positional encodings and their relative-position property
- [ ] Build a full transformer block from the two sub-layers
- [ ] Explain the parallelism and $O(n^2)$ trade-off
- [ ] Explain why the transformer scales where the RNN could not
