# 08.05 — Exercises: Vision Transformers

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Show that patch embedding (extract $P\times P$ patches, flatten, linear-project) is identical to
a `Conv2d` with kernel = stride = $P$. Give the weight-reshaping that maps one to the other.

**D2.** Prove that single-head self-attention is permutation-equivariant:
$\text{attn}(\Pi X) = \Pi\,\text{attn}(X)$ for any permutation matrix $\Pi$. Where does the proof use
that $Q, K, V$ are computed per-token?

**D3.** Show that adding position-dependent embeddings breaks permutation equivariance, and explain why
that is necessary for vision.

**D4.** Count the number of tokens $N$ for an $H\times W$ image with patch size $P$, and the FLOPs of one
self-attention layer as a function of $N$ and $D$. Identify the $O(N^2)$ term.

**D5.** Derive the cost of windowed attention with window size $w$ patches and show it is linear in $N$.
Compute the speedup vs full attention.

**D6.** Explain the inductive-bias argument: why does a CNN's locality + translation-equivariance prior
make it more data-efficient, and why can a ViT surpass it with enough data?

**D7.** Explain the role of the `[CLS]` token and contrast classifying from it vs global-average-pooling
the patch tokens.

**D8.** Explain how shifted windows (Swin) recover global information despite local attention, and why
shifting is needed.

**D9.** Describe MAE's masked-reconstruction objective and why masking 75% of patches is a good pretext
task for a ViT.

**D10.** Explain CLIP's contrastive objective and how it enables zero-shot classification.

---

## Tier 2 — Implementation

**I1.** Implement patch embedding and verify it equals `Conv2d(kernel=P, stride=P)` to machine precision
(Experiment 1).

**I2.** Implement single-head self-attention and verify permutation equivariance (Experiment 2).

**I3.** Add learned positional embeddings and reproduce Experiment 3 (shuffling now changes the output).

**I4.** Reproduce Experiment 4: show a single attention layer attends to all patches (dense attention
matrix), and contrast with a conv's local receptive field.

**I5.** Reproduce Experiment 5: tabulate full vs windowed attention cost across image sizes.

**I6.** Implement multi-head attention and a full transformer encoder block (attention + MLP +
LayerNorm + residual) and run a forward pass on patch tokens.

**I7.** Implement a `[CLS]` token and a classification head; build a tiny ViT and forward-pass an image.

**I8.** Implement windowed attention (partition tokens into windows, attend within each) and measure the
speedup vs full attention.

**I9.** Implement MAE-style masking (randomly drop 75% of patch tokens) and a reconstruction loss on a
toy dataset.

**I10.** *(Comparison.)* Train a small ViT and a small CNN on a limited-data image task and compare
sample efficiency, illustrating the inductive-bias trade.

---

## Tier 3 — Interview

**Q1.** How does a Vision Transformer process an image?

**Q2.** What is patch embedding, and how does it relate to convolution?

**Q3.** Why do transformers need positional embeddings?

**Q4.** What is the difference in inductive bias between a CNN and a ViT?

**Q5.** Why do ViTs need more data than CNNs, and how did DeiT reduce that?

**Q6.** Why is self-attention expensive for high-resolution images?

**Q7.** What problem does Swin's windowed attention solve, and how?

**Q8.** What is the `[CLS]` token?

**Q9.** How does MAE (or DINO) pretrain a ViT without labels?

**Q10.** What is CLIP and how does it enable zero-shot classification?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Describe the full ViT pipeline patch → token → transformer → head
- [ ] Show patch embedding is a strided convolution
- [ ] Prove attention's permutation equivariance and explain why position embeddings are needed
- [ ] Explain the global receptive field and the inductive-bias/data trade
- [ ] Compute attention's quadratic cost and windowed attention's linear cost
- [ ] Explain when ViTs beat CNNs and when they don't
- [ ] Describe MAE / DINO / CLIP self-supervised pretraining
