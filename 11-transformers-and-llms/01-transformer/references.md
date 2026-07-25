# 11.01 — References: The Transformer

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§7 | The transformer | Vaswani et al. (2017) |
| §2 | Scaled dot-product attention | Vaswani et al. (2017) §3.2.1 |
| §3 | Multi-head attention | Vaswani et al. (2017) §3.2.2 |
| §5 | Positional encodings | Vaswani et al. (2017) §3.5 |
| §6 | Pre-norm vs post-norm | Xiong et al. (2020); Baevski & Auli (2019) |
| §6 | LayerNorm, residuals | Ba et al. (2016); He et al. (2016) |

---

## The paper

**Vaswani, A. et al. (2017). "Attention Is All You Need." *NeurIPS*.** — the transformer: scaled
dot-product attention, multi-head attention, positional encodings, the encoder–decoder block, and the
parallelism argument. Everything in this chapter derives from it. Free at
<https://arxiv.org/abs/1706.03762>.

---

## Understanding and implementing it

- **Alammar, J. (2018).** "The Illustrated Transformer." — the standard visual explanation.
  <https://jalammar.github.io/illustrated-transformer/>.
- **Rush, A. et al. "The Annotated Transformer."** — a line-by-line PyTorch implementation of the paper.
  <https://nlp.seas.harvard.edu/annotated-transformer/>.
- **Karpathy, A. "Let's build GPT" / nanoGPT.** — building a decoder-only transformer from scratch.
  <https://github.com/karpathy/nanoGPT>, <https://www.youtube.com/watch?v=kCc8FmEb1nY>.
- **Phuong, M. & Hutter, M. (2022).** "Formal Algorithms for Transformers." — precise pseudocode for
  every component. <https://arxiv.org/abs/2207.09238>.

## Architecture refinements

- **Ba, J., Kiros, J. & Hinton, G. (2016).** "Layer Normalization." — the normalization used in
  transformers (§6). <https://arxiv.org/abs/1607.06450>.
- **Xiong, R. et al. (2020).** "On Layer Normalization in the Transformer Architecture." *ICML*. —
  **pre-norm vs post-norm** and training stability (§6). <https://arxiv.org/abs/2002.04745>.
- **He, K. et al. (2016).** "Deep Residual Learning." — the residual connections the block relies on
  (§6). <https://arxiv.org/abs/1512.03385>.
- **Hendrycks, D. & Gimpel, K. (2016).** "Gaussian Error Linear Units (GELUs)." — the MLP activation
  (§6). <https://arxiv.org/abs/1606.08415>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | the attention op, verified against here |
| [`torch.nn.MultiheadAttention`](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py) | reference MHA, verified against here |
| [`torch.nn.TransformerEncoderLayer`](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py) | the full block, verified against here |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | a clean, complete decoder-only transformer |

---

## Deferred to later chapters

- **Pretraining objectives (BERT/GPT/T5)** → [11.02](../02-pretraining/)
- **Efficient attention, KV cache, RoPE/ALiBi, long context** → [11.03](../03-efficient-attention/)
- **Seq2seq attention, the ancestor** → [09.03](../../09-sequence-models/03-seq2seq-and-attention/)
- **Vision transformers** → [08.05](../../08-computer-vision/05-vision-transformers/)
- **Normalization and residuals** → [07.07](../../07-deep-learning/07-normalization/), [08.02](../../08-computer-vision/02-cnn-architectures/)
