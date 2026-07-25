# 11.03 — References: Efficient Attention

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2 | KV cache | folklore; Pope et al. (2022) |
| §3 | MQA / GQA | Shazeer (2019); Ainslie et al. (2023) |
| §4 | RoPE | Su et al. (2021) |
| §4 | ALiBi | Press et al. (2022) |
| §5 | FlashAttention | Dao et al. (2022, 2023) |
| §6 | Sparse / linear / SSM | Child et al. (2019); Katharopoulos et al. (2020); Gu & Dao (2023) |
| §7 | Long context, lost-in-the-middle | Liu et al. (2023); Peng et al. (2023, YaRN) |

---

## KV cache and grouped attention

- **Pope, R. et al. (2022).** "Efficiently Scaling Transformer Inference." — the KV cache, batching, and
  serving analysis (§2). <https://arxiv.org/abs/2211.05102>.
- **Shazeer, N. (2019).** "Fast Transformer Decoding: One Write-Head is All You Need" (**MQA**). — the
  single shared K,V head (§3). <https://arxiv.org/abs/1911.02150>.
- **Ainslie, J. et al. (2023).** "GQA: Training Generalized Multi-Query Transformer Models from
  Multi-Head Checkpoints." *EMNLP*. — **Grouped-Query Attention** (§3). <https://arxiv.org/abs/2305.13245>.

## Positional methods

- **Su, J. et al. (2021).** "RoFormer: Enhanced Transformer with Rotary Position Embedding" (**RoPE**).
  — rotary embeddings and the relative-position property (§4). <https://arxiv.org/abs/2104.09864>.
- **Press, O., Smith, N. & Lewis, M. (2022).** "Train Short, Test Long: Attention with Linear Biases
  Enables Input Length Extrapolation" (**ALiBi**). *ICLR*. (§4). <https://arxiv.org/abs/2108.12409>.
- **Peng, B. et al. (2023).** "YaRN: Efficient Context Window Extension of Large Language Models." —
  RoPE frequency scaling to extend context (§7). <https://arxiv.org/abs/2309.00071>.

## FlashAttention and complexity reduction

- **Dao, T. et al. (2022).** "FlashAttention: Fast and Memory-Efficient Exact Attention with
  IO-Awareness." *NeurIPS*. — the online-softmax tiling (§5). <https://arxiv.org/abs/2205.14135>.
- **Dao, T. (2023).** "FlashAttention-2." — the faster follow-up. <https://arxiv.org/abs/2307.08691>.
- **Child, R. et al. (2019).** "Generating Long Sequences with Sparse Transformers." (§6).
  <https://arxiv.org/abs/1904.10509>.
- **Beltagy, I. et al. (2020).** "Longformer: The Long-Document Transformer." — windowed + global
  attention (§6). <https://arxiv.org/abs/2004.05150>.
- **Katharopoulos, A. et al. (2020).** "Transformers are RNNs: Fast Autoregressive Transformers with
  Linear Attention." *ICML*. (§6). <https://arxiv.org/abs/2006.16236>.
- **Gu, A. & Dao, T. (2023).** "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." —
  the state-space challenger to attention (§6). <https://arxiv.org/abs/2312.00752>.

## Long-context behavior

- **Liu, N. et al. (2023).** "Lost in the Middle: How Language Models Use Long Contexts." *TACL*. —
  models under-use the middle of long prompts (§7). <https://arxiv.org/abs/2307.03172>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`flash-attention`](https://github.com/Dao-AILab/flash-attention) | the reference CUDA kernels |
| [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | dispatches to FlashAttention when available |
| [vLLM PagedAttention](https://github.com/vllm-project/vllm) | production KV-cache management (§2, §7) |
| [`transformers` RoPE/GQA](https://github.com/huggingface/transformers) | RoPE and GQA in Llama/Mistral code |

---

## Deferred to later chapters

- **The attention it optimizes** → [11.01](../01-transformer/)
- **Scaling laws and architecture** → [11.04](../04-scaling-and-architecture/)
- **Inference and serving (KV cache, batching)** → [11.07](../07-inference/)
- **Windowed attention in vision** → [08.05](../../08-computer-vision/05-vision-transformers/)
