# 11.07 — References: Inference & Serving

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2 | Nucleus (top-p) sampling | Holtzman et al. (2020) |
| §3 | Speculative decoding | Leviathan et al. (2023); Chen et al. (2023) |
| §4-§5 | Batching, PagedAttention | Kwon et al. (2023, vLLM); Yu et al. (2022, Orca) |
| §1, §4 | Serving efficiency / roofline | Pope et al. (2022) |

---

## Decoding

- **Holtzman, A. et al. (2020).** "The Curious Case of Neural Text Degeneration." *ICLR*. — **top-p
  (nucleus) sampling** and why greedy/beam produce degenerate text (§2).
  <https://arxiv.org/abs/1904.09751>.
- **Fan, A. et al. (2018).** "Hierarchical Neural Story Generation." — **top-k sampling** (§2).
  <https://arxiv.org/abs/1805.04833>.

## Speculative decoding

- **Leviathan, Y., Kalman, M. & Matias, Y. (2023).** "Fast Inference from Transformers via Speculative
  Decoding." *ICML*. — the accept/reject scheme and its exactness proof (§3).
  <https://arxiv.org/abs/2211.17192>.
- **Chen, C. et al. (2023).** "Accelerating Large Language Model Decoding with Speculative Sampling."
  (DeepMind, concurrent) (§3). <https://arxiv.org/abs/2302.01318>.
- **Cai, T. et al. (2024).** "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding
  Heads." — self-speculation without a separate draft model (§3). <https://arxiv.org/abs/2401.10774>.

## Serving systems

- **Kwon, W. et al. (2023).** "Efficient Memory Management for Large Language Model Serving with
  PagedAttention" (**vLLM**). *SOSP*. — paged KV cache and continuous batching (§4-§5).
  <https://arxiv.org/abs/2309.06180>.
- **Yu, G.-I. et al. (2022).** "Orca: A Distributed Serving System for Transformer-Based Generative
  Models." *OSDI*. — **continuous (iteration-level) batching** (§4).
- **Pope, R. et al. (2022).** "Efficiently Scaling Transformer Inference." — the serving roofline, KV
  cache, and parallelism (§1, §4-§5). <https://arxiv.org/abs/2211.05102>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | PagedAttention, continuous batching, speculative decoding |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | optimized inference kernels and speculative decoding |
| [`transformers` generation](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py) | reference decoding strategies |
| [SGLang](https://github.com/sgl-project/sglang) | structured generation and fast serving |

---

## Deferred to later chapters

- **Efficient attention & KV cache** → [11.03](../03-efficient-attention/)
- **Quantization** → [11.05](../05-adaptation/)
- **RAG & agents** → [11.08](../08-rag-and-agents/)
- **Beam search for seq2seq** → [09.03](../../09-sequence-models/03-seq2seq-and-attention/)
- **Production MLOps** → [Part 19](../../19-mlops/)
