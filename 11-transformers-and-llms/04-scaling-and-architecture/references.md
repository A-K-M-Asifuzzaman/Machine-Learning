# 11.04 — References: Scaling Laws & Modern Architecture

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1 | Scaling laws | Kaplan et al. (2020) |
| §2-§3 | Compute-optimal, Chinchilla | Hoffmann et al. (2022) |
| §4 | Emergence and its critique | Wei et al. (2022); Schaeffer et al. (2023) |
| §5 | Mixture-of-Experts | Shazeer et al. (2017); Fedus et al. (2022); Jiang et al. (2024, Mixtral) |
| §6 | RMSNorm, SwiGLU, modern recipe | Zhang & Sennrich (2019); Shazeer (2020); Touvron et al. (2023, Llama) |

---

## Scaling laws

- **Kaplan, J. et al. (2020).** "Scaling Laws for Neural Language Models." — the original power-law
  discovery over 7 orders of magnitude (§1). <https://arxiv.org/abs/2001.08361>.
- **Hoffmann, J. et al. (2022).** "Training Compute-Optimal Large Language Models" (**Chinchilla**). —
  the compute-optimal frontier, the parametric fit used here, and the GPT-3-undertraining result
  (§2-§3). <https://arxiv.org/abs/2203.15556>.

## Emergence

- **Wei, J. et al. (2022).** "Emergent Abilities of Large Language Models." *TMLR*. — the emergence
  claim (§4). <https://arxiv.org/abs/2206.07682>.
- **Schaeffer, R., Miranda, B. & Koyejo, S. (2023).** "Are Emergent Abilities of Large Language Models a
  Mirage?" *NeurIPS*. — the discontinuous-metric critique (§4). <https://arxiv.org/abs/2304.15004>.

## Mixture-of-Experts

- **Shazeer, N. et al. (2017).** "Outrageously Large Neural Networks: The Sparsely-Gated
  Mixture-of-Experts Layer." *ICLR*. — the modern MoE layer and load-balancing loss (§5).
  <https://arxiv.org/abs/1701.06538>.
- **Fedus, W., Zoph, B. & Shazeer, N. (2022).** "Switch Transformers." *JMLR*. — top-1 routing at
  trillion-parameter scale (§5). <https://arxiv.org/abs/2101.03961>.
- **Jiang, A. et al. (2024).** "Mixtral of Experts." — an open 8×7B top-2 MoE (§5).
  <https://arxiv.org/abs/2401.04088>.

## Modern architecture

- **Zhang, B. & Sennrich, R. (2019).** "Root Mean Square Layer Normalization" (**RMSNorm**). *NeurIPS*.
  (§6). <https://arxiv.org/abs/1910.07467>.
- **Shazeer, N. (2020).** "GLU Variants Improve Transformer" (**SwiGLU**). (§6).
  <https://arxiv.org/abs/2002.05202>.
- **Touvron, H. et al. (2023).** "LLaMA: Open and Efficient Foundation Language Models." — the modern
  recipe: RMSNorm + SwiGLU + RoPE + pre-norm (§6). <https://arxiv.org/abs/2302.13971>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [Llama (Meta)](https://github.com/meta-llama/llama) | RMSNorm, SwiGLU, RoPE, GQA in one modern model |
| [Mixtral / `transformers` MoE](https://github.com/huggingface/transformers) | MoE routing and load balancing |
| [EleutherAI cookbook](https://github.com/EleutherAI/cookbook) | scaling-law and training-cost calculators |
| [`chinchilla` calculators](https://arxiv.org/abs/2203.15556) | compute-optimal N/D given a budget |

---

## Deferred to later chapters

- **Efficient attention (RoPE/GQA/Flash) in the recipe** → [11.03](../03-efficient-attention/)
- **Adaptation (fine-tuning, LoRA)** → [11.05](../05-adaptation/)
- **Alignment** → [11.06](../06-alignment/)
- **Inference and serving cost** → [11.07](../07-inference/)
- **The transformer being scaled** → [11.01](../01-transformer/)
