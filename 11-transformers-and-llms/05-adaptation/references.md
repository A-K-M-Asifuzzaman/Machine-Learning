# 11.05 — References: Adaptation

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2 | Instruction tuning | Wei et al. (2022, FLAN); Ouyang et al. (2022, InstructGPT) |
| §4 | LoRA | Hu et al. (2021) |
| §5 | Intrinsic dimension | Aghajanyan et al. (2020) |
| §6 | Quantization | Dettmers et al. (2022, LLM.int8()) |
| §7 | QLoRA | Dettmers et al. (2023) |
| §1, §7 | PEFT family | Houlsby et al. (2019); Li & Liang (2021); Liu et al. (2022) |

---

## LoRA and PEFT

- **Hu, E. et al. (2021).** "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*. — the method,
  the no-op init, merging, and the parameter savings (§4). <https://arxiv.org/abs/2106.09685>.
- **Aghajanyan, A. et al. (2020).** "Intrinsic Dimensionality Explains the Effectiveness of Language
  Model Fine-Tuning." *ACL*. — the low-intrinsic-dimension premise behind LoRA (§5).
  <https://arxiv.org/abs/2012.13255>.
- **Houlsby, N. et al. (2019).** "Parameter-Efficient Transfer Learning for NLP" (**adapters**). *ICML*.
  (§1, §7). <https://arxiv.org/abs/1902.00751>.
- **Li, X. & Liang, P. (2021).** "Prefix-Tuning." *ACL*. — soft-prompt tuning (§1).
  <https://arxiv.org/abs/2101.00190>.
- **Liu, H. et al. (2022).** "Few-Shot Parameter-Efficient Fine-Tuning ($(IA)^3$)." *NeurIPS*.
  <https://arxiv.org/abs/2205.05638>.

## Quantization and QLoRA

- **Dettmers, T. et al. (2022).** "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale."
  *NeurIPS*. — int8 inference with outlier handling (§6). <https://arxiv.org/abs/2208.07339>.
- **Dettmers, T. et al. (2023).** "QLoRA: Efficient Finetuning of Quantized LLMs." *NeurIPS*. — 4-bit
  NF4, double quantization, and fine-tuning a 65B model on one GPU (§6-§7).
  <https://arxiv.org/abs/2305.14314>.
- **Frantar, E. et al. (2023).** "GPTQ: Accurate Post-Training Quantization for GPT." *ICLR*. — accurate
  4-bit post-training quantization (§6). <https://arxiv.org/abs/2210.17323>.

## Instruction tuning

- **Wei, J. et al. (2022).** "Finetuned Language Models Are Zero-Shot Learners" (**FLAN**). *ICLR*. —
  instruction tuning improves zero-shot generalization (§2). <https://arxiv.org/abs/2109.01652>.
- **Ouyang, L. et al. (2022).** "Training language models to follow instructions with human feedback"
  (**InstructGPT**). *NeurIPS*. — the SFT (instruction-tuning) stage before RLHF (§2).
  <https://arxiv.org/abs/2203.02155>.
- **Wang, Y. et al. (2023).** "Self-Instruct." *ACL*. — model-generated instruction data (§2).
  <https://arxiv.org/abs/2212.10560>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [Hugging Face PEFT](https://github.com/huggingface/peft) | LoRA, QLoRA, prefix/prompt tuning |
| [`bitsandbytes`](https://github.com/bitsandbytes-foundation/bitsandbytes) | int8 / 4-bit (NF4) quantization kernels |
| [microsoft/LoRA](https://github.com/microsoft/LoRA) | the original LoRA code |
| [artidoro/qlora](https://github.com/artidoro/qlora) | the QLoRA reference |

---

## Deferred to later chapters

- **Alignment (RLHF/DPO)** → [11.06](../06-alignment/)
- **Scaling laws** → [11.04](../04-scaling-and-architecture/)
- **Inference-time quantization and serving** → [11.07](../07-inference/)
- **Transfer learning in vision** → [08.03](../../08-computer-vision/03-transfer-learning/)
