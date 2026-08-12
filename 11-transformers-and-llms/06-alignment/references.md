# 11.06 — References: Alignment

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2-§3 | RLHF, reward model | Christiano et al. (2017); Ouyang et al. (2022) |
| §3 | Bradley-Terry model | Bradley & Terry (1952) |
| §4-§5 | KL-constrained objective, PPO | Schulman et al. (2017); Stiennon et al. (2020) |
| §5 | Reward overoptimization | Gao et al. (2023) |
| §6 | DPO | Rafailov et al. (2023) |
| §7 | Constitutional AI / RLAIF | Bai et al. (2022); Lee et al. (2023) |

---

## RLHF

- **Christiano, P. et al. (2017).** "Deep Reinforcement Learning from Human Preferences." *NeurIPS*. —
  the RLHF framework and preference-based reward learning (§2-§3). <https://arxiv.org/abs/1706.03741>.
- **Stiennon, N. et al. (2020).** "Learning to summarize from human feedback." *NeurIPS*. — RLHF for
  summarization; the KL-penalized objective (§4). <https://arxiv.org/abs/2009.01325>.
- **Ouyang, L. et al. (2022).** "Training language models to follow instructions with human feedback"
  (**InstructGPT**). *NeurIPS*. — the full SFT → RM → PPO pipeline (§2). <https://arxiv.org/abs/2203.02155>.
- **Bradley, R. & Terry, M. (1952).** "Rank Analysis of Incomplete Block Designs." *Biometrika*. — the
  preference model (§3).
- **Schulman, J. et al. (2017).** "Proximal Policy Optimization Algorithms" (**PPO**). — the RL
  algorithm RLHF uses (§5). <https://arxiv.org/abs/1707.06347>.

## Reward hacking and DPO

- **Gao, L., Schulman, J. & Hilton, J. (2023).** "Scaling Laws for Reward Model Overoptimization."
  *ICML*. — the true-vs-proxy reward divergence under optimization (§5).
  <https://arxiv.org/abs/2210.10760>.
- **Rafailov, R. et al. (2023).** "Direct Preference Optimization: Your Language Model is Secretly a
  Reward Model." *NeurIPS*. — the DPO derivation and loss (§6). <https://arxiv.org/abs/2305.18290>.
- **Azar, M. et al. (2023).** "A General Theoretical Paradigm to Understand Learning from Human
  Preferences" (**IPO**). — a DPO variant (§6). <https://arxiv.org/abs/2310.12036>.
- **Ethayarajh, K. et al. (2024).** "KTO: Model Alignment as Prospect Theoretic Optimization." (§6).
  <https://arxiv.org/abs/2402.01306>.

## Constitutional AI

- **Bai, Y. et al. (2022).** "Constitutional AI: Harmlessness from AI Feedback." — self-critique against
  a written constitution; RLAIF (§7). <https://arxiv.org/abs/2212.08073>.
- **Lee, H. et al. (2023).** "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI
  Feedback." (§7). <https://arxiv.org/abs/2309.00267>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [Hugging Face TRL](https://github.com/huggingface/trl) | reference RLHF (PPO), DPO, reward-model training |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | scalable RLHF/DPO pipelines |
| [Anthropic HH-RLHF dataset](https://github.com/anthropics/hh-rlhf) | human preference data for helpful/harmless |

---

## Deferred to later chapters

- **PPO and policy-gradient RL** → [Part 13](../../13-reinforcement-learning/)
- **Instruction tuning** → [11.05](../05-adaptation/)
- **Inference/serving** → [11.07](../07-inference/)
- **Evaluation and safety** → [11.08](../08-rag-and-agents/)
- **Fairness and values** → [Part 18](../../18-fairness-privacy-robustness/)
