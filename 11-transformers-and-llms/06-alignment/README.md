# 11.06 — Alignment: RLHF and DPO

> **A capable model is not automatically a good one.** A pretrained, instruction-tuned LLM can write
> fluent text, but it may be unhelpful, dishonest, or harmful — because "predict the next token" and
> "be a good assistant" are different objectives. Alignment closes that gap by optimizing the model
> against human *preferences*. This chapter derives the two dominant methods — RLHF (reward model +
> PPO) and DPO — shows they target the *same* optimum, and measures the reward-hacking failure that
> makes the KL leash essential.

Instruction tuning ([11.05 §2](../05-adaptation/)) teaches format by *imitation*. Alignment goes
further: it optimizes for what humans *prefer*, which is easier to *compare* ("A is better than B") than
to *demonstrate* or *score*. Turning those comparisons into a training signal is the whole game.

## Table of contents

1. [The alignment problem](#1-the-alignment-problem)
2. [The RLHF pipeline](#2-the-rlhf-pipeline)
3. [The reward model](#3-the-reward-model)
4. [KL-constrained reward maximization](#4-kl-constrained-reward-maximization)
5. [PPO and reward hacking](#5-ppo-and-reward-hacking)
6. [DPO: alignment without RL](#6-dpo-alignment-without-rl)
7. [Constitutional AI and RLAIF](#7-constitutional-ai-and-rlaif)
8. [Common misconceptions](#8-common-misconceptions)

## 1. The alignment problem

The goal is often summarized as **HHH**: **helpful, honest, harmless**. None of these is captured by the
pretraining objective — the internet contains helpful *and* harmful text, truth *and* falsehood.
Alignment is a *post-training* stage that steers the capable base model toward preferred behavior using
human (or AI) feedback. It does not add knowledge; it shapes *behavior*.

## 2. The RLHF pipeline

Reinforcement Learning from Human Feedback (Christiano et al., 2017; Ouyang et al., 2022) has three
stages:

1. **SFT** — supervised fine-tuning on demonstrations (instruction tuning, [11.05](../05-adaptation/)).
2. **Reward model (RM)** — collect human *preferences* between response pairs and fit a model that
   scores responses (§3).
3. **RL (PPO)** — optimize the SFT policy to maximize the reward model's score, kept close to the SFT
   model by a KL penalty (§4–§5).

DPO ([§6](#6-dpo-alignment-without-rl)) collapses stages 2–3 into one supervised loss.

## 3. The reward model

Humans can't reliably assign a numeric score, but they *can* pick the better of two responses. The
**Bradley-Terry** model turns comparisons into a reward: the probability that response $a$ is preferred
over $b$ is

$$
P(a \succ b) = \sigma\big(r(a) - r(b)\big).
$$

Fitting $r$ to maximize the likelihood of observed preferences recovers a reward function that **ranks**
responses. Experiment 1 fits a reward from 3,000 noisy pairwise preferences over 8 responses and
recovers the true quality ranking at **Spearman 0.976**. This learned reward turns sparse human
comparisons into a dense score for *any* response — the training signal for RL.

## 4. KL-constrained reward maximization

The RL objective is not "maximize reward" — that would let the model drift into degenerate text that
games the reward. It is **maximize reward while staying close to the reference (SFT) policy**:

$$
\max_{\pi}\; \mathbb{E}_{y \sim \pi}\big[r(y)\big] - \beta\, \mathrm{KL}\big(\pi \,\Vert\, \pi_{\text{ref}}\big).
$$

This has an **exact closed-form solution**:

$$
\pi^*(y) \;\propto\; \pi_{\text{ref}}(y)\,\exp\!\big(r(y)/\beta\big).
$$

Reweight the reference policy by the exponentiated reward. Experiment 2 verifies direct optimization
matches this formula to machine precision ($10^{-15}$). The coefficient $\beta$ is the central knob —
Experiment 3 sweeps it:

| $\beta$ | E[reward] | KL from ref |
|:--:|:--:|:--:|
| 10 | 0.48 | 0.008 |
| 1.0 | 1.43 | 0.46 |
| 0.1 | 1.80 | 1.12 |

Large $\beta$ stays near the reference (safe, small gain); small $\beta$ chases reward (large gain, but
drifts far from the well-behaved reference). The KL term keeps the aligned model fluent and
on-distribution. **PPO is just a way to approximate $\pi^*$** with gradient steps when the response space
is astronomically large (real text) and you can't normalize the closed form.

## 5. PPO and reward hacking

**PPO** (Proximal Policy Optimization, [Part 13](../../13-reinforcement-learning/)) is the RL algorithm
that climbs toward $\pi^*$: sample responses, score them with the reward model, and take clipped policy-
gradient steps, with the KL penalty as a regularizer. Its central danger is **reward hacking**: the
reward model is a *proxy* for human preference and has errors, so hard optimization exploits those
errors. Experiment 4 makes it vivid — the reward model wrongly rates a bad response highest, and we
optimize the proxy at shrinking $\beta$:

| $\beta$ | Proxy reward | **True reward** | P(hack) |
|:--:|:--:|:--:|:--:|
| 10 | −0.12 | −0.23 | 0.04 |
| 1.0 | 0.94 | **0.21** (peak) | 0.23 |
| 0.5 | 1.91 | 0.04 | 0.58 |
| 0.1 | 2.43 | **−0.80** | 1.00 |

The **proxy reward keeps rising** while the **true reward peaks then crashes** — the policy piles onto
the bad-but-high-proxy "hack" response. This is **Goodhart's law** ("when a measure becomes a target, it
ceases to be a good measure"). It is why RLHF keeps a KL leash, why reward models must be robust, and why
*more optimization is not always better*.

## 6. DPO: alignment without RL

RLHF is complex and unstable — a separate reward model, on-policy sampling, PPO tuning. **DPO** (Rafailov
et al., 2023) removes all of it with one observation. Invert the closed-form optimum (§4) to express the
reward in terms of the policy:

$$
r(y) = \beta \log \frac{\pi(y)}{\pi_{\text{ref}}(y)} + \text{const}.
$$

Substitute this into the Bradley-Terry likelihood (§3), and the reward model *cancels out*. What remains
is a simple **classification loss on preference pairs**:

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\!\left(\beta \log\frac{\pi(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log\frac{\pi(y_l)}{\pi_{\text{ref}}(y_l)}\right),
$$

trained directly on chosen ($y_w$) vs rejected ($y_l$) pairs — **no reward model, no sampling, no RL**.
Experiment 5 trains DPO on preference pairs and confirms it lands on the **same optimum** as full RLHF
(identical ranking, within finite-sample noise of the closed form). DPO is simpler, more stable, and has
become the default alignment method; variants (IPO, KTO, ORPO) tweak the loss.

## 7. Constitutional AI and RLAIF

Human feedback is expensive and inconsistent. **Constitutional AI** (Bai et al., 2022) and **RLAIF**
replace much of it with *AI feedback*: the model critiques and revises its own responses against a
written set of principles (a "constitution"), and an AI judge generates the preference labels. This
scales feedback cheaply and makes the values explicit and auditable. Most frontier labs now blend human
and AI feedback. The open problems remain: whose preferences, reward-model robustness (§5), and
**scalable oversight** — how to align models on tasks humans can no longer easily evaluate.

## 8. Common misconceptions

- **"Alignment adds capabilities."** It shapes *behavior* (helpful/honest/harmless); the knowledge comes
  from pretraining (§1).
- **"RLHF just maximizes a reward."** It maximizes reward *minus a KL penalty*; without the leash it
  reward-hacks (§4–§5).
- **"A higher reward-model score is always better."** The reward is a proxy; over-optimizing it lowers
  true quality (§5, Goodhart).
- **"DPO and RLHF are different objectives."** DPO optimizes the *same* KL-constrained objective, just
  reparameterized to skip the reward model and RL (§6).
- **"Alignment is solved."** Reward hacking, scalable oversight, and whose-values questions are open
  (§7).

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — the alignment machinery in an exactly-computable discrete
  setting. Five experiments: (1) a Bradley-Terry reward model recovers the ranking (Spearman 0.98);
  (2) the RLHF optimum equals $\pi_{\text{ref}}\exp(r/\beta)$ to $10^{-15}$; (3) the KL/reward
  trade-off; (4) reward hacking — true reward peaks then crashes as the proxy is over-optimized;
  (5) DPO recovering the RLHF optimum without RL.
- **[exercises.md](exercises.md)** — derive the closed form and the DPO loss, implement the reward model
  and reward hacking.
- **[references.md](references.md)** — RLHF, PPO, DPO, and constitutional-AI papers.

## Where this leads

- **PPO and policy-gradient RL in depth** → [Part 13](../../13-reinforcement-learning/)
- **Instruction tuning, the stage before alignment** → [11.05 §2](../05-adaptation/)
- **Inference and serving the aligned model** → [11.07](../07-inference/)
- **Evaluation and safety of aligned models** → [11.08](../08-rag-and-agents/)
- **Fairness and whose-values questions** → [Part 18](../../18-fairness-privacy-robustness/)
