# 11.05 — Adaptation: Fine-Tuning, LoRA, Quantization

> **A pretrained LLM is a generalist; adaptation makes it a specialist — cheaply.** Full fine-tuning
> updates all the weights and needs a copy of the optimizer state for each, which is unaffordable for
> billion-parameter models. The insight that broke this open: **fine-tuning updates are low-rank**, so
> you can train a tiny adapter (LoRA) instead of the whole model — and if you also **quantize** the
> frozen base to 4 bits (QLoRA), a 65B model fine-tunes on a single GPU. This chapter derives those
> methods and measures exactly how much they save.

Pretraining ([11.02](../02-pretraining/)) gives a model that knows language; adaptation teaches it a
*task* (classification), a *format* (following instructions), or a *behavior* (a persona). This is the
vision transfer-learning idea ([08.03](../../08-computer-vision/03-transfer-learning/)) at LLM scale,
where full fine-tuning is often infeasible.

## Table of contents

1. [The spectrum of adaptation](#1-the-spectrum-of-adaptation)
2. [Instruction tuning](#2-instruction-tuning)
3. [The problem with full fine-tuning](#3-the-problem-with-full-fine-tuning)
4. [LoRA: low-rank adaptation](#4-lora-low-rank-adaptation)
5. [Why LoRA works: updates are low-rank](#5-why-lora-works-updates-are-low-rank)
6. [Quantization](#6-quantization)
7. [QLoRA and the full picture](#7-qlora-and-the-full-picture)
8. [Common misconceptions](#8-common-misconceptions)

## 1. The spectrum of adaptation

From cheapest to most expensive:

| Method | What changes | Cost | When |
|---|---|---|---|
| **Prompting / in-context** | nothing (examples in the prompt) | ~0 | quick tasks, no training |
| **Prompt/prefix tuning** | a few soft-prompt vectors | tiny | lightweight steering |
| **LoRA / adapters (PEFT)** | small added modules | small | most fine-tuning today |
| **Full fine-tuning** | all weights | large | max quality, ample compute |

The middle ground — **parameter-efficient fine-tuning (PEFT)** — is where most real adaptation happens,
and LoRA is its dominant form.

## 2. Instruction tuning

A base LLM trained on raw text *completes* text; it does not naturally *follow instructions*.
**Instruction tuning** fine-tunes on `(instruction, response)` pairs so the model learns the format
"given a request, produce a helpful answer" (FLAN, InstructGPT's SFT stage). It is ordinary supervised
fine-tuning on a curated dataset of demonstrations, and it is the bridge from a raw language model to a
usable assistant — the step *before* alignment ([11.06](../06-alignment/)). The data can be human-written
or model-generated (self-instruct); quality and diversity matter more than quantity.

## 3. The problem with full fine-tuning

Full fine-tuning updates every weight, and the *optimizer* is the real cost: Adam
([07.06](../../07-deep-learning/06-optimizers/)) stores two moments per parameter, so training memory is
several times the model size. Experiment 5 estimates it for a 7B model:

$$
\text{weights} + \text{gradients} + \text{2 Adam moments} \approx 8 \times \text{params} = 56\text{ GB},
$$

far beyond a consumer GPU — and you must store a *full copy* of the model per task. PEFT attacks both.

## 4. LoRA: low-rank adaptation

LoRA (Hu et al., 2021) freezes the pretrained weight $W$ and learns a **low-rank** update:

$$
W' = W + \Delta W, \qquad \Delta W = \frac{\alpha}{r} B A, \quad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times d},\; r \ll d.
$$

Only $A$ and $B$ train — $2dr$ parameters instead of $d^2$. Experiment 1 quantifies it:

| $d$ | $r$ | Full ($d^2$) | LoRA ($2dr$) | % of full |
|:--:|:--:|--:|--:|:--:|
| 4096 | 8 | 16,777,216 | 65,536 | **0.39%** |
| 4096 | 64 | 16,777,216 | 524,288 | 3.12% |

Two properties make it practical (Experiment 2, verified to machine precision):

- **No-op at initialization.** $B = 0$, so $\Delta W = 0$ and the adapted model starts *exactly* equal
  to the base — training begins from the pretrained solution with zero disruption.
- **Mergeable.** After training, $W \leftarrow W + \frac{\alpha}{r}BA$ folds the adapter into the
  weights, so inference has **zero extra latency**. And many task adapters share one frozen base — swap a
  few megabytes to switch tasks.

## 5. Why LoRA works: updates are low-rank

LoRA's premise is empirical: **fine-tuning changes the weights in only a few directions.** A task update
is not a random full-rank matrix. Experiment 3 takes an update with intrinsic rank ~6 and measures how
much of its "energy" (Frobenius norm²) the best rank-$r$ approximation captures:

| Rank $r$ | Energy captured | Params ($2dr$) |
|:--:|:--:|:--:|
| 1 | 24.1% | 256 |
| 2 | 44.2% | 512 |
| 4 | 78.1% | 1,024 |
| **6** | **100.0%** | 1,536 |

A rank-6 approximation captures 100% of a rank-6 update, and even rank-4 gets most of it. Aghajanyan et
al. (2020) showed real fine-tuning has a low "intrinsic dimension" — which is precisely why a small-$r$
$BA$ reproduces the update at a fraction of the parameters. Typical $r$ is 8–64.

## 6. Quantization

The other lever: store weights in fewer bits. **Quantization** maps floating-point weights into a small
integer range with a scale factor, per block to handle outliers. Experiment 4 (512×512 weight, per-block
absmax):

| Precision | Bits | Rel. error | Memory vs fp32 |
|---|:--:|:--:|:--:|
| fp16 | 16 | 0.0002 | 50% |
| int8 | 8 | 0.006 | 25% |
| int4 | 4 | 0.107 | 12.5% |

**int8** (¼ the memory) is effectively lossless (~0.6% error). Plain uniform **int4** is lossier (~11%
here), but production 4-bit methods — QLoRA's **NF4** (a non-uniform grid matched to the Gaussian weight
distribution) with small blocks — roughly halve that and are usable. Quantization is how a 70B model
(140 GB in fp16) fits in ~35 GB (int4) and runs on one GPU. (Weight-only quantization for *inference*;
training needs higher precision for the parts that get gradients.)

## 7. QLoRA and the full picture

QLoRA (Dettmers et al., 2023) combines both tricks: **freeze the base model in 4-bit** (NF4) and train
**LoRA adapters in higher precision**. The frozen base has no gradients and no optimizer state; only the
tiny adapter does. Experiment 5:

| Method | Memory to fine-tune 7B |
|---|:--:|
| full fine-tune (fp16 + Adam) | 56 GB |
| **QLoRA (4-bit base + LoRA)** | **3.9 GB** (14× less) |

A 7B — even a 65B — model now fine-tunes on a single consumer GPU. QLoRA is what **democratized LLM
fine-tuning**: the base is quantized (cheap to store), gradients flow through it into the adapters (which
stay in bf16), and only the adapters are trained and saved. The rest of the PEFT family — adapters
(bottleneck modules), prefix/prompt tuning (soft prompts), $(IA)^3$ — trades off similarly; LoRA/QLoRA
won on simplicity and the zero-latency merge.

## 8. Common misconceptions

- **"Fine-tuning means updating all weights."** PEFT (LoRA) updates ~0.4% of them and often matches full
  fine-tuning (§4).
- **"LoRA hurts quality."** For most tasks it matches full fine-tuning; the low-rank assumption holds
  empirically (§5).
- **"Quantization always degrades the model."** int8 is effectively lossless; 4-bit (NF4) is usable, and
  QLoRA fine-tunes *through* it with no quality loss vs 16-bit LoRA (§6–§7).
- **"Instruction tuning is the same as alignment."** Instruction tuning is supervised imitation of good
  answers; alignment (RLHF/DPO, [11.06](../06-alignment/)) optimizes *preferences* beyond imitation.
- **"You need a data center to fine-tune an LLM."** QLoRA fine-tunes a 65B model on one GPU (§7).

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — LoRA and quantization in NumPy. Five experiments: (1) LoRA's
  $2dr$ vs $d^2$ parameter count; (2) LoRA is a no-op at init and merges exactly; (3) fine-tuning updates
  are low-rank (rank-6 captures 100%); (4) quantization round-trip error int8 vs int4; (5) QLoRA's 14×
  memory reduction.
- **[exercises.md](exercises.md)** — derive LoRA, implement quantization, reason about intrinsic
  dimension and memory.
- **[references.md](references.md)** — LoRA, QLoRA, instruction tuning, and PEFT papers.

## Where this leads

- **Alignment: RLHF and DPO** → [11.06](../06-alignment/)
- **Scaling laws that make small-and-adapted attractive** → [11.04](../04-scaling-and-architecture/)
- **Quantization for inference/serving** → [11.07](../07-inference/)
- **Transfer learning, the vision analogue** → [08.03](../../08-computer-vision/03-transfer-learning/)
- **Adam optimizer state (the memory QLoRA avoids)** → [07.06](../../07-deep-learning/06-optimizers/)
