# 11.05 — Exercises: Adaptation

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Rank the adaptation methods (prompting, prompt tuning, LoRA, full fine-tuning) by cost and
explain what each changes.

**D2.** Explain instruction tuning: the data format and how it turns a base LM into an instruction
follower.

**D3.** Compute the memory of full fine-tuning (weights + gradients + Adam moments) for an $N$-parameter
model, and show why it is several times the model size.

**D4.** Write LoRA's update $\Delta W = \frac{\alpha}{r}BA$ and derive its parameter count $2dr$ vs
$d^2$.

**D5.** Show that with $B = 0$ at initialization, LoRA is a no-op, and that the trained adapter can be
merged into $W$ with no inference overhead.

**D6.** Explain the low-rank / intrinsic-dimension premise and why it makes small-$r$ LoRA sufficient.

**D7.** Derive absmax quantization to $b$ bits with a per-block scale, and the round-trip error.

**D8.** Explain why NF4 (a non-uniform grid) beats uniform int4 for Gaussian-distributed weights.

**D9.** Explain how QLoRA fine-tunes through a 4-bit frozen base while keeping adapters in higher
precision, and why this saves optimizer memory.

**D10.** Contrast instruction tuning with alignment (RLHF/DPO): imitation vs preference optimization.

---

## Tier 2 — Implementation

**I1.** Implement a `LoRALinear` layer; reproduce Experiment 1's parameter counts.

**I2.** Verify LoRA is a no-op at init and merges exactly (Experiment 2).

**I3.** Reproduce Experiment 3: measure how much of a low-rank update the best rank-$r$ approximation
captures.

**I4.** Implement per-block absmax quantization (int8, int4) and reproduce Experiment 4.

**I5.** Estimate QLoRA vs full fine-tuning memory for a 7B model (Experiment 5).

**I6.** Fine-tune a small model with LoRA on a task and compare accuracy and trainable-parameter count to
full fine-tuning.

**I7.** Implement NF4-style non-uniform 4-bit quantization and compare error to uniform int4.

**I8.** Implement adapter modules (bottleneck) and compare to LoRA.

**I9.** Merge a trained LoRA adapter into the base weights and verify identical outputs before/after
merge.

**I10.** *(Multi-task.)* Train two LoRA adapters on two tasks over one frozen base and hot-swap them at
inference.

---

## Tier 3 — Interview

**Q1.** What is parameter-efficient fine-tuning and why does it matter?

**Q2.** What is instruction tuning?

**Q3.** How does LoRA work, and how many parameters does it train?

**Q4.** Why is LoRA a no-op at initialization?

**Q5.** Can LoRA be merged for inference? What's the benefit?

**Q6.** Why does LoRA work — what is the low-rank assumption?

**Q7.** What is quantization and how lossy is int8 vs int4?

**Q8.** What is QLoRA and what does it enable?

**Q9.** How much memory does full fine-tuning need vs QLoRA?

**Q10.** When would you choose full fine-tuning over LoRA?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Rank adaptation methods by cost and applicability
- [ ] Derive LoRA's parameter count and no-op/merge properties
- [ ] Explain the low-rank premise behind LoRA
- [ ] Implement per-block quantization and reason about int8 vs int4
- [ ] Explain QLoRA and its memory savings
- [ ] Distinguish instruction tuning from alignment
- [ ] Choose an adaptation method for a given budget
