# 08.03 — Exercises: Transfer Learning

Three tiers. **Derivation/reasoning** (paper), **implementation** (code), **interview** (explain out
loud).

---

## Tier 1 — Derivation & reasoning

**D1.** Explain the split of a trained network into a feature extractor and a head, and why the
extractor is largely task-agnostic while the head is task-specific.

**D2.** Using the feature hierarchy (edges → textures → parts), predict which layers transfer best to
(a) a similar task, (b) a very different domain. Justify a per-layer freezing/fine-tuning schedule.

**D3.** Explain why a linear probe on frozen features is a clean measure of *feature quality*, and why
comparing it to a random-feature probe isolates what pretraining contributed.

**D4.** Define negative transfer. Under what conditions does pretraining hurt relative to training from
scratch? Relate to Experiment 1's unrelated-source result.

**D5.** Explain the data-efficiency curve (Experiment 2): why the transfer gain is largest when target
data is scarce and can go negative with abundant data.

**D6.** Explain catastrophic forgetting mechanistically: why does a large fine-tuning learning rate
destroy the source-task performance (Experiment 4)?

**D7.** Justify using a learning rate 10–100× smaller for fine-tuning than for pretraining, and why
discriminative (per-layer) learning rates help.

**D8.** Compare feature extraction and fine-tuning by parameter count and overfitting risk, and state
the rule for choosing between them by dataset size.

**D9.** Explain how the pretrain-then-fine-tune paradigm in NLP/LLMs is the same idea as image transfer
learning, and what plays the role of "generic features" there.

**D10.** A model pretrained on natural images is applied to grayscale medical scans. Predict which
layers still help and design an adaptation strategy.

---

## Tier 2 — Implementation

**I1.** Reproduce Experiment 1: build related, unrelated, and random feature extractors and compare
linear-probe accuracy on the target. Confirm related ≫ unrelated ≈ random.

**I2.** Reproduce Experiment 2: measure the fine-tuning-vs-scratch gain across target-data sizes,
averaged over several draws. Confirm the gain shrinks (and reverses) as data grows.

**I3.** Reproduce Experiment 3: compare freeze vs fine-tune across data sizes; show fine-tuning's edge
grows with data.

**I4.** Reproduce Experiment 4: fine-tune at several learning rates and measure the retained source
accuracy; find where catastrophic forgetting sets in.

**I5.** Implement discriminative learning rates (smaller LR for early layers) in a multi-layer net and
compare to a single LR.

**I6.** Implement a linear probe and a full fine-tune on a real pretrained backbone (e.g. a torchvision
ResNet) for a small image dataset; compare accuracy and training time.

**I7.** Implement a layer-freezing schedule (freeze all → unfreeze last block → unfreeze more) and plot
accuracy vs number of unfrozen layers.

**I8.** Construct a negative-transfer example: pretrain on a source whose features are misaligned with
the target and show it underperforms training from scratch.

**I9.** Measure feature quality per layer: probe the frozen output of each layer of a pretrained net on
a new task and plot accuracy vs depth.

**I10.** *(Efficiency.)* Compare wall-clock time and accuracy of feature-extraction (cache features
once, train head) vs full fine-tuning.

---

## Tier 3 — Interview

**Q1.** What is transfer learning and why is it so widely used?

**Q2.** What is the difference between feature extraction and fine-tuning?

**Q3.** When would you freeze the backbone vs fine-tune it?

**Q4.** Why do you use a smaller learning rate when fine-tuning?

**Q5.** What is catastrophic forgetting?

**Q6.** When does transfer learning *not* help — or hurt?

**Q7.** Which layers of a CNN transfer best, and why?

**Q8.** How does dataset size change your transfer strategy?

**Q9.** What is a linear probe and what does it measure?

**Q10.** How does transfer learning in NLP/LLMs relate to image transfer learning?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Explain the feature-extractor / head split and why features transfer
- [ ] Predict which layers transfer using the feature hierarchy
- [ ] Choose feature extraction vs fine-tuning from dataset size
- [ ] Explain and avoid catastrophic forgetting
- [ ] Recognize negative transfer and its cause
- [ ] Design a layer-freezing + discriminative-LR fine-tuning schedule
- [ ] Connect image transfer to the pretrain-fine-tune paradigm in NLP
