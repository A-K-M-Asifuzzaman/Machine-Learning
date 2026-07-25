# 08.03 — Transfer Learning

> **Do not learn features from scratch — steal them.** A network trained on a large source task
> (ImageNet's 1.2M images) has already learned edges, textures, and parts that almost any vision task
> needs. Transfer learning reuses those features so a new task can be solved with a few thousand
> examples instead of a million. This chapter measures exactly *when* that works, *when* it backfires,
> and *how* to do it — in a controlled simulation where source and target genuinely share features.

Training a deep network from scratch needs a lot of data and compute. Most real problems have neither.
The single most important practical fact in applied deep learning is this: **you almost never train
from scratch.** You download a pretrained backbone ([08.02](../02-cnn-architectures/)) and adapt it.
Understanding why — and its failure modes — is what this chapter is about.

## Table of contents

1. [The idea: features are reusable](#1-the-idea-features-are-reusable)
2. [Why it works: the feature hierarchy](#2-why-it-works-the-feature-hierarchy)
3. [Do features actually transfer?](#3-do-features-actually-transfer)
4. [When transfer helps: data efficiency](#4-when-transfer-helps-data-efficiency)
5. [Feature extraction vs fine-tuning](#5-feature-extraction-vs-fine-tuning)
6. [Fine-tune gently: catastrophic forgetting](#6-fine-tune-gently-catastrophic-forgetting)
7. [The practical recipe](#7-the-practical-recipe)
8. [Domain shift and its limits](#8-domain-shift-and-its-limits)
9. [Common misconceptions](#9-common-misconceptions)

## 1. The idea: features are reusable

A trained classifier is two parts: a **feature extractor** (all the conv layers) that maps an image to
a vector, and a **head** (the final classifier) that maps that vector to labels. The head is
task-specific; the features are largely *generic*. Transfer learning keeps the extractor and swaps the
head:

$$
\text{pretrained: } x \to \underbrace{f_\theta(x)}_{\text{keep}} \to \underbrace{g_\phi}_{\text{replace}} \to y_{\text{source}}
\qquad\Longrightarrow\qquad
\text{transfer: } x \to f_\theta(x) \to g_{\phi'} \to y_{\text{target}}.
$$

Two ways to do it: **feature extraction** (freeze $\theta$, train a new head $\phi'$) or **fine-tuning**
(train both, starting from the pretrained $\theta$). §5 measures the trade-off.

## 2. Why it works: the feature hierarchy

A CNN learns a hierarchy ([08.01 §2](../01-convolution/), Zeiler & Fergus): early layers detect
**edges and colours**, middle layers detect **textures and simple shapes**, late layers detect
**object parts and whole objects**. The early and middle features are *generic* — every natural image
is made of edges and textures — so they transfer to almost any vision task. Only the last layers are
specific to the source task's classes. This is the structural reason transfer works, and it predicts
the recipe: **the later the layer, the more task-specific it is, and the more you should be willing to
retrain it** (§7).

The simulation in [`from_scratch.py`](from_scratch.py) makes this concrete: inputs map through a shared
hidden feature basis $z = \text{ReLU}(xW_{\text{shared}})$; the **source** is a 20-way task that needs
*all* those features, and the **target** is a binary task that is another readout of the same $z$. A
from-scratch model must relearn the basis from scarce data; a pretrained model already has it.

## 3. Do features actually transfer?

The core claim is testable: put a **linear probe** (freeze the features, train only a new linear head)
on the target and compare feature sources. Experiment 1 ($n_{\text{target}} = 100$):

| Frozen features | Target test accuracy |
|---|:--:|
| random (no pretraining) | 0.555 |
| **unrelated**-source pretraining | 0.566 |
| **related**-source pretraining | **0.632** |

Two lessons in one table. Features from a **related** source beat random features decisively — the
source task taught a basis the target reuses. But features from an **unrelated** source are **no better
than random**: transfer only helps when source and target share structure. This is why ImageNet
pretraining boosts almost any natural-image task (shared edges/textures) but does little for, say, raw
audio spectra or medical modalities that look nothing like ImageNet. **Transfer is not magic; it is
shared structure.**

## 4. When transfer helps: data efficiency

Transfer's value depends entirely on how much target data you have. Experiment 2 measures the
fine-tuning gain over from-scratch, averaged over 6 target training sets:

| $n_{\text{target}}$ | From scratch | Fine-tuned | Transfer gain |
|:--:|:--:|:--:|:--:|
| 25 | 0.571 | 0.574 | +0.003 |
| 50 | 0.616 | 0.637 | **+0.021** |
| 100 | 0.648 | 0.659 | +0.011 |
| 300 | 0.740 | 0.737 | −0.003 |
| 1000 | 0.802 | 0.782 | −0.020 |
| 3000 | 0.827 | 0.795 | −0.032 |

The gain is **positive when data is scarce** and **decreases monotonically** as data grows, eventually
going *negative* — once from-scratch has enough examples, it learns features specialized to the target
that beat the generic transferred ones. **The less target data you have, the more transfer is worth.**
(The margins are small here because the toy features are easy to learn; with real deep features that
need millions of images, the scarce-data gap is enormous — the difference between a usable model and
noise.)

## 5. Feature extraction vs fine-tuning

The two strategies differ in how many parameters they fit. Experiment 3 (mean over 6 sets):

| $n_{\text{target}}$ | Freeze (linear probe) | Fine-tune | Fine-tune edge |
|:--:|:--:|:--:|:--:|
| 25 | 0.567 | 0.574 | +0.007 |
| 50 | 0.604 | 0.637 | +0.033 |
| 100 | 0.612 | 0.659 | +0.048 |
| 300 | 0.680 | 0.737 | +0.057 |
| 1000 | 0.717 | 0.782 | **+0.066** |

- **Feature extraction (freeze)** trains only a small head — few parameters, fast, cheap, hard to
  overfit. A great first baseline.
- **Fine-tuning** also updates the features to fit the target. It needs more data (more parameters to
  fit safely) but adapts the representation — and its **edge over a frozen probe grows with target
  data**. With 1000 examples, fine-tuning is +6.6 points ahead.

## 6. Fine-tune gently: catastrophic forgetting

Fine-tuning has a failure mode. Update the pretrained weights too aggressively and you **overwrite the
very features you came for**. Experiment 4 fine-tunes the features on the target, then re-measures the
*source* task (with its original, unchanged head):

| Fine-tune LR | Source accuracy after | Kept |
|:--:|:--:|:--:|
| 0.00 (freeze) | 0.555 | 100% |
| 0.02 | 0.553 | 100% |
| 0.10 | 0.544 | 98% |
| 0.50 | 0.535 | 96% |
| 2.00 | **0.382** | **69%** |

A small learning rate barely disturbs the features — the source task is still solved. A large learning
rate destroys them: the source accuracy collapses toward chance. This is **catastrophic forgetting**,
and it is why the fine-tuning recipe is always **a learning rate 10–100× smaller than pretraining**,
often with the early layers frozen entirely and only the later, task-specific layers unfrozen
(discriminative learning rates). You adapt the top without demolishing the foundation.

## 7. The practical recipe

Putting the measurements together, the standard fine-tuning workflow:

1. **Start from a pretrained backbone** relevant to your domain (ImageNet for natural images; a
   domain-specific model if one exists — §3).
2. **Replace the head** with one sized for your classes.
3. **Feature-extract first:** freeze the backbone, train only the head. Fast baseline (§5).
4. **Then fine-tune** if you have the data: unfreeze some or all layers with a **small learning rate**
   (§6), later layers first.
5. **Scale effort to data:** little data → freeze / tune only the head; lots of data → fine-tune deeper
   (§4–§5).
6. **Use discriminative learning rates:** smaller LR for early (generic) layers, larger for late
   (specific) ones.

## 8. Domain shift and its limits

Transfer degrades as the target domain drifts from the source. The extreme is §3's unrelated source
(no benefit at all). In between, a **domain shift** — target images from a different distribution
(different sensor, lighting, medical scanner) — makes the pretrained features progressively less
reliable, and you must fine-tune more layers to compensate. When the gap is large enough (natural
images → medical/satellite/audio), the generic early features may still help but the late ones do not,
and heavy fine-tuning or a domain-specific pretrained model is needed. Quantifying and correcting
domain shift is its own field (**domain adaptation**), deferred to [18.xx].

## 9. Common misconceptions

- **"Pretraining always helps."** Only if the source is *related* (§3). An unrelated source is no
  better than random init, and a *mismatched* one plus aggressive fine-tuning can underperform training
  from scratch (negative transfer).
- **"Fine-tuning is always better than feature extraction."** Only with enough target data (§5). When
  data is scarce, freezing (fewer parameters) is safer and nearly as good — and far cheaper.
- **"Use the same learning rate as pretraining."** No — that causes catastrophic forgetting (§6). Use
  10–100× smaller.
- **"More target data always means fine-tune everything."** Roughly yes, but early layers are generic
  enough that freezing them costs little and saves compute.
- **"Transfer learning is only for images."** The same logic drives NLP and LLMs — pretrain on a huge
  corpus, fine-tune on the task ([11.02](../../11-transformers-and-llms/02-pretraining/)). It is
  the dominant paradigm across all of deep learning.

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — a controlled transfer simulation with a shared feature
  basis. Four experiments: (1) related-source features beat random while unrelated ones don't;
  (2) the transfer gain vs target-data size, positive when scarce and reversing with abundance;
  (3) feature-extraction vs fine-tuning and how fine-tuning's edge grows with data; (4) catastrophic
  forgetting when the fine-tuning learning rate is too high.
- **[exercises.md](exercises.md)** — reason about feature hierarchies, design a fine-tuning schedule,
  analyze negative transfer and domain shift.
- **[references.md](references.md)** — the transferability and fine-tuning literature.

## Where this leads

- **The backbones you transfer from** → [08.02](../02-cnn-architectures/)
- **The features being reused** → [08.01](../01-convolution/)
- **Learning-rate schedules for fine-tuning** → [07.06](../../07-deep-learning/06-optimizers/)
- **Pretraining + fine-tuning for language (the same paradigm)** → [11.02](../../11-transformers-and-llms/02-pretraining/)
- **Self-supervised pretraining without labels (DINO, MAE, CLIP)** → [08.05](../05-vision-transformers/)
- **Domain adaptation and distribution shift** → [18.xx], [05.xx]
