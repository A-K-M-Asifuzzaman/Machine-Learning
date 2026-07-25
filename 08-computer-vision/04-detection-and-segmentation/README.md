# 08.04 — Object Detection & Segmentation

> **Classification asks "what is in this image?"; detection asks "what, and where?"; segmentation asks
> "which pixels?"** The architectures that answer these — R-CNN, YOLO, SSD, FPN, U-Net, Mask R-CNN,
> DETR — differ mostly in how they *propose* regions. Underneath, they all share five primitives:
> IoU, non-max suppression, box coding, mAP, and pixel-mask metrics. This chapter builds those
> primitives (verified against torchvision) and surveys the architectures that assemble them.

A classifier outputs one label per image. Detection outputs a **variable number** of (box, class,
score) triples; segmentation outputs a **per-pixel** label map. That change in output structure — from
a fixed vector to a set or a dense map — is what makes these tasks their own subfield.

## Table of contents

1. [Three tasks](#1-three-tasks)
2. [IoU — the currency of detection](#2-iou--the-currency-of-detection)
3. [Two-stage detectors: the R-CNN family](#3-two-stage-detectors-the-r-cnn-family)
4. [Anchors and box coding](#4-anchors-and-box-coding)
5. [One-stage detectors: YOLO and SSD](#5-one-stage-detectors-yolo-and-ssd)
6. [FPN, DETR, and beyond](#6-fpn-detr-and-beyond)
7. [Evaluating detection: mAP](#7-evaluating-detection-map)
8. [Segmentation](#8-segmentation)
9. [Common misconceptions](#9-common-misconceptions)

## 1. Three tasks

| Task | Output | Example architecture |
|---|---|---|
| **Classification** | one label per image | ResNet ([08.02](../02-cnn-architectures/)) |
| **Object detection** | a set of (box, class, score) | Faster R-CNN, YOLO, DETR |
| **Semantic segmentation** | a class per pixel | FCN, U-Net |
| **Instance segmentation** | a mask per object | Mask R-CNN |

All are built on a CNN (or transformer) **backbone** that extracts features
([08.02](../02-cnn-architectures/), [08.05](../05-vision-transformers/)); detection and segmentation
add a **head** that reads those features into boxes or masks.

## 2. IoU — the currency of detection

Everything in detection is measured by **Intersection over Union**:

$$
\text{IoU}(A, B) = \frac{\lvert A \cap B \rvert}{\lvert A \cup B \rvert}.
$$

It is $0$ for disjoint boxes, $1$ for identical ones. Two half-overlapping $10\times10$ boxes
(intersection $25$, union $175$) give IoU $= 25/175 \approx 0.143$ — Experiment 1 computes exactly
this and matches `torchvision.ops.box_iou` to $0$. IoU decides whether a prediction *matches* a
ground-truth box (§7), which boxes to suppress as duplicates (§5), and often appears directly in the
loss (GIoU/DIoU). It is the one number to understand first.

## 3. Two-stage detectors: the R-CNN family

The original deep detectors split the problem in two:

1. **Propose** regions that might contain objects.
2. **Classify** each region and refine its box.

- **R-CNN** (2014) — run a region-proposal algorithm (selective search), warp each of ~2000 crops, and
  push each through a CNN. Accurate but painfully slow (one CNN pass per crop).
- **Fast R-CNN** (2015) — run the CNN *once* on the whole image, then crop *features* (RoI pooling) per
  proposal. Orders of magnitude faster.
- **Faster R-CNN** (2015) — replace selective search with a **Region Proposal Network (RPN)** that
  predicts proposals from the same features. Now fully end-to-end and the template for two-stage
  detection.

Two-stage detectors are accurate (the second stage refines each box) but heavier. The refinement is a
box regression on top of **anchors** (§4).

## 4. Anchors and box coding

Detectors do not regress absolute pixel coordinates. They place a grid of reference **anchor** boxes
(various scales/aspect ratios) and regress an **offset** from the nearest anchor:

$$
t_x = \frac{x - x_a}{w_a}, \quad t_y = \frac{y - y_a}{h_a}, \quad t_w = \log\frac{w}{w_a}, \quad t_h = \log\frac{h}{h_a},
$$

with the inverse (decode) recovering $(x,y,w,h)$. Experiment 3 confirms the transform is exactly
invertible (round-trip error $\sim 10^{-14}$) and, crucially, that the **targets are well-scaled**:

| Regression target | Std. dev. |
|---|---:|
| raw pixel coordinates | 112.1 (spans the image, scale-dependent) |
| **anchor offsets** | **1.2** (normalized, ~$O(1)$, scale-free) |

Normalized $O(1)$ targets are far easier for a network to learn than coordinates spanning hundreds of
pixels — the same reason we standardize inputs ([02.xx], [07.07](../../07-deep-learning/07-normalization/)).
Every anchor-based detector (Faster R-CNN, SSD, YOLO) regresses offsets, not coordinates. Anchor-free
detectors (FCOS, CenterNet) instead regress distances to the box edges from each point.

## 5. One-stage detectors: YOLO and SSD

One-stage detectors skip the proposal step and predict boxes **directly** on a dense grid — faster,
and (with modern tricks) just as accurate:

- **YOLO** (2016→) — divide the image into a grid; each cell predicts a few boxes, their objectness,
  and class probabilities in **one forward pass**. Real-time. Successive versions (v2–v8) add anchors,
  multi-scale features, and better training.
- **SSD** (2016) — predict boxes from **multiple feature-map scales** so different layers catch
  different object sizes.

Because a dense grid fires **many overlapping boxes per object**, one-stage detectors rely on **non-max
suppression** to clean up. Experiment 2 implements NMS — keep the highest-scoring box, suppress every
box overlapping it above an IoU threshold, repeat — collapsing 5 boxes (4 on one object, 1 on another)
to 2, matching `torchvision.ops.nms` exactly. One-stage detectors also fight **class imbalance** (most
grid cells are background); **focal loss** ([07.04](../../07-deep-learning/04-loss-functions/))
was invented for exactly this (RetinaNet).

## 6. FPN, DETR, and beyond

- **FPN (Feature Pyramid Network)** — objects appear at many scales, but deep features are
  low-resolution and shallow features are semantically weak. FPN builds a **top-down pyramid** with
  lateral connections so every scale has strong features. It is a near-universal detection neck.
- **DETR (Detection Transformer, 2020)** — casts detection as **set prediction**: a transformer
  ([11.01](../../11-transformers-and-llms/01-transformer/)) attends over the image and
  outputs a fixed set of boxes, matched to ground truth by the Hungarian algorithm. **No anchors, no
  NMS** — the model learns to emit one box per object directly. This removed two hand-designed
  components (anchors and NMS) and reframed detection as a sequence problem.

## 7. Evaluating detection: mAP

A detector's quality is summarized by **mean Average Precision**. For one class: sort all predictions
by confidence, walk down the list matching each to an unclaimed ground-truth box (IoU ≥ threshold, a
true positive; otherwise a false positive), trace the precision–recall curve, and integrate it — that
area is the **Average Precision (AP)**. Average AP over classes to get **mAP**. Experiment 4 measures
how AP responds:

| Detector on 3 objects (IoU thr 0.5) | AP |
|---|:--:|
| perfect (3 correct boxes) | 1.000 |
| 3 correct + 1 false positive **ranked high** | 0.833 |
| 3 correct + the **same FP ranked low** | 1.000 |
| boxes shifted, all IoU < 0.5 | 0.000 |

The lesson is subtle and important: **AP measures the ranking.** The *same* false positive costs
$0.167$ AP when it outranks real detections, but **nothing** when it sits below them. A detector that
keeps its junk at low confidence is barely penalized. Boxes that miss the IoU threshold are pure false
positives (AP $= 0$). COCO reports mAP averaged over IoU thresholds $0.5{:}0.05{:}0.95$, rewarding
tight localization.

## 8. Segmentation

Segmentation is **per-pixel classification**, so its architectures are encoder–decoders that recover
spatial resolution, and its metrics compare mask *sets*:

- **FCN** (2015) — a CNN with the dense head removed and upsampling added; the first end-to-end pixel
  labeler.
- **U-Net** (2015) — an encoder–decoder with **skip connections** from encoder to decoder that restore
  fine detail lost to downsampling. Dominant in medical imaging.
- **Mask R-CNN** (2017) — Faster R-CNN plus a per-RoI mask branch: **instance** segmentation (a mask
  per detected object), enabled by **RoIAlign** (bilinear, non-quantized RoI pooling).

The metrics (Experiment 5), for two $12\times12$ masks offset by $(2,2)$:

$$
\text{IoU} = \frac{|A\cap B|}{|A\cup B|} = \frac{100}{188} = 0.532, \qquad \text{Dice} = \frac{2|A\cap B|}{|A|+|B|} = 0.694.
$$

Dice is the **F1 score of the pixels**, and it always relates to IoU by
$\text{Dice} = \frac{2\,\text{IoU}}{1+\text{IoU}}$ (confirmed to the digit). Dice is the usual
segmentation *loss*, especially in medical imaging: unlike pixel accuracy, it is **not fooled by a
mostly-background image** where predicting "all background" scores 99% accuracy but 0 Dice.

## 9. Common misconceptions

- **"Detection is just classification on crops."** The hard part is proposing *where* to crop and
  producing a *variable* number of outputs — the whole architecture is about that (§3, §5).
- **"Higher confidence threshold always means better mAP."** mAP integrates over all thresholds; it
  rewards good *ranking*, not a single cutoff (§7). A high-confidence false positive is what hurts.
- **"NMS is a detail."** It is essential for anchor/grid detectors and a real source of errors (it can
  delete a true box that overlaps another object). DETR exists partly to remove it (§6).
- **"Pixel accuracy measures segmentation quality."** On imbalanced masks it is meaningless — use IoU
  or Dice (§8).
- **"Bigger anchors/more scales are always better."** Anchor design is a hand-tuned prior; anchor-free
  and set-prediction methods (FCOS, DETR) drop it entirely (§4, §6).

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — the five detection/segmentation primitives in NumPy:
  (1) IoU, verified against `torchvision.ops.box_iou`; (2) non-max suppression, verified against
  `torchvision.ops.nms`; (3) box encode/decode, invertible to $10^{-14}$ and well-scaled; (4) average
  precision with greedy IoU matching and all-point interpolation; (5) pixel IoU and the Dice
  coefficient, with the Dice–IoU identity.
- **[exercises.md](exercises.md)** — derive IoU/Dice, implement NMS and AP, reason about anchors and
  set prediction.
- **[references.md](references.md)** — the landmark detection and segmentation papers.

## Where this leads

- **The backbones underneath detectors** → [08.02](../02-cnn-architectures/)
- **Focal loss for one-stage class imbalance** → [07.04](../../07-deep-learning/04-loss-functions/)
- **The transformer DETR is built on** → [11.01](../../11-transformers-and-llms/01-transformer/)
- **Vision transformers as detection backbones** → [08.05](../05-vision-transformers/)
- **Precision, recall, and the PR curve in general** → [05.03](../../05-model-evaluation/03-classification-metrics/)
