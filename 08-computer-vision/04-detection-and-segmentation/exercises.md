# 08.04 — Exercises: Object Detection & Segmentation

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Derive the IoU of two axis-aligned boxes from their corners, including the empty-intersection
case. Show IoU $\in [0, 1]$.

**D2.** Derive the anchor box-coding equations $(t_x, t_y, t_w, t_h)$ and their inverse. Explain why
$t_w, t_h$ use a log scale and why the offsets are scale-free.

**D3.** Prove the Dice–IoU identity $\text{Dice} = \frac{2\,\text{IoU}}{1 + \text{IoU}}$ from their
definitions.

**D4.** Given a ranked list of predictions with TP/FP labels, derive the precision–recall points and
the all-point-interpolated AP. Work a small example by hand.

**D5.** Show why a false positive ranked *below* all true positives does not change AP, while one
ranked *above* them does. Relate to Experiment 4.

**D6.** Explain why pixel accuracy is a poor segmentation metric on class-imbalanced masks, and why Dice
/ IoU are not.

**D7.** Explain the failure mode of greedy NMS: construct a case where two distinct objects overlap and
NMS wrongly deletes a correct box. What do Soft-NMS / DETR do about it?

**D8.** Contrast two-stage (Faster R-CNN) and one-stage (YOLO/SSD) detectors in speed, accuracy, and
where the class-imbalance problem arises.

**D9.** Explain the foreground/background imbalance in one-stage detectors and how focal loss
([07.04](../../07-deep-learning/04-loss-functions/)) addresses it.

**D10.** Explain DETR's set-prediction formulation and why it needs neither anchors nor NMS. What role
does the Hungarian matching play?

---

## Tier 2 — Implementation

**I1.** Implement vectorized `box_iou` for an $N\times M$ matrix; verify against
`torchvision.ops.box_iou` (Experiment 1).

**I2.** Implement greedy NMS; verify against `torchvision.ops.nms` (Experiment 2). Then implement
Soft-NMS and compare.

**I3.** Implement box `encode`/`decode` against anchors; verify the round-trip to machine precision and
measure the target scale (Experiment 3).

**I4.** Implement `average_precision` with greedy IoU matching and all-point interpolation; reproduce
Experiment 4's four detectors.

**I5.** Compute mAP averaged over IoU thresholds $0.5{:}0.05{:}0.95$ (COCO-style) for a set of
predictions.

**I6.** Implement pixel IoU and Dice for masks; verify the Dice–IoU identity (Experiment 5).

**I7.** Implement an anchor generator (grid × scales × aspect ratios) and assign each anchor to a
ground-truth box by IoU (positive/negative/ignore).

**I8.** Build a tiny single-scale one-stage detector head (objectness + box offsets) on a toy dataset,
train it, and run NMS on its outputs.

**I9.** Implement Dice loss and train a small U-Net-style segmenter on a synthetic mask dataset;
compare Dice loss vs cross-entropy on imbalanced masks.

**I10.** *(DETR-lite.)* Implement Hungarian matching between predicted and ground-truth boxes and use
it as a training target for a fixed-set predictor.

---

## Tier 3 — Interview

**Q1.** What is IoU and where is it used in detection?

**Q2.** What is non-max suppression and why is it needed?

**Q3.** Why do detectors regress offsets from anchors instead of raw coordinates?

**Q4.** What is the difference between one-stage and two-stage detectors?

**Q5.** How is mAP computed, and what does it reward?

**Q6.** Why can a high-confidence false positive hurt mAP more than a low-confidence one?

**Q7.** What problem does FPN solve?

**Q8.** How does DETR remove anchors and NMS?

**Q9.** What is the difference between semantic and instance segmentation?

**Q10.** Why is Dice preferred over pixel accuracy for segmentation?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Compute IoU and explain its central role
- [ ] Implement and reason about NMS and its failure modes
- [ ] Derive and use anchor box coding
- [ ] Compute AP/mAP and explain what ranking it rewards
- [ ] Contrast one-stage, two-stage, and set-prediction detectors
- [ ] Compute pixel IoU and Dice and prove their identity
- [ ] Explain why Dice beats pixel accuracy on imbalanced masks
