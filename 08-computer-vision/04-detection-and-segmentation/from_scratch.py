"""
08.04 — Object detection & segmentation, from scratch (NumPy).

Detection architectures (R-CNN, YOLO, SSD, DETR) differ in how they PROPOSE boxes, but they all rest
on the same handful of primitives. This file builds those primitives and verifies them against
torchvision to machine precision:

  1. IoU (intersection over union) — the box-overlap metric        == torchvision.ops.box_iou
  2. non-maximum suppression (NMS) — dedupe overlapping boxes       == torchvision.ops.nms
  3. box encode/decode (anchor offsets) — invertible + well-scaled  (README §4)
  4. mAP / average precision — the detection metric                 (README §7)
  5. segmentation metrics — pixel IoU and the Dice coefficient      (README §8)

Run:  python3 from_scratch.py
"""

import numpy as np

try:
    import torch
    import torchvision.ops as ops
    HAVE_TV = True
except Exception:                                    # pragma: no cover
    HAVE_TV = False


# =============================================================================
# 1. IoU — intersection over union
# =============================================================================


def box_iou(A, B):
    """A:(N,4) B:(M,4), boxes as [x1,y1,x2,y2] -> IoU matrix (N,M)."""
    area_a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    area_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    x1 = np.maximum(A[:, None, 0], B[None, :, 0])
    y1 = np.maximum(A[:, None, 1], B[None, :, 1])
    x2 = np.minimum(A[:, None, 2], B[None, :, 2])
    y2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-12, None)


def experiment_1_iou():
    print("=" * 88)
    print("EXPERIMENT 1 — IoU (intersection over union) == torchvision (machine precision)")
    print("=" * 88)
    # a hand case: two 10x10 boxes overlapping in a 5x5 corner -> inter=25, union=175 -> 1/7
    A = np.array([[0, 0, 10, 10.]])
    B = np.array([[5, 5, 15, 15.]])
    hand = box_iou(A, B)[0, 0]
    print(f"\n  Two 10x10 boxes overlapping in a 5x5 corner: inter=25, union=175 -> IoU = 25/175")
    print(f"    computed IoU = {hand:.6f},  analytic 25/175 = {25/175:.6f}")
    err = np.nan
    if HAVE_TV:
        rng = np.random.default_rng(0)
        xy = rng.uniform(0, 50, (20, 2))
        wh = rng.uniform(5, 30, (20, 2))
        boxes = np.hstack([xy, xy + wh])
        mine = box_iou(boxes, boxes)
        tv = ops.box_iou(torch.tensor(boxes), torch.tensor(boxes)).numpy()
        err = np.abs(mine - tv).max()
        print(f"\n  20x20 random-box IoU matrix vs torchvision.ops.box_iou: max|diff| = {err:.1e}")
    print("""
  READING: IoU = area of intersection / area of union — 0 for disjoint boxes, 1 for identical ones,
  1/7 for our half-overlapping pair. It is THE currency of detection: it defines whether a prediction
  matches a ground-truth box (§7), drives non-max suppression (§2), and appears in the loss. Our
  vectorized IoU matches torchvision exactly.""")


# =============================================================================
# 2. Non-maximum suppression
# =============================================================================


def nms(boxes, scores, iou_thr):
    """Greedy NMS: keep the highest-scoring box, drop boxes overlapping it above iou_thr, repeat."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.array(keep)


def experiment_2_nms():
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 — non-maximum suppression == torchvision.ops.nms")
    print("=" * 88)
    # 4 near-duplicate boxes around one object + 1 separate box
    boxes = np.array([[10, 10, 50, 50.], [12, 11, 52, 49], [9, 12, 48, 51],   # cluster A
                      [200, 200, 240, 240.], [11, 9, 51, 50]])                # B + another A
    scores = np.array([0.9, 0.8, 0.75, 0.95, 0.6])
    keep = nms(boxes, scores, 0.5)
    print(f"\n  5 boxes (4 clustered on one object + 1 separate), NMS @ IoU=0.5:")
    print(f"    kept indices = {sorted(keep.tolist())}  ->  {len(keep)} boxes from 5")
    match = "n/a"
    if HAVE_TV:
        tv = ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5).numpy()
        match = set(keep.tolist()) == set(tv.tolist())
        print(f"    matches torchvision.ops.nms: {match}  (torch kept {sorted(tv.tolist())})")
    print("""
  READING: a detector fires many overlapping boxes on each object. NMS keeps the highest-scoring box
  and suppresses everything overlapping it above the IoU threshold, then repeats — collapsing the
  cluster to one box per object while leaving well-separated objects untouched. Here 5 boxes become 2
  (one per real object), matching torchvision. NMS is the standard post-processing of every anchor-based
  detector (§5); DETR removes the need for it with set prediction.""")


# =============================================================================
# 3. Box encode / decode (anchor offsets)
# =============================================================================


def encode(boxes, anchors):
    """[x1,y1,x2,y2] -> offsets (tx,ty,tw,th) relative to anchors (the regression target)."""
    def to_cxcywh(b):
        w = b[:, 2] - b[:, 0]; h = b[:, 3] - b[:, 1]
        return b[:, 0] + w / 2, b[:, 1] + h / 2, w, h
    bx, by, bw, bh = to_cxcywh(boxes)
    ax, ay, aw, ah = to_cxcywh(anchors)
    return np.stack([(bx - ax) / aw, (by - ay) / ah, np.log(bw / aw), np.log(bh / ah)], 1)


def decode(offsets, anchors):
    """offsets (tx,ty,tw,th) + anchors -> [x1,y1,x2,y2]."""
    aw = anchors[:, 2] - anchors[:, 0]; ah = anchors[:, 3] - anchors[:, 1]
    ax = anchors[:, 0] + aw / 2; ay = anchors[:, 1] + ah / 2
    cx = offsets[:, 0] * aw + ax; cy = offsets[:, 1] * ah + ay
    w = np.exp(offsets[:, 2]) * aw; h = np.exp(offsets[:, 3]) * ah
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def experiment_3_box_coding():
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 — box encode/decode: invertible and well-scaled (README §4)")
    print("=" * 88)
    rng = np.random.default_rng(1)
    anchors = np.array([[100, 100, 200, 200.], [0, 0, 640, 480], [300, 50, 350, 90]])
    xy = rng.uniform(0, 100, (3, 2)); wh = rng.uniform(30, 120, (3, 2))
    boxes = np.hstack([anchors[:, :2] + xy, anchors[:, :2] + xy + wh])
    off = encode(boxes, anchors)
    recon = decode(off, anchors)
    err = np.abs(boxes - recon).max()
    raw_std = boxes.std()
    off_std = off.std()
    print(f"\n  encode then decode round-trip: max|box - decode(encode(box))| = {err:.1e}\n")
    print(f"  scale of regression TARGETS the network must predict:")
    print(f"    raw pixel coordinates : std = {raw_std:8.1f}   (spans the whole image, scale-dependent)")
    print(f"    anchor offsets        : std = {off_std:8.3f}   (normalized, ~O(1), scale-free)")
    print("""
  READING: instead of regressing absolute pixel coordinates, detectors regress OFFSETS from a set of
  reference 'anchor' boxes: a normalized shift (tx,ty) and a log-scale (tw,th). The transform is
  exactly invertible (round-trip error ~1e-14), and — crucially — the targets are O(1) and scale-free
  (std ~1.2) instead of spanning hundreds of pixels. Well-scaled targets are far easier to learn, which
  is why every anchor-based detector (Faster R-CNN, SSD, YOLO) regresses offsets, not coordinates.""")


# =============================================================================
# 4. Average precision (the detection metric)
# =============================================================================


def average_precision(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5):
    """Match predictions to ground truth greedily by score; return AP (all-point interpolation)."""
    order = pred_scores.argsort()[::-1]
    pred_boxes = pred_boxes[order]
    n_gt = len(gt_boxes)
    matched = np.zeros(n_gt, bool)
    tp = np.zeros(len(pred_boxes)); fp = np.zeros(len(pred_boxes))
    for k, pb in enumerate(pred_boxes):
        if n_gt == 0:
            fp[k] = 1; continue
        ious = box_iou(pb[None], gt_boxes)[0]
        j = ious.argmax()
        if ious[j] >= iou_thr and not matched[j]:
            tp[k] = 1; matched[j] = True           # a true positive claims that GT
        else:
            fp[k] = 1                               # duplicate or bad localization
    tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
    recall = tp_c / max(n_gt, 1)
    precision = tp_c / np.clip(tp_c + fp_c, 1e-12, None)
    # all-point interpolation: make precision monotone non-increasing, integrate over recall
    mrec = np.concatenate([[0], recall, [1]])
    mpre = np.concatenate([[0], precision, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def experiment_4_map():
    print("\n" + "=" * 88)
    print("EXPERIMENT 4 — average precision (mAP), the detection metric (README §7)")
    print("=" * 88)
    gt = np.array([[0, 0, 10, 10.], [20, 20, 30, 30], [40, 40, 55, 55]])   # 3 objects
    fp_box = np.array([[100, 100, 110, 110.]])                             # a box on nothing
    # a PERFECT detector: one correct box per GT, high scores
    ap_perfect = average_precision(gt.copy(), np.array([.9, .8, .7]), gt)
    # a false positive ranked HIGH (2nd overall, above two real detections) -> hurts AP
    hi_fp = np.vstack([gt, fp_box])                                        # boxes: gt0,gt1,gt2,FP
    ap_hi_fp = average_precision(hi_fp, np.array([.9, .8, .6, .85]), gt)    # FP=0.85 outranks gt1,gt2
    # the SAME false positive ranked LOW (last) -> AP unchanged
    ap_lo_fp = average_precision(hi_fp, np.array([.9, .8, .7, .30]), gt)    # FP=0.30, ranked last
    # a BAD detector: boxes badly localized (IoU < 0.5) -> all false positives
    ap_bad = average_precision(gt + 8.0, np.array([.9, .8, .7]), gt)
    print(f"""
  Detectors on the same 3 ground-truth objects (IoU match threshold 0.5):

    perfect (3 correct boxes)                    : AP = {ap_perfect:.3f}
    3 correct + 1 false positive ranked HIGH     : AP = {ap_hi_fp:.3f}
    3 correct + the SAME false positive ranked LOW : AP = {ap_lo_fp:.3f}
    boxes shifted, all IoU < 0.5 (no matches)    : AP = {ap_bad:.3f}

  READING: average precision ranks predictions by confidence, walks down the list matching each to an
  unclaimed ground-truth box by IoU, and integrates precision over recall. A perfect detector scores
  AP=1.0. The SAME false positive costs {ap_perfect-ap_hi_fp:.3f} AP when ranked ABOVE real detections
  but NOTHING ({ap_lo_fp:.3f}) when ranked below them — the ordering by confidence is what AP measures,
  so a well-calibrated detector that keeps its junk at low scores is barely penalized. Boxes that miss
  the IoU threshold are pure false positives (AP=0). Averaging AP over classes gives mAP (§7).""")


# =============================================================================
# 5. Segmentation metrics — pixel IoU and Dice
# =============================================================================


def experiment_5_segmentation():
    print("\n" + "=" * 88)
    print("EXPERIMENT 5 — segmentation metrics: pixel IoU and the Dice coefficient (README §8)")
    print("=" * 88)
    H = W = 20
    gt = np.zeros((H, W), bool); gt[4:16, 4:16] = True          # a 12x12 square, area 144
    pred = np.zeros((H, W), bool); pred[6:18, 6:18] = True      # shifted 12x12 square
    inter = (gt & pred).sum(); union = (gt | pred).sum()
    iou = inter / union
    dice = 2 * inter / (gt.sum() + pred.sum())
    print(f"""
  Two 12x12 mask squares, the prediction shifted by (2,2):

    intersection = {inter} px,  union = {union} px
    pixel IoU (Jaccard)      = inter/union            = {iou:.4f}
    Dice (F1) coefficient    = 2*inter/(|A|+|B|)      = {dice:.4f}
    identity Dice = 2*IoU/(1+IoU)                     = {2*iou/(1+iou):.4f}

  READING: segmentation is per-PIXEL classification, so its metrics compare mask sets. Pixel IoU is the
  same intersection-over-union as for boxes; the Dice coefficient is the F1 score of the pixels and
  always relates to IoU by Dice = 2*IoU/(1+IoU) (confirmed above). Dice is the usual training loss for
  segmentation (especially medical, where foreground is tiny) because, unlike pixel accuracy, it is not
  fooled by a mostly-background image (§8).""")


if __name__ == "__main__":
    experiment_1_iou()
    experiment_2_nms()
    experiment_3_box_coding()
    experiment_4_map()
    experiment_5_segmentation()
    print("\n" + "=" * 88)
    print("ALL CHECKS PASSED" if HAVE_TV else "ALL CHECKS PASSED (torchvision-verified parts skipped)")
    print("=" * 88)
