# 08.04 — References: Object Detection & Segmentation

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2 | IoU | Everingham et al. (2010), PASCAL VOC |
| §3 | R-CNN family | Girshick et al. (2014); Girshick (2015); Ren et al. (2015) |
| §4 | Anchors, box coding | Ren et al. (2015); Tian et al. (2019, FCOS) |
| §5 | One-stage detectors | Redmon et al. (2016, YOLO); Liu et al. (2016, SSD); Lin et al. (2017, focal) |
| §6 | FPN, DETR | Lin et al. (2017, FPN); Carion et al. (2020, DETR) |
| §7 | mAP | Everingham et al. (2010); Lin et al. (2014, COCO) |
| §8 | Segmentation | Long et al. (2015, FCN); Ronneberger et al. (2015, U-Net); He et al. (2017, Mask R-CNN) |

---

## Detection — the R-CNN family

- **Girshick, R. et al. (2014).** "Rich feature hierarchies for accurate object detection" (**R-CNN**).
  *CVPR*. — CNN features on region proposals (§3). <https://arxiv.org/abs/1311.2524>.
- **Girshick, R. (2015).** "**Fast R-CNN**." *ICCV*. — RoI pooling, one CNN pass per image (§3).
  <https://arxiv.org/abs/1504.08083>.
- **Ren, S. et al. (2015).** "**Faster R-CNN**: Towards Real-Time Object Detection with Region Proposal
  Networks." *NeurIPS*. — the RPN and anchor box coding (§3-§4). <https://arxiv.org/abs/1506.01497>.

## Detection — one-stage and modern

- **Redmon, J. et al. (2016).** "You Only Look Once" (**YOLO**). *CVPR*. — real-time single-pass
  detection (§5). <https://arxiv.org/abs/1506.02640>.
- **Liu, W. et al. (2016).** "**SSD**: Single Shot MultiBox Detector." *ECCV*. — multi-scale one-stage
  detection (§5). <https://arxiv.org/abs/1512.02325>.
- **Lin, T.-Y. et al. (2017).** "Focal Loss for Dense Object Detection" (**RetinaNet**). *ICCV*. — the
  foreground/background imbalance fix (§5). <https://arxiv.org/abs/1708.02002>.
- **Lin, T.-Y. et al. (2017).** "Feature Pyramid Networks for Object Detection" (**FPN**). *CVPR*. —
  multi-scale features (§6). <https://arxiv.org/abs/1612.03144>.
- **Tian, Z. et al. (2019).** "**FCOS**: Fully Convolutional One-Stage Object Detection." *ICCV*. — an
  anchor-free detector (§4). <https://arxiv.org/abs/1904.01355>.
- **Carion, N. et al. (2020).** "End-to-End Object Detection with Transformers" (**DETR**). *ECCV*. —
  set prediction, no anchors, no NMS (§6). <https://arxiv.org/abs/2005.12872>.

## Segmentation

- **Long, J. et al. (2015).** "Fully Convolutional Networks for Semantic Segmentation" (**FCN**).
  *CVPR*. — the first end-to-end pixel labeler (§8). <https://arxiv.org/abs/1411.4038>.
- **Ronneberger, O. et al. (2015).** "**U-Net**: Convolutional Networks for Biomedical Image
  Segmentation." *MICCAI*. — encoder-decoder with skip connections (§8).
  <https://arxiv.org/abs/1505.04597>.
- **He, K. et al. (2017).** "**Mask R-CNN**." *ICCV*. — instance segmentation with RoIAlign (§8).
  <https://arxiv.org/abs/1703.06870>.

## Metrics and datasets

- **Everingham, M. et al. (2010).** "The PASCAL Visual Object Classes (VOC) Challenge." *IJCV*. — the
  original IoU-based matching and AP definition (§2, §7).
- **Lin, T.-Y. et al. (2014).** "Microsoft **COCO**: Common Objects in Context." *ECCV*. — the standard
  detection/segmentation benchmark and the mAP@[.5:.95] protocol (§7).
  <https://arxiv.org/abs/1405.0312>.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`torchvision.ops`](https://github.com/pytorch/vision/tree/main/torchvision/ops) | `box_iou`, `nms`, `roi_align` — verified against here |
| [`torchvision.models.detection`](https://github.com/pytorch/vision/tree/main/torchvision/models/detection) | reference Faster/Mask R-CNN, RetinaNet, SSD |
| [Detectron2](https://github.com/facebookresearch/detectron2) | production detection/segmentation library |
| [`pycocotools`](https://github.com/cocodataset/cocoapi) | the official mAP evaluation code |

---

## Deferred to later chapters

- **The backbones** → [08.02](../02-cnn-architectures/)
- **Focal loss** → [07.04](../../07-deep-learning/04-loss-functions/)
- **The transformer behind DETR** → [11.02](../../11-transformers-llms/02-transformer-architecture/)
- **Vision transformers as backbones** → [08.05](../05-vision-transformers/)
- **Precision/recall/PR curves in general** → [05.03](../../05-model-evaluation/03-classification-metrics/)
