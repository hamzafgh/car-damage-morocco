# Stage 1 — Parts Segmentation

Detects and segments **23 car-part classes** on the input photo using YOLOv8s-seg, producing a pixel mask for each visible part.

Module: `src/car_damage_morocco/stage1_parts.py`.

---

## Why segmentation (not detection)

We don't want a bounding box around the door — we want the **pixel mask** of the door. Stage 2 (damage) also returns pixel masks, and pairing damages with parts requires computing **area overlap** ([IoMin](fusion.md)), which only works with masks — bounding boxes overlap even when the actual regions don't.

---

## Dataset

We used the **Ultralytics built-in `carparts-seg` dataset** (`carparts-seg.yaml`), which Ultralytics ships as a standard benchmark. It contains annotated car images with pixel-level masks for 23 part categories.

**Why this dataset:**
- Covers a wide range of car makes (not region-specific — parts look the same everywhere)
- Pre-annotated at pixel level — no manual labelling needed
- Directly compatible with the YOLOv8 training pipeline

The dataset is not downloaded manually — Ultralytics auto-downloads it on first use.

---

## Model

| Property | Value |
|---|---|
| Architecture | **YOLOv8s-seg** (Ultralytics) |
| Starting weights | `yolov8s-seg.pt` (COCO pretrained) |
| Input size | 640 × 640 |
| Output | Bounding box + pixel mask per detection |
| Classes | 23 |
| Weights file | `models/stage1/best.pt` |

YOLOv8s-seg was chosen over larger variants (m, l, x) because:
- The T4 Kaggle GPU fits it comfortably at batch 16 with AMP
- The "s" model generalises better than "m"/"l" on this dataset size
- Inference speed matters — Stage 1 runs in real time in the app

---

## Classes (23)

Class indices from [`data/stage1_classes.json`](https://github.com/hamzafgh/car-damage-morocco/blob/main/data/stage1_classes.json):

| Front | Back | Other |
|---|---|---|
| front_bumper | back_bumper | hood |
| front_door | back_door | tailgate |
| front_glass | back_glass | trunk |
| front_left_door | back_left_door | wheel |
| front_left_light | back_left_light | left_mirror |
| front_light | back_light | right_mirror |
| front_right_door | back_right_door | `object` *(excluded)* |
| front_right_light | back_right_light | |

!!! note "The `object` class"
    The original carparts-seg config includes a catch-all `object` class (id 18) for ambiguous parts. It has only **1 test image** and 0 mAP — we keep it in the model but exclude it from pricing. We never bill a customer for repairing "object".

---

## Training (Kaggle T4)

YOLOv8s-seg fine-tuned for **100 epochs** on `carparts-seg`. Notebook: [`stage1_parts_seg_train.ipynb`](../notebooks.md).

| Hyperparameter | Value | Rationale |
|---|---|---|
| `epochs` | 100 | Sufficient for convergence with patience=30 |
| `imgsz` | 640 | YOLOv8 default; best accuracy/speed trade-off |
| `batch` | 16 | Fits T4 16 GB VRAM with `amp=True` |
| `optimizer` | AdamW | More stable than SGD for fine-tuning from COCO |
| `lr0` | 1e-3 | Conservative learning rate for transfer learning |
| `cos_lr` | True | Cosine annealing — smooth LR decay to final epoch |
| `mosaic` | 1.0 | Strong multi-image augmentation for the full run |
| `close_mosaic` | 10 | Disable mosaic for the last 10 epochs to stabilise |
| `patience` | 30 | Early stop if mask mAP50 stagnates for 30 epochs |

---

## Results

### Overall (401 images, 2 042 instances)

| Metric | Box | Mask |
|---|---|---|
| **Precision** | 0.652 | 0.675 |
| **Recall** | 0.805 | 0.797 |
| **mAP50** | **0.699** | **0.722** |
| **mAP50-95** | 0.578 | 0.573 |

The model is **recall-biased** (finds more parts than it confidently labels) — which is the right trade-off for damage assessment. Missing a damaged part is worse than a slightly lower precision.

### Per-class results

| Class | Images | Instances | Box mAP50 | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|---|
| **front_bumper** | 208 | 208 | **0.967** | **0.967** | 0.867 |
| **front_glass** | 214 | 214 | **0.956** | **0.956** | 0.894 |
| **hood** | 214 | 214 | 0.950 | 0.950 | 0.841 |
| **back_glass** | 114 | 115 | 0.949 | 0.934 | 0.763 |
| **front_door** | 167 | 167 | 0.929 | 0.929 | 0.826 |
| **back_door** | 158 | 159 | 0.925 | 0.923 | 0.806 |
| **back_bumper** | 94 | 94 | 0.941 | 0.941 | 0.720 |
| **back_light** | 161 | 226 | 0.843 | 0.878 | 0.623 |
| **front_light** | 248 | 373 | 0.899 | 0.895 | 0.655 |
| trunk | 9 | 9 | 0.792 | 0.734 | 0.627 |
| right_mirror | 31 | 31 | 0.629 | 0.629 | 0.445 |
| back_left_door | 15 | 15 | 0.676 | 0.681 | 0.577 |
| back_left_light | 19 | 19 | 0.695 | 0.721 | 0.513 |
| back_right_door | 12 | 12 | 0.715 | 0.715 | 0.602 |
| back_right_light | 13 | 13 | 0.597 | 0.597 | 0.493 |
| tailgate | 5 | 5 | 0.526 | 0.838 | 0.516 |
| left_mirror | 31 | 31 | 0.509 | 0.509 | 0.336 |
| front_right_light | 26 | 26 | 0.698 | 0.698 | 0.590 |
| front_left_door | 15 | 15 | 0.600 | 0.600 | 0.524 |
| wheel | 34 | 53 | 0.405 | 0.626 | 0.291 |
| front_left_light | 30 | 30 | 0.444 | 0.444 | 0.318 |
| front_right_door | 12 | 12 | 0.435 | 0.435 | 0.353 |
| object *(excluded)* | 1 | 1 | 0 | 0 | 0 |

### Why some classes perform poorly

The pattern is consistent: **high-instance, symmetric classes perform best; low-instance, side-specific classes perform worst.**

| Group | Examples | mAP50 | Reason |
|---|---|---|---|
| Large symmetric parts | front_bumper, front_glass, hood | 0.95–0.97 | Hundreds of instances, always visible, distinctive shape |
| Generic lights/doors | front_light, back_light, front_door | 0.85–0.93 | Good instance count; some shape variation |
| Side-specific doors | front_left_door, front_right_door | 0.44–0.60 | Only 12–15 test images; model conflates with generic `front_door` |
| Mirrors | left_mirror, right_mirror | 0.51–0.63 | Small region, easily occluded |
| Wheel | wheel | 0.41 (box) / 0.63 (mask) | Multiple wheels per image with overlapping bboxes |
| object | object | 0 | 1 test image; intentionally ignored |

!!! tip "In practice"
    The parts that matter most for damage assessment — bumpers, doors, glass, hood — are exactly the ones with the highest mAP. Side-specific classes (front_left_door etc.) are a nice-to-have; when they fail, the fusion step still attaches the damage to the generic `front_door` mask.

---

## API

```python
from car_damage_morocco.stage1_parts import PartsSegmenter, PartDetection

seg = PartsSegmenter(
    weights_path="models/stage1/best.pt",
    classes_json="data/stage1_classes.json",
)
parts: list[PartDetection] = seg.predict(image_bgr, conf=0.25, imgsz=640)
```

Each `PartDetection`:

```python
@dataclass
class PartDetection:
    class_id:   int
    class_name: str
    confidence: float
    bbox:       tuple[int, int, int, int]   # (x1, y1, x2, y2) pixel coords
    mask:       np.ndarray                  # bool array, shape (H, W)
```

Masks are resized to the **original image resolution** so they align exactly with Stage 2 output for pixel-level fusion.

---

## What the output feeds

The list of `PartDetection`s goes into [Fusion](fusion.md), which pairs each Stage 2 damage mask with the Stage 1 part it overlaps — using **IoMin** (intersection over minimum area) as the overlap metric.
