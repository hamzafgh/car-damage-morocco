# Stage 1 — Parts segmentation

Detects and segments **23 car-part classes** with YOLOv8s-seg.

Module: `src/car_damage_morocco/stage1_parts.py`.

## Why segmentation (and not detection)

We don't want a bounding box around the door — we want the **pixel mask** of the door. The next stage (damage) returns pixel masks too, and pairing damages with parts requires computing **area overlap** ([IoMin](fusion.md)), which requires masks.

## Model

| Property | Value |
|---|---|
| Architecture | YOLOv8s-seg (Ultralytics) |
| Pretrained on | Ultralytics `carparts-seg` dataset |
| Input | 640×640 |
| Classes | 23 |
| Weights file | `models/stage1/best.pt` |

The 24th class in the original Ultralytics carparts-seg config (`object`, id 18) is a catch-all and is **excluded from pricing** — we never want to bill a customer for repairing "object".

## Classes (23)

Class indices `0..22` from [`data/stage1_classes.json`](https://github.com/hamzafgh/car-damage-morocco/blob/main/data/stage1_classes.json):

| Front | Back | Other |
|---|---|---|
| front_bumper | back_bumper | hood |
| front_door | back_door | tailgate |
| front_glass | back_glass | trunk |
| front_left_door | back_left_door | wheel |
| front_left_light | back_left_light | left_mirror |
| front_light | back_light | right_mirror |
| front_right_door | back_right_door | object (excluded) |
| front_right_light | back_right_light | |

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

Masks are resized to the original image resolution so they line up exactly with Stage 2's output for fusion.

## Training (Kaggle T4)

YOLOv8s-seg fine-tuned for 100 epochs on the Ultralytics built-in `carparts-seg` dataset (`carparts-seg.yaml`). Notebook: [`stage1_parts_seg_train.ipynb`](../notebooks.md).

Hyperparameters worth defending:

| Param | Value | Why |
|---|---|---|
| imgsz | 640 | YOLOv8 default; balance of accuracy vs speed |
| batch | 16 | Fits in T4 16 GB VRAM with `amp=True` |
| optimizer | AdamW | More stable than SGD for fine-tuning |
| lr0 | 1e-3 | Conservative for transfer learning |
| cos_lr | True | Cosine annealing |
| mosaic | 1.0 | Strong augmentation for small dataset |
| close_mosaic | 10 | Disable mosaic for last 10 epochs |
| patience | 30 | Early stop if no improvement |

## What the output feeds

The list of `PartDetection`s is consumed by [Fusion](fusion.md), which pairs each Stage-2 damage mask with the Stage-1 part mask it overlaps.
