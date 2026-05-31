# Stage 2 — Damage segmentation

Segments **4 damage types** with YOLOv8s-seg.

Module: `src/car_damage_morocco/stage2_damage.py`.

## The 4 classes

| ID | Class | French | Action recommended |
|---|---|---|---|
| 0 | `dent` | enfoncement | repair (à débosseler) |
| 1 | `scratch` | rayure | repair (à repeindre) |
| 2 | `glass` | bris de vitre | replace (à remplacer) |
| 3 | `broken_part` | casse | replace (à remplacer) |

These come from a remapping: the Roboflow `is_it_damaged` v6 dataset originally has 7 classes; the training notebook collapses them into these 4 (e.g. `crack` → `glass` if on a window, `broken_part` otherwise).

## Model

| Property | Value |
|---|---|
| Architecture | YOLOv8s-seg (Ultralytics) |
| Source dataset | Roboflow `is_it_damaged` v6 (7 → 4 classes) |
| Input | 640×640 |
| Classes | 4 |
| Weights file | `models/stage2/best.pt` |
| Reported mask mAP50 | **0.711** |

## Overlay colors

Each class has a distinct color used in the Streamlit dashboard's annotated image. The color table is defined in two places (`detector.py` and the dashboard CSS); changing one without the other will create a mismatch.

| Class | BGR (`detector.py`) | CSS hex (dashboard) |
|---|---|---|
| `dent` | `(60, 60, 230)` | `#E63C3C` (red) |
| `scratch` | `(60, 200, 230)` | `#E6C83C` (yellow) |
| `glass` | `(230, 200, 60)` | `#3CC8E6` (cyan) |
| `broken_part` | `(60, 60, 130)` | `#823C3C` (dark red) |

## API

```python
from car_damage_morocco.stage2_damage import DamageSegmenter, DamageDetection

seg = DamageSegmenter(
    weights_path="models/stage2/best.pt",
    classes_json="data/stage2_classes.json",
)
damages: list[DamageDetection] = seg.predict(image_bgr, conf=0.25, imgsz=640)
```

Each `DamageDetection`:

```python
@dataclass
class DamageDetection:
    class_id:   int
    class_name: str         # 'dent' | 'scratch' | 'glass' | 'broken_part'
    confidence: float
    bbox:       tuple[int, int, int, int]
    mask:       np.ndarray  # bool array, shape (H, W)
```

## Training notes

The training notebook (`stage2_damage_seg_train.ipynb`) does three things:

1. **Downloads** the Roboflow dataset via API.
2. **Remaps** the 7 original classes → 4 by rewriting every YOLO label file in train/val/test.
3. **Trains** YOLOv8s-seg for ~80 epochs with strong augmentation.

The remapping happens **before** YAML is written, so the model only ever sees the 4-class universe.

## Confidence threshold

The Streamlit dashboard exposes `damage_conf` (default `0.25`) in the Inspection page's parameters expander. Lower it to catch faint scratches; raise it to suppress noise.

In production an insurance app would calibrate this against a precision/recall budget per damage class.
