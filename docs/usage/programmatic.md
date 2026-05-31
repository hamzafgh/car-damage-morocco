# Programmatic API

Use the library from any Python code, no Streamlit needed.

## Minimum example

```python
import cv2
from car_damage_morocco.detector import default_detector

detector = default_detector()                   # reads from models/ + data/
result   = detector.predict(cv2.imread("car.jpg"), render=True)

print(result.car_display, f"{result.car_confidence:.1%}")
print(result.total_MAD, "MAD")
print(result.report_fr)

for f in result.findings:
    print(f"  {f.part:25s} {f.damage_type:12s} {f.cost_MAD or 0} MAD")
```

## Custom paths

`default_detector()` assumes the repo's standard layout. To use weights or data files in other locations:

```python
from car_damage_morocco.detector import DamageDetector

detector = DamageDetector(
    stage0_weights      = "/path/to/best.keras",
    stage1_weights      = "/path/to/parts.pt",
    stage2_weights      = "/path/to/damages.pt",
    stage0_classes_json = "/path/to/stage0_classes.json",
    stage1_classes_json = "/path/to/stage1_classes.json",
    stage2_classes_json = "/path/to/stage2_classes.json",
    prices_csv          = "/path/to/prix.csv",
    tiers_json          = "/path/to/tiers.json",
    parts_conf          = 0.25,
    damage_conf         = 0.25,
    iomin_threshold     = 0.35,
    multi_part          = True,
)
```

## Skipping Stage 0

If you already know the car model (e.g. the user entered the VIN), pass `car_label_override` to skip Stage 0:

```python
result = detector.predict(
    image_bgr,
    car_label_override="Dacia_Logan",   # one of the 20 Stage-0 classes
)
```

Stage 0 isn't run, `car_confidence` is set to `1.0`, `topk` contains just the override.

## Tuning thresholds at runtime

The three inference knobs are simple attributes — mutate them between calls:

```python
detector.parts_conf       = 0.15
detector.damage_conf      = 0.20
detector.iomin_threshold  = 0.30
result = detector.predict(image_bgr)
```

## Output shape

```python
@dataclass
class DetectionResult:
    car_label:      str                              # 'Dacia_Logan'
    car_display:    str                              # 'Dacia Logan'
    car_confidence: float                            # 0.94
    car_topk:       list[tuple[str, float]]
    tier:           str | None                       # 'economy' | 'mid_range' | 'premium' | None
    findings:       list[DamageRecord]
    total_MAD:      int
    report_fr:      str                              # French multi-sentence report
    overlay:        np.ndarray | None                # annotated BGR (only if render=True)
    raw:            dict[str, Any]                   # n_parts_detected, n_damages_detected

    def to_dict(self) -> dict:
        """Same as the JSON output in the dashboard. Drops overlay+raw."""
```

And per finding:

```python
@dataclass
class DamageRecord:
    part:            str                             # 'front_left_door' | 'unknown'
    damage_type:     str                             # 'dent' | 'scratch' | 'glass' | 'broken_part'
    damage_conf:     float
    part_conf:       float
    iomin:           float
    area_ratio:      float
    action:          str | None                      # 'repair' | 'replace' | None
    part_cost_MAD:   int | None
    labor_MAD:       int | None
    cost_MAD:        int | None
    description_fr:  str                             # one French sentence
    is_estimate:     bool                            # True if from fallback_estimate
    estimate_note:   str
```

## Saving the overlay to disk

```python
import cv2
result = detector.predict(image_bgr, render=True)
if result.overlay is not None:
    cv2.imwrite("annotated.png", result.overlay)
```

## Batch processing

For an offline job over many images, instantiate the detector **once** and reuse it:

```python
detector = default_detector()
for path in image_paths:
    img = cv2.imread(str(path))
    if img is None:
        continue
    result = detector.predict(img)
    save_json(path.with_suffix('.json'), result.to_dict())
```

Each `default_detector()` call loads all three model weights from disk (~30 seconds total on CPU). Don't put it in a loop.
