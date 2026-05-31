# Architecture

End-to-end pipeline: a single photo in, a structured damage assessment out.

## Diagram

```mermaid
flowchart TD
    A[Input image] --> S0[Stage 0<br/>EfficientNetB0<br/>car classifier]
    A --> S1[Stage 1<br/>YOLOv8s-seg<br/>parts]
    A --> S2[Stage 2<br/>YOLOv8s-seg<br/>damages]

    S0 --> M[car_model<br/>1 of 20]
    M --> T{tier<br/>economy / mid_range / premium}

    S1 --> F[Fusion<br/>IoMin ≥ 0.35]
    S2 --> F
    F --> Findings[(part, damage)<br/>pairs]

    Findings --> P[Pricing lookup<br/>part × damage × tier]
    T --> P

    Findings --> N[NLP<br/>French templates]

    P --> R[Final report]
    N --> R

    R -.-> J[JSON output]
    R -.-> FR[French markdown]
    R -.-> O[Overlay PNG]
```

## Why three separate models

Each stage solves a problem it's well-suited for:

- **Classification** for the car model (one of 20 fixed classes — Stage 0).
- **Instance segmentation** for car parts (Stage 1) and damages (Stage 2) — we need *pixel masks*, not just bounding boxes, to compute area overlap.

Stages don't share weights or features. They're independent specialists. The orchestrator just feeds them all the same image and merges the outputs.

## Data flow

| Step | Component | Output |
|---|---|---|
| 1 | [Stage 0](pipeline/stage0.md) | `car_label`, top-3 confidences |
| 2 | [Stage 1](pipeline/stage1.md) | N part masks `(class_name, conf, mask, bbox)` |
| 3 | [Stage 2](pipeline/stage2.md) | M damage masks `(class_name, conf, mask, bbox)` |
| 4 | [Fusion](pipeline/fusion.md) | List of `Finding(damage, part, iomin, area_ratio)` |
| 5 | [Pricing](pipeline/pricing.md) | Per finding: `(action, part_cost_MAD, labor_MAD, total_MAD)` |
| 6 | [NLP](nlp/templates.md) | Per finding: French sentence ; full report |

## Orchestrator — `DamageDetector`

`DamageDetector.predict(image_bgr)` returns a [`DetectionResult`](https://github.com/hamzafgh/car-damage-morocco/blob/main/src/car_damage_morocco/detector.py):

```python
@dataclass
class DetectionResult:
    car_label:      str
    car_display:    str
    car_confidence: float
    car_topk:       list[tuple[str, float]]
    tier:           str | None
    findings:       list[DamageRecord]
    total_MAD:      int
    report_fr:      str
    overlay:        np.ndarray | None    # annotated BGR when render=True
    raw:            dict[str, Any]
```

Construct once at app startup (`default_detector(repo_root)`), call `.predict()` per request.

## Repository layout

```
car-damage-morocco/
├── README.md
├── ARCHITECTURE.md
├── LICENSE
├── requirements.txt
├── mkdocs.yml
├── .readthedocs.yaml
│
├── src/car_damage_morocco/        ← library code
│   ├── __init__.py
│   ├── detector.py                ← DamageDetector
│   ├── stage0_classifier.py
│   ├── stage1_parts.py
│   ├── stage2_damage.py
│   ├── fusion.py                  ← IoMin
│   ├── pricing.py                 ← MAD lookup
│   └── nlp/
│       ├── describe_damage.py     ← templates
│       └── caption_model.py       ← CNN+LSTM
│
├── data/
│   ├── prix_reparation_maroc.csv
│   ├── car_model_tiers.json
│   ├── stage0_classes.json        ← 20 car models
│   ├── stage1_classes.json        ← 23 parts
│   └── stage2_classes.json        ← 4 damages
│
├── models/
│   ├── stage0/best.keras          (gitignored, fetch from Kaggle)
│   ├── stage1/best.pt
│   └── stage2/best.pt
│
├── notebooks/                     ← Kaggle training notebooks
│   ├── stage0_car_classifier.ipynb
│   ├── stage1_parts_seg_train.ipynb
│   └── stage2_damage_seg_train.ipynb
│
├── scripts/                       ← standalone CLI tools
│   ├── generate_pricing_csv.py
│   ├── train_caption_model.py
│   ├── inspect_stage2.py
│   └── verify_weights.py
│
├── app/
│   ├── streamlit_app.py           ← dashboard
│   └── static/theme.css
│
├── tests/
│   └── test_pipeline.py           ← 10 invariants
│
└── docs/                          ← THIS site
```
