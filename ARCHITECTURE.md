# Architecture — car-damage-morocco

End-to-end car-damage assessment pipeline for the Moroccan market. Given a photo of a damaged car, the system identifies the car model, locates which parts are damaged and how, then estimates repair cost in MAD using a local pricing table — and produces a French-language damage report.

```
                    +----------------------------+
   Input image ---->|  Stage 0: car classifier   |---> car_model (1 of 20)
                    |  EfficientNetB0 / Keras    |     -> tier (economy|mid|premium)
                    +----------------------------+
                                  |
                    +----------------------------+
                    |  Stage 1: parts segmenter  |---> 23 part masks
                    |  YOLOv8s-seg (carparts)    |
                    +----------------------------+
                                  |
                    +----------------------------+
                    |  Stage 2: damage segmenter |---> 4 damage masks
                    |  YOLOv8s-seg (custom)      |
                    +----------------------------+
                                  |
                    +----------------------------+
                    |  Fusion (IoMin >= 0.35)    |---> (part, damage) pairs
                    +----------------------------+
                                  |
                +----------+--------------------+--------+
                |                                        |
   +-----------------------+              +-----------------------+
   |  Pricing lookup       |              |  NLP description      |
   |  prix_reparation_     |              |  describe_damage.py   |
   |  maroc.csv            |              |  (French templates)   |
   +-----------------------+              +-----------------------+
                |                                        |
                +-------------------+--------------------+
                                    |
                            Final report (JSON + FR text + overlay PNG)
```

## Stages

### Stage 0 — Car model classifier (DONE on Kaggle)
- **Model:** EfficientNetB0 + GAP + BatchNorm + Dropout(0.4) + Dense(256, ReLU, L2=0.01) + Dropout(0.3) + Dense(20, softmax)
- **Input:** 224×224×3, uint8, NO Rescaling (EfficientNet handles preprocessing internally)
- **Output:** 20-way softmax over Moroccan-market models (see `data/stage0_classes.json`)
- **Training:** Phase 1 frozen backbone (20 epochs Adam 1e-3) → Phase 2 fine-tune last 30 layers (30 epochs Adam 1e-4) → Phase 3 full unfreeze with strong aug + label smoothing (40 epochs)
- **Output file:** `car_classifier_efficientnet_b0.keras` → place at `models/stage0/best.keras`
- **Class order:** alphabetical from `image_dataset_from_directory` — frozen in `data/stage0_classes.json`

### Stage 1 — Parts segmentation (IN PROGRESS)
- **Model:** YOLOv8s-seg fine-tuned on Ultralytics built-in `carparts-seg` (23 classes)
- **Input:** 640×640
- **Output:** Per-image list of (class_id, mask, bbox, conf)
- **Training:** `notebooks/stage1_parts_seg_train.ipynb` (T4, ~2–3h)
- **Output file:** `parts_seg_best.pt` → place at `models/stage1/best.pt`
- **Classes:** `data/stage1_classes.json`

### Stage 2 — Damage segmentation (DONE on Kaggle)
- **Model:** YOLOv8s-seg fine-tuned on remapped Roboflow `is_it_damaged` v6 (7 → 4 classes)
- **Input:** 640×640
- **Output:** Per-image list of (class_id, mask, bbox, conf) where class ∈ {0:dent, 1:scratch, 2:glass, 3:broken_part}
- **Reported mAP:** mask mAP50 ≈ 0.711
- **Output file:** `damage_segmenter_yolov8s.pt` → place at `models/stage2/best.pt`
- **Classes:** `data/stage2_classes.json`

## Fusion logic (`src/car_damage_morocco/fusion.py`)
Pair each damage mask with the part mask it overlaps using **Intersection over Minimum** (IoMin), which is robust when one mask is much smaller than the other (the common case — a small scratch on a large door):

```
IoMin(d, p) = |d ∩ p| / min(|d|, |p|)
```

- Default threshold: `0.35` (configurable).
- A damage may map to multiple parts if IoMin exceeds threshold for several (e.g. a long scratch crossing two panels) — we emit one finding per (damage, part) pair.
- If no part exceeds threshold, the damage is emitted with `part = "unknown"` and excluded from pricing.

## Pricing (`src/car_damage_morocco/pricing.py`)
Lookup key = `(part, damage_type, tier)`. Tier is derived from `car_model` via `data/car_model_tiers.json`. Returns `(action, part_cost_MAD, labor_MAD, total_MAD)` from `data/prix_reparation_maroc.csv` (162 rows covering applicable combinations).

## NLP — French damage description (`src/car_damage_morocco/nlp/`)
Two modes:

1. **Template engine** (`describe_damage.py`): deterministic French generator using structured pipeline outputs. Grammatically correct (gender agreement on damage noun + severity adjective). Works the moment the pipeline runs.
2. **CNN+LSTM captioner** (`caption_model.py` + `scripts/train_caption_model.py`): MobileNetV3-Small encoder + LSTM decoder, trained via distillation on captions auto-generated by mode 1. End-to-end image→French at inference, no pipeline dependency. Trained on Kaggle.

## Orchestrator (`src/car_damage_morocco/detector.py`)
`DamageDetector` is the single public class. Constructed once with all model paths; `detector.predict(image)` returns:
```python
{
  "car_model":  "Dacia_Logan",
  "tier":       "economy",
  "findings": [
    {"part": "front_left_door", "damage_type": "scratch",
     "area_ratio": 0.08, "iomin": 0.51,
     "action": "repair", "cost_MAD": 380,
     "description_fr": "La portière avant gauche présente une rayure modérée ..."},
    ...
  ],
  "total_MAD":  1230,
  "report_fr":  "Dacia Logan : 2 dommages détectés. ...",
  "overlay":    <numpy array, the annotated image>
}
```

## File map

```
car-damage-morocco/
├── ARCHITECTURE.md                ← you are here
├── README.md                      setup, training, inference, demo
├── requirements.txt
├── .gitignore
├── models/                        weights (gitignored)
│   ├── stage0/best.keras
│   ├── stage1/best.pt
│   └── stage2/best.pt
├── src/car_damage_morocco/
│   ├── __init__.py
│   ├── stage0_classifier.py       Keras wrapper for EfficientNetB0
│   ├── stage1_parts.py            YOLO parts-seg wrapper
│   ├── stage2_damage.py           YOLO damage-seg wrapper
│   ├── fusion.py                  IoMin fusion of damage<->part masks
│   ├── pricing.py                 CSV lookup + tier mapping
│   ├── detector.py                DamageDetector (the orchestrator)
│   └── nlp/
│       ├── describe_damage.py     French template generator
│       └── caption_model.py       CNN+LSTM end-to-end captioner
├── data/
│   ├── prix_reparation_maroc.csv  162 pricing rows
│   ├── car_model_tiers.json       20 models -> tier
│   ├── stage0_classes.json        20 car-model class names
│   ├── stage1_classes.json        24 carparts-seg class names
│   └── stage2_classes.json        4 damage class names
├── notebooks/
│   ├── stage0_car_classifier.ipynb
│   ├── stage1_parts_seg_train.ipynb
│   └── stage2_damage_seg_train.ipynb
├── scripts/
│   ├── generate_pricing_csv.py    regenerate the pricing CSV
│   └── train_caption_model.py     train the CNN+LSTM captioner
├── app/
│   └── streamlit_app.py           demo UI
├── tests/
│   └── test_pipeline.py
└── docs/
    └── (extra design notes)
```
