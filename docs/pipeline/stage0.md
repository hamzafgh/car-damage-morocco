# Stage 0 — Car classifier

Identifies which of 20 Moroccan-market car models is in the photo.

Module: `src/car_damage_morocco/stage0_classifier.py`.

## Why a classifier (and not detection)

We only need the **make/model**, not where the car is in the frame. The image is assumed to contain one focal car (insurance photo, claim photo). A 20-way softmax is simpler, smaller and more accurate than a detection head.

## Model

EfficientNetB0 + GAP + BatchNorm + Dropout(0.4) + Dense(256, ReLU, L2=0.01) + Dropout(0.3) + Dense(20, softmax).

| Input | `(224, 224, 3)` uint8 in `[0, 255]` |
| --- | --- |
| Preprocessing | **None** — EfficientNetB0 handles it internally |
| Output | softmax over 20 classes (alphabetically sorted) |
| Reported accuracy | ~92% test |
| Weights file | `models/stage0/best.keras` |

!!! note "Why no Rescaling layer"
    EfficientNetB0 (and B1–B7) include a `Normalization` layer inside the backbone in TF/Keras. Passing already-normalized floats `[0, 1]` would double-normalize and crash accuracy. We pass raw uint8 in `[0, 255]`.

## Classes (20)

Class indices `0..19` are alphabetical, frozen in [`data/stage0_classes.json`](https://github.com/hamzafgh/car-damage-morocco/blob/main/data/stage0_classes.json):

`Citroen_C4L` · `Citroen_Elysee` · `Dacia_Duster` · `Dacia_Logan` · `Dacia_Sandero` · `Fiat_Punto` · `Hyundai_Accent` · `Hyundai_Tucson` · `Mercedes_C_Class` · `Mercedes_E_Class` · `Peugeot_208` · `Peugeot_301` · `Peugeot_308` · `Renault_Captur` · `Renault_Clio` · `Toyota_Corolla` · `Toyota_Yaris` · `VW_Golf` · `VW_Polo` · `VW_Tiguan`

## Training (Kaggle T4, three phases)

| Phase | Backbone | Epochs | LR | Notes |
|---|---|---|---|---|
| 1 | Frozen | 20 | Adam 1e-3 | Train head only |
| 2 | Last 30 layers unfrozen | 30 | Adam 1e-4 | Light fine-tune |
| 3 | Fully unfrozen | 40 | Adam 1e-4 | Heavy aug + label smoothing 0.1 |

Data was a CompCars subset filtered for the Moroccan parc auto. Notebook: [`stage0_car_classifier.ipynb`](../notebooks.md).

## API

```python
from car_damage_morocco.stage0_classifier import CarClassifier

clf = CarClassifier(
    weights_path="models/stage0/best.keras",
    classes_json="data/stage0_classes.json",
)
result = clf.predict(image_bgr_or_rgb, top_k=3)
```

Returns:

```python
{
    "label":         "Dacia_Logan",       # raw class name
    "display_label": "Dacia Logan",       # pretty name for UI
    "confidence":    0.94,
    "topk": [
        ("Dacia_Logan",   0.94),
        ("Dacia_Sandero", 0.04),
        ("Peugeot_208",   0.01),
    ],
}
```

## What the output feeds

The `label` is the key that the [Pricing](pricing.md) module uses to look up the **tier** in [`car_model_tiers.json`](https://github.com/hamzafgh/car-damage-morocco/blob/main/data/car_model_tiers.json):

| Tier | Examples |
|---|---|
| **economy** | Dacia (Logan/Sandero/Duster), Peugeot 208/301, Hyundai Accent, Toyota Yaris, Fiat Punto, Citroen Elysee |
| **mid_range** | VW (Golf/Polo/Tiguan), Peugeot 308, Renault Captur, Hyundai Tucson, Citroen C4L, Toyota Corolla |
| **premium** | Mercedes C-Class, Mercedes E-Class |

The tier doesn't change *which* repairs are needed — only how much they cost. A bumper replacement on a Mercedes costs more than on a Dacia.
