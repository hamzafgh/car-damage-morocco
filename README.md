# car-damage-morocco

[![Docs](https://readthedocs.org/projects/car-damage-morocco/badge/?version=latest)](https://car-damage-morocco.readthedocs.io/)

End-to-end car damage assessment for the Moroccan market. Photo → car model → damaged parts → cost estimate in MAD → French-language report.

📖 **[Full documentation → car-damage-morocco.readthedocs.io](https://car-damage-morocco.readthedocs.io/)**

See [ARCHITECTURE.md](ARCHITECTURE.md) for the system design.

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows
# python -m venv .venv && source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### Drop trained weights in place
After training each stage on Kaggle, download the weights and copy them here:

| Stage | Source file (on Kaggle) | Place here |
|---|---|---|
| 0 | `/kaggle/working/car_classifier_efficientnet_b0.keras` | `models/stage0/best.keras` |
| 1 | `/kaggle/working/stage1_deliverables/parts_seg_best.pt` | `models/stage1/best.pt` |
| 2 | `/kaggle/working/damage_segmenter_yolov8s.pt` | `models/stage2/best.pt` |

### Run the Streamlit demo
```bash
streamlit run app/streamlit_app.py
```
Upload an image, watch detections, get a French report with MAD pricing.

### Programmatic use
```python
import cv2
from car_damage_morocco import DamageDetector
from car_damage_morocco.detector import default_detector

detector = default_detector()                       # reads from models/ + data/
result   = detector.predict(cv2.imread("car.jpg"), render=True)

print(result.car_display, result.car_confidence)
print(result.total_MAD, "MAD")
print(result.report_fr)
for f in result.findings:
    print(f.part, f.damage_type, f.cost_MAD)
```

## Tests
```bash
python -m pytest tests -v
```
Smoke tests run without weights — they validate CSV/JSON alignment, fusion math, and French templates.

## Training notebooks
Three Kaggle notebooks (T4 GPU). Pre-rendered HTML views are linked below if GitHub's notebook renderer struggles with them.

| Stage | Model | Dataset | View |
|---|---|---|---|
| `stage0_car_classifier.ipynb` | EfficientNetB0 | 20 Moroccan-market models | [HTML](https://raw.githack.com/hamzafgh/car-damage-morocco/main/docs/notebooks/stage0_car_classifier.html) · [Colab](https://colab.research.google.com/github/hamzafgh/car-damage-morocco/blob/main/notebooks/stage0_car_classifier.ipynb) |
| `stage1_parts_seg_train.ipynb` | YOLOv8s-seg | Ultralytics carparts-seg (23 classes) | [HTML](https://raw.githack.com/hamzafgh/car-damage-morocco/main/docs/notebooks/stage1_parts_seg_train.html) · [Colab](https://colab.research.google.com/github/hamzafgh/car-damage-morocco/blob/main/notebooks/stage1_parts_seg_train.ipynb) |
| `stage2_damage_seg_train.ipynb` | YOLOv8s-seg | Roboflow `is_it_damaged` v6, 7 → 4 classes | [HTML](https://raw.githack.com/hamzafgh/car-damage-morocco/main/docs/notebooks/stage2_damage_seg_train.html) · [Colab](https://colab.research.google.com/github/hamzafgh/car-damage-morocco/blob/main/notebooks/stage2_damage_seg_train.ipynb) |

## Pricing
`data/prix_reparation_maroc.csv` — 162 rows (part × damage × tier in MAD).
Regenerate after editing the price tables: `python scripts/generate_pricing_csv.py`.

## NLP — French damage description
- `src/car_damage_morocco/nlp/describe_damage.py` — template engine (works today, grammar-correct).
- `src/car_damage_morocco/nlp/caption_model.py` + `scripts/train_caption_model.py` — CNN+LSTM end-to-end captioner trained via distillation on auto-generated captions.

## Trained components
- **Stage 0** — EfficientNetB0, ~92% test accuracy on 20 Moroccan-market car models (Kaggle).
- **Stage 2** — YOLOv8s-seg, mask mAP50 = 0.711 on the 4-class damage dataset (Kaggle).
- **Stage 1** — YOLOv8s-seg fine-tuned on Ultralytics carparts-seg, 23 classes (Kaggle).
- **Fusion · Pricing · French NLP · Streamlit dashboard** — implemented and tested.
