# Training notebooks

Three notebooks, one per trained stage. All run on Kaggle T4.

| Notebook | Model | Dataset | View | Run |
|---|---|---|---|---|
| `stage0_car_classifier.ipynb` | EfficientNetB0 | 20 Moroccan-market cars | [HTML](https://raw.githack.com/hamzafgh/car-damage-morocco/main/docs/notebooks/stage0_car_classifier.html) | [Open in Colab](https://colab.research.google.com/github/hamzafgh/car-damage-morocco/blob/main/notebooks/stage0_car_classifier.ipynb) |
| `stage1_parts_seg_train.ipynb` | YOLOv8s-seg | Ultralytics carparts-seg (23 classes) | [HTML](https://raw.githack.com/hamzafgh/car-damage-morocco/main/docs/notebooks/stage1_parts_seg_train.html) | [Open in Colab](https://colab.research.google.com/github/hamzafgh/car-damage-morocco/blob/main/notebooks/stage1_parts_seg_train.ipynb) |
| `stage2_damage_seg_train.ipynb` | YOLOv8s-seg | Roboflow `is_it_damaged` v6, 7→4 classes | [HTML](https://raw.githack.com/hamzafgh/car-damage-morocco/main/docs/notebooks/stage2_damage_seg_train.html) | [Open in Colab](https://colab.research.google.com/github/hamzafgh/car-damage-morocco/blob/main/notebooks/stage2_damage_seg_train.ipynb) |

## Why pre-rendered HTML

GitHub's notebook renderer occasionally chokes on valid notebooks (cache issues, render queue, OOM). The pre-rendered HTML files in [`docs/notebooks/`](https://github.com/hamzafgh/car-damage-morocco/tree/main/docs/notebooks) are served via raw.githack — a CDN that delivers raw GitHub files as actual HTML. They always work.

## Stage 0 — Car classifier

Three-phase fine-tuning of EfficientNetB0:

1. Frozen backbone, 20 epochs (Adam 1e-3, head-only training)
2. Last 30 layers unfrozen, 30 epochs (Adam 1e-4)
3. Fully unfrozen, 40 epochs (Adam 1e-4, label smoothing 0.1)

Outputs: `car_classifier_efficientnet_b0.keras` → place at `models/stage0/best.keras`.

Final test accuracy: **87.2 %** (weighted avg F1: 0.87).

## Stage 1 — Parts segmentation

Single-phase fine-tune of `yolov8s-seg.pt` on the Ultralytics built-in `carparts-seg` dataset.

Key hyperparameters: `epochs=100`, `imgsz=640`, `batch=16`, `optimizer=AdamW`, `lr0=1e-3`, `cos_lr=True`, `mosaic=1.0`, `close_mosaic=10`, `patience=30`.

Outputs: `parts_seg/weights/best.pt` → place at `models/stage1/best.pt`.

## Stage 2 — Damage segmentation

1. **Download** the Roboflow `is_it_damaged` v6 dataset via API.
2. **Remap** the 7 original classes → 4 (`dent`, `scratch`, `glass`, `broken_part`).
3. **Train** YOLOv8s-seg.

Outputs: `damage_segmenter_yolov8s.pt` → place at `models/stage2/best.pt`.

Final mask mAP50: **0.711**.

## Re-rendering the HTML after re-training

If you re-run a notebook and want to refresh the rendered HTML in `docs/notebooks/`:

```bash
pip install nbconvert
python -m nbconvert --to html notebooks/stage0_car_classifier.ipynb --output-dir docs/notebooks/
git add docs/notebooks/ && git commit -m "docs: refresh stage0 rendered notebook" && git push
```

For a notebook that picked up large outputs (matplotlib plots, sample images), strip them first or the .ipynb itself bloats the repo:

```bash
pip install nbstripout
nbstripout notebooks/stage0_car_classifier.ipynb
```

Or install nbstripout as a git filter so this happens automatically on every commit:

```bash
nbstripout --install
```
