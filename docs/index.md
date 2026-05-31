# car-damage-morocco

End-to-end **car damage assessment pipeline** tailored for the Moroccan market.

> Photo → car model → segmented parts → segmented damages → MAD cost → French report.

---

## What this project does

You upload a photo of a damaged car. The pipeline:

1. **Identifies the model** (1 of 20 cars common in Morocco) using a fine-tuned EfficientNetB0.
2. **Segments the parts** (23 carparts classes) with YOLOv8s-seg.
3. **Segments the damages** (4 types: `dent`, `scratch`, `glass`, `broken_part`) with another YOLOv8s-seg.
4. **Pairs damages with parts** using IoMin (intersection over minimum area).
5. **Looks up the cost** for each (part, damage, tier) tuple in a Moroccan pricing CSV.
6. **Generates a French report** in grammatical French — gender agreement and all.

The whole thing runs through a dark **AI NEXUS** Streamlit dashboard.

## Where to start

<div class="grid cards" markdown>

- :material-sitemap: **[Architecture](architecture.md)**  
  The 3-stage pipeline with diagram and data flow.

- :material-cube-outline: **[Pipeline](pipeline/index.md)**  
  Module-by-module: Stage 0, Stage 1, Stage 2, fusion, pricing.

- :material-comment-text: **[NLP](nlp/index.md)**  
  French template engine + CNN+LSTM captioner trained via distillation.

- :material-rocket-launch: **[Usage](usage/index.md)**  
  Run the demo, use the Python API, run the tests.

</div>

## Stack

- **TensorFlow / Keras** — Stage 0 (EfficientNetB0)
- **Ultralytics YOLOv8** — Stages 1 & 2 (instance segmentation)
- **OpenCV** — image decoding and overlay rendering
- **Pandas** — pricing CSV and table operations
- **Streamlit** — interactive dashboard
- **Pytest** — 10 unit tests for data alignment + business logic

## Reported metrics

| Stage | Model | Metric | Value |
|---|---|---|---|
| 0 | EfficientNetB0 | Test accuracy | **87.2 %** |
| 1 | YOLOv8s-seg | Mask mAP50 | **0.722** |
| 2 | YOLOv8s-seg | Mask mAP50 | **0.714** |

## Repository

[github.com/hamzafgh/car-damage-morocco](https://github.com/hamzafgh/car-damage-morocco) — MIT licensed.

## Authors

**Hamza El Faghloumi** · **Achraf Lemrani**
