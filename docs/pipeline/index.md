# Pipeline

Three trained models, two business-logic modules, one orchestrator.

## Execution order

`DamageDetector.predict(image)` runs the following in sequence:

1. [Stage 0](stage0.md) — classify the car model (or skip if the user picked one manually in the UI).
2. [Stage 1](stage1.md) — segment the parts.
3. [Stage 2](stage2.md) — segment the damages.
4. [Fusion](fusion.md) — pair each damage with the part it sits on (IoMin overlap).
5. [Pricing](pricing.md) — look up the MAD cost for each `(part, damage, tier)` tuple.
6. NLP — per-finding French sentence via the [template engine](../nlp/templates.md), then concatenated into a report.

## Independence

Stages 0/1/2 don't share weights or features. They're three separate Keras / PyTorch models, each fine-tuned on its own dataset.

The orchestrator just feeds them all the same image. This means you can:

- Use Stage 2 standalone if you only want damages.
- Replace any single stage without retraining the others (e.g. swap Stage 1 for a different parts model).
- Run Stages 1 and 2 in parallel on multi-GPU setups (current implementation is sequential).

## Per-stage summary

| Stage | Task | Model | Classes | Input |
|---|---|---|---|---|
| [0](stage0.md) | classification | EfficientNetB0 | 20 cars | 224×224 uint8 |
| [1](stage1.md) | instance seg | YOLOv8s-seg | 23 parts | 640×640 |
| [2](stage2.md) | instance seg | YOLOv8s-seg | 4 damages | 640×640 |

## Business logic

- [Fusion](fusion.md) — `_iomin(d_mask, p_mask)` is the only function with non-trivial math.
- [Pricing](pricing.md) — precomputed `(part, damage, tier) → row` dict for O(1) lookup; fallback to category averages when the part is unknown.
