# Fusion — IoMin

Pairs each Stage-2 damage mask with the Stage-1 part mask it sits on.

Module: `src/car_damage_morocco/fusion.py`.

## The problem

Stages 1 and 2 produce two independent lists of pixel masks. To bill a customer, you need to know *which part is damaged* — i.e. attribute each damage to a part. The standard metric for "how much does mask A overlap mask B" is **IoU (Intersection over Union)**:

$$\text{IoU}(d, p) = \frac{|d \cap p|}{|d \cup p|}$$

But IoU fails for our case: a 100-px scratch on a 50,000-px door has a tiny IoU (because the union is dominated by the door area), even though the scratch is *entirely inside* the door.

## The solution

**IoMin** — intersection divided by the smaller of the two areas:

$$\text{IoMin}(d, p) = \frac{|d \cap p|}{\min(|d|, |p|)}$$

In the scratch-on-door case, `min(100, 50000) = 100`, so IoMin = `100 / 100 = 1.0` — exactly what we want.

```python
def _iomin(d_mask: np.ndarray, p_mask: np.ndarray) -> tuple[float, int]:
    inter = int(np.logical_and(d_mask, p_mask).sum())
    if inter == 0:
        return 0.0, 0
    d_area = int(d_mask.sum())
    p_area = int(p_mask.sum())
    return inter / max(1, min(d_area, p_area)), inter
```

## Threshold

Default `iomin_threshold = 0.35`. Below that, the damage is treated as "unattached" — it's still emitted but with `part = None` (and excluded from exact pricing; the fallback estimate is used).

The dashboard exposes this slider so users can tune it per-image:

- **Lower (0.10–0.30)** — catch more damages, even partially-overlapping ones. Risk: false attribution.
- **Higher (0.50–0.90)** — only obvious overlaps. Risk: damages sliding off the edge of the part go unattributed.

## Multi-part damages

A long scratch can span two panels (e.g. a key drag from the front fender across the door). With `multi_part=True` (default), the fuser emits **one `Finding` per (damage, part) pair** that exceeds the threshold. The pricing module then bills both panels — which is the correct insurance behavior.

With `multi_part=False`, only the best-matching part is kept (highest IoMin).

## Output

```python
@dataclass
class Finding:
    damage:       DamageDetection
    part:         PartDetection | None         # None if no part exceeded threshold
    iomin:        float                         # 0..1
    area_ratio:   float                         # damage_area_inside_part / part_area
    centroid_xy:  tuple[float, float]           # damage centroid in image pixels
```

- `iomin` — the score from above. Useful for debugging.
- `area_ratio` — `intersection / part_area`. This is what the [NLP module](../nlp/templates.md) consumes for severity classification (small ratio = minor damage, large = severe).
- `centroid_xy` — image-space centroid of the damage mask. Used to position the French description ("dans la partie supérieure gauche du capot").

## Tested invariant

There's a real test in `tests/test_pipeline.py`:

```python
def test_fusion_iomin_threshold():
    # Damage entirely inside the part → IoMin ≈ 1.0, gets attached
    # Damage entirely outside the part → IoMin = 0, emitted with part=None
    findings = fuse([part], [d_inside, d_outside], iomin_threshold=0.3)
    assert len(findings) == 2
    assert inside_f.part is not None and inside_f.iomin > 0.9
    assert outside_f.part is None
```
