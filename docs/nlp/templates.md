# Template engine — `describe_damage.py`

Deterministic French generator that turns structured pipeline output into one French sentence per damage, plus an overall multi-sentence report.

Module: `src/car_damage_morocco/nlp/describe_damage.py`.

## Input

A `Damage` dataclass per detection:

```python
@dataclass
class Damage:
    part:        str                          # carparts label (e.g. 'front_left_door')
    damage_type: str                          # 'dent' | 'scratch' | 'glass' | 'broken_part'
    area_ratio:  float                        # in [0, 1]
    position:    tuple[str, str] | None       # ('top'|'middle'|'bottom', 'left'|'center'|'right')
    severity:    Literal[...] | None          # if None, derived from area_ratio
```

## Four French lexicons

### `PART_FR` — 22 entries

Maps each Stage-1 class to a French noun phrase **with its grammatical gender**:

```python
PART_FR = {
    "front_left_door":   ("la portière avant gauche", "f"),
    "hood":              ("le capot", "m"),
    "front_glass":       ("le pare-brise", "m"),
    "wheel":             ("la roue", "f"),
    # ...
}
```

Gender is the linchpin: it propagates into the severity adjective so the sentence agrees.

### `DAMAGE_FR` — 4 entries

```python
DAMAGE_FR = {
    "dent":        {"noun": "un enfoncement",   "gender": "m",
                    "verb": "présente", "action": "à débosseler"},
    "scratch":     {"noun": "une rayure",       "gender": "f",
                    "verb": "présente", "action": "à repeindre"},
    "glass":       {"noun": "un bris de vitre", "gender": "m",
                    "verb": "présente", "action": "à remplacer"},
    "broken_part": {"noun": "une casse",        "gender": "f",
                    "verb": "présente", "action": "à remplacer"},
}
```

### `SEVERITY_FR` — 3 levels × 2 genders

```python
SEVERITY_FR = {
    "minor":    {"m": "léger",     "f": "légère"},
    "moderate": {"m": "modéré",    "f": "modérée"},
    "severe":   {"m": "important", "f": "importante"},
}
```

### `POSITION_FR` — 9 grid positions

```python
POSITION_FR = {
    ("top",    "left"):   "dans la partie supérieure gauche",
    ("middle", "right"):  "sur le côté droit",
    ("bottom", "center"): "dans la partie inférieure",
    # ... etc, 3×3 grid
}
```

## Severity from area ratio

When no `severity` is provided on the `Damage`, it's derived from `area_ratio`:

```python
def severity_from_area(r: float) -> Literal["minor", "moderate", "severe"]:
    if r < 0.05:  return "minor"
    if r < 0.20:  return "moderate"
    return "severe"
```

Thresholds tuned by hand. You can defend them as "fraction of the panel area covered = how big the repair job is".

## Position from centroid

The damage centroid `(cx, cy)` (computed in [fusion](../pipeline/fusion.md)) is quantized into a 3×3 grid relative to the part's bounding box:

```python
def position_from_centroid(cx, cy, x1, y1, x2, y2):
    rx = (cx - x1) / max(1e-6, x2 - x1)
    ry = (cy - y1) / max(1e-6, y2 - y1)
    col = "left"   if rx < 1/3 else "right"  if rx > 2/3 else "center"
    row = "top"    if ry < 1/3 else "bottom" if ry > 2/3 else "middle"
    return (row, col)
```

## The template

```text
{Part_phrase.capitalize()} {verb} {damage_noun} {severity_adj}{position},
couvrant environ {pct}% de la surface.
```

## Concrete example

```python
from car_damage_morocco.nlp.describe_damage import Damage, describe

d = Damage(
    part="front_left_door",
    damage_type="scratch",
    area_ratio=0.12,
    position=("middle", "right"),
)
print(describe(d))
```

Step by step:

1. `PART_FR["front_left_door"]` → `("la portière avant gauche", "f")` — take noun: `"la portière avant gauche"`.
2. `DAMAGE_FR["scratch"]` → `{noun: "une rayure", gender: "f", verb: "présente"}`.
3. `severity_from_area(0.12)` → `"moderate"`.
4. `SEVERITY_FR["moderate"]["f"]` → `"modérée"` (matches *rayure*).
5. `POSITION_FR[("middle", "right")]` → `"sur le côté droit"`.
6. `pct = max(1, round(0.12 * 100))` = `12`.

Final sentence:

> **La portière avant gauche présente une rayure modérée sur le côté droit, couvrant environ 12% de la surface.**

## Full report

```python
def report(damages: Iterable[Damage], car_label: str | None = None) -> str:
    """Concatenate per-damage sentences with a heading."""
```

Returns:

> **Dacia Logan : 2 dommages détectés.** La portière avant gauche présente une rayure modérée sur le côté droit, couvrant environ 12% de la surface. Le pare-chocs avant présente un enfoncement important au centre, couvrant environ 25% de la surface.

## Tested invariants

```python
def test_french_description_gender_agreement():
    # 'casse' is feminine → 'importante' (not 'important')
    assert "casse importante" in describe(Damage("back_right_light", "broken_part", 0.5))
    # 'enfoncement' is masculine → 'important'
    assert "enfoncement important" in describe(Damage("hood", "dent", 0.5))

def test_french_report_no_damages():
    assert "Aucun dommage" in report([], car_label="Dacia Logan")
```

## Why templates win for production

- **Deterministic** — same input always produces the same sentence. No hallucination.
- **Grammatically correct** — gender agreement table is the only "linguistic intelligence" needed.
- **Auditable** — every word in the output traces back to a structured field.
- **Cheap** — pure dict lookups, no model inference.
