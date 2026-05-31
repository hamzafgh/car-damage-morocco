# Pricing

MAD cost lookup keyed by `(part, damage_type, tier)`.

Module: `src/car_damage_morocco/pricing.py`.

## Data

[`data/prix_reparation_maroc.csv`](https://github.com/hamzafgh/car-damage-morocco/blob/main/data/prix_reparation_maroc.csv) — 162 rows. Schema:

| Column | Type | Notes |
|---|---|---|
| `part` | str | Stage 1 class name (e.g. `front_bumper`) |
| `yolo_class_id` | int | Class index in `stage1_classes.json` |
| `category` | str | Grouping (e.g. `body`, `glass`, `light`) |
| `damage_type` | str | One of `dent`, `scratch`, `glass`, `broken_part` |
| `action` | str | `repair` or `replace` |
| `tier` | str | `economy`, `mid_range`, `premium` |
| `part_cost_MAD` | int | Cost of the part itself |
| `labor_MAD` | int | Man-hour cost for the action |
| `total_MAD` | int | `part_cost_MAD + labor_MAD` |

A test asserts the totals are consistent: `test_pricing_totals_consistent` verifies `total_MAD == part_cost_MAD + labor_MAD` for every row.

## Tier mapping

`data/car_model_tiers.json` maps each Stage-0 class to a tier:

```python
"Dacia_Logan"     → "economy"
"Peugeot_308"     → "mid_range"
"Mercedes_C_Class"→ "premium"
```

The mapping accepts three key formats so callers don't have to worry about underscores vs. spaces vs. hyphens:

- `Dacia_Logan` (Stage 0 raw label)
- `Dacia Logan` (display name)
- `dacia_logan` (lowercase)

## API

```python
from car_damage_morocco.pricing import PricingTable

pt = PricingTable(
    csv_path="data/prix_reparation_maroc.csv",
    tiers_json="data/car_model_tiers.json",
)

# Two-step
tier = pt.tier_for("Dacia_Logan")           # → "economy"
price = pt.lookup("front_bumper", "dent", tier)
# Price(part='front_bumper', damage_type='dent', tier='economy',
#       action='repair', part_cost_MAD=850, labor_MAD=300, total_MAD=1150,
#       is_estimate=False, note='')

# One-step (skips intermediate tier var)
price = pt.estimate("Dacia_Logan", "front_bumper", "dent")
```

## Fallback for unknown parts

When Stage 1 fails to identify the part (no part mask exceeds IoMin threshold with the damage), the orchestrator falls back to an **average** across all parts in the same `(damage_type, tier)` bucket:

```python
def fallback_estimate(self, damage_type, tier):
    sub = self.df[(self.df.damage_type == damage_type) &
                  (self.df.tier == tier)]
    if sub.empty:
        return None
    return Price(
        part="(pièce non identifiée)",
        damage_type=damage_type,
        tier=tier,
        action=sub.action.mode().iat[0],
        part_cost_MAD=int(round(sub.part_cost_MAD.mean())),
        labor_MAD=int(round(sub.labor_MAD.mean())),
        total_MAD=int(round(sub.total_MAD.mean())),
        is_estimate=True,                          # <- flagged
        note=f"Estimation moyenne sur {len(sub)} pièce(s) "
             f"(min={sub.total_MAD.min()} MAD, max={sub.total_MAD.max()} MAD).",
    )
```

The dashboard surfaces this with a yellow row in the findings table and an info banner explaining the estimate.

## Performance

Lookups are O(1). The constructor precomputes a `dict[(part, damage, tier) → row]` from the CSV; `lookup()` is one dict access.

Tier mapping is also O(1) — dict access into `self._tier_map`.

## Tested invariants

- `test_pricing_csv_loads` — schema columns are present, row count > 100, damage_type/tier/action values are in the expected set.
- `test_pricing_totals_consistent` — total = part + labor for all 162 rows.
- `test_stage2_classes_match_csv_damage_types` — exactly the same 4 damage types in the CSV as in `stage2_classes.json`.
- `test_stage1_classes_match_csv_parts` — every part in the CSV is one of the 23 Stage 1 classes.
- `test_tier_mapping_covers_stage0_classes` — every Stage 0 class has a tier.
- `test_pricing_lookup` — `Dacia_Logan` → `economy`, looking up `(front_bumper, dent, economy)` returns a valid `Price` with `action=repair` and `total_MAD > 0`.

These tests run on every commit; if you edit the CSV the suite will catch desynchronization with the class JSONs.
