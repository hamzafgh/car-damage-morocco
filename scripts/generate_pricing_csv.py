"""Generate prix_reparation_maroc.csv for the car-damage-morocco pipeline.

Schema:
  part            : part name (matches Stage 1 / carparts-seg class name)
  yolo_class_id   : integer class id from carparts-seg (0..23)
  category        : exterior_panel | door | glass | lighting | mirror | wheel
  damage_type     : dent | scratch | glass | broken_part   (Stage 2 classes)
  action          : repair | replace
  tier            : economy | mid_range | premium          (driven by car model)
  part_cost_MAD   : OEM/aftermarket part cost in MAD (0 if action=repair)
  labor_MAD       : labor + paint in MAD
  total_MAD       : final price = part_cost_MAD + labor_MAD

Pricing rules (kept explicit so you can edit easily):
- dent       -> action=repair  on body panels, doors, wheel
- scratch    -> action=repair  on body panels, doors, mirrors, wheel
- glass      -> action=replace on glass parts only
- broken_part-> action=replace on any part
"""
from __future__ import annotations
import csv
from pathlib import Path

# ------------------------------------------------------------------
# Carparts-seg class id -> name  (Ultralytics built-in dataset)
# ------------------------------------------------------------------
PARTS = [
    # Class ids match the actual Ultralytics carparts-seg.yaml (23 classes confirmed from the downloaded zip).
    # NOTE: there is NO back_window or front_window in this dataset.
    # 'object' (id 18) is a catch-all and is intentionally excluded from pricing.
    (0,  "back_bumper",       "exterior_panel"),
    (1,  "back_door",         "door"),
    (2,  "back_glass",        "glass"),
    (3,  "back_left_door",    "door"),
    (4,  "back_left_light",   "lighting"),
    (5,  "back_light",        "lighting"),
    (6,  "back_right_door",   "door"),
    (7,  "back_right_light",  "lighting"),
    (8,  "front_bumper",      "exterior_panel"),
    (9,  "front_door",        "door"),
    (10, "front_glass",       "glass"),
    (11, "front_left_door",   "door"),
    (12, "front_left_light",  "lighting"),
    (13, "front_light",       "lighting"),
    (14, "front_right_door",  "door"),
    (15, "front_right_light", "lighting"),
    (16, "hood",              "exterior_panel"),
    (17, "left_mirror",       "mirror"),
    # id 18 = 'object' — catch-all, no pricing
    (19, "right_mirror",      "mirror"),
    (20, "tailgate",          "exterior_panel"),
    (21, "trunk",             "exterior_panel"),
    (22, "wheel",             "wheel"),
]

# ------------------------------------------------------------------
# Per-part base costs in MAD (repair_labor, full_replacement) per tier
# Numbers are Morocco-market estimates (Casablanca/Rabat garages, 2025).
# Edit freely.
# ------------------------------------------------------------------
# key = part_name
# value = {tier: (repair_labor_MAD, part_cost_MAD, replace_labor_MAD)}
COSTS: dict[str, dict[str, tuple[int, int, int]]] = {
    # ---- body panels (front/back bumpers, hood, trunk, tailgate) ----
    "front_bumper": {"economy": ( 700,  800,  600), "mid_range": (1000, 1500,  800), "premium": (2200, 4500, 1500)},
    "back_bumper":  {"economy": ( 650,  750,  550), "mid_range": ( 950, 1400,  750), "premium": (2000, 4200, 1400)},
    "hood":         {"economy": ( 900, 1200,  800), "mid_range": (1300, 2200, 1100), "premium": (2800, 6500, 2000)},
    "trunk":        {"economy": ( 850, 1100,  750), "mid_range": (1200, 2000, 1000), "premium": (2600, 6000, 1900)},
    "tailgate":     {"economy": ( 950, 1300,  800), "mid_range": (1400, 2300, 1100), "premium": (2900, 6800, 2100)},
    # ---- doors ----
    "front_door":       {"economy": (1000, 1500,  900), "mid_range": (1500, 2800, 1200), "premium": (3200, 8500, 2300)},
    "front_left_door":  {"economy": (1000, 1500,  900), "mid_range": (1500, 2800, 1200), "premium": (3200, 8500, 2300)},
    "front_right_door": {"economy": (1000, 1500,  900), "mid_range": (1500, 2800, 1200), "premium": (3200, 8500, 2300)},
    "back_door":        {"economy": ( 950, 1400,  850), "mid_range": (1450, 2700, 1150), "premium": (3100, 8200, 2200)},
    "back_left_door":   {"economy": ( 950, 1400,  850), "mid_range": (1450, 2700, 1150), "premium": (3100, 8200, 2200)},
    "back_right_door":  {"economy": ( 950, 1400,  850), "mid_range": (1450, 2700, 1150), "premium": (3100, 8200, 2200)},
    # ---- glass ----
    # Only front_glass and back_glass exist in carparts-seg (no side windows).
    "front_glass":  {"economy": (   0,  900,  400), "mid_range": (   0, 1600,  500), "premium": (   0, 4800,  900)},
    "back_glass":   {"economy": (   0,  700,  350), "mid_range": (   0, 1300,  450), "premium": (   0, 3800,  800)},
    # ---- lighting ----
    "front_light":       {"economy": (   0,  350, 150), "mid_range": (   0,  700, 200), "premium": (   0, 3500, 400)},
    "front_left_light":  {"economy": (   0,  350, 150), "mid_range": (   0,  700, 200), "premium": (   0, 3500, 400)},
    "front_right_light": {"economy": (   0,  350, 150), "mid_range": (   0,  700, 200), "premium": (   0, 3500, 400)},
    "back_light":        {"economy": (   0,  250, 120), "mid_range": (   0,  500, 180), "premium": (   0, 2200, 350)},
    "back_left_light":   {"economy": (   0,  250, 120), "mid_range": (   0,  500, 180), "premium": (   0, 2200, 350)},
    "back_right_light":  {"economy": (   0,  250, 120), "mid_range": (   0,  500, 180), "premium": (   0, 2200, 350)},
    # ---- mirrors ----
    "left_mirror":  {"economy": ( 150,  250, 100), "mid_range": ( 250,  500, 150), "premium": ( 700, 1800, 300)},
    "right_mirror": {"economy": ( 150,  250, 100), "mid_range": ( 250,  500, 150), "premium": ( 700, 1800, 300)},
    # ---- wheel ----
    "wheel": {"economy": ( 250,  600, 200), "mid_range": ( 400, 1200, 300), "premium": (1000, 3500, 500)},
}

DAMAGE_TYPES = ["dent", "scratch", "glass", "broken_part"]
TIERS        = ["economy", "mid_range", "premium"]

# Which (category, damage_type) pairs are valid
APPLICABLE = {
    ("exterior_panel", "dent"):        "repair",
    ("exterior_panel", "scratch"):     "repair",
    ("exterior_panel", "broken_part"): "replace",
    ("door",           "dent"):        "repair",
    ("door",           "scratch"):     "repair",
    ("door",           "broken_part"): "replace",
    ("glass",          "glass"):       "replace",
    ("glass",          "broken_part"): "replace",
    ("lighting",       "broken_part"): "replace",
    ("mirror",         "scratch"):     "repair",
    ("mirror",         "broken_part"): "replace",
    ("wheel",          "dent"):        "repair",
    ("wheel",          "scratch"):     "repair",
    ("wheel",          "broken_part"): "replace",
}

# Scratch is cheaper than dent (paint only vs body work + paint)
SCRATCH_DISCOUNT = 0.4


def compute_row(part: str, cls_id: int, category: str, dmg: str, tier: str) -> dict | None:
    action = APPLICABLE.get((category, dmg))
    if action is None:
        return None
    repair_labor, part_cost, replace_labor = COSTS[part][tier]
    if action == "repair":
        part_c, labor = 0, repair_labor
        if dmg == "scratch":
            labor = int(round(repair_labor * SCRATCH_DISCOUNT))
    else:  # replace
        part_c, labor = part_cost, replace_labor
    return {
        "part":           part,
        "yolo_class_id":  cls_id,
        "category":       category,
        "damage_type":    dmg,
        "action":         action,
        "tier":           tier,
        "part_cost_MAD":  part_c,
        "labor_MAD":      labor,
        "total_MAD":      part_c + labor,
    }


def main() -> None:
    rows: list[dict] = []
    for cls_id, name, category in PARTS:
        for dmg in DAMAGE_TYPES:
            for tier in TIERS:
                r = compute_row(name, cls_id, category, dmg, tier)
                if r is not None:
                    rows.append(r)
    out = Path(__file__).resolve().parents[1] / "data" / "prix_reparation_maroc.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
