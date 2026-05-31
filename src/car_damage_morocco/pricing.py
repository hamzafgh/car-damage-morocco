"""Pricing — MAD cost lookup keyed by (part, damage_type, tier).

Loads prix_reparation_maroc.csv (162 rows) and car_model_tiers.json (20 models -> tier).
Stage 0 returns a label like 'Dacia_Logan'; this module maps it to a tier and then
to a CSV row. Lookups are O(1) via a precomputed dict.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

Tier = Literal["economy", "mid_range", "premium"]


@dataclass
class Price:
    part:          str
    damage_type:   str
    tier:          Tier
    action:        Literal["repair", "replace"]
    part_cost_MAD: int
    labor_MAD:     int
    total_MAD:     int
    is_estimate:   bool = False     # True when this was a fallback (e.g. unknown part)
    note:          str  = ""        # explanation shown to the user when is_estimate=True


class PricingTable:
    def __init__(self, csv_path: str | Path, tiers_json: str | Path):
        self.df = pd.read_csv(csv_path)
        info = json.loads(Path(tiers_json).read_text(encoding="utf-8"))
        # accept both 'Dacia_Logan' and 'Dacia Logan' as keys -> tier
        self._tier_map: dict[str, Tier] = {}
        for m in info["models"]:
            label = m["label"]                       # e.g. "Dacia Logan"
            tier  = m["tier"]
            self._tier_map[label] = tier
            self._tier_map[label.replace(" ", "_").replace("-", "_")] = tier
            self._tier_map[label.lower()] = tier

        # precompute (part, damage_type, tier) -> row dict
        self._lookup: dict[tuple[str, str, Tier], dict] = {}
        for _, r in self.df.iterrows():
            self._lookup[(r["part"], r["damage_type"], r["tier"])] = r.to_dict()

    def tier_for(self, car_label: str) -> Tier | None:
        return self._tier_map.get(car_label) or self._tier_map.get(car_label.lower())

    def lookup(self, part: str, damage_type: str, tier: Tier) -> Price | None:
        row = self._lookup.get((part, damage_type, tier))
        if row is None:
            return None
        return Price(
            part=row["part"],
            damage_type=row["damage_type"],
            tier=row["tier"],
            action=row["action"],
            part_cost_MAD=int(row["part_cost_MAD"]),
            labor_MAD=int(row["labor_MAD"]),
            total_MAD=int(row["total_MAD"]),
            is_estimate=False,
        )

    def fallback_estimate(self, damage_type: str, tier: Tier) -> Price | None:
        """Average cost across all parts in (damage_type, tier).

        Used when the pipeline detects a damage but Stage 1 couldn't identify
        which part it sits on. We average across applicable parts so the user
        still gets a useful ballpark figure instead of 0.
        """
        sub = self.df[(self.df.damage_type == damage_type) & (self.df.tier == tier)]
        if sub.empty:
            return None
        part_cost = int(round(sub.part_cost_MAD.mean()))
        labor     = int(round(sub.labor_MAD.mean()))
        total     = part_cost + labor
        # Pick the most-common action for this (damage_type, tier) bucket
        action = sub.action.mode().iat[0]
        return Price(
            part="(pièce non identifiée)",
            damage_type=damage_type,
            tier=tier,
            action=action,
            part_cost_MAD=part_cost,
            labor_MAD=labor,
            total_MAD=total,
            is_estimate=True,
            note=(
                f"Estimation moyenne sur {len(sub)} pièce(s) éligibles "
                f"(min={sub.total_MAD.min()} MAD, max={sub.total_MAD.max()} MAD)."
            ),
        )

    def estimate(self, car_label: str, part: str, damage_type: str) -> Price | None:
        tier = self.tier_for(car_label)
        if tier is None:
            return None
        return self.lookup(part, damage_type, tier)
