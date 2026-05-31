"""Smoke tests for the pipeline modules.

Most tests don't require the trained weights — they validate data files,
class-name alignment, fusion math, and the French NLP templates.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


# ----------------------------------------------------------------------
# Data files
# ----------------------------------------------------------------------
def test_pricing_csv_loads():
    df = pd.read_csv(REPO / "data" / "prix_reparation_maroc.csv")
    expected_cols = {"part", "yolo_class_id", "category", "damage_type",
                     "action", "tier", "part_cost_MAD", "labor_MAD", "total_MAD"}
    assert expected_cols.issubset(df.columns)
    assert len(df) > 100
    assert set(df.damage_type) <= {"dent", "scratch", "glass", "broken_part"}
    assert set(df.tier) == {"economy", "mid_range", "premium"}
    assert set(df.action) == {"repair", "replace"}


def test_pricing_totals_consistent():
    df = pd.read_csv(REPO / "data" / "prix_reparation_maroc.csv")
    diff = df["total_MAD"] - (df["part_cost_MAD"] + df["labor_MAD"])
    assert (diff == 0).all(), f"total_MAD should equal part_cost_MAD + labor_MAD; offenders: {diff.value_counts()}"


def test_classes_files_present():
    for name in ("stage0_classes.json", "stage1_classes.json",
                 "stage2_classes.json", "car_model_tiers.json"):
        p = REPO / "data" / name
        assert p.exists(), f"Missing {p}"
        json.loads(p.read_text(encoding="utf-8"))


def test_stage2_classes_match_csv_damage_types():
    classes = json.loads((REPO / "data" / "stage2_classes.json").read_text(encoding="utf-8"))
    df = pd.read_csv(REPO / "data" / "prix_reparation_maroc.csv")
    assert set(classes["class_names"]) == set(df.damage_type), \
        "Stage 2 class names must match damage_type values in the pricing CSV"


def test_stage1_classes_match_csv_parts():
    classes = json.loads((REPO / "data" / "stage1_classes.json").read_text(encoding="utf-8"))
    df = pd.read_csv(REPO / "data" / "prix_reparation_maroc.csv")
    assert set(df.part) <= set(classes["class_names"]), \
        "All parts in CSV must be a subset of carparts-seg class names"


def test_tier_mapping_covers_stage0_classes():
    tiers   = json.loads((REPO / "data" / "car_model_tiers.json").read_text(encoding="utf-8"))
    classes = json.loads((REPO / "data" / "stage0_classes.json").read_text(encoding="utf-8"))
    tier_labels = {m["label"].replace(" ", "_").replace("-", "_") for m in tiers["models"]}
    assert set(classes["class_names"]) == tier_labels, \
        "Every Stage-0 class must have an entry in car_model_tiers.json"


# ----------------------------------------------------------------------
# Pricing logic
# ----------------------------------------------------------------------
def test_pricing_lookup():
    from car_damage_morocco.pricing import PricingTable
    pt = PricingTable(REPO / "data" / "prix_reparation_maroc.csv",
                      REPO / "data" / "car_model_tiers.json")
    assert pt.tier_for("Dacia_Logan") == "economy"
    assert pt.tier_for("Mercedes_C_Class") == "premium"
    p = pt.estimate("Dacia_Logan", "front_bumper", "dent")
    assert p is not None
    assert p.action == "repair"
    assert p.total_MAD > 0


# ----------------------------------------------------------------------
# Fusion
# ----------------------------------------------------------------------
def test_fusion_iomin_threshold():
    from car_damage_morocco.fusion       import fuse
    from car_damage_morocco.stage1_parts import PartDetection
    from car_damage_morocco.stage2_damage import DamageDetection

    H = W = 100
    # part = top half
    part_mask = np.zeros((H, W), bool); part_mask[:50, :] = True
    part = PartDetection(class_id=10, class_name="front_door", confidence=0.9,
                         bbox=(0, 0, W, 50), mask=part_mask)
    # damage entirely inside the part (high IoMin)
    d_inside_mask = np.zeros((H, W), bool); d_inside_mask[10:20, 10:20] = True
    d_inside = DamageDetection(class_id=1, class_name="scratch", confidence=0.8,
                                bbox=(10, 10, 20, 20), mask=d_inside_mask)
    # damage entirely outside the part (zero overlap)
    d_outside_mask = np.zeros((H, W), bool); d_outside_mask[70:80, 70:80] = True
    d_outside = DamageDetection(class_id=0, class_name="dent", confidence=0.7,
                                 bbox=(70, 70, 80, 80), mask=d_outside_mask)

    findings = fuse([part], [d_inside, d_outside], iomin_threshold=0.3)
    assert len(findings) == 2
    inside_f  = next(f for f in findings if f.damage.class_name == "scratch")
    outside_f = next(f for f in findings if f.damage.class_name == "dent")
    assert inside_f.part is not None and inside_f.iomin > 0.9
    assert outside_f.part is None  # below threshold -> no part attached


# ----------------------------------------------------------------------
# NLP — French templates
# ----------------------------------------------------------------------
def test_french_description_gender_agreement():
    from car_damage_morocco.nlp.describe_damage import Damage, describe
    # "casse" is feminine -> "importante" not "important"
    s = describe(Damage(part="back_right_light", damage_type="broken_part", area_ratio=0.5))
    assert "casse importante" in s, s
    # "enfoncement" is masculine -> "important"
    s = describe(Damage(part="hood", damage_type="dent", area_ratio=0.5))
    assert "enfoncement important" in s, s


def test_french_report_no_damages():
    from car_damage_morocco.nlp.describe_damage import report
    assert "Aucun dommage" in report([], car_label="Dacia Logan")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
