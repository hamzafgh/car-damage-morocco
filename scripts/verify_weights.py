"""Verify that downloaded model weights load and run a dummy inference.

Use after dropping weight files into models/. Catches:
  - Corrupted downloads
  - Class-count mismatches with the class JSON files
  - Stage 0 input-size / dtype mismatches
  - Wrong Ultralytics version (incompatible checkpoint)

Run:
  python scripts/verify_weights.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

if sys.platform == "win32":
    OK, FAIL, SKIP = "[OK]  ", "[FAIL]", "[SKIP]"
else:
    OK   = "\033[92m✓\033[0m"
    FAIL = "\033[91m✗\033[0m"
    SKIP = "\033[93m·\033[0m"


def check_stage0() -> bool:
    print("\n--- Stage 0 (car classifier, EfficientNetB0) ---")
    weights = REPO / "models" / "stage0" / "best.keras"
    classes = REPO / "data"   / "stage0_classes.json"
    if not weights.exists():
        print(f"  {SKIP} weights missing: {weights}")
        return False
    try:
        from car_damage_morocco.stage0_classifier import CarClassifier
        clf = CarClassifier(weights, classes)
        print(f"  {OK} loaded ({clf.model.count_params():,} params)")
        # Dummy inference: random uint8 image
        dummy = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        out = clf.predict(dummy, top_k=3)
        print(f"  {OK} dummy predict OK -> label={out['label']!r} conf={out['confidence']:.3f}")
        print(f"      classes ({len(clf.class_names)}): {clf.class_names[:3]} ... {clf.class_names[-2:]}")
        return True
    except Exception as e:
        print(f"  {FAIL} {type(e).__name__}: {e}")
        return False


def check_stage1() -> bool:
    print("\n--- Stage 1 (parts seg, YOLOv8s-seg) ---")
    weights = REPO / "models" / "stage1" / "best.pt"
    classes = REPO / "data"   / "stage1_classes.json"
    if not weights.exists():
        print(f"  {SKIP} weights missing: {weights}  (still training?)")
        return False
    try:
        from car_damage_morocco.stage1_parts import PartsSegmenter
        seg = PartsSegmenter(weights, classes)
        print(f"  {OK} loaded ({len(seg.class_names)} classes)")
        dummy = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        dets = seg.predict(dummy, conf=0.01)
        print(f"  {OK} dummy predict OK -> {len(dets)} detections")
        return True
    except Exception as e:
        print(f"  {FAIL} {type(e).__name__}: {e}")
        return False


def check_stage2() -> bool:
    print("\n--- Stage 2 (damage seg, YOLOv8s-seg) ---")
    weights = REPO / "models" / "stage2" / "best.pt"
    classes = REPO / "data"   / "stage2_classes.json"
    if not weights.exists():
        print(f"  {SKIP} weights missing: {weights}")
        return False
    try:
        from car_damage_morocco.stage2_damage import DamageSegmenter
        seg = DamageSegmenter(weights, classes)
        print(f"  {OK} loaded ({len(seg.class_names)} classes: {seg.class_names})")
        dummy = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        dets = seg.predict(dummy, conf=0.01)
        print(f"  {OK} dummy predict OK -> {len(dets)} detections")
        return True
    except Exception as e:
        print(f"  {FAIL} {type(e).__name__}: {e}")
        return False


def check_pricing() -> bool:
    print("\n--- Pricing + tier mapping ---")
    try:
        from car_damage_morocco.pricing import PricingTable
        pt = PricingTable(REPO / "data" / "prix_reparation_maroc.csv",
                          REPO / "data" / "car_model_tiers.json")
        tier = pt.tier_for("Dacia_Logan")
        price = pt.estimate("Dacia_Logan", "front_bumper", "dent")
        assert tier == "economy" and price is not None
        print(f"  {OK} Dacia_Logan -> {tier}, dent front_bumper = {price.total_MAD} MAD")
        return True
    except Exception as e:
        print(f"  {FAIL} {type(e).__name__}: {e}")
        return False


def check_nlp() -> bool:
    print("\n--- NLP (French templates) ---")
    try:
        from car_damage_morocco.nlp.describe_damage import Damage, report
        text = report([
            Damage("front_left_door", "scratch", 0.08, position=("middle", "right")),
            Damage("back_right_light", "broken_part", 0.45),
        ], car_label="Dacia Logan")
        print(f"  {OK} {text[:120]}{'...' if len(text) > 120 else ''}")
        return True
    except Exception as e:
        print(f"  {FAIL} {type(e).__name__}: {e}")
        return False


def main() -> int:
    print("=" * 70)
    print("car-damage-morocco — weight + module verification")
    print("=" * 70)

    results = {
        "Stage 0":  check_stage0(),
        "Stage 1":  check_stage1(),
        "Stage 2":  check_stage2(),
        "Pricing":  check_pricing(),
        "NLP":      check_nlp(),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        mark = OK if ok else FAIL
        print(f"  {mark}  {name}")
    n_ok = sum(results.values())
    print(f"\n{n_ok}/{len(results)} checks passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
