"""Inspect Stage 2 (damage segmentation) by running inference on a single image.

Doesn't need the full validation dataset — just give it an image path and it
prints the detected damages + saves an annotated visualization.

Usage:
    python scripts/inspect_stage2.py path/to/car.jpg
    python scripts/inspect_stage2.py path/to/car.jpg --conf 0.15
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from car_damage_morocco.stage2_damage import DamageSegmenter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path, help="Path to an image to test on")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Confidence threshold (default 0.25). Lower = more detections.")
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "stage2_inspect.png",
                    help="Where to save the annotated image")
    args = ap.parse_args()

    if not args.image.exists():
        sys.exit(f"Image not found: {args.image}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading Stage 2 model...")
    seg = DamageSegmenter(
        REPO / "models" / "stage2" / "best.pt",
        REPO / "data"   / "stage2_classes.json",
    )
    print(f"  Classes: {seg.class_names}\n")

    img = cv2.imread(str(args.image))
    if img is None:
        sys.exit(f"Failed to read image: {args.image}")
    H, W = img.shape[:2]
    print(f"Image: {args.image.name}  {W}x{H}")

    detections = seg.predict(img, conf=args.conf)
    print(f"\nFound {len(detections)} damage detections (conf >= {args.conf}):")

    # Per-class summary
    per_class: dict[str, list] = {}
    for d in detections:
        per_class.setdefault(d.class_name, []).append(d)
    for cls in seg.class_names:
        ds = per_class.get(cls, [])
        print(f"  {cls:12s}: {len(ds):2d} instance(s)")
        for d in ds:
            x1, y1, x2, y2 = d.bbox
            area_pct = 100 * d.mask.sum() / (H * W)
            print(f"      conf={d.confidence:.3f}  bbox=({x1},{y1},{x2},{y2})  area={area_pct:.2f}% of image")

    # Render: damage masks as colored overlays + bboxes + labels
    COLORS = {
        "dent":        (60, 60, 230),
        "scratch":     (60, 200, 230),
        "glass":       (230, 200, 60),
        "broken_part": (60, 60, 130),
    }
    out = img.copy()
    for d in detections:
        color = COLORS.get(d.class_name, (0, 0, 255))
        # Translucent mask
        layer = np.zeros_like(out)
        layer[d.mask] = color
        out = cv2.addWeighted(out, 1.0, layer, 0.45, 0)
        # bbox + label
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(out, (x1, y1 - lh - 6), (x1 + lw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imwrite(str(args.out), out)
    print(f"\nAnnotated image saved to: {args.out}")


if __name__ == "__main__":
    main()
