"""DamageDetector — orchestrates Stages 0/1/2 + fusion + pricing + NLP."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np

from .stage0_classifier import CarClassifier
from .stage1_parts       import PartsSegmenter
from .stage2_damage      import DamageSegmenter
from .fusion             import fuse, Finding
from .pricing            import PricingTable
from .nlp.describe_damage import Damage, describe, report, position_from_centroid


# ----------------------------------------------------------------------
# Public dataclasses
# ----------------------------------------------------------------------
@dataclass
class DamageRecord:
    part:            str
    damage_type:     str
    damage_conf:     float
    part_conf:       float
    iomin:           float
    area_ratio:      float
    action:          str | None        # 'repair' | 'replace' | None if no price
    part_cost_MAD:   int | None
    labor_MAD:       int | None
    cost_MAD:        int | None
    description_fr:  str
    is_estimate:     bool = False      # True when cost was a fallback average
    estimate_note:   str  = ""         # explanation shown when is_estimate=True


@dataclass
class DetectionResult:
    car_label:        str
    car_display:      str
    car_confidence:   float
    car_topk:         list[tuple[str, float]]
    tier:             str | None
    findings:         list[DamageRecord]
    total_MAD:        int
    report_fr:        str
    overlay:          np.ndarray | None = None         # annotated BGR image (set when render=True)
    raw:              dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("overlay")
        d.pop("raw")
        return d


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------
class DamageDetector:
    """End-to-end pipeline: image -> structured assessment.

    Construct once at app startup; call .predict(image) per request.
    """

    def __init__(
        self,
        # Model weights
        stage0_weights: str | Path,
        stage1_weights: str | Path,
        stage2_weights: str | Path,
        # Class metadata
        stage0_classes_json: str | Path,
        stage1_classes_json: str | Path,
        stage2_classes_json: str | Path,
        # Pricing
        prices_csv:    str | Path,
        tiers_json:    str | Path,
        # Inference knobs
        parts_conf:     float = 0.25,
        damage_conf:    float = 0.25,
        iomin_threshold: float = 0.35,
        multi_part:     bool = True,
    ) -> None:
        self.car      = CarClassifier(stage0_weights, stage0_classes_json)
        self.parts    = PartsSegmenter(stage1_weights, stage1_classes_json)
        self.damages  = DamageSegmenter(stage2_weights, stage2_classes_json)
        self.prices   = PricingTable(prices_csv, tiers_json)
        self.parts_conf  = parts_conf
        self.damage_conf = damage_conf
        self.iomin_threshold = iomin_threshold
        self.multi_part = multi_part

    # ----------------------------------------------------------------
    # Main entry point
    # ----------------------------------------------------------------
    def predict(
        self,
        image_bgr: np.ndarray,
        render: bool = False,
        car_label_override: str | None = None,
    ) -> DetectionResult:
        """Run the full pipeline.

        If car_label_override is given (e.g. user manually picked the model in the UI),
        Stage 0 is skipped and the override label is used for tier lookup. The result
        still reports car_label as the override and confidence as 1.0.
        """
        # Stage 0 — car classifier (or manual override)
        if car_label_override:
            car = {
                "label":         car_label_override,
                "display_label": self.car.display_names.get(
                    car_label_override, car_label_override.replace("_", " ")
                ),
                "confidence":    1.0,
                "topk":          [(car_label_override, 1.0)],
            }
        else:
            car = self.car.predict(image_bgr)
        car_label = car["label"]
        tier      = self.prices.tier_for(car_label)

        # Stages 1 & 2 — segmenters
        parts   = self.parts.predict(image_bgr, conf=self.parts_conf)
        damages = self.damages.predict(image_bgr, conf=self.damage_conf)

        # Fusion
        findings_raw = fuse(parts, damages,
                            iomin_threshold=self.iomin_threshold,
                            multi_part=self.multi_part)

        # Pricing + NLP per finding
        findings: list[DamageRecord] = []
        damage_objs_for_report: list[Damage] = []
        total = 0
        for f in findings_raw:
            part_name = f.part.class_name if f.part else "unknown"
            dmg_name  = f.damage.class_name
            # Position from damage centroid inside part bbox (or full image if no part)
            if f.part is not None:
                x1, y1, x2, y2 = f.part.bbox
            else:
                H, W = image_bgr.shape[:2]
                x1, y1, x2, y2 = 0, 0, W, H
            position = position_from_centroid(
                f.centroid_xy[0], f.centroid_xy[1], x1, y1, x2, y2
            )

            # Pricing
            #   - If we know the part: exact lookup
            #   - If we don't but we know the tier + damage type: fall back to the average
            #     across all parts in that bucket (clearly marked as an estimate)
            price = None
            if f.part is not None and tier is not None:
                price = self.prices.lookup(part_name, dmg_name, tier)
            if price is None and tier is not None:
                price = self.prices.fallback_estimate(dmg_name, tier)
            cost = price.total_MAD if price else None
            if cost is not None:
                total += cost

            # French description
            if f.part is None:
                if price is not None and price.is_estimate:
                    desc = (
                        f"Un dommage de type '{dmg_name}' a été détecté sans pouvoir "
                        f"l'associer à une pièce précise. Coût estimé à partir de la "
                        f"moyenne des pièces concernées (~{price.total_MAD} MAD)."
                    )
                else:
                    desc = (f"Un dommage de type '{dmg_name}' a été détecté mais "
                            f"sans pouvoir l'associer à une pièce avec certitude.")
            else:
                dmg_obj = Damage(part=part_name, damage_type=dmg_name,
                                 area_ratio=f.area_ratio, position=position)
                desc = describe(dmg_obj)
                damage_objs_for_report.append(dmg_obj)

            findings.append(DamageRecord(
                part=part_name,
                damage_type=dmg_name,
                damage_conf=f.damage.confidence,
                part_conf=f.part.confidence if f.part else 0.0,
                iomin=f.iomin,
                area_ratio=f.area_ratio,
                action=price.action if price else None,
                part_cost_MAD=price.part_cost_MAD if price else None,
                labor_MAD=price.labor_MAD if price else None,
                cost_MAD=cost,
                description_fr=desc,
                is_estimate=(price is not None and price.is_estimate),
                estimate_note=(price.note if price is not None else ""),
            ))

        # Build the French report from EVERY finding's description (including unknown-part ones)
        car_display = car["display_label"]
        if not findings:
            report_fr = f"{car_display} : Aucun dommage détecté."
        else:
            n = len(findings)
            head = f"{car_display} : {n} dommage" + ("s" if n > 1 else "") + " détecté"
            head += "s. " if n > 1 else ". "
            report_fr = head + " ".join(r.description_fr for r in findings)

        result = DetectionResult(
            car_label=car_label,
            car_display=car_display,
            car_confidence=car["confidence"],
            car_topk=car["topk"],
            tier=tier,
            findings=findings,
            total_MAD=total,
            report_fr=report_fr,
        )
        if render:
            result.overlay = self._render(image_bgr, parts, damages, findings_raw)
        result.raw = {"n_parts_detected": len(parts), "n_damages_detected": len(damages)}
        return result

    # ----------------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------------
    @staticmethod
    def _render(image_bgr, parts, damages, findings_raw):
        """Overlay part masks (cool colors) + damage masks (warm colors)."""
        import cv2
        img = image_bgr.copy()
        H, W = img.shape[:2]

        # Damage masks first, in saturated red/orange
        DMG_COLORS = {
            "dent":        (60, 60, 230),
            "scratch":     (60, 200, 230),
            "glass":       (230, 200, 60),
            "broken_part": (60, 60, 130),
        }
        for d in damages:
            color = DMG_COLORS.get(d.class_name, (0, 0, 255))
            mask3 = np.zeros_like(img)
            mask3[d.mask] = color
            img = cv2.addWeighted(img, 1.0, mask3, 0.45, 0)
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{d.class_name} {d.confidence:.2f}",
                        (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Findings legend at top
        for i, f in enumerate(findings_raw):
            if f.part is None: continue
            txt = f"{f.part.class_name} <- {f.damage.class_name}  IoMin={f.iomin:.2f}"
            cv2.putText(img, txt, (10, 20 + 18 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img, txt, (10, 20 + 18 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return img


# ----------------------------------------------------------------------
# Convenience constructor with default repo paths
# ----------------------------------------------------------------------
def default_detector(repo_root: str | Path | None = None) -> DamageDetector:
    """Build a DamageDetector from default paths under the repo root."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    r = Path(repo_root)
    return DamageDetector(
        stage0_weights      = r / "models" / "stage0" / "best.keras",
        stage1_weights      = r / "models" / "stage1" / "best.pt",
        stage2_weights      = r / "models" / "stage2" / "best.pt",
        stage0_classes_json = r / "data"   / "stage0_classes.json",
        stage1_classes_json = r / "data"   / "stage1_classes.json",
        stage2_classes_json = r / "data"   / "stage2_classes.json",
        prices_csv          = r / "data"   / "prix_reparation_maroc.csv",
        tiers_json          = r / "data"   / "car_model_tiers.json",
    )
