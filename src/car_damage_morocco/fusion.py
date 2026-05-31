"""Fusion — attach each damage mask to the underlying car part using IoMin overlap."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .stage1_parts import PartDetection
from .stage2_damage import DamageDetection


@dataclass
class Finding:
    """A (damage, part) pair after fusion, with geometry features ready for pricing + NLP."""
    damage:        DamageDetection
    part:          PartDetection | None       # None if no part overlapped
    iomin:         float                       # 0..1
    area_ratio:    float                       # damage_area_inside_part / part_area  (0..1)
    centroid_xy:   tuple[float, float]         # damage centroid in image pixels


def _iomin(d_mask: np.ndarray, p_mask: np.ndarray) -> tuple[float, int]:
    """Return (IoMin, intersection_area). Robust to small/large area ratios."""
    inter = int(np.logical_and(d_mask, p_mask).sum())
    if inter == 0:
        return 0.0, 0
    d_area = int(d_mask.sum())
    p_area = int(p_mask.sum())
    return inter / max(1, min(d_area, p_area)), inter


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0.0, 0.0)
    return (float(xs.mean()), float(ys.mean()))


def fuse(
    parts:    Sequence[PartDetection],
    damages:  Sequence[DamageDetection],
    iomin_threshold: float = 0.35,
    multi_part: bool = True,
) -> list[Finding]:
    """Assign each damage to one or more parts via IoMin overlap.

    multi_part=True : emit one Finding per (damage, part) pair exceeding the threshold
                      (useful for long scratches crossing two panels).
    multi_part=False: keep only the best matching part per damage.

    Damages with no part exceeding the threshold are still emitted with part=None.
    """
    findings: list[Finding] = []

    for d in damages:
        d_area = int(d.mask.sum())
        if d_area == 0:
            continue
        centroid = _centroid(d.mask)

        scored: list[tuple[float, int, PartDetection]] = []
        for p in parts:
            iomin, inter = _iomin(d.mask, p.mask)
            if iomin >= iomin_threshold:
                scored.append((iomin, inter, p))

        if not scored:
            findings.append(Finding(damage=d, part=None, iomin=0.0,
                                    area_ratio=0.0, centroid_xy=centroid))
            continue

        scored.sort(key=lambda t: t[0], reverse=True)
        chosen = scored if multi_part else [scored[0]]
        for iomin, inter, p in chosen:
            p_area = max(1, int(p.mask.sum()))
            findings.append(Finding(
                damage=d, part=p,
                iomin=iomin,
                area_ratio=inter / p_area,
                centroid_xy=centroid,
            ))

    return findings
