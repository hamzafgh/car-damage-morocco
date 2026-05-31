"""French damage description — template engine.

This is the rule-based generator used in two places:
  1. Production inference path (DamageDetector calls describe()).
  2. Training-time caption generator for the CNN+LSTM captioner
     (caption_model.py / train_caption_model.py).

Inputs are the structured outputs of the multi-stage pipeline:
  - part name (carparts-seg label, e.g. 'front_left_door')
  - damage type ('dent' | 'scratch' | 'glass' | 'broken_part')
  - area_ratio = damage_mask_area / part_mask_area  (float in 0..1)
  - position   = ('top'|'middle'|'bottom', 'left'|'center'|'right')
                 derived from damage centroid inside the part bbox.

Output: one French sentence per damage, plus a multi-sentence report.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Iterable

# ----------------------------------------------------------------------
# French lexicon
# ----------------------------------------------------------------------
PART_FR: dict[str, tuple[str, str]] = {
    # (article + noun, gender) — gender used for adjective agreement
    "back_bumper":       ("le pare-chocs arrière", "m"),
    "back_door":         ("la portière arrière", "f"),
    "back_glass":        ("la lunette arrière", "f"),
    "back_left_door":    ("la portière arrière gauche", "f"),
    "back_left_light":   ("le feu arrière gauche", "m"),
    "back_light":        ("le feu arrière", "m"),
    "back_right_door":   ("la portière arrière droite", "f"),
    "back_right_light":  ("le feu arrière droit", "m"),
    "front_bumper":      ("le pare-chocs avant", "m"),
    "front_door":        ("la portière avant", "f"),
    "front_glass":       ("le pare-brise", "m"),
    "front_left_door":   ("la portière avant gauche", "f"),
    "front_left_light":  ("le phare avant gauche", "m"),
    "front_light":       ("le phare avant", "m"),
    "front_right_door":  ("la portière avant droite", "f"),
    "front_right_light": ("le phare avant droit", "m"),
    "hood":              ("le capot", "m"),
    "left_mirror":       ("le rétroviseur gauche", "m"),
    "right_mirror":      ("le rétroviseur droit", "m"),
    "tailgate":          ("le hayon", "m"),
    "trunk":             ("le coffre", "m"),
    "wheel":             ("la roue", "f"),
}

DAMAGE_FR: dict[str, dict] = {
    # noun = full damage noun phrase; gender = gender of the damage noun
    # (used for severity-adjective agreement, e.g. "casse importante" vs "enfoncement important")
    "dent":        {"noun": "un enfoncement",   "gender": "m",
                    "verb": "présente", "action": "à débosseler"},
    "scratch":     {"noun": "une rayure",       "gender": "f",
                    "verb": "présente", "action": "à repeindre"},
    "glass":       {"noun": "un bris de vitre", "gender": "m",
                    "verb": "présente", "action": "à remplacer"},
    "broken_part": {"noun": "une casse",        "gender": "f",
                    "verb": "présente", "action": "à remplacer"},
}

SEVERITY_FR = {
    "minor":    {"m": "léger",    "f": "légère"},
    "moderate": {"m": "modéré",   "f": "modérée"},
    "severe":   {"m": "important","f": "importante"},
}

POSITION_FR = {
    ("top",    "left"):   "dans la partie supérieure gauche",
    ("top",    "center"): "dans la partie supérieure",
    ("top",    "right"):  "dans la partie supérieure droite",
    ("middle", "left"):   "sur le côté gauche",
    ("middle", "center"): "au centre",
    ("middle", "right"):  "sur le côté droit",
    ("bottom", "left"):   "dans la partie inférieure gauche",
    ("bottom", "center"): "dans la partie inférieure",
    ("bottom", "right"):  "dans la partie inférieure droite",
}

# ----------------------------------------------------------------------
# Severity from area ratio (used when no severity classifier is available)
# ----------------------------------------------------------------------
def severity_from_area(area_ratio: float) -> Literal["minor", "moderate", "severe"]:
    if area_ratio < 0.05:
        return "minor"
    if area_ratio < 0.20:
        return "moderate"
    return "severe"


def position_from_centroid(
    cx: float, cy: float, x1: float, y1: float, x2: float, y2: float
) -> tuple[str, str]:
    """Quantize a damage centroid (cx,cy) within a part bbox (x1,y1,x2,y2)
    into a 3x3 grid label ('top'|'middle'|'bottom', 'left'|'center'|'right')."""
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    rx = (cx - x1) / w
    ry = (cy - y1) / h
    col = "left"   if rx < 1/3 else "right"  if rx > 2/3 else "center"
    row = "top"    if ry < 1/3 else "bottom" if ry > 2/3 else "middle"
    return (row, col)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
@dataclass
class Damage:
    part: str                       # carparts-seg label
    damage_type: str                # one of DAMAGE_FR keys
    area_ratio: float               # in [0,1]
    position: tuple[str, str] | None = None  # ('top'|'middle'|'bottom', 'left'|'center'|'right')
    severity: Literal["minor","moderate","severe"] | None = None  # if None, derived from area_ratio


def describe(d: Damage) -> str:
    """One French sentence describing a single damage."""
    if d.part not in PART_FR:
        raise KeyError(f"Unknown part: {d.part}")
    if d.damage_type not in DAMAGE_FR:
        raise KeyError(f"Unknown damage_type: {d.damage_type}")

    part_phrase, _part_gender = PART_FR[d.part]
    dmg          = DAMAGE_FR[d.damage_type]
    dmg_noun     = dmg["noun"]
    verb         = dmg["verb"]
    dmg_gender   = dmg["gender"]   # severity adjective agrees with the damage noun

    sev = d.severity or severity_from_area(d.area_ratio)
    sev_adj = SEVERITY_FR[sev][dmg_gender]

    pos_phrase = ""
    if d.position is not None:
        pos_phrase = " " + POSITION_FR.get(d.position, "")

    pct = max(1, int(round(d.area_ratio * 100)))
    return (
        f"{part_phrase.capitalize()} {verb} {dmg_noun} {sev_adj}{pos_phrase}, "
        f"couvrant environ {pct}% de la surface."
    )


def report(damages: Iterable[Damage], car_label: str | None = None) -> str:
    """Multi-sentence French report covering all detected damages."""
    damages = list(damages)
    if not damages:
        head = "Aucun dommage détecté."
        return f"{car_label + ' : ' if car_label else ''}{head}"
    head = f"{car_label} : " if car_label else ""
    head += f"{len(damages)} dommage" + ("s" if len(damages) > 1 else "") + " détecté"
    head += ("s." if len(damages) > 1 else ".")
    return head + " " + " ".join(describe(d) for d in damages)


# ----------------------------------------------------------------------
# Quick self-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        Damage("front_left_door",  "scratch",     0.08, position=("middle", "right")),
        Damage("front_bumper",     "dent",        0.25, position=("bottom", "center")),
        Damage("front_glass",      "glass",       0.12),
        Damage("back_right_light", "broken_part", 0.45),
    ]
    print(report(samples, car_label="Dacia Logan"))
