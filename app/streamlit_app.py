"""Streamlit demo — upload a car photo, get a French damage assessment + MAD estimate.

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from car_damage_morocco.detector import default_detector, DetectionResult  # noqa: E402


# ── Page config + theme ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI NEXUS — Car Damage Morocco",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEME_CSS = (Path(__file__).resolve().parent / "static" / "theme.css").read_text(encoding="utf-8")
st.markdown(f"<style>{THEME_CSS}</style>", unsafe_allow_html=True)


# ── Resource loaders ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement des modèles…")
def get_detector():
    return default_detector(REPO)


def _missing_weights() -> list[str]:
    expected = [
        REPO / "models" / "stage0" / "best.keras",
        REPO / "models" / "stage1" / "best.pt",
        REPO / "models" / "stage2" / "best.pt",
    ]
    return [str(p.relative_to(REPO)) for p in expected if not p.exists()]


@st.cache_data
def load_json(name: str) -> dict:
    return json.loads((REPO / "data" / name).read_text(encoding="utf-8"))


@st.cache_data
def load_prices() -> pd.DataFrame:
    return pd.read_csv(REPO / "data" / "prix_reparation_maroc.csv")


# Stage-2 overlay colors (BGR in detector.py, CSS hex here)
DMG_COLORS_HEX = {
    "dent":        "#E63C3C",
    "scratch":     "#E6C83C",
    "glass":       "#3CC8E6",
    "broken_part": "#823C3C",
}


# ── HTML helpers ─────────────────────────────────────────────────────────────
def md(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


def spacer(h: int = 24) -> None:
    md(f"<div style='height:{h}px'></div>")


def render_topbar(title: str, dataset: str = "car-damage-morocco") -> None:
    md(
        f'<div class="topbar"><div class="topbar-title">{title}</div>'
        f'<div class="topbar-actions">'
        f'<span class="chip">📂 {dataset}</span>'
        f'<span class="chip purple">⬆ Export Report</span>'
        f'</div></div>'
    )


def section_header(title: str, subtitle: str = "") -> None:
    md(f'<div class="section-h">{title}</div>')
    if subtitle:
        md(f'<div class="section-sub">{subtitle}</div>')


def metric_card(
    title: str,
    accent: str = "base",
    *,
    rows: list[tuple] | None = None,
    badge: str | None = None,
    badge_class: str = "",
    body: str = "",
) -> str:
    """Build an HTML metric card.

    accent      : 'base' | 'green' | 'purple' | 'amber' | 'red'
    rows        : list of (label, value) or (label, value, val_class) tuples
    badge_class : '' | 'warn' | 'neg'  (controls badge color)
    body        : raw HTML injected between the title and the rows
    """
    badge_html = f'<span class="badge {badge_class}">{badge}</span>' if badge else ""
    rows_html = ""
    for r in rows or []:
        lbl, val = r[0], r[1]
        cls = r[2] if len(r) > 2 else ""
        rows_html += (
            f'<div class="metric-row">'
            f'<span class="lbl">{lbl}</span>'
            f'<span class="val {cls}">{val}</span>'
            f'</div>'
        )
    return (
        f'<div class="metric-card {accent}">'
        f'<div class="card-title"><span class="bar"></span> {title} {badge_html}</div>'
        f'{body}{rows_html}'
        f'</div>'
    )


def big_card(title: str, value: str, sub: str, accent: str = "base", value_color: str = "#E6EAF3") -> str:
    return (
        f'<div class="metric-card {accent}">'
        f'<div class="card-title"><span class="bar"></span> {title}</div>'
        f'<div class="big-val" style="color:{value_color}">{value}</div>'
        f'<div class="big-sub">{sub}</div>'
        f'</div>'
    )


# ── Session state ────────────────────────────────────────────────────────────
def init_state() -> None:
    defaults = {
        "parts_conf": 0.25,
        "damage_conf": 0.25,
        "iomin_thr": 0.35,
        "car_override_display": "🔮 Détection automatique",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────
def page_overview(detector_ready: bool, missing: list[str]) -> None:
    render_topbar("Overview")
    section_header(
        "Pipeline d'évaluation de dommages",
        "Pipeline en trois étapes : classification du modèle de voiture (Stage 0), "
        "segmentation des pièces (Stage 1), segmentation des dommages (Stage 2). "
        "La fusion par IoMin associe chaque dommage à une pièce, puis la table "
        "tarifaire renvoie un coût en dirhams marocains.",
    )

    s0 = load_json("stage0_classes.json")
    s1 = load_json("stage1_classes.json")
    s2 = load_json("stage2_classes.json")
    prices = load_prices()

    def weight_row(rel: str, label: str) -> tuple:
        ok = rel not in missing
        return (label, "✓ présent" if ok else "✗ absent", "pos" if ok else "neg")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        md(metric_card(
            "PIPELINE CV", "base",
            badge="OPÉRATIONNEL" if detector_ready else "POIDS MANQUANTS",
            badge_class="" if detector_ready else "neg",
            rows=[
                weight_row("models/stage0/best.keras", "Stage 0 — Classifier (EfficientNetB0)"),
                weight_row("models/stage1/best.pt",    "Stage 1 — Pièces (YOLOv8s-seg)"),
                weight_row("models/stage2/best.pt",    "Stage 2 — Dommages (YOLOv8s-seg)"),
                ("Fusion", "IoMin (intersection / min area)"),
                ("Langue rapport", "Français 🇲🇦"),
            ],
        ))
    with c2:
        md(metric_card(
            "BASE TARIFAIRE", "green", badge="CHARGÉE",
            rows=[
                ("Lignes prix", str(len(prices)), "cyan"),
                ("Devise", "MAD (Dirham marocain)"),
                ("Tiers", "economy · mid_range · premium"),
                ("Source", "prix_reparation_maroc.csv"),
                ("Schéma", "part / damage / tier / part_cost / labor / total"),
            ],
        ))

    spacer(28)

    c3, c4, c5 = st.columns(3, gap="medium")
    with c3:
        md(big_card("MODÈLES SUPPORTÉS", str(len(s0["class_names"])),
                    "Stage 0 — marché marocain", "purple", "#FF3D9A"))
    with c4:
        md(big_card("CLASSES DE PIÈCES", str(len(s1["class_names"])),
                    "Stage 1 — carparts-seg", "base", "#00D4FF"))
    with c5:
        md(big_card("TYPES DE DOMMAGE", str(len(s2["class_names"])),
                    " · ".join(s2["class_names"]), "amber", "#FFB547"))

    if not detector_ready:
        spacer(28)
        st.error("⚠️  Téléchargez les poids depuis Kaggle et placez-les sous `models/stage{0,1,2}/`, puis rechargez la page.")


def page_inspection(detector, detector_ready: bool) -> None:
    render_topbar("Inspection")
    section_header(
        "Analyser une photo",
        "Uploadez une photo de voiture endommagée. Le pipeline détecte le modèle, "
        "segmente les pièces et les dommages, calcule le coût en MAD et rédige un "
        "rapport en français.",
    )

    if not detector_ready:
        st.error("Pipeline indisponible — voir l'onglet Vue d'ensemble pour la liste des poids manquants.")
        return

    s0 = load_json("stage0_classes.json")
    AUTO_LABEL = "🔮 Détection automatique"
    DISPLAY_TO_CLASS = {
        s0["display_names"].get(c, c.replace("_", " ")): c
        for c in s0["class_names"]
    }
    dropdown = [AUTO_LABEL] + sorted(DISPLAY_TO_CLASS.keys())

    with st.expander("⚙️  Paramètres de détection", expanded=False):
        col_a, col_b, col_c, col_d = st.columns(4, gap="medium")
        with col_a:
            st.session_state.parts_conf = st.slider(
                "Confiance pièces", 0.05, 0.9, st.session_state.parts_conf, 0.05,
            )
        with col_b:
            st.session_state.damage_conf = st.slider(
                "Confiance dommages", 0.05, 0.9, st.session_state.damage_conf, 0.05,
            )
        with col_c:
            st.session_state.iomin_thr = st.slider(
                "Seuil IoMin (fusion)", 0.10, 0.90, st.session_state.iomin_thr, 0.05,
            )
        with col_d:
            current = st.session_state.car_override_display
            idx = dropdown.index(current) if current in dropdown else 0
            st.session_state.car_override_display = st.selectbox(
                "Modèle de voiture", options=dropdown, index=idx,
            )

    detector.parts_conf = st.session_state.parts_conf
    detector.damage_conf = st.session_state.damage_conf
    detector.iomin_threshold = st.session_state.iomin_thr
    car_override = (
        None if st.session_state.car_override_display == AUTO_LABEL
        else DISPLAY_TO_CLASS[st.session_state.car_override_display]
    )

    uploaded = st.file_uploader(
        "Téléchargez une photo de la voiture endommagée",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded is None:
        st.info("📷 Glissez-déposez une image pour démarrer l'analyse.")
        return

    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("Impossible de décoder l'image.")
        return

    with st.spinner("Analyse en cours…"):
        result: DetectionResult = detector.predict(
            image_bgr, render=True, car_label_override=car_override,
        )

    tier = result.tier or "inconnu"
    tier_class = tier if tier in ("economy", "mid_range", "premium") else "economy"
    n_findings = len(result.findings)
    n_est = sum(1 for f in result.findings if f.is_estimate)
    cost_str = f"{result.total_MAD:,}".replace(",", " ")
    veh_sub = "Sélection manuelle" if car_override else f"Confiance {result.car_confidence*100:.1f}%"

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        md(metric_card("VÉHICULE", "base", body=(
            f'<div style="color:#E6EAF3;font-size:20px;font-weight:700;margin:6px 0 4px">{result.car_display}</div>'
            f'<div class="big-sub">{veh_sub}</div>'
        )))
    with c2:
        md(metric_card("TIER DE PRIX", "green", body=(
            f'<div style="margin:8px 0 8px"><span class="tier {tier_class}">{tier}</span></div>'
            f'<div class="big-sub">Détermine la grille tarifaire</div>'
        )))
    with c3:
        md(big_card(
            "DOMMAGES DÉTECTÉS", str(n_findings),
            f"{n_est} estimation(s) · {n_findings - n_est} certifié(s)",
            "amber", "#FFB547",
        ))
    with c4:
        cost_color = "#00E5A0" if n_findings else "#5C6A85"
        cost_value = f'{cost_str} <span style="font-size:18px;color:#8B95B0;font-weight:600">MAD</span>'
        md(big_card("COÛT TOTAL", cost_value, "Pièces + main d'œuvre", "purple", cost_color))

    spacer(28)

    col_img, col_topk = st.columns([3, 2], gap="large")
    with col_img:
        md('<div class="card-title" style="--accent:#00D4FF"><span class="bar"></span> IMAGE ANNOTÉE</div>')
        bgr = result.overlay if result.overlay is not None else image_bgr
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col_topk:
        if not car_override:
            md('<div class="card-title" style="--accent:#FF3D9A"><span class="bar"></span> TOP-3 STAGE 0</div>')
            for lbl, conf in result.car_topk[:3]:
                display = s0["display_names"].get(lbl, lbl.replace("_", " "))
                st.progress(int(conf * 100), text=f"{display} — {conf*100:.1f}%")
        else:
            md('<div class="card-title" style="--accent:#FF3D9A"><span class="bar"></span> MODE MANUEL</div>')
            st.info(f"Stage 0 ignoré. Modèle sélectionné : **{result.car_display}**")

        spacer(14)
        md('<div class="card-title" style="--accent:#FFB547"><span class="bar"></span> DIAGNOSTIC</div>')
        n_parts = result.raw.get("n_parts_detected", 0)
        n_damages = result.raw.get("n_damages_detected", 0)
        n_unmatched = sum(1 for f in result.findings if f.part == "unknown")
        md(
            f'<div class="metric-row"><span class="lbl">Pièces détectées (≥ {detector.parts_conf:.2f})</span><span class="val">{n_parts}</span></div>'
            f'<div class="metric-row"><span class="lbl">Dommages détectés (≥ {detector.damage_conf:.2f})</span><span class="val">{n_damages}</span></div>'
            f'<div class="metric-row"><span class="lbl">Seuil IoMin</span><span class="val">{detector.iomin_threshold:.2f}</span></div>'
            f'<div class="metric-row"><span class="lbl">Dommages sans pièce</span><span class="val {"warn" if n_unmatched else ""}">{n_unmatched}</span></div>'
        )
        if n_unmatched:
            st.warning(f"⚠️ {n_unmatched} dommage(s) sans pièce associée — baissez le seuil IoMin pour en récupérer.")
        if n_damages > 0 and n_parts == 0:
            st.warning("⚠️ Aucune pièce détectée — baissez le seuil de confiance des pièces.")

    spacer(28)

    md(
        f'<div class="section-h" style="font-size:18px">Détail des dommages '
        f'<span style="color:#5C6A85;font-size:13px;font-weight:500;margin-left:8px">({n_findings})</span></div>'
    )
    if not result.findings:
        st.success("✅ Aucun dommage détecté.")
    else:
        df = pd.DataFrame([{
            "Pièce":       f.part if not f.is_estimate else "🟡 pièce inconnue",
            "Dommage":     f.damage_type,
            "Action":      f.action or "—",
            "Surface":     f"{int(f.area_ratio*100)}%",
            "IoMin":       f"{f.iomin:.2f}",
            "Pièce (MAD)": f.part_cost_MAD or 0,
            "M-O (MAD)":   f.labor_MAD or 0,
            "Total (MAD)": (
                f"~{f.cost_MAD}" if f.is_estimate and f.cost_MAD is not None
                else (f.cost_MAD or 0)
            ),
            "Confiance":   f"{f.damage_conf*100:.0f}%",
        } for f in result.findings])
        st.dataframe(df, use_container_width=True, hide_index=True)
        if n_est:
            st.info(
                f"🟡 {n_est} ligne(s) marquée(s) comme estimation : la pièce exacte "
                "n'a pas pu être identifiée par Stage 1, donc le coût est une "
                "moyenne sur les pièces compatibles."
            )

    spacer(20)
    md('<div class="section-h" style="font-size:18px">Rapport en français</div>')
    md(f'<div class="metric-card base" style="font-size:14px;line-height:1.7;color:#E6EAF3">{result.report_fr}</div>')

    with st.expander("📦 Sortie JSON brute"):
        st.json(result.to_dict())


def page_parts() -> None:
    render_topbar("Pièces (Stage 1)")
    section_header(
        "Catalogue des pièces",
        "Stage 1 utilise le dataset Ultralytics carparts-seg. Le modèle prédit un "
        "mask par instance et un score de confiance pour chacune des classes ci-dessous.",
    )

    classes = load_json("stage1_classes.json")["class_names"]
    front = sum(1 for c in classes if c.startswith("front"))
    back  = sum(1 for c in classes if c.startswith("back"))

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        md(big_card("CLASSES", str(len(classes)), "YOLOv8s-seg · 640×640", "base", "#00D4FF"))
    with c2:
        md(big_card("PIÈCES AVANT", str(front), "bumper, glass, doors, lights…", "green", "#00E5A0"))
    with c3:
        md(big_card("PIÈCES ARRIÈRE", str(back), "bumper, glass, doors, lights…", "purple", "#FF3D9A"))

    spacer()

    df = pd.DataFrame([
        {"ID": i, "Classe": c, "Famille": (
            "avant"  if c.startswith("front") else
            "arrière" if c.startswith("back")  else
            "exclu"  if c == "object"         else "autre"
        )}
        for i, c in enumerate(classes)
    ])
    q = st.text_input("Filtrer une classe", placeholder="ex: door, bumper, glass…")
    view = df[df["Classe"].str.contains(q, case=False)] if q else df
    st.dataframe(view, use_container_width=True, hide_index=True, height=560)


FR_DAMAGE_LABELS = {
    "dent": "Bosse / enfoncement",
    "scratch": "Rayure",
    "glass": "Bris de vitre",
    "broken_part": "Pièce cassée",
}


def page_damages() -> None:
    render_topbar("Dommages (Stage 2)")
    section_header(
        "Catalogue des dommages",
        "Stage 2 a été ré-étiqueté depuis Roboflow is_it_damaged v6 vers 4 types. "
        "Chaque type s'affiche avec sa propre couleur dans l'image annotée de la "
        "page Inspection.",
    )

    for cls in load_json("stage2_classes.json")["class_names"]:
        color = DMG_COLORS_HEX.get(cls, "#888888")
        fr = FR_DAMAGE_LABELS.get(cls, cls)
        md(
            f'<div class="metric-card" style="margin-bottom:14px">'
            f'<div style="display:flex;align-items:center;justify-content:space-between">'
            f'<div>'
            f'<span class="swatch" style="background:{color}"></span>'
            f'<span style="color:#E6EAF3;font-size:16px;font-weight:600">{cls}</span>'
            f'<span style="color:#8B95B0;font-size:13px;margin-left:14px">{fr}</span>'
            f'</div>'
            f'<span style="color:#5C6A85;font-family:ui-monospace,monospace;font-size:12px">{color.upper()}</span>'
            f'</div></div>'
        )


def _filter_select(label: str, df: pd.DataFrame, col: str) -> str:
    """Selectbox over unique values in df[col], with 'tous' as the default."""
    if col not in df.columns:
        return "tous"
    opts = ["tous"] + sorted(df[col].dropna().unique().tolist())
    return st.selectbox(label, opts)


def page_pricing() -> None:
    render_topbar("Tarification")
    section_header(
        "Grille tarifaire — MAD",
        "Table prix_reparation_maroc.csv. Chaque ligne associe (pièce, type de "
        "dommage, tier) à un coût pièce, un coût main d'œuvre et une action recommandée.",
    )

    df = load_prices()

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        md(big_card("LIGNES", str(len(df)), "Combinaisons tarifées", "base", "#00D4FF"))
    with c2:
        if "total_MAD" in df.columns:
            avg = f"{int(df['total_MAD'].mean()):,}".replace(",", " ")
            md(big_card("COÛT MOYEN", avg, "MAD / réparation", "green", "#00E5A0"))
    with c3:
        if "tier" in df.columns:
            md(big_card(
                "TIERS", str(df["tier"].nunique()),
                " · ".join(sorted(df["tier"].dropna().unique())),
                "purple", "#FF3D9A",
            ))
    with c4:
        if "damage_type" in df.columns:
            md(big_card(
                "TYPES DOMMAGE", str(df["damage_type"].nunique()),
                "Couverts par la grille", "amber", "#FFB547",
            ))

    spacer()

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        tier_filter = _filter_select("Tier", df, "tier")
    with fc2:
        dmg_filter = _filter_select("Type de dommage", df, "damage_type")
    with fc3:
        action_filter = _filter_select("Action", df, "action")

    view = df
    if tier_filter   != "tous": view = view[view["tier"]        == tier_filter]
    if dmg_filter    != "tous": view = view[view["damage_type"] == dmg_filter]
    if action_filter != "tous": view = view[view["action"]      == action_filter]

    st.dataframe(view, use_container_width=True, hide_index=True, height=420)


def page_models() -> None:
    render_topbar("Modèles supportés")
    section_header(
        "Stage 0 — 20 modèles marché marocain",
        "Le classifieur EfficientNetB0 a été entraîné sur un sous-ensemble de "
        "CompCars filtré pour le parc marocain (Dacia, Renault, Peugeot, VW, "
        "Mercedes…). Chaque modèle est mappé à un tier qui détermine la grille tarifaire.",
    )

    tiers = load_json("car_model_tiers.json")
    by_tier: dict[str, list[str]] = {"economy": [], "mid_range": [], "premium": []}
    for m in tiers.get("models", []):
        by_tier.setdefault(m.get("tier", "economy"), []).append(m.get("label", "—"))

    tier_meta = [
        ("economy",   "#00D4FF", "ÉCONOMIQUE",    "Dacia, Peugeot 208/301, Yaris…"),
        ("mid_range", "#00E5A0", "GAMME MOYENNE", "VW, Peugeot 308, Captur, Tucson…"),
        ("premium",   "#FF3D9A", "PREMIUM",       "Mercedes C/E Class"),
    ]
    for col, (t, color, head, sub) in zip(st.columns(3, gap="medium"), tier_meta):
        with col:
            models = sorted(by_tier.get(t, []))
            rows = "".join(
                f'<div class="metric-row">'
                f'<span class="lbl">{name}</span>'
                f'<span class="val" style="color:{color}">{t}</span>'
                f'</div>'
                for name in models
            )
            body = f'<div class="big-sub" style="margin-bottom:14px">{sub}</div>{rows}'
            md(
                f'<div class="metric-card" style="--accent:{color}">'
                f'<div class="card-title"><span class="bar"></span> {head}'
                f'<span class="badge" style="background:rgba(0,0,0,0);color:{color}">{len(models)}</span>'
                f'</div>{body}</div>'
            )


def page_config(detector_ready: bool, missing: list[str]) -> None:
    render_topbar("Configuration")
    section_header(
        "Paramètres système",
        "État des fichiers de poids et chemins importants. Modifiez les seuils sur "
        "la page Inspection — ils sont persistants tant que l'application reste ouverte.",
    )

    files = [
        ("models/stage0/best.keras",       "Stage 0 weights"),
        ("models/stage1/best.pt",          "Stage 1 weights"),
        ("models/stage2/best.pt",          "Stage 2 weights"),
        ("data/prix_reparation_maroc.csv", "Pricing CSV"),
        ("data/car_model_tiers.json",      "Tiers JSON"),
        ("data/stage0_classes.json",       "Stage 0 classes"),
        ("data/stage1_classes.json",       "Stage 1 classes"),
        ("data/stage2_classes.json",       "Stage 2 classes"),
    ]
    rows = []
    for path, label in files:
        ok = (REPO / path).exists()
        rows.append((
            f'{label} <code style="color:#5C6A85;font-size:11px">{path}</code>',
            "✓ trouvé" if ok else "✗ absent",
            "pos" if ok else "neg",
        ))
    md(metric_card("FICHIERS", "base", rows=rows))

    spacer()
    md(metric_card("SEUILS COURANTS", "green", rows=[
        ("Confiance pièces (Stage 1)",   f"{st.session_state.parts_conf:.2f}",  "cyan"),
        ("Confiance dommages (Stage 2)", f"{st.session_state.damage_conf:.2f}", "cyan"),
        ("Seuil IoMin (fusion)",         f"{st.session_state.iomin_thr:.2f}",   "cyan"),
        ("Override modèle",              st.session_state.car_override_display),
    ]))

    spacer()
    with st.expander("📦 Configuration brute (JSON)"):
        st.json({
            "repo_root": str(REPO),
            "detector_ready": detector_ready,
            "missing_weights": missing,
            "parts_conf": st.session_state.parts_conf,
            "damage_conf": st.session_state.damage_conf,
            "iomin_thr": st.session_state.iomin_thr,
            "car_override": st.session_state.car_override_display,
        })


def page_about() -> None:
    render_topbar("À propos")
    section_header("Car Damage Morocco")
    presentation = (
        '<p style="color:#E6EAF3;font-size:14px;line-height:1.75;margin:0 0 16px">'
        "Pipeline <b>Computer Vision</b> d'évaluation de dommages automobile pour le marché marocain. "
        'Trois étapes : <b style="color:#00D4FF">Stage 0</b> (EfficientNetB0, 20 modèles), '
        '<b style="color:#00E5A0">Stage 1</b> (YOLOv8s-seg, 23 pièces, dataset carparts-seg), '
        '<b style="color:#FF3D9A">Stage 2</b> (YOLOv8s-seg, 4 types de dommages, dataset is_it_damaged). '
        "Fusion par <b>IoMin</b>, tarification en <b>MAD</b>, rapport en <b>français</b>."
        "</p>"
    )
    md(metric_card(
        "PRÉSENTATION", "purple", body=presentation,
        rows=[
            ("Auteur",         "Hamza El Faghloumi"),
            ("Stack",          "Streamlit · TensorFlow · Ultralytics · OpenCV"),
            ("Langue rapport", "Français 🇲🇦"),
            ("Devise",         "MAD (Dirham marocain)"),
            ("Pipeline",       "Photo → Stage 0/1/2 → Fusion → Pricing → NLP"),
        ],
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
NAV = [
    ("📊", "Vue d'ensemble"),
    ("🔍", "Inspection"),
    ("🧩", "Pièces"),
    ("💥", "Dommages"),
    ("💰", "Tarification"),
    ("🚙", "Modèles"),
    ("⚙️", "Configuration"),
    ("ℹ️", "À propos"),
]

PAGES = {
    "Vue d'ensemble": lambda d, r, m: page_overview(r, m),
    "Inspection":     lambda d, r, m: page_inspection(d, r),
    "Pièces":         lambda d, r, m: page_parts(),
    "Dommages":       lambda d, r, m: page_damages(),
    "Tarification":   lambda d, r, m: page_pricing(),
    "Modèles":        lambda d, r, m: page_models(),
    "Configuration":  lambda d, r, m: page_config(r, m),
    "À propos":       lambda d, r, m: page_about(),
}


def main() -> None:
    init_state()
    missing = _missing_weights()
    detector_ready = not missing
    detector = get_detector() if detector_ready else None

    with st.sidebar:
        md(
            '<div class="brand">'
            '<div class="brand-logo">AI</div>'
            '<div class="brand-name">NEXUS</div>'
            '</div>'
            '<div class="sidebar-section">Pipeline CV</div>'
        )

        labels = [f"{icon}  {name}" for icon, name in NAV]
        choice = st.radio("Navigation", labels, label_visibility="collapsed")

        dot = "green" if detector_ready else "red"
        status = "Pipeline online" if detector_ready else "Weights missing"
        md(
            f'<div class="status-pill">'
            f'<div class="dot {dot}"></div>'
            f'<div>'
            f'<div class="label">System Status</div>'
            f'<div class="value">{status}</div>'
            f'</div></div>'
        )

    page_name = choice.split("  ", 1)[1] if "  " in choice else choice
    PAGES[page_name](detector, detector_ready, missing)


if __name__ == "__main__":
    main()
