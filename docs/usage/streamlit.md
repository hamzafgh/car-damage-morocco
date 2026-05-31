# Streamlit dashboard

Dark **AI NEXUS** themed interface. 8 pages routed from a left sidebar.

## Run

```bash
streamlit run app/streamlit_app.py
```

Or on Windows where `streamlit` isn't on PATH:

```bash
python -m streamlit run app/streamlit_app.py
```

Opens on http://localhost:8501.

## Page-by-page

| Icon | Page | What it shows |
|---|---|---|
| :material-chart-box: | **Vue d'ensemble** | Pipeline status (3 weight files), pricing CSV summary, class counts (20/23/4). |
| :material-magnify: | **Inspection** | Main flow: upload → 4 KPI cards (Véhicule / Tier / Dommages / Coût MAD) → annotated image + Top-3 + diagnostic → findings table → French report → JSON. |
| :material-puzzle: | **Pièces** | Catalogue of all 23 carparts classes, searchable. |
| :material-flash: | **Dommages** | The 4 damage classes with their overlay colors. |
| :material-cash: | **Tarification** | `prix_reparation_maroc.csv` with tier/damage/action filters + 4 summary cards. |
| :material-car: | **Modèles** | 20 Stage-0 cars grouped into 3 tier columns (économique / gamme moyenne / premium). |
| :material-cog: | **Configuration** | File existence checks, current thresholds, raw JSON dump. |
| :material-information: | **À propos** | Project credits. |

## Inference parameters

The Inspection page exposes (in a collapsible "Paramètres de détection" expander):

| Slider | Default | What it controls |
|---|---|---|
| Confiance pièces | 0.25 | Stage 1 confidence threshold (lower = more parts detected) |
| Confiance dommages | 0.25 | Stage 2 confidence threshold (lower = more damages detected) |
| Seuil IoMin (fusion) | 0.35 | Min IoMin overlap to attach a damage to a part |
| Modèle de voiture | Auto | Pick a model manually to skip Stage 0 |

Values persist across page changes via `st.session_state`.

## Theme

- Base: dark `#0A0E1A`
- Primary: cyan `#00D4FF`
- Accent: pink `#FF3D9A`
- Font: Inter (Google Fonts)
- Sidebar nav: custom-styled `st.radio` with `:has()` selector for the active item
- Cards: HTML helpers `metric_card()` and `big_card()` defined in `streamlit_app.py`
- CSS lives in `app/static/theme.css` (loaded once at startup), so the Python file isn't bloated with styles.

## File structure

```
app/
├── streamlit_app.py          ← page routing, helpers, all page functions
└── static/
    └── theme.css             ← extracted dark theme

.streamlit/
└── config.toml               ← base dark theme color seed
```

`.streamlit/config.toml` sets Streamlit's own dark theme (primary color, backgrounds) so the framework's defaults already match the custom CSS — no flash of light styles on first paint.
