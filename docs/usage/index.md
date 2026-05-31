# Usage

Three ways to use this project:

<div class="grid cards" markdown>

- :material-monitor-dashboard: **[Streamlit dashboard](streamlit.md)**  
  Upload a photo, get a French report, see annotated overlays. The AI NEXUS look.

- :material-code-braces: **[Programmatic API](programmatic.md)**  
  `from car_damage_morocco import DamageDetector` — use it from your own Python code.

- :material-test-tube: **[Tests](tests.md)**  
  10 pytest invariants that run without weights — data alignment, fusion math, French grammar.

</div>

## Prerequisites

- Python 3.10+
- The three weight files (`best.keras`, `best.pt`, `best.pt`) downloaded from Kaggle into `models/stage{0,1,2}/`. See [README](https://github.com/hamzafgh/car-damage-morocco#drop-trained-weights-in-place).
- `pip install -r requirements.txt`
