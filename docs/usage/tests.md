# Tests

10 pytest unit tests in `tests/test_pipeline.py`. They validate the **data layer** and **business logic** without needing the trained weights.

## Run

```bash
python -m pytest tests -v
```

Expected output:

```
test_pipeline.py::test_pricing_csv_loads                    PASSED
test_pipeline.py::test_pricing_totals_consistent            PASSED
test_pipeline.py::test_classes_files_present                PASSED
test_pipeline.py::test_stage2_classes_match_csv_damage_types PASSED
test_pipeline.py::test_stage1_classes_match_csv_parts       PASSED
test_pipeline.py::test_tier_mapping_covers_stage0_classes   PASSED
test_pipeline.py::test_pricing_lookup                       PASSED
test_pipeline.py::test_fusion_iomin_threshold               PASSED
test_pipeline.py::test_french_description_gender_agreement  PASSED
test_pipeline.py::test_french_report_no_damages             PASSED
============================= 10 passed in 6.45s =============================
```

## What each test verifies

| Test | Invariant |
|---|---|
| `test_pricing_csv_loads` | Pricing CSV has the expected columns; row count > 100; damage_type/tier/action values are in their expected sets. |
| `test_pricing_totals_consistent` | For every row: `total_MAD == part_cost_MAD + labor_MAD`. |
| `test_classes_files_present` | All four data JSONs exist and are valid JSON. |
| `test_stage2_classes_match_csv_damage_types` | The 4 damage classes in `stage2_classes.json` are exactly the unique `damage_type` values in the pricing CSV. |
| `test_stage1_classes_match_csv_parts` | Every `part` value in the pricing CSV is one of the 23 Stage 1 classes. |
| `test_tier_mapping_covers_stage0_classes` | Every Stage 0 class has an entry in `car_model_tiers.json`. |
| `test_pricing_lookup` | `Dacia_Logan → economy`; lookup of `(front_bumper, dent, economy)` returns a `Price` with `action='repair'` and `total_MAD > 0`. |
| `test_fusion_iomin_threshold` | IoMin fusion attaches a damage that overlaps a part (IoMin > 0.9) and emits `part=None` for a disjoint damage. |
| `test_french_description_gender_agreement` | `"casse importante"` (feminine) and `"enfoncement important"` (masculine) — proves the gender table actually drives the adjective. |
| `test_french_report_no_damages` | Empty list of damages produces a sentence containing `"Aucun dommage"`. |

## Why these tests matter

The data-alignment tests are the safety net for **CSV/JSON edits**. If you edit the pricing CSV and accidentally add a `damage_type` that isn't in `stage2_classes.json`, the test suite catches it before the dashboard crashes in front of a user.

The fusion test asserts the IoMin behavior we care about — without it, a refactor that broke the threshold logic would silently ship.

The French tests assert linguistic correctness — without them, a "fix" to the lexicon could break gender agreement and produce ungrammatical sentences.

## What's NOT tested

- The three trained models themselves (no GPU in CI, weights aren't in git).
- End-to-end pipeline on a real image (requires weights).
- The Streamlit UI (would need playwright or similar).

These are deliberate. The tests run on every commit, fast, with no GPU. For weight-level verification use `scripts/verify_weights.py` locally after dropping the `.keras` / `.pt` files into `models/`.

## Running tests in CI

The repo doesn't ship a GitHub Actions workflow yet. The minimum file would be:

```yaml
# .github/workflows/test.yml
name: tests
on: [push, pull_request]
jobs:
  pytest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt pytest
      - run: python -m pytest tests -v
```

Drop that file in if you want a green checkmark on the README.
