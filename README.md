# Assessment Tiered Screener

The Streamlit application now has two tabs:

- **Design Assessment** — the existing Design Assessment triage workflow is preserved as-is.
- **Manufacturing Assessment** — a new deterministic model identifies potential manufacturing issues based on the provided Medtronic definition.

## Manufacturing Assessment tab

The Manufacturing Assessment model flags potential manufacturing issues where an aspect of the manufacturing process may have contributed to the device issue, including external supplier manufacturing processes. It looks for definition signals including sterile packaging compromise, foreign material inside a sterile package, missing/damaged/detached package contents, labeling issues, out-of-box failures, supplier/assembly/process language, and prior manufacturing-related Further Action or Investigation language.

Batch score a file for Manufacturing Assessment:

```bash
python score_manufacturing_file.py "input.xlsx" --output "manufacturing_assessment_scored.xlsx"
```

Optional threshold:

```bash
python score_manufacturing_file.py "input.xlsx" --review-threshold 0.50 --output "manufacturing_assessment_scored.xlsx"
```

# Design Assessment Tiered Screener

This package scores Medtronic-style Product Event / PLI extracts for Design Assessment triage.

## What changed in this version

The output is now split into three actionable levels instead of one broad review flag:

1. **DA REQUIRED**
   - `Decision = Death - Reportable` or `Decision = Serious Injury - Reportable`
   - This is a hard override. These rows are flagged no matter what the historical task value says.

2. **DA REVIEW - HIGH CONFIDENCE**
   - Non-mandatory rows with model probability at or above the high-confidence threshold.
   - Default threshold: `0.60`.
   - These rows should go to the DA reviewer queue.

3. **DA WATCHLIST - LOW CONFIDENCE**
   - Rows with a definition text signal or model probability at or above the watchlist threshold.
   - Complaint rows (`Complaint? – PE = Y`) whose `Code/LLT Desc – PE PLI` contains any configured low-confidence Code/LLT complaint rule, such as `ARTICULATION INSUFFICIENT`, `CLIP APPLIER DID NOT FIRE`, `CORD/CABLE FAILURE/DAMAGE`, `DEVICE MISSING BARBS`, `SIGNIA ADAPTER SLOW TO RECOGNIZE`, `SUTURE APPEARANCE`, `SUTURES ARE TOO LOOSE`, or `WILL NOT ROTATE`.
   - Default threshold: `0.0485`.
   - These are for trending, sampling, or a second look. They are not automatic Design Assessments.

Rows that do not meet those rules are **NO DA FLAG**.

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload an Excel or CSV file with the same column names as the historical extract, including `Decision`.

## Batch score a file

```bash
python score_file.py "input.xlsx" --output "design_assessment_tiered_scored.xlsx"
```

Optional thresholds:

```bash
python score_file.py "input.xlsx" --high-threshold 0.60 --watchlist-threshold 0.0485 --output "tiered_scored.xlsx"
```

## Retrain and validate with a random 70/30 split

```bash
python train_model.py "historical_data.xlsx" --output design_assessment_model.joblib
```

Default behavior now randomly splits the historical file by **PE - PLI # group**:

- 70% of grouped PE - PLI #s are used to train the model.
- The remaining 30% are held out for testing.
- Rows from the same PE - PLI # stay together, so the same PLI does not leak into both train and test.
- Mandatory `Death - Reportable` and `Serious Injury - Reportable` rules are applied during validation, but the model itself trains only on non-mandatory rows.
- The metrics also include 3 simultaneous rotated grouped 70/30 holdouts, repeated grouped 70/30 holdouts, and grouped k-fold validation, so the reported result is less dependent on one lucky or unlucky split.

Outputs from retraining:

- `design_assessment_model.joblib` — saved model package
- `model_metrics.json` — true holdout validation metrics, repeated validation summaries, and per-run/fold details
- `validation_holdout_scored.xlsx` — the first 30% test set scored by the tool
- `validation_rotations_scored.xlsx` — each rotated 30% prediction set on its own sheet, plus a `combined` sheet with all rotated predictions together

Use a different random split or percentage if needed:

```bash
python train_model.py "historical_data.xlsx" --train-size 0.70 --random-state 123
```

Run exactly the 3-way rotated 70/30 validation view with each prediction set and a combined sheet:

```bash
python train_model.py "historical_data.xlsx" --rotation-runs 3 --n-jobs 3
```

Change the grouping column or the number of rotated/repeated validation runs/folds if needed:

```bash
python train_model.py "historical_data.xlsx" --group-column "PE - PLI #" --rotation-runs 3 --cv-repeats 10 --cv-folds 5
```

For final production after you are satisfied with validation, you can validate on 70/30 and then save a model refit on all non-mandatory historical rows:

```bash
python train_model.py "historical_data.xlsx" --fit-final-on-all
```

## Main output columns

- `DA Triage Tier`
- `DA Required Flag`
- `DA High Confidence Review Flag`
- `DA Watchlist Flag`
- `DA Review Queue Flag` — Required + High Confidence
- `DA Broad Attention Flag` — Required + High Confidence + Watchlist
- `Recommended DA Action`
- `Tiered DA Reason`

The older output columns are also retained for compatibility:

- `Design Assessment Hard Required Flag`
- `Design Assessment Review Flag`
- `DA Triage Recommendation`
- `Final DA Reason`

## Notes

This is a screening/triage tool. It should not replace the quality/design review decision. The hard mandatory Decision rule is intentionally conservative and should not be lowered by the model.
