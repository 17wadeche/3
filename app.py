import streamlit as st
import pandas as pd
from design_assessment_engine import (
    read_input_file,
    score_dataframe,
    load_model,
    to_excel_bytes,
    summarize_scored,
    HIGH_CONFIDENCE_THRESHOLD,
    WATCHLIST_THRESHOLD,
)

st.set_page_config(page_title="Design Assessment Tiered Screener", layout="wide")
st.title("Design Assessment Tiered Screener")
st.caption("Mandatory rule: Death - Reportable or Serious Injury - Reportable is always DA REQUIRED, even if prior history did not include a DA task.")


def _format_pct(value):
    if value is None:
        return ""
    return f"{float(value):.1%}"

def _scorecard_row(label: str, scorecard: dict) -> dict:
    return {
        "Validation set": label,
        "Accuracy": _format_pct(scorecard.get("accuracy")),
        "Precision": _format_pct(scorecard.get("precision")),
        "Recall": _format_pct(scorecard.get("recall")),
        "F1": _format_pct(scorecard.get("f1")),
        "TP": scorecard.get("tp", 0),
        "FP": scorecard.get("fp", 0),
        "FN": scorecard.get("fn", 0),
        "TN": scorecard.get("tn", 0),
    }

def _combined_scorecard_row(label: str, section: dict, scorecard_name: str) -> dict | None:
    scorecard = section.get(scorecard_name, {})
    total_counts = scorecard.get("total_counts", {})
    if not total_counts:
        return None
    return {
        "Validation set": label,
        "Accuracy": _format_pct(scorecard.get("accuracy", {}).get("mean")),
        "Precision": _format_pct(scorecard.get("precision", {}).get("mean")),
        "Recall": _format_pct(scorecard.get("recall", {}).get("mean")),
        "F1": _format_pct(scorecard.get("f1", {}).get("mean")),
        "TP": total_counts.get("tp", 0),
        "FP": total_counts.get("fp", 0),
        "FN": total_counts.get("fn", 0),
        "TN": total_counts.get("tn", 0),
    }

def show_validation_results(metrics: dict) -> None:
    if not metrics:
        return

    with st.expander("Current model validation results", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train rows", f"{metrics.get('train_rows', metrics.get('training_rows', 0)):,}")
        c2.metric("Test rows", f"{metrics.get('test_rows', metrics.get('validation_rows', 0)):,}")
        c3.metric("Mandatory Decision rows", f"{metrics.get('mandatory_decision_rows', 0):,}")
        c4.metric("Overturned historical rows", f"{metrics.get('overturned_rows', 0):,}")

        holdout = metrics.get("heldout_review_queue_scorecard", {})
        if holdout:
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Holdout accuracy", f"{holdout.get('accuracy', 0):.1%}")
            v2.metric("Holdout precision", f"{holdout.get('precision', 0):.1%}")
            v3.metric("Holdout recall", f"{holdout.get('recall', 0):.1%}")
            v4.metric("Holdout false negatives", f"{holdout.get('fn', 0):,}")

        rows = []
        if holdout:
            rows.append(_scorecard_row("Primary holdout", holdout))

        validation_section = metrics.get("rotated_grouped_holdouts", {})
        run_label = "Rotation"
        combined_label = "Rotations combined"
        if not validation_section.get("run_details"):
            validation_section = metrics.get("repeated_grouped_holdout", {})
            run_label = "Repeated holdout"
            combined_label = "Repeated holdouts combined"

        for run in validation_section.get("run_details", []):
            rows.append(_scorecard_row(f"{run_label} {run.get('run')}", run.get("review_queue_scorecard", {})))
        combined = _combined_scorecard_row(combined_label, validation_section, "review_queue_scorecard")
        if combined:
            rows.append(combined)

        if rows:
            st.markdown("**Review queue validation by set**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if run_label != "Rotation":
                st.info("Retrain with the latest `train_model.py` to populate the 3 rotated 70/30 validation rows. Showing repeated holdout runs from the current saved model until then.")

        st.caption(metrics.get("validation_method", "Validation metrics"))


payload = load_model("design_assessment_model.joblib")
metrics = payload.get("metrics", {})

with st.sidebar:
    st.header("Tier rules")
    high_threshold = st.slider("High-confidence model threshold", 0.01, 0.99, float(payload.get("high_confidence_threshold", payload.get("high_threshold", HIGH_CONFIDENCE_THRESHOLD))), 0.01)
    watch_threshold = st.slider("Watchlist model threshold", 0.01, 0.50, float(payload.get("watchlist_threshold", payload.get("review_threshold", WATCHLIST_THRESHOLD))), 0.01)
    st.markdown("**DA REQUIRED**: Death - Reportable or Serious Injury - Reportable")
    st.markdown("**DA REVIEW - HIGH CONFIDENCE**: model probability above high-confidence threshold")
    st.markdown("**DA WATCHLIST - LOW CONFIDENCE**: definition signal or model probability above watchlist threshold")
    st.warning("This is a triage tool. Final DA disposition should stay with the quality/design reviewer.")

show_validation_results(metrics)

uploaded = st.file_uploader("Upload a new Excel or CSV file to score", type=["xlsx", "xls", "csv"])
if uploaded is None:
    st.info("Upload a file with the same columns as the historical extract, including Decision.")
    st.stop()

try:
    df = read_input_file(uploaded)
    scored = score_dataframe(df, payload, high_confidence_threshold=high_threshold, watchlist_threshold=watch_threshold)
except Exception as e:
    st.error(f"Could not score the file: {e}")
    st.stop()

summary = summarize_scored(scored)
st.success(f"Scored {len(scored):,} rows")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("DA REQUIRED", f"{summary['tier_counts'].get('DA REQUIRED', 0):,}")
c2.metric("High-confidence review", f"{summary['tier_counts'].get('DA REVIEW - HIGH CONFIDENCE', 0):,}")
c3.metric("Watchlist", f"{summary['tier_counts'].get('DA WATCHLIST - LOW CONFIDENCE', 0):,}")
c4.metric("Review queue", f"{summary['review_queue_rows']:,}")
c5.metric("Mandatory overrides", f"{summary['mandatory_overrides']:,}")

if "review_queue_scorecard" in summary:
    sc = summary["review_queue_scorecard"]
    st.caption(f"Review queue score vs corrected label: accuracy {sc['accuracy']:.1%}, precision {sc['precision']:.1%}, recall {sc['recall']:.1%}.")

filter_choice = st.selectbox(
    "View",
    ["All rows", "DA review queue", "DA REQUIRED only", "High-confidence only", "Watchlist only", "No DA flag", "Mandatory overrides only"],
)
view = scored
if filter_choice == "DA review queue":
    view = scored[scored["DA Review Queue Flag"] == 1]
elif filter_choice == "DA REQUIRED only":
    view = scored[scored["DA Required Flag"] == 1]
elif filter_choice == "High-confidence only":
    view = scored[scored["DA High Confidence Review Flag"] == 1]
elif filter_choice == "Watchlist only":
    view = scored[scored["DA Watchlist Flag"] == 1]
elif filter_choice == "No DA flag":
    view = scored[scored["DA Broad Attention Flag"] == 0]
elif filter_choice == "Mandatory overrides only":
    view = scored[scored["Override Applied to Historical Data"] == 1]

cols = [c for c in [
    "Product Event ID", "PE - PLI #", "Decision", "Type - PE PLI Task", "Model DA Probability",
    "DA Triage Tier", "DA Review Queue Flag", "DA Watchlist Flag", "Recommended DA Action", "Tiered DA Reason",
    "Brief Description – PE", "Event Description – PE"
] if c in view.columns]
st.dataframe(view[cols], use_container_width=True, height=540)

st.download_button(
    "Download tiered scored Excel",
    data=to_excel_bytes(scored),
    file_name="design_assessment_tiered_scored.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
st.download_button(
    "Download tiered scored CSV",
    data=scored.to_csv(index=False).encode("utf-8"),
    file_name="design_assessment_tiered_scored.csv",
    mime="text/csv",
)
