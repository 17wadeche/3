from __future__ import annotations

import re
from pathlib import Path
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import joblib

MANDATORY_DECISIONS = {"Death - Reportable", "Serious Injury - Reportable"}
HIGH_CONFIDENCE_THRESHOLD = 0.60
WATCHLIST_THRESHOLD = 0.0485
LOW_CONFIDENCE_CODE_LLT_DESCRIPTIONS = {
    "ARTICULATION INSUFFICIENT",
    "CLIP APPLIER DID NOT FIRE",
    "CORD/CABLE FAILURE/DAMAGE",
    "DEVICE MISSING BARBS",
    "SIGNIA ADAPTER SLOW TO RECOGNIZE",
    "SUTURE APPEARANCE",
    "SUTURES ARE TOO LOOSE",
    "WILL NOT ROTATE",
}

TEXT_COLUMNS = [
    "Product Description – PE PLI",
    "Brief Description – PE",
    "Event Description – PE",
    "Event Context – PE",
    "Code/LLT Desc – PE PLI",
    "Complaint? – PE",
    "Reportable?",
    "Decision",
    "Product Returned to MDT? – PE PLI",
    "Rationale for no return – PE PLI",
]

BASE_COLUMNS = [
    "Product Event ID", "PE - PLI #", "Serial Number – PE PLI", "Lot Number – PE PLI",
    "Product Description – PE PLI", "Labeled for Single Use – PE PLI PM", "Brief Description – PE",
    "Event Description – PE", "Event Context – PE", "Code/LLT Desc – PE PLI", "Complaint? – PE",
    "Reportable?", "Decision", "Product Returned to MDT? – PE PLI", "Rationale for no return – PE PLI",
    "Type - PE PLI Task",
]

RULE_PATTERNS = {
    "Possible new/unanticipated issue": [
        r"\bunanticipated\b", r"\bunexpected\b", r"\bnew failure mode\b", r"\bnew issue\b",
        r"\bnot previously observed\b", r"\bpreviously investigated\b", r"\bnew model\b", r"\bnew product\b",
    ],
    "Software/display/control issue signal": [
        r"\bsoftware issue\b", r"\bsoftware defect\b", r"\bsoftware bug\b", r"\bfirmware\b",
        r"\berror message\b", r"\bdisplay(?:ed|s)?\b", r"\bscreen\b", r"\bfroze\b", r"\bfrozen\b",
        r"\bfreeze\b", r"\bcrash(?:ed)?\b", r"\breboot(?:ed)?\b", r"\bshut(?:\s|-)?down\b",
        r"\bcalibration\b", r"\bincorrect calculation\b", r"\bwrong value\b", r"\bwrong dose\b",
    ],
    "Prior action/scope mismatch signal": [
        r"\boutside scope\b", r"\bout of scope\b", r"\bfield action\b", r"\bfurther action\b",
        r"\brecall\b", r"\bcapa\b", r"\bsame failure mode\b", r"\bprevious action\b",
    ],
}
EXCLUSION_PATTERNS = [r"\bstandard software analysis\b"]


def _normalize_text(value: Any) -> str:
    """Normalize free text for case-insensitive rule matching."""
    return re.sub(r"\s+", " ", str(value or "").strip()).upper()


def low_confidence_code_llt_reason(row: pd.Series) -> str:
    """Return a low-confidence rule reason when Code/LLT and Complaint? match."""
    complaint = _normalize_text(row.get("Complaint? – PE", ""))
    if complaint not in {"Y", "YES", "TRUE"}:
        return ""

    code_llt = _normalize_text(row.get("Code/LLT Desc – PE PLI", ""))
    if not code_llt:
        return ""

    matched = [
        desc
        for desc in sorted(LOW_CONFIDENCE_CODE_LLT_DESCRIPTIONS)
        if desc in code_llt
    ]
    if not matched:
        return ""
    return "Low-confidence Code/LLT complaint rule: " + "; ".join(matched)

KEY_EXPORT_COLUMNS = [
    "Product Event ID", "PE - PLI #", "Decision", "Type - PE PLI Task", "Model DA Probability",
    "DA Triage Tier", "Recommended DA Action", "Tiered DA Reason", "Product Description – PE PLI",
    "Brief Description – PE", "Event Description – PE", "Event Context – PE", "Text Rule Reason",
]

def read_input_file(file_or_path: Any) -> pd.DataFrame:
    """Read a CSV/XLSX export. Handles normal headers and PowerBI-style row-4 headers."""
    name = getattr(file_or_path, "name", str(file_or_path))
    if str(name).lower().endswith(".csv"):
        df = pd.read_csv(file_or_path)
    else:
        df = pd.read_excel(file_or_path, header=0)
        if "Product Event ID" not in df.columns and "Decision" not in df.columns:
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            df = pd.read_excel(file_or_path, header=3)
    return df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in BASE_COLUMNS + TEXT_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    for c in df.columns:
        df[c] = df[c].fillna("").astype(str).str.strip()
    return df


def build_text(df: pd.DataFrame) -> list[str]:
    df = clean_df(df)
    texts: list[str] = []
    for vals in df[TEXT_COLUMNS].itertuples(index=False, name=None):
        parts = [f"{c}: {v}" for c, v in zip(TEXT_COLUMNS, vals) if v]
        texts.append(" | ".join(parts))
    return texts


def rule_score_text(text: str) -> tuple[int, str]:
    lower = (text or "").lower()
    reasons: list[str] = []
    for reason, pats in RULE_PATTERNS.items():
        if any(re.search(p, lower, flags=re.I) for p in pats):
            reasons.append(reason)
    exclusion = any(re.search(p, lower, flags=re.I) for p in EXCLUSION_PATTERNS)
    if exclusion and not reasons:
        return 0, "Exclusion note: standard software analysis alone is not a DA trigger"
    if reasons:
        return 1, "; ".join(dict.fromkeys(reasons))
    return 0, ""


def load_model(model_path: str | Path = "design_assessment_model.joblib") -> dict:
    payload = joblib.load(model_path)
    if isinstance(payload, dict) and "model" in payload:
        payload.setdefault("mandatory_decisions", sorted(MANDATORY_DECISIONS))
        payload.setdefault("high_confidence_threshold", payload.get("high_threshold", HIGH_CONFIDENCE_THRESHOLD))
        payload.setdefault("watchlist_threshold", payload.get("review_threshold", WATCHLIST_THRESHOLD))
        return payload
    return {
        "model": payload,
        "mandatory_decisions": sorted(MANDATORY_DECISIONS),
        "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
        "watchlist_threshold": WATCHLIST_THRESHOLD,
    }


def _as_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def score_dataframe(
    df: pd.DataFrame,
    model_payload: dict | None = None,
    high_confidence_threshold: float | None = None,
    watchlist_threshold: float | None = None,
) -> pd.DataFrame:
    """Score a dataset and assign one of four tiers:
    DA REQUIRED, DA REVIEW - HIGH CONFIDENCE, DA WATCHLIST - LOW CONFIDENCE, or NO DA FLAG.
    """
    df = clean_df(df)
    if model_payload is None:
        model_payload = load_model(Path(__file__).with_name("design_assessment_model.joblib"))

    model = model_payload["model"]
    high_threshold = float(high_confidence_threshold if high_confidence_threshold is not None else model_payload.get("high_confidence_threshold", model_payload.get("high_threshold", HIGH_CONFIDENCE_THRESHOLD)))
    watch_threshold = float(watchlist_threshold if watchlist_threshold is not None else model_payload.get("watchlist_threshold", model_payload.get("review_threshold", WATCHLIST_THRESHOLD)))
    mandatory_decisions = set(model_payload.get("mandatory_decisions", sorted(MANDATORY_DECISIONS)))

    X = build_text(df)
    probs = model.predict_proba(X)[:, 1]
    mandatory = df["Decision"].isin(mandatory_decisions).to_numpy()
    historical = (df["Type - PE PLI Task"].str.lower() == "design assessment").astype(int).to_numpy()
    rule_results = [rule_score_text(t) for t in X]
    rule_signal = np.array([x[0] for x in rule_results], dtype=int)
    rule_reason = [x[1] for x in rule_results]
    low_confidence_reasons = [low_confidence_code_llt_reason(row) for _, row in df.iterrows()]
    low_confidence_signal = np.array([1 if reason else 0 for reason in low_confidence_reasons], dtype=int)
    rule_signal = np.maximum(rule_signal, low_confidence_signal)
    rule_reason = [
        "; ".join(part for part in [base_reason, low_confidence_reason] if part)
        for base_reason, low_confidence_reason in zip(rule_reason, low_confidence_reasons)
    ]
    exclusion = (df["Decision"].str.lower() == "not a complaint") | (df["Complaint? – PE"].str.upper().isin(["N", "NO", "FALSE"]))

    tier: list[str] = []
    required_flag: list[int] = []
    high_flag: list[int] = []
    watch_flag: list[int] = []
    review_queue_flag: list[int] = []
    broad_attention_flag: list[int] = []
    action: list[str] = []
    reason: list[str] = []
    sort_order: list[int] = []

    for i in range(len(df)):
        if mandatory[i]:
            tier.append("DA REQUIRED")
            required_flag.append(1); high_flag.append(0); watch_flag.append(0)
            review_queue_flag.append(1); broad_attention_flag.append(1); sort_order.append(1)
            action.append("Open Design Assessment / route to DA owner")
            reason.append(f"Mandatory Decision = {df.iloc[i]['Decision']}; overrides historical task value")
        elif bool(exclusion.iloc[i]):
            tier.append("NO DA FLAG")
            required_flag.append(0); high_flag.append(0); watch_flag.append(0)
            review_queue_flag.append(0); broad_attention_flag.append(0); sort_order.append(4)
            action.append("No DA triage flag unless reviewer identifies new device/software issue")
            reason.append("Exclusion: not a complaint unless mandatory Decision applies")
        elif low_confidence_signal[i] == 1:
            tier.append("DA WATCHLIST - LOW CONFIDENCE")
            required_flag.append(0); high_flag.append(0); watch_flag.append(1)
            review_queue_flag.append(0); broad_attention_flag.append(1); sort_order.append(3)
            action.append("Watchlist/trending review; not a required DA without reviewer confirmation")
            reason.append(low_confidence_reasons[i])
        elif probs[i] >= high_threshold:
            tier.append("DA REVIEW - HIGH CONFIDENCE")
            required_flag.append(0); high_flag.append(1); watch_flag.append(0)
            review_queue_flag.append(1); broad_attention_flag.append(1); sort_order.append(2)
            action.append("Route to DA review queue")
            reason.append(f"Model probability >= {high_threshold:.2f}; likely historical DA pattern")
        elif rule_signal[i] == 1 or probs[i] >= watch_threshold:
            tier.append("DA WATCHLIST - LOW CONFIDENCE")
            required_flag.append(0); high_flag.append(0); watch_flag.append(1)
            review_queue_flag.append(0); broad_attention_flag.append(1); sort_order.append(3)
            action.append("Watchlist/trending review; not a required DA without reviewer confirmation")
            if rule_signal[i] == 1:
                reason.append(f"Low-confidence definition signal: {rule_reason[i]}" if rule_reason[i] else "Low-confidence definition signal")
            else:
                reason.append(f"Model probability >= {watch_threshold:.2f} watchlist threshold")
        else:
            tier.append("NO DA FLAG")
            required_flag.append(0); high_flag.append(0); watch_flag.append(0)
            review_queue_flag.append(0); broad_attention_flag.append(0); sort_order.append(4)
            action.append("No DA flag")
            reason.append("No mandatory Decision, high-confidence model score, text rule signal, or watchlist threshold")

    scored = df.copy()
    scored["Historical DA in Original Data"] = historical
    scored["Mandatory Decision DA Flag"] = mandatory.astype(int)
    scored["Override Applied to Historical Data"] = ((mandatory) & (historical == 0)).astype(int)
    scored["Corrected DA Training Label"] = np.where(mandatory, 1, historical)
    scored["Text Rule Signal"] = rule_signal
    scored["Text Rule Reason"] = rule_reason
    scored["Model DA Probability"] = probs

    # New tiered outputs.
    scored["DA Triage Tier"] = tier
    scored["DA Required Flag"] = required_flag
    scored["DA High Confidence Review Flag"] = high_flag
    scored["DA Watchlist Flag"] = watch_flag
    scored["DA Review Queue Flag"] = review_queue_flag
    scored["DA Broad Attention Flag"] = broad_attention_flag
    scored["Recommended DA Action"] = action
    scored["Tiered DA Reason"] = reason
    scored["Tier Sort"] = sort_order

    # Backward-compatible names from the earlier version.
    scored["Design Assessment Hard Required Flag"] = scored["DA Required Flag"]
    scored["Design Assessment Review Flag"] = scored["DA Broad Attention Flag"]
    scored["DA Triage Recommendation"] = scored["DA Triage Tier"]
    scored["Final DA Reason"] = scored["Tiered DA Reason"]
    return scored


def summarize_scored(scored: pd.DataFrame) -> dict:
    total = int(len(scored))
    tier_counts = scored.get("DA Triage Tier", pd.Series(dtype=str)).value_counts().to_dict()
    out = {
        "total_rows": total,
        "tier_counts": {k: int(tier_counts.get(k, 0)) for k in ["DA REQUIRED", "DA REVIEW - HIGH CONFIDENCE", "DA WATCHLIST - LOW CONFIDENCE", "NO DA FLAG"]},
        "review_queue_rows": int(_as_float_series(scored.get("DA Review Queue Flag", pd.Series([0]*total))).sum()),
        "broad_attention_rows": int(_as_float_series(scored.get("DA Broad Attention Flag", pd.Series([0]*total))).sum()),
        "mandatory_overrides": int(_as_float_series(scored.get("Override Applied to Historical Data", pd.Series([0]*total))).sum()),
    }
    if "Corrected DA Training Label" in scored.columns:
        y = _as_float_series(scored["Corrected DA Training Label"]).astype(int) == 1
        for label, col in [("review_queue", "DA Review Queue Flag"), ("broad_attention", "DA Broad Attention Flag")]:
            pred = _as_float_series(scored[col]).astype(int) == 1
            tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            accuracy = (tp + tn) / total if total else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            out[label + "_scorecard"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    return out


def to_excel_bytes(scored: pd.DataFrame) -> bytes:
    """Return a multi-sheet scored workbook as bytes."""
    output = BytesIO()
    summary = summarize_scored(scored)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        scored.to_excel(writer, index=False, sheet_name="Scored")
        scored[scored["DA Review Queue Flag"] == 1][[c for c in KEY_EXPORT_COLUMNS if c in scored.columns]].to_excel(writer, index=False, sheet_name="Reviewer_Queue")
        scored[scored["DA Watchlist Flag"] == 1][[c for c in KEY_EXPORT_COLUMNS if c in scored.columns]].to_excel(writer, index=False, sheet_name="Watchlist")
        pd.DataFrame([
            {"Metric": "Total rows", "Value": summary["total_rows"]},
            {"Metric": "DA REQUIRED", "Value": summary["tier_counts"].get("DA REQUIRED", 0)},
            {"Metric": "DA REVIEW - HIGH CONFIDENCE", "Value": summary["tier_counts"].get("DA REVIEW - HIGH CONFIDENCE", 0)},
            {"Metric": "DA WATCHLIST - LOW CONFIDENCE", "Value": summary["tier_counts"].get("DA WATCHLIST - LOW CONFIDENCE", 0)},
            {"Metric": "NO DA FLAG", "Value": summary["tier_counts"].get("NO DA FLAG", 0)},
            {"Metric": "Review queue rows", "Value": summary["review_queue_rows"]},
            {"Metric": "Broad attention rows", "Value": summary["broad_attention_rows"]},
            {"Metric": "Mandatory overrides", "Value": summary["mandatory_overrides"]},
        ]).to_excel(writer, index=False, sheet_name="Dashboard")
    return output.getvalue()
