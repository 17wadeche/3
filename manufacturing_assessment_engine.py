from __future__ import annotations

import re
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd

try:
    from design_assessment_engine import BASE_COLUMNS, TEXT_COLUMNS, clean_df, read_input_file
except Exception:
    BASE_COLUMNS: list[str] = []
    TEXT_COLUMNS: list[str] = []

    def clean_df(df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        for column in cleaned.columns:
            if cleaned[column].dtype == "object":
                cleaned[column] = cleaned[column].fillna("").astype(str).str.replace("\xa0", " ", regex=False).str.strip()
        return cleaned

    def read_input_file(file: Any) -> pd.DataFrame:
        return pd.read_excel(file)

try:
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # sklearn is optional; deterministic rule-only mode still works.
    HashingVectorizer = None
    TfidfTransformer = None
    SGDClassifier = None
    Pipeline = None
    StratifiedGroupKFold = None

# IMPORTANT:
# Overall accuracy is misleading for this dataset because only ~3.6% of rows are historical Mfg Assessment.
# A model that flags nothing is already ~96.4% "accurate" but misses every Mfg Assessment.
# These thresholds are review-routing presets from the uploaded workbook's grouped out-of-fold validation.
MANUFACTURING_REVIEW_THRESHOLDS = {
    # Best starting point when the business goal is "do not miss Mfg Assessment".
    # On the provided workbook, this was approximately 95% historical recall and a large review queue.
    "high_recall": 0.037,
    # Smaller queue; materially better than the original rules, but lower recall.
    "balanced": 0.20,
    # Highest confidence rows only; high overall accuracy, but many missed Mfg rows.
    "high_confidence": 0.50,
}

MANUFACTURING_REVIEW_THRESHOLD = MANUFACTURING_REVIEW_THRESHOLDS["balanced"]
RULE_ONLY_REVIEW_THRESHOLD = 0.55
HIGH_PRIORITY_THRESHOLD = 0.50
MEDIUM_PRIORITY_THRESHOLD = 0.20

HISTORICAL_MFG_LABEL_RE = re.compile(r"\b(?:mfg|manufactur(?:e|ed|ing))\s*assessment\b", re.I)

KEY_EXPORT_COLUMNS = [
    "Product Event ID", "PE - PLI #", "Decision", "Type - PE PLI Task",
    "Historical Manufacturing Assessment in Original Data",
    "Manufacturing Assessment Probability", "Manufacturing Assessment Priority",
    "Manufacturing Assessment Recommendation", "Manufacturing Assessment Review Flag",
    "Manufacturing Assessment Reason", "Product Description – PE PLI", "Brief Description – PE",
    "Event Description – PE", "Event Context – PE", "Code/LLT Desc – PE PLI",
    "Product Returned to MDT? – PE PLI", "Rationale for no return – PE PLI",
]

MFG_TEXT_COLUMNS = [
    "Product Description – PE PLI",
    "Brief Description – PE",
    "Event Description – PE",
    "Event Context – PE",
    "Code/LLT Desc – PE PLI",
    "Product Returned to MDT? – PE PLI",
    "Rationale for no return – PE PLI",
    "Labeled for Single Use – PE PLI PM",
    "Reportable?",
    "Complaint? – PE",
]

# These columns are also learned as smoothed historical-rate lookups.
# They help the model remember high-risk recurring LLT/product/brief patterns without needing a huge text model.
MFG_RATE_COLUMNS = [
    "Code/LLT Desc – PE PLI",
    "Brief Description – PE",
    "Product Description – PE PLI",
    "Event Context – PE",
    "Product Returned to MDT? – PE PLI",
    "Rationale for no return – PE PLI",
    "Complaint? – PE",
]

# Definition/quality signals plus the recurring historical false-negative families from the workbook.
MANUFACTURING_RULE_PATTERNS = {
    "Explicit manufacturing/process/supplier signal": [
        r"\bmanufactur(?:e|ed|ing)\b",
        r"\bassembly\b|\bassembled\b|\bassembling\b",
        r"\bsupplier\b|\bvendor\b|\bexternal supplier\b",
        r"\bprocess(?:ing)? (?:issue|defect|deviation|nonconformance)\b",
        r"\binvestigation\b.*\bmanufactur",
        r"\bCAPA\b.*\bmanufactur",
    ],
    "Sterile packaging compromised": [
        r"\bsterile packaging\b",
        r"\bpackag(?:e|ing) (?:compromised|breach(?:ed)?|damaged|torn|open|unsealed|incorrect|wrong|seal(?:ed)? issue|seal failure)\b",
        r"\bseal (?:breach|failure|damage|open|compromised|disengaged|partial)\b",
        r"\bsterility (?:breach|compromised|issue|questioned)\b",
    ],
    "Foreign material/contamination/particulate": [
        r"\bforeign (?:material|matter|object|body|debris|object found)\b",
        r"\bcontaminat(?:ed|ion)\b",
        r"\bdebris inside\b",
        r"\bparticulate(?: matter)?\b",
        r"\bhair in package\b",
        r"\bwhite powdery substance\b",
        r"\bbroken pieces?\b",
    ],
    "Missing/damaged/detached/loose/broken component": [
        r"\bmissing (?:component|part|device|item|piece|screw|cap|clip|barb|barbs|tack|needle|suture|thread)\b",
        r"\b(?:component|part|device|item|piece|screw|cap|clip|needle|button|jaw|knife|blade|anvil|port|balloon|seal|tack|barb) (?:missing|damaged|broken|cracked|detached|detatch(?:ed)?|loose|disengaged|bent|frayed)\b",
        r"\bdetached (?:component|part|device|item|piece|needle)\b",
        r"\bdamaged (?:component|part|device|item|piece|package contents)\b",
        r"\b(?:needle|suture|thread) (?:detached|detatch(?:ed)?|broke|broken|frayed|missing|bent|appearance)\b",
        r"\b(?:component|parts?) (?:loose|disengaged)\b",
        r"\b(?:suture|thread) not secured\b",
    ],
    "Labeling/UDI/expiration/lot issue": [
        r"\blabel(?:ing|ling)? (?:issue|error|incorrect|wrong|mismatch|missing)\b",
        r"\bincorrect label\b|\bwrong label\b|\bmissing label\b",
        r"\bUDI\b|\bexpiration date\b|\bexpiry\b|\blot (?:mismatch|incorrect|wrong)\b",
    ],
    "Out-of-box/prior-to-use failure": [
        r"\bOOB\b",
        r"\bout[ -]?of[ -]?box\b",
        r"\bout of box failure\b",
        r"\bfailed (?:out of box|upon opening|before use|during setup)\b",
        r"\bprior to (?:use|patient contact)\b",
        r"\bduring setup\b",
    ],
    "Device malfunction/failure historically routed to Mfg": [
        r"\bdevice (?:did not|does not|would not|won't|wont) (?:activate|work|fire|respond|function|detect|recognize|recognise)\b",
        r"\b(?:did not|does not|would not|won't|wont) (?:activate|work|fire|deploy|load|unload|retract|detect|recognize|recognise|open|close|turn on)\b",
        r"\b(?:not detected|not recognized|not recognised|device stopped working|defective|stopped working)\b",
        r"\b(?:instrument|stapler|tacker|skin stapler|egia|endo gia) (?:did not|would not) fire\b",
        r"\b(?:staples?|clips?|tacks?) (?:did not deploy|not loading|spitting|slippage|failed to deploy)\b",
        r"\b(?:alarm activation|error code|red led activated|yellow led|led activated)\b",
        r"\b(?:switch|knob|button|battery|latch|toggle) (?:failure|broken|stuck|not responding|did not respond|would not respond)\b",
        r"\b(?:leak|leaks|leaking|gas leaked|air leak|leaking air|would not inflate|balloon broken|unraveled balloon)\b",
        r"\b(?:knife blade|jaws?) (?:did not advance|difficult to advance|will not open|would not open|locked|stuck|broke)\b",
        r"\b(?:suture|thread) (?:broke|broken|frayed|missing|not secured|did not release|unable to remove|difficult to remove)\b",
        r"\b(?:needle) (?:detached|detatch|prematur|bent|appearance)\b",
        r"\b(?:corrosion|rust) found\b",
        r"\bunable to perform clamp test\b",
        r"\bdifficult to (?:load|retract|remove|advance)\b",
    ],
}

MANUFACTURING_EXCLUSION_PATTERNS = [
    r"\bnot a complaint\b",
    r"\buser error\b",
    r"\bmisuse\b",
    r"\bduplicate of\b",
]

COMPILED_MANUFACTURING_RULE_PATTERNS = {
    reason: re.compile("|".join(f"(?:{pattern})" for pattern in patterns), re.I)
    for reason, patterns in MANUFACTURING_RULE_PATTERNS.items()
}
COMPILED_MANUFACTURING_EXCLUSION_RE = re.compile(
    "|".join(f"(?:{pattern})" for pattern in MANUFACTURING_EXCLUSION_PATTERNS), re.I
)


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def _norm_key(value: Any) -> str:
    return _safe_str(value).lower()


def _series_or_blank(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype="object")


def _returned_yes(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper().isin(["Y", "YES", "TRUE", "1"])


def historical_mfg_label(series: pd.Series) -> pd.Series:
    """True for historical labels such as 'Mfg Assessment' or 'Manufacturing Assessment'."""
    return series.fillna("").astype(str).str.contains(HISTORICAL_MFG_LABEL_RE, regex=True, na=False)


def build_manufacturing_text(df: pd.DataFrame) -> list[str]:
    df = clean_df(df)
    available = [c for c in MFG_TEXT_COLUMNS if c in df.columns]
    texts: list[str] = []
    for _, row in df[available].iterrows():
        parts: list[str] = []
        for column in available:
            value = _safe_str(row.get(column, ""))
            if not value:
                continue
            label = column.lower().replace(" – ", "_").replace(" ", "_").replace("/", "_").replace("?", "")
            repeat = 1
            if column in {"Code/LLT Desc – PE PLI", "Brief Description – PE"}:
                repeat = 4
            elif column == "Product Description – PE PLI":
                repeat = 3
            elif column in {"Product Returned to MDT? – PE PLI", "Event Context – PE", "Complaint? – PE"}:
                repeat = 2
            if column == "Event Description – PE":
                value = value[:900]
            parts.extend([f"{label}: {value}"] * repeat)

        returned = _safe_str(row.get("Product Returned to MDT? – PE PLI", "")).upper()
        complaint = _safe_str(row.get("Complaint? – PE", "")).upper()
        if returned in {"Y", "YES", "TRUE", "1"}:
            parts.extend(["returned_yes"] * 10)
        else:
            parts.extend(["returned_no_or_unknown"] * 4)
        if complaint in {"Y", "YES", "TRUE", "1"}:
            parts.extend(["complaint_yes"] * 3)
        else:
            parts.extend(["complaint_no_or_unknown"] * 3)
        texts.append(" | ".join(parts).lower())
    return texts




def build_manufacturing_rule_text(df: pd.DataFrame) -> list[str]:
    """Short, non-repeated text used only for regex rules. Keeps scoring fast."""
    df = clean_df(df)
    available = [c for c in MFG_TEXT_COLUMNS if c in df.columns]
    texts: list[str] = []
    for row in df[available].itertuples(index=False, name=None):
        parts: list[str] = []
        for column, value in zip(available, row):
            value_text = _safe_str(value)
            if not value_text:
                continue
            if column == "Event Description – PE":
                value_text = value_text[:500]
            parts.append(value_text)
        texts.append(" | ".join(parts).lower())
    return texts

def rule_score_text(text: str, product_returned: str = "") -> tuple[float, str]:
    lower = (text or "").lower()[:5000]
    returned_yes = str(product_returned or "").strip().upper() in {"Y", "YES", "TRUE", "1"}
    reasons: list[str] = []
    for reason, compiled_pattern in COMPILED_MANUFACTURING_RULE_PATTERNS.items():
        if compiled_pattern.search(lower):
            reasons.append(reason)

    if not reasons:
        # Returned product alone is not enough to route, but it is weak signal for the supervised model/ranking.
        return (0.025 if returned_yes else 0.0), ""

    unique_reasons = list(dict.fromkeys(reasons))
    has_exclusion = bool(COMPILED_MANUFACTURING_EXCLUSION_RE.search(lower))
    has_definition_signal = any(
        r in unique_reasons
        for r in [
            "Explicit manufacturing/process/supplier signal",
            "Sterile packaging compromised",
            "Foreign material/contamination/particulate",
            "Missing/damaged/detached/loose/broken component",
            "Labeling/UDI/expiration/lot issue",
            "Out-of-box/prior-to-use failure",
        ]
    )
    has_malfunction_signal = "Device malfunction/failure historically routed to Mfg" in unique_reasons

    if has_definition_signal:
        probability = 0.62
    elif has_malfunction_signal and returned_yes:
        probability = 0.18  # above high-recall and below balanced unless supported by ML/history
    elif has_malfunction_signal:
        probability = 0.07
    else:
        probability = 0.03

    probability = min(0.95, probability + 0.08 * (len(unique_reasons) - 1))
    if has_exclusion:
        probability = min(probability, 0.02)
        return probability, "Potential exclusion present; reviewer should confirm: " + "; ".join(unique_reasons)

    return probability, "; ".join(unique_reasons)


def _build_rate_tables(df: pd.DataFrame, labels: np.ndarray, smoothing: float = 25.0) -> dict[str, Any]:
    prevalence = float(np.mean(labels)) if len(labels) else 0.0
    tables: dict[str, dict[str, tuple[float, int]]] = {}
    for column in MFG_RATE_COLUMNS:
        if column not in df.columns:
            continue
        temp = pd.DataFrame({"key": df[column].map(_norm_key), "label": labels})
        stats = temp.groupby("key", dropna=False).agg(n=("label", "size"), pos=("label", "sum"))
        rate = (stats["pos"] + smoothing * prevalence) / (stats["n"] + smoothing)
        tables[column] = {str(key): (float(rate.loc[key]), int(stats.loc[key, "n"])) for key in stats.index}
    return {"prevalence": prevalence, "smoothing": smoothing, "tables": tables}


def _rate_table_probabilities(df: pd.DataFrame, rate_tables: dict[str, Any] | None) -> np.ndarray:
    if not rate_tables:
        return np.zeros(len(df), dtype=float)
    prevalence = float(rate_tables.get("prevalence", 0.0))
    tables = rate_tables.get("tables", {})
    out = np.full(len(df), prevalence, dtype=float)
    for column, table in tables.items():
        if column not in df.columns:
            continue
        values = df[column].map(_norm_key)
        col_probs = []
        for value in values:
            item = table.get(value)
            if not item:
                col_probs.append(prevalence)
            else:
                rate, n = item
                # Trust recurring categories more than one-off categories.
                credibility = min(1.0, np.log1p(n) / np.log1p(75))
                col_probs.append((credibility * rate) + ((1.0 - credibility) * prevalence))
        out = np.maximum(out, np.array(col_probs, dtype=float))
    return out


def train_manufacturing_model(training_df: pd.DataFrame) -> dict[str, Any]:
    """Train a high-recall model from historical rows.

    The returned object is a small dictionary with:
      - a text model, when scikit-learn is installed
      - smoothed historical-rate tables for recurring LLT/product/brief patterns
      - prevalence metadata
    """
    df = clean_df(training_df)
    if "Type - PE PLI Task" not in df.columns:
        raise ValueError("Training data must include 'Type - PE PLI Task'.")

    labels = historical_mfg_label(df["Type - PE PLI Task"]).astype(int).to_numpy()
    positives = int(labels.sum())
    if positives < 25:
        raise ValueError(f"Not enough historical Mfg Assessment rows to train a model: found {positives}.")

    rate_tables = _build_rate_tables(df, labels)
    model: Any = None
    if Pipeline is not None:
        texts = build_manufacturing_text(df)
        model = Pipeline([
            ("hash", HashingVectorizer(
                n_features=2 ** 17,
                alternate_sign=False,
                ngram_range=(1, 2),
                norm=None,
                lowercase=True,
            )),
            ("tfidf", TfidfTransformer(sublinear_tf=True)),
            ("clf", SGDClassifier(
                loss="log_loss",
                alpha=1e-5,
                class_weight="balanced",
                max_iter=1000,
                tol=1e-3,
                random_state=42,
            )),
        ])
        model.fit(texts, labels)

    return {
        "model_type": "manufacturing_assessment_v3_high_recall",
        "text_model": model,
        "rate_tables": rate_tables,
        "prevalence": float(labels.mean()),
        "positive_rows": positives,
        "total_rows": int(len(df)),
    }


def _predict_model_probabilities(model_bundle: Any, texts: list[str], df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    text_probabilities = np.zeros(len(df), dtype=float)
    rate_probabilities = np.zeros(len(df), dtype=float)

    if model_bundle is None:
        return text_probabilities, rate_probabilities

    # Backward compatibility: accept either the new dict bundle or a plain sklearn Pipeline.
    if isinstance(model_bundle, dict):
        text_model = model_bundle.get("text_model")
        rate_probabilities = _rate_table_probabilities(df, model_bundle.get("rate_tables"))
    else:
        text_model = model_bundle

    if text_model is not None and hasattr(text_model, "predict_proba"):
        text_probabilities = text_model.predict_proba(texts)[:, 1].astype(float)

    return text_probabilities, rate_probabilities


def is_complaint_excluded(df: pd.DataFrame) -> pd.Series:
    decision = _series_or_blank(df, "Decision").str.lower()
    complaint = _series_or_blank(df, "Complaint? – PE").str.upper()
    return decision.eq("not a complaint") | complaint.isin(["N", "NO", "FALSE", "0"])


def _resolve_threshold(review_threshold: float | None, mode: str) -> float:
    if review_threshold is not None:
        return float(review_threshold)
    return float(MANUFACTURING_REVIEW_THRESHOLDS.get(mode, MANUFACTURING_REVIEW_THRESHOLD))


def _priority(probability: float) -> str:
    if probability >= HIGH_PRIORITY_THRESHOLD:
        return "High"
    if probability >= MEDIUM_PRIORITY_THRESHOLD:
        return "Medium"
    if probability >= MANUFACTURING_REVIEW_THRESHOLDS["high_recall"]:
        return "Low"
    return "None"


def score_dataframe(
    df: pd.DataFrame,
    review_threshold: float | None = None,
    manufacturing_model: Any | None = None,
    mode: str = "balanced",
) -> pd.DataFrame:
    """Score rows for potential Manufacturing Assessment review.

    Recommended production setup:
      1) Train once on historical data: mfg_model = train_manufacturing_model(historical_df)
      2) Score incoming data: score_dataframe(new_df, manufacturing_model=mfg_model, mode="balanced")

    Modes:
      - high_recall: catches approximately 95% of historical Mfg rows in grouped validation, but creates a large queue.
      - balanced: smaller queue with lower recall.
      - high_confidence: high-confidence queue only; not appropriate if misses are costly.
    """
    df = clean_df(df)
    texts = build_manufacturing_text(df)
    rule_texts = build_manufacturing_rule_text(df)
    returned = _series_or_blank(df, "Product Returned to MDT? – PE PLI")

    scored_rules = [rule_score_text(text, ret) for text, ret in zip(rule_texts, returned)]
    rule_probabilities = np.array([score for score, _reason in scored_rules], dtype=float)
    rule_reasons = [reason for _score, reason in scored_rules]

    text_probabilities, rate_probabilities = _predict_model_probabilities(manufacturing_model, texts, df)
    probabilities = np.maximum.reduce([rule_probabilities, text_probabilities, rate_probabilities])

    active_threshold = _resolve_threshold(review_threshold, mode)
    if manufacturing_model is None:
        active_threshold = max(active_threshold, RULE_ONLY_REVIEW_THRESHOLD)

    complaint_exclusion = is_complaint_excluded(df)
    review_flag = ((probabilities >= active_threshold) & (~complaint_exclusion.to_numpy())).astype(int)
    recommendation = np.where(
        review_flag == 1,
        "Route to Manufacturing Assessment review",
        "No Manufacturing Assessment flag",
    )

    priorities = [_priority(float(p)) if flag else "None" for p, flag in zip(probabilities, review_flag)]

    final_reasons: list[str] = []
    for flag, excluded, probability, text_p, rate_p, rule_p, rule_reason in zip(
        review_flag, complaint_exclusion, probabilities, text_probabilities, rate_probabilities, rule_probabilities, rule_reasons
    ):
        if excluded:
            final_reasons.append("Exclusion: not a complaint unless manufacturing reviewer identifies a device issue")
        elif flag:
            reason_parts: list[str] = []
            if text_p >= active_threshold:
                reason_parts.append(f"ML historical-pattern score {text_p:.2f}")
            if rate_p >= active_threshold:
                reason_parts.append(f"Historical LLT/product/brief rate score {rate_p:.2f}")
            if rule_p >= active_threshold and rule_reason:
                reason_parts.append(f"rule signal: {rule_reason}")
            if not reason_parts:
                reason_parts.append(f"combined score {probability:.2f} exceeded {mode} threshold {active_threshold:.3f}")
            final_reasons.append("; ".join(reason_parts))
        else:
            if rule_reason:
                final_reasons.append(f"Below review threshold ({probability:.2f}); weak/insufficient signal: {rule_reason}")
            else:
                final_reasons.append("No manufacturing, returned-device malfunction, sterile packaging, foreign material, component, labeling, OOB, supplier, or prior manufacturing-related signal")

    scored = df.copy()
    if "Type - PE PLI Task" in scored.columns:
        historical = historical_mfg_label(scored["Type - PE PLI Task"]).astype(int)
    else:
        historical = pd.Series([0] * len(scored), index=scored.index, dtype=int)

    scored["Historical Manufacturing Assessment in Original Data"] = historical
    scored["Manufacturing Assessment Rule Probability"] = rule_probabilities
    scored["Manufacturing Assessment ML Probability"] = text_probabilities
    scored["Manufacturing Assessment Historical Rate Probability"] = rate_probabilities
    scored["Manufacturing Assessment Probability"] = probabilities
    scored["Manufacturing Assessment Threshold Used"] = active_threshold
    scored["Manufacturing Assessment Priority"] = priorities
    scored["Manufacturing Assessment Review Flag"] = review_flag
    scored["Manufacturing Assessment Recommendation"] = recommendation
    scored["Manufacturing Assessment Reason"] = final_reasons
    return scored


def summarize_scored(scored: pd.DataFrame) -> dict[str, int | float]:
    total = int(len(scored))
    flags = pd.to_numeric(scored.get("Manufacturing Assessment Review Flag", pd.Series([0] * total)), errors="coerce").fillna(0).astype(int)
    historical = pd.to_numeric(scored.get("Historical Manufacturing Assessment in Original Data", pd.Series([0] * total)), errors="coerce").fillna(0).astype(int)
    out: dict[str, int | float] = {
        "total_rows": total,
        "review_rows": int(flags.sum()),
        "no_flag_rows": int(total - flags.sum()),
        "historical_mfg_rows": int(historical.sum()),
    }
    if historical.sum() > 0:
        tp = int(((flags == 1) & (historical == 1)).sum())
        fp = int(((flags == 1) & (historical == 0)).sum())
        fn = int(((flags == 0) & (historical == 1)).sum())
        tn = int(((flags == 0) & (historical == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        accuracy = (tp + tn) / total if total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out.update({
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return out


def threshold_report_from_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = [0.037, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
    rows: list[dict[str, float | int]] = []
    labels = labels.astype(int)
    for threshold in thresholds:
        pred = (probabilities >= threshold).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "review_rows": int(pred.sum()),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
        })
    return pd.DataFrame(rows)


def validate_manufacturing_model(
    historical_df: pd.DataFrame,
    n_splits: int = 5,
    group_column: str = "Product Event ID",
) -> pd.DataFrame:
    """Grouped out-of-fold validation to pick an operating threshold.

    This is slower than normal scoring, but it prevents same-event leakage when estimating performance.
    """
    if StratifiedGroupKFold is None:
        raise ImportError("scikit-learn is required for validate_manufacturing_model.")
    df = clean_df(historical_df)
    labels = historical_mfg_label(df["Type - PE PLI Task"]).astype(int).to_numpy()
    if group_column in df.columns:
        groups = df[group_column].fillna(df.index.to_series()).astype(str).to_numpy()
    else:
        groups = np.arange(len(df))

    probabilities = np.zeros(len(df), dtype=float)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in cv.split(df, labels, groups):
        model = train_manufacturing_model(df.iloc[train_idx].copy())
        scored = score_dataframe(df.iloc[test_idx].copy(), manufacturing_model=model, review_threshold=0.0)
        probabilities[test_idx] = scored["Manufacturing Assessment Probability"].to_numpy(dtype=float)

    thresholds = sorted(set([
        0.037, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90,
        *[float(x) for x in np.quantile(probabilities, np.linspace(0.01, 0.99, 50))],
    ]))
    return threshold_report_from_probabilities(probabilities, labels, thresholds)


def to_excel_bytes(scored: pd.DataFrame) -> bytes:
    output = BytesIO()
    summary = summarize_scored(scored)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        scored.to_excel(writer, index=False, sheet_name="Scored")
        scored[scored["Manufacturing Assessment Review Flag"] == 1][[c for c in KEY_EXPORT_COLUMNS if c in scored.columns]].to_excel(
            writer, index=False, sheet_name="Reviewer_Queue"
        )
        pd.DataFrame([{"Metric": key, "Value": value} for key, value in summary.items()]).to_excel(
            writer, index=False, sheet_name="Dashboard"
        )
    return output.getvalue()
