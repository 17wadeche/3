from __future__ import annotations

import re
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd

from design_assessment_engine import BASE_COLUMNS, TEXT_COLUMNS, clean_df, read_input_file

MANUFACTURING_REVIEW_THRESHOLD = 0.50

MANUFACTURING_RULE_PATTERNS = {
    "Potential manufacturing process contribution": [
        r"\bmanufactur(?:e|ed|ing)\b",
        r"\bassembly\b|\bassembled\b|\bassembling\b",
        r"\bsupplier\b|\bvendor\b|\bexternal supplier\b",
        r"\bprocess(?:ing)? (?:issue|defect|deviation|nonconformance)\b",
    ],
    "Sterile packaging compromised": [
        r"\bsterile packaging\b",
        r"\bpackag(?:e|ing) (?:compromised|breach(?:ed)?|damaged|torn|open|unsealed|seal(?:ed)? issue|seal failure)\b",
        r"\bseal (?:breach|failure|damage|open|compromised)\b",
        r"\bsterility (?:breach|compromised|issue)\b",
    ],
    "Foreign material inside sterile package": [
        r"\bforeign (?:material|matter|object|body|debris)\b",
        r"\bcontaminat(?:ed|ion)\b",
        r"\bdebris inside\b",
        r"\bparticulate\b",
    ],
    "Missing, damaged, or detached component/device in package": [
        r"\bmissing (?:component|part|device|item|piece|screw|cap|clip)\b",
        r"\b(?:component|part|device|item|piece|screw|cap|clip) (?:missing|damaged|broken|cracked|detached|loose)\b",
        r"\bdetached (?:component|part|device|item|piece)\b",
        r"\bdamaged (?:component|part|device|item|piece|package contents)\b",
    ],
    "Labeling issue": [
        r"\blabel(?:ing|ling)? (?:issue|error|incorrect|wrong|mismatch|missing)\b",
        r"\bincorrect label\b|\bwrong label\b|\bmissing label\b",
        r"\bUDI\b|\bexpiration date\b|\bexpiry\b|\blot (?:mismatch|incorrect|wrong)\b",
    ],
    "Failure of product OOB": [
        r"\bOOB\b",
        r"\bout[ -]?of[ -]?box\b",
        r"\bout of box failure\b",
        r"\bfailed (?:out of box|upon opening|before use|during setup)\b",
    ],
    "Previously identified as potentially manufacturing related": [
        r"\bpreviously identified\b.*\bmanufactur",
        r"\bpotentially related to manufactur",
        r"\bfurther action\b.*\bmanufactur",
        r"\binvestigation\b.*\bmanufactur",
        r"\bCAPA\b.*\bmanufactur",
        r"\bassembly\b.*\bprevious",
    ],
}

MANUFACTURING_EXCLUSION_PATTERNS = [
    r"\bnot a complaint\b",
    r"\buser error\b",
    r"\bmisuse\b",
]

KEY_EXPORT_COLUMNS = [
    "Product Event ID", "PE - PLI #", "Decision", "Type - PE PLI Task", "Manufacturing Assessment Probability",
    "Manufacturing Assessment Recommendation", "Manufacturing Assessment Review Flag", "Manufacturing Assessment Reason",
    "Product Description – PE PLI", "Brief Description – PE", "Event Description – PE", "Event Context – PE",
    "Code/LLT Desc – PE PLI",
]


def build_manufacturing_text(df: pd.DataFrame) -> list[str]:
    df = clean_df(df)
    texts: list[str] = []
    for vals in df[TEXT_COLUMNS].itertuples(index=False, name=None):
        parts = [f"{c}: {v}" for c, v in zip(TEXT_COLUMNS, vals) if v]
        texts.append(" | ".join(parts))
    return texts


def rule_score_text(text: str) -> tuple[float, str]:
    lower = (text or "").lower()
    reasons: list[str] = []
    for reason, patterns in MANUFACTURING_RULE_PATTERNS.items():
        if any(re.search(pattern, lower, flags=re.I) for pattern in patterns):
            reasons.append(reason)

    if not reasons:
        return 0.0, ""

    if any(re.search(pattern, lower, flags=re.I) for pattern in MANUFACTURING_EXCLUSION_PATTERNS):
        return 0.25, "Potential exclusion present; reviewer should confirm: " + "; ".join(dict.fromkeys(reasons))

    # Deterministic probability-like score for prioritization. Multiple definition signals increase confidence.
    probability = min(0.95, 0.50 + (0.15 * len(set(reasons))))
    return probability, "; ".join(dict.fromkeys(reasons))


def score_dataframe(
    df: pd.DataFrame,
    review_threshold: float = MANUFACTURING_REVIEW_THRESHOLD,
) -> pd.DataFrame:
    """Score rows for potential Manufacturing Assessment review.

    This deterministic model implements the Medtronic definition that a potential manufacturing issue is a
    device issue where manufacturing, assembling, packaging, labeling, supplier manufacturing, or prior
    manufacturing-related investigation signals may have contributed to the issue.
    """
    df = clean_df(df)
    texts = build_manufacturing_text(df)
    scored_rules = [rule_score_text(text) for text in texts]
    probabilities = np.array([score for score, _reason in scored_rules], dtype=float)
    reasons = [reason for _score, reason in scored_rules]
    complaint_exclusion = (df["Decision"].str.lower() == "not a complaint") | (df["Complaint? – PE"].str.upper().isin(["N", "NO", "FALSE"]))

    review_flag = ((probabilities >= review_threshold) & (~complaint_exclusion.to_numpy())).astype(int)
    recommendation = np.where(
        review_flag == 1,
        "Route to Manufacturing Assessment review",
        "No Manufacturing Assessment flag",
    )
    final_reasons = []
    for flag, excluded, reason in zip(review_flag, complaint_exclusion, reasons):
        if excluded:
            final_reasons.append("Exclusion: not a complaint unless manufacturing reviewer identifies a device issue")
        elif flag:
            final_reasons.append(reason or "Potential manufacturing issue definition signal")
        else:
            final_reasons.append("No sterile packaging, foreign material, missing/damaged/detached component, labeling, OOB failure, supplier, assembly, or prior manufacturing-related signal")

    scored = df.copy()
    historical = scored["Type - PE PLI Task"].str.lower().str.contains("manufacturing assessment", na=False).astype(int)
    scored["Historical Manufacturing Assessment in Original Data"] = historical
    scored["Manufacturing Assessment Probability"] = probabilities
    scored["Manufacturing Assessment Review Flag"] = review_flag
    scored["Manufacturing Assessment Recommendation"] = recommendation
    scored["Manufacturing Assessment Reason"] = final_reasons
    return scored


def summarize_scored(scored: pd.DataFrame) -> dict:
    total = int(len(scored))
    flags = pd.to_numeric(scored.get("Manufacturing Assessment Review Flag", pd.Series([0] * total)), errors="coerce").fillna(0).astype(int)
    return {
        "total_rows": total,
        "review_rows": int(flags.sum()),
        "no_flag_rows": int(total - flags.sum()),
    }


def to_excel_bytes(scored: pd.DataFrame) -> bytes:
    output = BytesIO()
    summary = summarize_scored(scored)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        scored.to_excel(writer, index=False, sheet_name="Scored")
        scored[scored["Manufacturing Assessment Review Flag"] == 1][[c for c in KEY_EXPORT_COLUMNS if c in scored.columns]].to_excel(
            writer, index=False, sheet_name="Reviewer_Queue"
        )
        pd.DataFrame([
            {"Metric": "Total rows", "Value": summary["total_rows"]},
            {"Metric": "Manufacturing Assessment review rows", "Value": summary["review_rows"]},
            {"Metric": "No Manufacturing Assessment flag", "Value": summary["no_flag_rows"]},
        ]).to_excel(writer, index=False, sheet_name="Dashboard")
    return output.getvalue()
