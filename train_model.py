import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

from design_assessment_engine import (
    clean_df,
    build_text,
    score_dataframe,
    summarize_scored,
    MANDATORY_DECISIONS,
    HIGH_CONFIDENCE_THRESHOLD,
    WATCHLIST_THRESHOLD,
    TEXT_COLUMNS,
)


def read_source(path):
    if str(path).lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, header=0)
        if "Product Event ID" not in df.columns and "Decision" not in df.columns:
            df = pd.read_excel(path, header=3)
    return df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")


def make_groups(df: pd.DataFrame) -> np.ndarray:
    """Group rows so the same Product Event does not appear in both train and test."""
    product_event = df.get("Product Event ID", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
    pli = df.get("PE - PLI #", pd.Series(np.arange(len(df)).astype(str))).fillna("").astype(str).str.strip()
    row_fallback = pd.Series([f"row_{i}" for i in range(len(df))])
    groups = np.where(product_event != "", product_event, np.where(pli != "", pli, row_fallback))
    return groups


def threshold_metrics(y_true, probs, threshold):
    pred = probs >= threshold
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "flag_rate": float(pred.mean()) if len(pred) else 0.0,
    }


def safe_average_precision(y_true, probs):
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, probs))


def safe_roc_auc(y_true, probs):
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, probs))


def build_model(random_state: int):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.95,
            max_features=15000,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("clf", SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            class_weight="balanced",
            max_iter=1000,
            tol=1e-3,
            random_state=random_state,
        )),
    ])


def main():
    parser = argparse.ArgumentParser(description="Retrain and validate the Design Assessment tiered triage model with a random 70/30 holdout split.")
    parser.add_argument("input", help="Historical .xlsx/.csv file")
    parser.add_argument("--output", default="design_assessment_model.joblib", help="Output model .joblib")
    parser.add_argument("--metrics-output", default="model_metrics.json", help="Validation metrics JSON output")
    parser.add_argument("--split-output", default="validation_holdout_scored.xlsx", help="Scored test-set workbook output")
    parser.add_argument("--train-size", type=float, default=0.70, help="Random training share. Default: 0.70")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for repeatable splits. Default: 42")
    parser.add_argument("--high-threshold", type=float, default=HIGH_CONFIDENCE_THRESHOLD)
    parser.add_argument("--watchlist-threshold", type=float, default=WATCHLIST_THRESHOLD)
    parser.add_argument("--fit-final-on-all", action="store_true", help="After measuring the 70/30 holdout, refit the saved model on all non-mandatory rows. Leave off when you want the saved model to be the 70% training model.")
    args = parser.parse_args()

    if not 0.0485 <= args.train_size <= 0.95:
        raise ValueError("--train-size must be between 0.0485 and 0.95")

    df = clean_df(read_source(args.input))
    X = build_text(df)
    mandatory = df["Decision"].isin(MANDATORY_DECISIONS).to_numpy()
    y_hist = (df["Type - PE PLI Task"].str.lower() == "design assessment").astype(int).to_numpy()
    y_corrected = np.where(mandatory, 1, y_hist).astype(int)
    groups = make_groups(df)

    splitter = GroupShuffleSplit(n_splits=1, train_size=args.train_size, random_state=args.random_state)
    train_idx, test_idx = next(splitter.split(X, y_corrected, groups=groups))

    train_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    # Train the model only on non-mandatory training rows so the mandatory rule remains explicit and auditable.
    model_train_mask = train_mask & (~mandatory)
    X_train = [X[i] for i in np.where(model_train_mask)[0]]
    y_train = y_hist[model_train_mask]

    model = build_model(args.random_state)
    model.fit(X_train, y_train)

    payload_for_validation = {
        "model": model,
        "mandatory_decisions": sorted(MANDATORY_DECISIONS),
        "text_columns": TEXT_COLUMNS,
        "high_confidence_threshold": args.high_threshold,
        "watchlist_threshold": args.watchlist_threshold,
        "high_threshold": args.high_threshold,
        "review_threshold": args.watchlist_threshold,
    }

    test_df = df.iloc[test_idx].copy()
    scored_test = score_dataframe(
        test_df,
        payload_for_validation,
        high_confidence_threshold=args.high_threshold,
        watchlist_threshold=args.watchlist_threshold,
    )
    test_summary = summarize_scored(scored_test)

    # Non-mandatory model-only metrics on the held-out rows.
    test_nonmand_mask = test_mask & (~mandatory)
    X_test_nm = [X[i] for i in np.where(test_nonmand_mask)[0]]
    y_test_nm = y_hist[test_nonmand_mask]
    probs_nm = model.predict_proba(X_test_nm)[:, 1] if len(X_test_nm) else np.array([])

    split_col = np.where(train_mask, "TRAIN", "TEST")
    split_counts = pd.Series(split_col).value_counts().to_dict()

    metrics = {
        "validation_method": "Random 70/30 holdout by Product Event ID group",
        "train_size_requested": float(args.train_size),
        "test_size_requested": float(1 - args.train_size),
        "random_state": int(args.random_state),
        "fit_final_on_all_nonmandatory_rows": bool(args.fit_final_on_all),
        "total_rows": int(len(df)),
        "train_rows": int(split_counts.get("TRAIN", 0)),
        "test_rows": int(split_counts.get("TEST", 0)),
        "train_groups": int(len(np.unique(groups[train_mask]))),
        "test_groups": int(len(np.unique(groups[test_mask]))),
        "historical_da_rows": int(y_hist.sum()),
        "mandatory_decision_rows": int(mandatory.sum()),
        "overturned_rows": int((mandatory & (y_hist == 0)).sum()),
        "corrected_da_positive_rows": int(y_corrected.sum()),
        "model_training_rows_nonmandatory": int(model_train_mask.sum()),
        "model_training_positive_rows_nonmandatory": int(y_hist[model_train_mask].sum()),
        "test_positive_rows_corrected": int(y_corrected[test_mask].sum()),
        "test_mandatory_rows": int(mandatory[test_mask].sum()),
        "heldout_review_queue_scorecard": test_summary.get("review_queue_scorecard", {}),
        "heldout_broad_attention_scorecard": test_summary.get("broad_attention_scorecard", {}),
        "heldout_tier_counts": test_summary.get("tier_counts", {}),
        "nonmandatory_model_only": {
            "test_rows": int(len(y_test_nm)),
            "test_positive_rows": int(y_test_nm.sum()) if len(y_test_nm) else 0,
            "average_precision": safe_average_precision(y_test_nm, probs_nm) if len(y_test_nm) else None,
            "roc_auc": safe_roc_auc(y_test_nm, probs_nm) if len(y_test_nm) else None,
            "thresholds": {
                str(args.high_threshold): threshold_metrics(y_test_nm, probs_nm, args.high_threshold) if len(y_test_nm) else {},
                str(args.watchlist_threshold): threshold_metrics(y_test_nm, probs_nm, args.watchlist_threshold) if len(y_test_nm) else {},
            },
        },
    }

    if args.split_output:
        scored_test.to_excel(args.split_output, index=False)

    # By default, save the actual 70% training model. Optionally, validate with 70/30 then refit for production.
    saved_model = model
    if args.fit_final_on_all:
        saved_model = build_model(args.random_state)
        all_nonmand = ~mandatory
        saved_model.fit([x for x, m in zip(X, all_nonmand) if m], y_hist[all_nonmand])

    payload = {
        "model": saved_model,
        "mandatory_decisions": sorted(MANDATORY_DECISIONS),
        "text_columns": TEXT_COLUMNS,
        "high_confidence_threshold": args.high_threshold,
        "watchlist_threshold": args.watchlist_threshold,
        "high_threshold": args.high_threshold,
        "review_threshold": args.watchlist_threshold,
        "metrics": metrics,
    }
    joblib.dump(payload, args.output)

    Path(args.metrics_output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
