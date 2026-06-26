import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
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


def make_groups(df: pd.DataFrame, group_column: str = "PE - PLI #") -> np.ndarray:
    """Group rows so the same PLI/Event does not appear in both train and test."""
    preferred = df.get(group_column, pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
    product_event = df.get("Product Event ID", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
    pli = df.get("PE - PLI #", pd.Series(np.arange(len(df)).astype(str))).fillna("").astype(str).str.strip()
    row_fallback = pd.Series([f"row_{i}" for i in range(len(df))])
    groups = np.where(preferred != "", preferred, np.where(pli != "", pli, np.where(product_event != "", product_event, row_fallback)))
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


def make_validation_payload(model, high_threshold: float, watch_threshold: float) -> dict:
    return {
        "model": model,
        "mandatory_decisions": sorted(MANDATORY_DECISIONS),
        "text_columns": TEXT_COLUMNS,
        "high_confidence_threshold": high_threshold,
        "watchlist_threshold": watch_threshold,
        "high_threshold": high_threshold,
        "review_threshold": watch_threshold,
    }


def evaluate_split(
    df: pd.DataFrame,
    X: list[str],
    y_hist: np.ndarray,
    mandatory: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    random_state: int,
    high_threshold: float,
    watch_threshold: float,
) -> dict:
    train_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    # Train the model only on non-mandatory training rows so the mandatory rule remains explicit and auditable.
    model_train_mask = train_mask & (~mandatory)
    X_train = [X[i] for i in np.where(model_train_mask)[0]]
    y_train = y_hist[model_train_mask]

    model = build_model(random_state)
    model.fit(X_train, y_train)

    payload_for_validation = make_validation_payload(model, high_threshold, watch_threshold)
    test_df = df.iloc[test_idx].copy()
    scored_test = score_dataframe(
        test_df,
        payload_for_validation,
        high_confidence_threshold=high_threshold,
        watchlist_threshold=watch_threshold,
    )
    test_summary = summarize_scored(scored_test)

    # Non-mandatory model-only metrics on the held-out rows.
    test_nonmand_mask = test_mask & (~mandatory)
    X_test_nm = [X[i] for i in np.where(test_nonmand_mask)[0]]
    y_test_nm = y_hist[test_nonmand_mask]
    probs_nm = model.predict_proba(X_test_nm)[:, 1] if len(X_test_nm) else np.array([])

    return {
        "model": model,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "model_train_mask": model_train_mask,
        "scored_test": scored_test,
        "test_summary": test_summary,
        "nonmandatory_model_only": {
            "test_rows": int(len(y_test_nm)),
            "test_positive_rows": int(y_test_nm.sum()) if len(y_test_nm) else 0,
            "average_precision": safe_average_precision(y_test_nm, probs_nm) if len(y_test_nm) else None,
            "roc_auc": safe_roc_auc(y_test_nm, probs_nm) if len(y_test_nm) else None,
            "thresholds": {
                str(high_threshold): threshold_metrics(y_test_nm, probs_nm, high_threshold) if len(y_test_nm) else {},
                str(watch_threshold): threshold_metrics(y_test_nm, probs_nm, watch_threshold) if len(y_test_nm) else {},
            },
        },
    }


def _mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def summarize_validation_run(run: dict, groups: np.ndarray) -> dict:
    train_mask = run["train_mask"]
    test_mask = run["test_mask"]
    return {
        "run": run.get("run"),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_groups": int(len(np.unique(groups[train_mask]))),
        "test_groups": int(len(np.unique(groups[test_mask]))),
        "tier_counts": run["test_summary"].get("tier_counts", {}),
        "review_queue_scorecard": run["test_summary"].get("review_queue_scorecard", {}),
        "broad_attention_scorecard": run["test_summary"].get("broad_attention_scorecard", {}),
        "nonmandatory_model_only": run["nonmandatory_model_only"],
    }


def aggregate_validation_runs(runs: list[dict], groups: np.ndarray) -> dict:
    out = {
        "runs": len(runs),
        "run_details": [summarize_validation_run(run, groups) for run in runs],
    }
    for scorecard_name in ["review_queue_scorecard", "broad_attention_scorecard"]:
        metric_values: dict[str, list[float]] = {k: [] for k in ["accuracy", "precision", "recall", "f1"]}
        count_values: dict[str, int] = {k: 0 for k in ["tp", "fp", "fn", "tn"]}
        for run in runs:
            scorecard = run["test_summary"].get(scorecard_name, {})
            for metric in metric_values:
                if metric in scorecard:
                    metric_values[metric].append(float(scorecard[metric]))
            for count in count_values:
                count_values[count] += int(scorecard.get(count, 0))
        out[scorecard_name] = {
            **{metric: _mean_std(values) for metric, values in metric_values.items()},
            "total_counts": count_values,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Retrain and validate the Design Assessment tiered triage model.")
    parser.add_argument("input", help="Historical .xlsx/.csv file")
    parser.add_argument("--output", default="design_assessment_model.joblib", help="Output model .joblib")
    parser.add_argument("--metrics-output", default="model_metrics.json", help="Validation metrics JSON output")
    parser.add_argument("--split-output", default="validation_holdout_scored.xlsx", help="Scored test-set workbook output")
    parser.add_argument("--train-size", type=float, default=0.70, help="Random training share. Default: 0.70")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for repeatable splits. Default: 42")
    parser.add_argument("--group-column", default="PE - PLI #", help="Column used to keep related rows together across train/test splits. Default: PE - PLI #")
    parser.add_argument("--cv-repeats", type=int, default=10, help="Number of repeated grouped 70/30 validations to run for all-inclusive metrics. Default: 10")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of grouped cross-validation folds to run for all-inclusive metrics. Default: 5")
    parser.add_argument("--high-threshold", type=float, default=HIGH_CONFIDENCE_THRESHOLD)
    parser.add_argument("--watchlist-threshold", type=float, default=WATCHLIST_THRESHOLD)
    parser.add_argument("--fit-final-on-all", action="store_true", help="After measuring the 70/30 holdout, refit the saved model on all non-mandatory rows. Leave off when you want the saved model to be the 70%% training model.")
    args = parser.parse_args()

    if not 0.0485 <= args.train_size <= 0.95:
        raise ValueError("--train-size must be between 0.0485 and 0.95")

    df = clean_df(read_source(args.input))
    X = build_text(df)
    mandatory = df["Decision"].isin(MANDATORY_DECISIONS).to_numpy()
    y_hist = (df["Type - PE PLI Task"].str.lower() == "design assessment").astype(int).to_numpy()
    y_corrected = np.where(mandatory, 1, y_hist).astype(int)
    groups = make_groups(df, args.group_column)
    unique_group_count = len(np.unique(groups))
    if unique_group_count < 2:
        raise ValueError(f"--group-column {args.group_column!r} must produce at least two groups")

    splitter = GroupShuffleSplit(n_splits=1, train_size=args.train_size, random_state=args.random_state)
    train_idx, test_idx = next(splitter.split(X, y_corrected, groups=groups))
    holdout = evaluate_split(
        df,
        X,
        y_hist,
        mandatory,
        train_idx,
        test_idx,
        args.random_state,
        args.high_threshold,
        args.watchlist_threshold,
    )
    model = holdout["model"]
    train_mask = holdout["train_mask"]
    test_mask = holdout["test_mask"]
    model_train_mask = holdout["model_train_mask"]
    scored_test = holdout["scored_test"]
    test_summary = holdout["test_summary"]

    split_col = np.where(train_mask, "TRAIN", "TEST")
    split_counts = pd.Series(split_col).value_counts().to_dict()

    metrics = {
        "validation_method": f"Random 70/30 holdout by {args.group_column} group plus repeated and k-fold grouped validation",
        "train_size_requested": float(args.train_size),
        "test_size_requested": float(1 - args.train_size),
        "random_state": int(args.random_state),
        "group_column": args.group_column,
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
        "nonmandatory_model_only": holdout["nonmandatory_model_only"],
    }

    repeated_runs = []
    if args.cv_repeats > 0:
        repeated_splitter = GroupShuffleSplit(n_splits=args.cv_repeats, train_size=args.train_size, random_state=args.random_state)
        for run_number, (cv_train_idx, cv_test_idx) in enumerate(repeated_splitter.split(X, y_corrected, groups=groups), start=1):
            run = evaluate_split(
                df,
                X,
                y_hist,
                mandatory,
                cv_train_idx,
                cv_test_idx,
                args.random_state + run_number,
                args.high_threshold,
                args.watchlist_threshold,
            )
            run["run"] = run_number
            repeated_runs.append(run)
    metrics["repeated_grouped_holdout"] = {
        "method": f"{args.cv_repeats} repeated grouped holdouts by {args.group_column}",
        "train_size_requested": float(args.train_size),
        **aggregate_validation_runs(repeated_runs, groups),
    }

    fold_runs = []
    n_splits = min(args.cv_folds, unique_group_count)
    if n_splits >= 2:
        fold_splitter = GroupKFold(n_splits=n_splits)
        for fold_number, (cv_train_idx, cv_test_idx) in enumerate(fold_splitter.split(X, y_corrected, groups=groups), start=1):
            run = evaluate_split(
                df,
                X,
                y_hist,
                mandatory,
                cv_train_idx,
                cv_test_idx,
                args.random_state + 1000 + fold_number,
                args.high_threshold,
                args.watchlist_threshold,
            )
            run["run"] = fold_number
            fold_runs.append(run)
    metrics["group_k_fold"] = {
        "method": f"{n_splits}-fold grouped cross-validation by {args.group_column}",
        **aggregate_validation_runs(fold_runs, groups),
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
