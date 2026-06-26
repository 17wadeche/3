import argparse
from pathlib import Path
import pandas as pd
from design_assessment_engine import read_input_file, score_dataframe, load_model, to_excel_bytes


def main():
    parser = argparse.ArgumentParser(description="Score a file for Design Assessment tiered triage.")
    parser.add_argument("input", help="Input .xlsx/.xls/.csv file")
    parser.add_argument("--model", default="design_assessment_model.joblib", help="Path to trained model joblib")
    parser.add_argument("--output", default="design_assessment_tiered_scored.xlsx", help="Output .xlsx or .csv")
    parser.add_argument("--high-threshold", type=float, default=None, help="High-confidence model threshold, default from model")
    parser.add_argument("--watchlist-threshold", type=float, default=None, help="Watchlist model threshold, default from model")
    args = parser.parse_args()

    payload = load_model(args.model)
    df = read_input_file(args.input)
    scored = score_dataframe(df, payload, high_confidence_threshold=args.high_threshold, watchlist_threshold=args.watchlist_threshold)

    out = Path(args.output)
    if out.suffix.lower() == ".csv":
        scored.to_csv(out, index=False)
    else:
        out.write_bytes(to_excel_bytes(scored))
    print(f"Scored {len(scored):,} rows -> {out}")


if __name__ == "__main__":
    main()
