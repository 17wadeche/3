import argparse
from pathlib import Path
import manufacturing_assessment_engine as manufacturing_engine
def main():
    parser = argparse.ArgumentParser(description="Score a file for Manufacturing Assessment triage.")
    parser.add_argument("input", help="Input .xlsx/.xls/.csv file")
    parser.add_argument("--output", default="manufacturing_assessment_scored.xlsx", help="Output .xlsx or .csv")
    parser.add_argument("--review-threshold", type=float, default=manufacturing_engine.MANUFACTURING_REVIEW_THRESHOLD)
    args = parser.parse_args()
    df = manufacturing_engine.read_input_file(args.input)
    scored = manufacturing_engine.score_dataframe(df, review_threshold=args.review_threshold)
    out = Path(args.output)
    if out.suffix.lower() == ".csv":
        scored.to_csv(out, index=False)
    else:
        out.write_bytes(manufacturing_engine.to_excel_bytes(scored))
    print(f"Scored {len(scored):,} rows -> {out}")
if __name__ == "__main__":
    main()
