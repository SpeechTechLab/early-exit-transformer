import argparse
import numpy as np
import pandas as pd


META_COLS = {"file_name", "speaker", "label", "task", "chapter", "utterance"}


def main():
    parser = argparse.ArgumentParser(description="Compare two QCP feature CSV files")
    parser.add_argument("--ref", required=True, help="Reference CSV (e.g., MATLAB output)")
    parser.add_argument("--pred", required=True, help="Predicted CSV (Python output)")
    parser.add_argument("--out", default="qcp_compare_report.csv", help="Per-column comparison report")
    args = parser.parse_args()

    ref = pd.read_csv(args.ref)
    pred = pd.read_csv(args.pred)

    if "file_name" not in ref.columns or "file_name" not in pred.columns:
        raise ValueError("Both CSVs must contain file_name column")

    m = ref.merge(pred, on="file_name", suffixes=("_ref", "_pred"))
    if m.empty:
        raise ValueError("No overlapping file_name values between CSVs")

    common_cols = [
        c for c in ref.columns
        if c in pred.columns and c not in META_COLS
    ]

    rows = []
    for c in common_cols:
        a = pd.to_numeric(m[f"{c}_ref"], errors="coerce").to_numpy(dtype=np.float64)
        b = pd.to_numeric(m[f"{c}_pred"], errors="coerce").to_numpy(dtype=np.float64)

        mask = np.isfinite(a) & np.isfinite(b)
        if np.sum(mask) == 0:
            rows.append({"column": c, "n": 0, "mae": np.nan, "rmse": np.nan, "corr": np.nan})
            continue

        d = a[mask] - b[mask]
        mae = float(np.mean(np.abs(d)))
        rmse = float(np.sqrt(np.mean(d ** 2)))
        if np.sum(mask) > 1 and np.std(a[mask]) > 0 and np.std(b[mask]) > 0:
            corr = float(np.corrcoef(a[mask], b[mask])[0, 1])
        else:
            corr = np.nan

        rows.append({"column": c, "n": int(np.sum(mask)), "mae": mae, "rmse": rmse, "corr": corr})

    report = pd.DataFrame(rows).sort_values(["rmse", "mae"], ascending=[True, True])
    report.to_csv(args.out, index=False)

    print(f"Compared {len(common_cols)} columns on {len(m)} overlapping files")
    print(f"Saved report to {args.out}")
    print("Overall means:")
    print(report[["mae", "rmse", "corr"]].mean(numeric_only=True))


if __name__ == "__main__":
    main()
