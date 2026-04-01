import argparse
import csv
import glob
import os
from collections import OrderedDict


def collect_train_clean_100_utt_ids(root):
    utt_ids = set()
    pattern = os.path.join(root, "*", "*", "*.trans.txt")
    for trans_path in glob.glob(pattern):
        with open(trans_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    utt_ids.add(parts[0])
    return utt_ids


def merge_csvs(csv_paths, output_path):
    merged_rows = OrderedDict()
    header = None

    for csv_path in csv_paths:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue

            if header is None:
                header = reader.fieldnames
            elif reader.fieldnames != header:
                raise ValueError(
                    f"Header mismatch in {csv_path}.\nExpected: {header}\nGot: {reader.fieldnames}"
                )

            if "file_name" not in reader.fieldnames:
                raise ValueError(f"Missing 'file_name' column in {csv_path}")

            for row in reader:
                utt_id = row["file_name"].strip()
                if not utt_id:
                    continue
                merged_rows[utt_id] = row

    if header is None:
        raise ValueError("No valid input CSVs found")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in merged_rows.values():
            writer.writerow(row)

    return len(merged_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-speaker glottal feature CSVs into one CSV for LibriSpeech 100h"
    )
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=["QCP", "IAIF", "TRLP"],
        help="Feature extraction algorithm to merge",
    )
    parser.add_argument(
        "--input_glob",
        default=None,
        help="Optional glob for input CSVs. Defaults to features_*_<algorithm>.csv",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output merged CSV path. Defaults to glottal_features_100h_<algorithm>.csv",
    )
    parser.add_argument(
        "--librispeech_root",
        default="LibriSpeech/train-clean-100",
        help="Path to LibriSpeech train-clean-100 root for coverage check",
    )
    args = parser.parse_args()

    input_glob = args.input_glob or f"features_*_{args.algorithm}.csv"
    output_path = args.output or f"glottal_features_100h_{args.algorithm}.csv"

    csv_paths = sorted(
        path for path in glob.glob(input_glob)
        if os.path.isfile(path) and os.path.basename(path) != os.path.basename(output_path)
    )

    if not csv_paths:
        raise SystemExit(f"No CSVs found matching: {input_glob}")

    print("Input CSVs:")
    for path in csv_paths:
        print(f"  {path}")

    merged_count = merge_csvs(csv_paths, output_path)
    print(f"\nWrote merged CSV: {output_path}")
    print(f"Merged utterances: {merged_count}")

    if os.path.isdir(args.librispeech_root):
        target_utt_ids = collect_train_clean_100_utt_ids(args.librispeech_root)
        if target_utt_ids:
            coverage = merged_count / len(target_utt_ids) * 100
            print(f"Target utterances in 100h split: {len(target_utt_ids)}")
            print(f"Coverage: {merged_count}/{len(target_utt_ids)} ({coverage:.2f}%)")

            missing = sorted(target_utt_ids - set())
            # Re-open merged file once to avoid holding full csv row map twice.
            merged_ids = set()
            with open(output_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    merged_ids.add(row["file_name"].strip())
            missing = sorted(target_utt_ids - merged_ids)
            if missing:
                print("Example missing utterances:")
                for utt_id in missing[:10]:
                    print(f"  {utt_id}")
        else:
            print("Could not compute coverage: no utterances found under LibriSpeech root")
    else:
        print(f"Skipped coverage check; LibriSpeech root not found: {args.librispeech_root}")


if __name__ == "__main__":
    main()
