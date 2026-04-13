#!/usr/bin/env python3
"""Export a TensorBoard scalar tag (e.g., loss) to CSV.

Example:
    python3 extract_tensorboard_losses.py \
        --logdir runs/Mar11_22-13-03_gre-lap-mac787 \
        --tag "Total loss" \
        --output loss_total.csv
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError as exc:
    print(
        "ERROR: tensorboard is not installed. Install it with: pip install tensorboard",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


def find_event_files(logdir: Path):
    return sorted(logdir.rglob("events.out.tfevents.*"))


def collect_scalars(event_files, tag):
    rows = []
    tags_seen = set()

    for ev_path in event_files:
        accumulator = event_accumulator.EventAccumulator(
            str(ev_path), size_guidance={"scalars": 0}
        )
        try:
            accumulator.Reload()
        except Exception as exc:  # pragma: no cover
            print(f"WARNING: failed to read {ev_path}: {exc}", file=sys.stderr)
            continue

        scalar_tags = accumulator.Tags().get("scalars", [])
        tags_seen.update(scalar_tags)

        if tag not in scalar_tags:
            continue

        for scalar in accumulator.Scalars(tag):
            rows.append(
                {
                    "step": int(scalar.step),
                    "value": float(scalar.value),
                    "wall_time": float(scalar.wall_time),
                    "event_file": str(ev_path),
                }
            )

    return rows, sorted(tags_seen)


def deduplicate_by_step(rows):
    last_by_step = {}
    for row in rows:
        step = row["step"]
        prev = last_by_step.get(step)
        if prev is None or row["wall_time"] >= prev["wall_time"]:
            last_by_step[step] = row

    return [last_by_step[s] for s in sorted(last_by_step)]


def write_rows(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["step", "value", "wall_time", "event_file"]
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export TensorBoard scalar data to CSV"
    )
    parser.add_argument("--logdir", required=True, help="TensorBoard run directory")
    parser.add_argument(
        "--tag",
        default="Total loss",
        help="Scalar tag to export (default: 'Total loss')",
    )
    parser.add_argument(
        "--output",
        default="loss_total.csv",
        help="Output CSV path (default: loss_total.csv)",
    )
    parser.add_argument(
        "--list_tags",
        action="store_true",
        help="List available scalar tags and exit",
    )
    parser.add_argument(
        "--keep_duplicates",
        action="store_true",
        help="Keep duplicate steps from multiple event files instead of deduplicating by latest wall_time",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logdir = Path(args.logdir)
    if not logdir.exists():
        raise SystemExit(f"ERROR: logdir not found: {logdir}")

    event_files = find_event_files(logdir)
    if not event_files:
        raise SystemExit(f"ERROR: no TensorBoard event files found under: {logdir}")

    rows, tags = collect_scalars(event_files, args.tag)

    if args.list_tags:
        if tags:
            print("Available scalar tags:")
            for tag in tags:
                print(f"- {tag}")
        else:
            print("No scalar tags found in the provided event files.")
        return

    if not rows:
        print(f"ERROR: no scalar values found for tag: {args.tag}", file=sys.stderr)
        if tags:
            print("Available scalar tags:", file=sys.stderr)
            for tag in tags:
                print(f"- {tag}", file=sys.stderr)
        raise SystemExit(1)

    if not args.keep_duplicates:
        rows = deduplicate_by_step(rows)
    else:
        rows = sorted(rows, key=lambda r: (r["step"], r["wall_time"]))

    out_path = Path(args.output)
    write_rows(rows, out_path)
    print(f"Wrote {len(rows)} points to {out_path}")


if __name__ == "__main__":
    main()
