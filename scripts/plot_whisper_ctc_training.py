#!/usr/bin/env python3
"""Plot Whisper CTC fine-tuning curves from HuggingFace trainer_state.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = (
    REPO_ROOT / "whisper_ctc_runs/bridge2ai_norm_partial_unfreeze/checkpoint-8000/trainer_state.json"
)
DEFAULT_OUT = REPO_ROOT / "classification_results/whisper_ctc_training_curve.pdf"


def load_history(path: Path) -> list[dict]:
    st = json.loads(path.read_text(encoding="utf-8"))
    return st.get("log_history", [])


def split_series(history: list[dict]) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    train_epochs: list[float] = []
    train_losses: list[float] = []
    eval_epochs: list[float] = []
    eval_losses: list[float] = []
    eval_wers: list[float] = []

    for e in history:
        if "epoch" not in e:
            continue
        if "loss" in e and "eval_loss" not in e:
            train_epochs.append(float(e["epoch"]))
            train_losses.append(float(e["loss"]))
        if "eval_loss" in e:
            eval_epochs.append(float(e["epoch"]))
            eval_losses.append(float(e["eval_loss"]))
            eval_wers.append(100.0 * float(e.get("eval_wer", float("nan"))))

    return train_epochs, train_losses, eval_epochs, eval_losses, eval_wers


def plot_curves(
    train_epochs: list[float],
    train_losses: list[float],
    eval_epochs: list[float],
    eval_losses: list[float],
    eval_wers: list[float],
    out_path: Path,
) -> None:
    fig, ax1 = plt.subplots(figsize=(6.2, 3.4))

    ax1.plot(train_epochs, train_losses, color="#2563eb", alpha=0.35, linewidth=1.0, label="Train loss (step)")
    ax1.plot(
        eval_epochs,
        eval_losses,
        color="#2563eb",
        marker="o",
        markersize=5,
        linewidth=1.8,
        label="Dev loss (eval)",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (CTC $-$ 0.05$\\times$entropy)")
    ax1.set_xlim(0, 3.05)
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(
        eval_epochs,
        eval_wers,
        color="#dc2626",
        marker="s",
        markersize=5,
        linewidth=1.8,
        label="Dev WER (%)",
    )
    ax2.set_ylabel("Dev WER (%)")
    ax2.set_ylim(50, 85)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(f"Wrote {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trainer-state",
        type=Path,
        default=DEFAULT_STATE,
        help="Path to checkpoint-*/trainer_state.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="Output figure path (.pdf); .png is written alongside",
    )
    args = parser.parse_args()

    if not args.trainer_state.is_file():
        raise SystemExit(
            f"Missing {args.trainer_state}\n"
            "Copy from cluster:\n"
            "  scp -J ... stek@digis:.../checkpoint-8000/trainer_state.json "
            "whisper_ctc_runs/bridge2ai_norm_partial_unfreeze/checkpoint-8000/"
        )

    history = load_history(args.trainer_state)
    series = split_series(history)
    plot_curves(*series, out_path=args.output)


if __name__ == "__main__":
    main()
