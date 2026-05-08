#!/usr/bin/env python3
"""Generate concise plots for the glottal+MFCC presentation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    out_dir = Path("presentation_assets")
    out_dir.mkdir(parents=True, exist_ok=True)

    exits = np.array([1, 2, 3, 4, 5, 6])
    wer_clean = np.array([37.72, 20.07, 17.25, 17.37, 17.28, 15.89])
    wer_other = np.array([65.30, 47.82, 43.93, 43.71, 43.58, 41.28])

    # Plot 1: WER by exit (best reported run with glottal+MFCC)
    plt.figure(figsize=(8, 5))
    plt.plot(exits, wer_clean, marker="o", label="test-clean")
    plt.plot(exits, wer_other, marker="o", label="test-other")
    plt.xticks(exits)
    plt.xlabel("Exit index")
    plt.ylabel("WER (%)")
    plt.title("Zipformer 2-layer exits: WER by exit")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "wer_by_exit_glottal_mfcc.png", dpi=160)
    plt.close()

    # Plot 2: Exit-6 comparison with MFCC-only baseline (reported as equal)
    labels = ["test-clean", "test-other"]
    glottal_mfcc = np.array([15.89, 41.28])
    mfcc_only = np.array([15.89, 41.28])
    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(7, 4.5))
    plt.bar(x - width / 2, mfcc_only, width, label="MFCC-only baseline (reported)")
    plt.bar(x + width / 2, glottal_mfcc, width, label="MFCC+glottal (best)")
    for i, v in enumerate(mfcc_only):
        plt.text(i - width / 2, v + 0.5, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(glottal_mfcc):
        plt.text(i + width / 2, v + 0.5, f"{v:.2f}", ha="center", fontsize=9)
    plt.xticks(x, labels)
    plt.ylabel("WER (%)")
    plt.title("Exit-6 comparison")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "wer_comparison_exit6.png", dpi=160)
    plt.close()

    # Plot 3: Relative improvement from exit 1 to exit 6
    rel_clean = (wer_clean[0] - wer_clean[-1]) / wer_clean[0] * 100.0
    rel_other = (wer_other[0] - wer_other[-1]) / wer_other[0] * 100.0

    plt.figure(figsize=(6.5, 4.2))
    vals = [rel_clean, rel_other]
    plt.bar(["test-clean", "test-other"], vals)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.7, f"{v:.1f}%", ha="center", fontsize=10)
    plt.ylabel("Relative WER reduction (%)")
    plt.title("Early-exit gain (Exit 1 to Exit 6)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "exit_gain_relative.png", dpi=160)
    plt.close()

    print(f"Saved plots in {out_dir}")


if __name__ == "__main__":
    main()
