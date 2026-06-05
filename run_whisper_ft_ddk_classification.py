#!/usr/bin/env python3
"""Extract Bridge2AI-finetuned Whisper embeddings on DDK data and run classification.

Uses the ASR finetune checkpoint (default: ``checkpoint-150``, ~14%% dev / 8.4%% test WER
on ``clean_short_en``). The encoder weights are reused for PD/HC classification on
German, Czech, and Colombian DDK — same protocol as base Whisper.

Example (Mac, after copying checkpoint into repo):

  export HF_HOME="$(pwd)/.hf_cache"
  .venv/bin/python run_whisper_ft_ddk_classification.py \\
    --checkpoint whisper_runs/bridge2ai_read_clean_short_en_ft_v2/checkpoint-150

Copy checkpoint from cluster (once):

  rsync -avP USER@CLUSTER:/stek/patsoura/early-exit-transformer/whisper_runs/bridge2ai_read_clean_short_en_ft_v2/checkpoint-150 \\
    whisper_runs/bridge2ai_read_clean_short_en_ft_v2/
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from extract_whisper_embeddings import DEFAULT_FT_CHECKPOINT, DEFAULT_MODEL_ID

_REPO = Path(__file__).resolve().parent


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=_REPO, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Whisper finetuned DDK embedding + classification pipeline")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_FT_CHECKPOINT,
        help=f"Local finetuned Whisper checkpoint directory (default: {DEFAULT_FT_CHECKPOINT})",
    )
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="Base model id for feature extractor fallback")
    parser.add_argument("--device", default="", help="Torch device (default: cuda if available)")
    parser.add_argument("--skip_extract", action="store_true", help="Skip embedding extraction (CSVs exist)")
    parser.add_argument("--skip_merge", action="store_true", help="Skip glottal merge")
    parser.add_argument("--skip_classify", action="store_true", help="Skip classify_tasks.py")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted extraction")
    parser.add_argument("--max_files", type=int, default=0, help="Limit wavs per dataset (debug)")
    args = parser.parse_args()

    ckpt = (_REPO / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    if not ckpt.is_dir():
        print(
            f"ERROR: Finetuned checkpoint not found: {ckpt}\n\n"
            "Copy from the cluster ASR run, e.g.:\n"
            "  mkdir -p whisper_runs/bridge2ai_read_clean_short_en_ft_v2\n"
            "  rsync -avP USER@HOST:/stek/patsoura/early-exit-transformer/"
            "whisper_runs/bridge2ai_read_clean_short_en_ft_v2/checkpoint-150 \\\n"
            "    whisper_runs/bridge2ai_read_clean_short_en_ft_v2/\n",
            file=sys.stderr,
        )
        return 1

    py = sys.executable
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(_REPO / ".hf_cache"))
    env.setdefault("HF_HUB_CACHE", str(_REPO / ".hf_cache" / "hub"))
    env.setdefault("MPLCONFIGDIR", str(_REPO / ".mpl_cache"))

    common = [
        "--local_model_dir",
        str(ckpt),
        "--model_id",
        args.model_id,
        "--save_every",
        "25",
    ]
    if args.device:
        common.extend(["--device", args.device])
    if args.resume:
        common.append("--resume")
    if args.max_files:
        common.extend(["--max_files", str(args.max_files)])

    if not args.skip_extract:
        _run(
            [py, "extract_whisper_german_czech.py", "--dataset", "both", "--ddk_only", "--output_tag", "ft", *common],
            env=env,
        )
        _run(
            [py, "extract_whisper_colombian_ddk.py", "--output_tag", "ft", *common],
            env=env,
        )

    expected = [
        "features_German_whisper_ft_DDK.csv",
        "features_Czech_whisper_ft_DDK.csv",
        "features_Colombian_whisper_ft_DDK.csv",
    ]
    missing = [f for f in expected if not (_REPO / f).exists()]
    if missing:
        print(f"ERROR: Missing feature CSVs after extraction: {missing}", file=sys.stderr)
        return 1

    if not args.skip_merge:
        _run([py, "merge_hubert_glottal_csvs.py", "--whisper_ft_defaults"], env=env)

    if not args.skip_classify:
        csvs = expected + [
            "features_German_whisper_ft_glottal_DDK.csv",
            "features_Czech_whisper_ft_glottal_DDK.csv",
            "features_Colombian_whisper_ft_glottal_DDK.csv",
        ]
        _run(
            [
                py,
                "classify_tasks.py",
                "--csv",
                *csvs,
                "--paper_min_sens",
                "0.9",
                "--models",
                "rf,xgb",
                "--paper_feature_subset",
                "whisper_plus_glottal_plus_direct",
            ],
            env=env,
        )

    print("\nDone. Update classification_results/ddk_classifier_f1_tables.md with the new run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
