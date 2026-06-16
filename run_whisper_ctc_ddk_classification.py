#!/usr/bin/env python3
"""Extract CTC-finetuned Whisper embeddings on DDK data and run classification.

Uses the SLAM-LLM partial-unfreeze CTC checkpoint (Bridge2AI NORM fine-tune).
Requires ``openai-whisper`` and the CTC ``.pth`` on disk (cluster path by default).

Example (cluster):

  export CUDA_VISIBLE_DEVICES=0
  python3 run_whisper_ctc_ddk_classification.py \\
    --checkpoint whisper_ctc_runs/bridge2ai_norm_partial_unfreeze/whisper_ctc_partial_unfreeze.pth
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from extract_whisper_embeddings import DEFAULT_CTC_CHECKPOINT

_REPO = Path(__file__).resolve().parent


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=_REPO, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Whisper CTC-finetuned DDK embedding + classification pipeline"
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CTC_CHECKPOINT,
        help=f"SLAM-LLM Whisper CTC .pth (default: {DEFAULT_CTC_CHECKPOINT})",
    )
    parser.add_argument("--ctc_encoder_name", default="large-v3-turbo")
    parser.add_argument("--device", default="", help="Torch device (default: cuda if available)")
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument("--skip_classify", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument(
        "--models",
        default="rf,xgb",
        help="Comma-separated classifiers for classify_tasks.py (default: rf,xgb)",
    )
    parser.add_argument(
        "--run_tag",
        default="whisper_ctc_ddk",
        help="Subfolder name under classification_results/ (via classify --run_name if supported)",
    )
    args = parser.parse_args()

    (_REPO / "logs").mkdir(parents=True, exist_ok=True)
    (_REPO / ".mpl_cache").mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = (_REPO / ckpt).resolve()
    if not ckpt.is_file():
        print(
            f"ERROR: CTC checkpoint not found: {ckpt}\n\n"
            "Train on cluster first, or copy the .pth locally:\n"
            "  whisper_ctc_runs/bridge2ai_norm_partial_unfreeze/whisper_ctc_partial_unfreeze.pth\n",
            file=sys.stderr,
        )
        return 1

    py = sys.executable
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(_REPO / ".mpl_cache"))

    common = [
        "--ctc_checkpoint",
        str(ckpt),
        "--ctc_encoder_name",
        args.ctc_encoder_name,
        "--output_tag",
        "ctc",
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
            [py, "extract_whisper_german_czech.py", "--dataset", "both", "--ddk_only", *common],
            env=env,
        )
        _run([py, "extract_whisper_colombian_ddk.py", *common], env=env)

    expected = [
        "features_German_whisper_ctc_DDK.csv",
        "features_Czech_whisper_ctc_DDK.csv",
        "features_Colombian_whisper_ctc_DDK.csv",
    ]
    missing = [f for f in expected if not (_REPO / f).exists()]
    if missing:
        print(f"ERROR: Missing feature CSVs after extraction: {missing}", file=sys.stderr)
        return 1

    if not args.skip_merge:
        _run([py, "merge_hubert_glottal_csvs.py", "--whisper_ctc_defaults"], env=env)

    if not args.skip_classify:
        csvs = expected + [
            "features_German_whisper_ctc_glottal_DDK.csv",
            "features_Czech_whisper_ctc_glottal_DDK.csv",
            "features_Colombian_whisper_ctc_glottal_DDK.csv",
        ]
        classify_cmd = [
            py,
            "classify_tasks.py",
            "--csv",
            *csvs,
            "--paper_min_sens",
            "0.9",
            "--models",
            str(args.models),
            "--paper_feature_subset",
            "whisper_plus_glottal_plus_direct",
            "--only_feature_subsets",
            "whisper_all,whisper_plus_glottal_plus_direct",
        ]
        device = (args.device or "").strip().lower()
        if not device:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        if device.startswith("cuda") and "xgb" in str(args.models).lower():
            classify_cmd.extend(["--xgb_device", "cuda"])
            env.setdefault("CLASSIFY_N_JOBS", "4")
        _run(classify_cmd, env=env)

        results_root = _REPO / "classification_results"
        runs = sorted(results_root.glob("run_*"), key=lambda p: p.stat().st_mtime)
        if runs:
            latest = runs[-1]
            link = results_root / args.run_tag
            if link.is_symlink() or link.is_dir():
                if link.is_symlink():
                    link.unlink()
                elif link.is_dir() and not any(link.iterdir()):
                    link.rmdir()
            if not link.exists():
                link.symlink_to(latest.name, target_is_directory=True)
            print(f"Linked {link} -> {latest.name}")

    print(
        "\nDone. Regenerate tables:\n"
        "  python3 build_ddk_f1_tables.py --write-latex --write-md --dual-level\n"
        f"  (point W-CTC column at latest run under classification_results/)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
