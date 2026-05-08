# Slide 1 - Project Summary

## Goal
- Port glottal feature extraction from MATLAB to Python.
- Train `early_zipformer_2layer_exits` on LibriSpeech 100h with acoustic + glottal features.
- Evaluate on `test-clean` and `test-other` and compare with MFCC-only baseline.

## Current Status
- End-to-end Python pipeline is implemented.
- Training and inference pipelines were updated to ingest per-utterance glottal CSV features.
- Strict normalization path (train stats reused at inference) is implemented.

---

# Slide 2 - MATLAB to Python Port

## What Was Implemented
- Ported QCP/glottal extraction pipeline to Python in `extract_glottal_features_qcp.py`.
- Added LibriSpeech extraction driver in `extract_librispeech_100h_glottal_features.py`.
- Generated glottal CSVs for train and test splits.

## Feature Output
- Per-utterance statistical glottal descriptors are produced.
- CSV keying uses LibriSpeech utterance IDs (`speaker-chapter-utt`).

## Note
- MFCC columns from the glottal extractor were intentionally disabled in earlier steps to avoid duplication inside that extractor.

---

# Slide 3 - Training/Inference Integration

## Data Loader and Config Updates
- `util/data_loader.py`:
  - Per-utterance glottal map loading.
  - Duplicate-row aggregation.
  - Standardization support.
  - Strict mode support with saved/loaded train normalization stats.
  - LibriSpeech ID normalization fix to match `speaker-chapter-utt`.
  - Inference collate bug fix so batches are returned correctly.
- `util/conf.py`:
  - Added `--glottal_standardize` / `--no_glottal_standardize`.
  - Added `--glottal_norm_stats_in` / `--glottal_norm_stats_out`.

## Result
- Zipformer input supports `n_mels + glottal_dim` consistently in train and inference.

---

# Slide 4 - Training Behavior (100h)

## Observed Loss Curve
- Total loss decreased from ~92.7 to ~1.86 over 353 epochs.
- Stable convergence with diminishing improvements after ~250 epochs.

![Loss Curve](presentation_assets/loss_total.png)

---

# Slide 5 - Inference Results and Comparison

## Best Reported WER (Glottal + MFCC Input Path)
- Inference model selection: **checkpoint averaging over the last 100 epochs** (`253` to `352`).
- `test-clean`: Exit 6 = **15.89%**
- `test-other`: Exit 6 = **41.28%**

## By Exit
![WER by Exit](presentation_assets/wer_by_exit_glottal_mfcc.png)

## Comparison vs MFCC-only Baseline
- Baseline values available in current notes were reported as matching these best values.
- No measurable gain observed yet from glottal augmentation in this setup.

![Exit 6 Comparison](presentation_assets/wer_comparison_exit6.png)

---

# Slide 6 - Next Steps to Improve WER

## Priority Actions
- Run a clean A/B experiment with fixed seed and identical checkpoint policy:
  - MFCC-only.
  - MFCC + glottal.
- Revisit glottal feature design:
  - Frame-level alignment instead of repeating utterance-level vectors across all frames.
  - Feature selection and ablation by groups (NAQ/QOQ/HRF/H1H2/etc.).
- Improve optimization:
  - Lower LR in late phase and checkpoint averaging windows.
  - Tune batch size and regularization for `test-other` robustness.
- Validate domain coverage:
  - Ensure consistent extraction quality and coverage across train/test splits.

## Expected Outcome
- Determine whether glottal features provide true incremental value or need different fusion strategy.

![Early-Exit Relative Gain](presentation_assets/exit_gain_relative.png)
