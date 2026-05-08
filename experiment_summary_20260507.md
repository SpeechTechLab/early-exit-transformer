# Experiment summary (ASR + OneVoice baseline)

## 1) LibriSpeech ASR (100h, 50 epochs) — Zipformer 2-layer early-exit

- **Setup**: trained `early_zipformer_2layer_exits` on **LibriSpeech `train-clean-100` (~100h)** for **50 epochs** and evaluated on **`test-clean` (2620 utts)** and **`test-other` (2939 utts)**.
- **Condition A (with glottal)**: appended frame-level **QCP-style glottal features** to the acoustic features (time-aligned, concatenated as extra channels).
- **Condition B (no glottal)**: same model/training recipe, **acoustic features only**.

### Results (WER %, lower is better)

**With glottal**

- **test-clean**: Exit6 **16.31**
- **test-other**: Exit6 **43.49**

**Without glottal**

- **test-clean**: Exit6 **15.80**
- **test-other**: Exit6 **42.37**

### Interpretation

- **Glottal features did not improve WER** in this run; performance was **slightly worse** with glottal on both `test-clean` (**+0.51 abs WER**) and `test-other` (**+1.12 abs WER**) at the final exit.
- This is still a useful negative result: it suggests either (a) the current glottal representation/normalization/alignment is not adding information beyond the baseline acoustics, or (b) the model/training recipe is not yet exploiting that information.

---

## 2) OneVoice-MSD 2026-style baseline — classical ML on QCP features

- **Goal**: establish a reproducible baseline using **glottal/QCP-derived features** (and related subsets) for the German and Czech datasets, aligned with typical “handcrafted feature + classical ML” pipelines.
- **Protocol**: `classify_tasks.py` with **SGKF** (**StratifiedGroupKFold**, **10 splits**, speaker grouping), reporting **sample-level** and **speaker-level** metrics.
- **Inputs**:
  - `features_Czech_QCP_python_DDK.csv` (**200 rows kept**, 109 feature columns after dropping all-NaN cols)
  - `features_German_QCP_python_DDK.csv` (**238 rows kept**, 109 feature columns after dropping all-NaN cols)
  - `features_Vowels_QCP_python.csv` (**1500 rows kept**, 63 feature columns after dropping all-NaN cols)

### Key results (speaker-level balanced accuracy)

**Czech (DDK)**

- **RandomForest, `glottal_plus_direct` (63 feats)**: **0.630 ± 0.185**
- **XGBoost, `glottal_plus_direct` (63 feats)**: **0.590 ± 0.176**

**German (DDK)**

- **RandomForest, `glottal_plus_direct` (63 feats)**: **0.680 ± 0.097**
- **XGBoost, `glottal_plus_direct` (63 feats)**: **0.640 ± 0.087**

**Colombian (vowels)**

- **RandomForest, `glottal_plus_direct` (63 feats)**: **0.720 ± 0.125**
- **XGBoost, `glottal_plus_direct` (63 feats)**: **0.730 ± 0.142**

### Interpretation

- These baselines are **directionally sensible**: German shows **higher and more stable** performance than Czech in this run, and adding “direct” acoustic features to glottal tends to be among the best subsets.
- The **large standard deviations** (especially Czech) indicate **high variance / small-data sensitivity**, so single numbers should be treated as preliminary.

---

## Next steps / potential improvements

- **ASR (glottal vs no-glottal)**
  - Verify glottal helps *at all* in controlled settings: ablate **feature scaling**, **dropout**, **glottal dimensionality**, and **where fusion happens** (early concat vs later fusion).
  - Check whether glottal extraction quality differs across splits (e.g., voicing, noise) and whether standardization stats are appropriate.

- **OneVoice classification**
  - Prefer **SGKF** (as above) for stable reporting; if using **LOSO**, compute **global aggregated metrics across folds** (to avoid undefined AUC/PR-AUC when a fold has a single class).
  - Add simple robustness improvements: repeated CV with different seeds, model calibration, and basic hyperparameter tuning (nested CV already supports this).
  - Consider stronger representations (if allowed by the challenge rules): SSL embeddings or task-specific fine-tuning, then compare against this handcrafted-feature baseline.

