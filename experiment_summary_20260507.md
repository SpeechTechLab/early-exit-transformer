# Experiment summary (ASR + OneVoice baseline)

## 1) LibriSpeech ASR (100h, 50 epochs) — Zipformer 2-layer early-exit

- **Setup**: trained `early_zipformer_2layer_exits` on **LibriSpeech `train-clean-100` (~100h)** for **50 epochs** and evaluated on **`test-clean` (2620 utts)** and **`test-other` (2939 utts)**.
- **Condition A (with glottal)**: appended frame-level **QCP-style glottal features** to the acoustic features (time-aligned, concatenated as extra channels).
- **Condition B (no glottal)**: same model/training recipe, **acoustic features only**.

### Results (WER %, lower is better)

**With glottal**

- **test-clean**: Exit6 **16.24**
- **test-other**: Exit6 **43.37**

#### With glottal (detailed per-exit WER + I/D/S)

**test-clean** (2620 utts)

- Exit1: WER **44.86**; **I/D/S = 2514 / 3006 / 18067**
- Exit2: WER **21.30**; **I/D/S = 1138 / 1232 / 8830**
- Exit3: WER **18.28**; **I/D/S = 1037 / 1125 / 7447**
- Exit4: WER **18.34**; **I/D/S = 1047 / 1136 / 7461**
- Exit5: WER **18.02**; **I/D/S = 1024 / 1122 / 7330**
- Exit6: WER **16.24**; **I/D/S = 911 / 961 / 6667**

**test-other** (2939 utts)

- Exit1: WER **68.75**; **I/D/S = 3008 / 5066 / 27910**
- Exit2: WER **49.62**; **I/D/S = 2367 / 3187 / 20420**
- Exit3: WER **45.82**; **I/D/S = 2203 / 3006 / 18773**
- Exit4: WER **45.76**; **I/D/S = 2193 / 3049 / 18711**
- Exit5: WER **45.52**; **I/D/S = 2177 / 3009 / 18643**
- Exit6: WER **43.37**; **I/D/S = 2073 / 2839 / 17788**

**Without glottal** (mels-only checkpoint `trained_model_zip2layer_100h_mels_only_50ep/mod049-transformer`)

- **test-clean**: Exit6 **15.88**
- **test-other**: Exit6 **42.44**

#### Without glottal (detailed per-exit WER + I/D/S)

**test-clean** (2620 utts)

- Exit1: WER **45.65**; **I/D/S = 2924 / 2752 / 18325**
- Exit2: WER **20.67**; **I/D/S = 1222 / 1088 / 8555**
- Exit3: WER **18.03**; **I/D/S = 1117 / 962 / 7401**
- Exit4: WER **18.03**; **I/D/S = 1125 / 958 / 7395**
- Exit5: WER **17.43**; **I/D/S = 1139 / 896 / 7129**
- Exit6: WER **15.88**; **I/D/S = 1001 / 785 / 6561**

**test-other** (2939 utts)

- Exit1: WER **69.22**; **I/D/S = 3488 / 4540 / 28204**
- Exit2: WER **48.94**; **I/D/S = 2642 / 2819 / 20156**
- Exit3: WER **45.52**; **I/D/S = 2497 / 2571 / 18760**
- Exit4: WER **45.52**; **I/D/S = 2482 / 2568 / 18775**
- Exit5: WER **44.95**; **I/D/S = 2544 / 2506 / 18480**
- Exit6: WER **42.44**; **I/D/S = 2373 / 2344 / 17499**

### Interpretation

- **Glottal features did not improve WER** in this matched comparison: at Exit6, **with glottal** was **slightly worse** than **mels-only** on `test-clean` (**+0.36 abs WER**, 16.24 vs 15.88) and on `test-other` (**+0.93 abs WER**, 43.37 vs 42.44).
- **I/D/S at Exit6** is consistent with that story: on **test-clean**, glottal trades **fewer insertions** (911 vs 1001) for **more deletions** (961 vs 785) and **slightly more substitutions** (6667 vs 6561). On **test-other**, glottal again has **fewer insertions** (2073 vs 2373) but **more deletions** (2839 vs 2344) and **more substitutions** (17788 vs 17499), so the net WER still moves against glottal.
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

