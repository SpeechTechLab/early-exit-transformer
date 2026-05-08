# Baselines + my current results (as of 2026-04-27)

## What I ran

- **Feature tables**: `features_German_QCP_python.csv`, `features_Czech_QCP_python.csv`
- **Classifier script**: `classify_tasks.py`
- **Evaluation**:
  - Nested CV with **speaker-disjoint folds** (`StratifiedGroupKFold`)
  - Outer CV: **10 folds**; Inner CV: **5 folds** (hyperparameter search)
  - Model selection score: **ROC-AUC**
  - Metrics reported **sample-level** and **speaker-level**
  - Speaker-level = mean probability per speaker, threshold at 0.5
- **Models**: Random Forest (RF), XGBoost (XGB)
- **Feature subsets** (script-defined):
  - `glottal_only` (28)
  - `glottal_plus_direct` (63)
  - `glottal_plus_direct_plus_mfcc` (63 in these runs)
  - `glottal_plus_mfcc` (28)

Outputs are under: `classification_results/run_20260423_141721/` (global summaries + per-task folders).

## Training run (ASR) + CTC loss curve

### Training command (CTC, early-zipformer, + frame-level glottal features)

```bash
python3 -u train.py \
  --decoder_mode ctc \
  --model_type early_zipformer_2layer_exits \
  --train_split 100h \
  --n_enc_layers_per_exit 2 \
  --n_enc_exits 6 \
  --n_epochs 50 \
  --batch_size 8 \
  --n_workers 0 \
  --save_model_dir trained_model_zip2layer_100h_glottal_framelevel_50ep \
  --append_glottal_features \
  --glottal_features_path /stek/patsoura/early-exit-transformer/data_release/glottal_features_100h_QCP_python_tuned_framelevel.csv \
  --n_glottal_features 10 \
  --glottal_drop_mfcc \
  --glottal_standardize \
  --glottal_norm_stats_out trained_model_zip2layer_100h_glottal_framelevel_50ep/glottal_norm_stats_train100h.npz \
  2>&1 | tee train_framelevel_50ep.log
```

### Loss extraction (TensorBoard → CSV) and curve generation

The training loss ("Total loss") is logged to TensorBoard under `runs/` and was extracted across resumed sessions:

```bash
python3 extract_tensorboard_losses.py --logdir runs/Apr21_11-27-31_caa7ff5a7579 >/dev/null && tail -n +2 loss_total.csv > /tmp/l0.csv && \
python3 extract_tensorboard_losses.py --logdir runs/Apr21_17-19-23_caa7ff5a7579 >/dev/null && tail -n +2 loss_total.csv > /tmp/l1.csv && \
python3 extract_tensorboard_losses.py --logdir runs/Apr24_10-22-37_9f0543a59905 >/dev/null && tail -n +2 loss_total.csv > /tmp/l2.csv && \
{ echo "epoch,loss"; cat /tmp/l0.csv /tmp/l1.csv /tmp/l2.csv | awk -F, '$1>=0 && $1<=49 {print $1","$2}' | sort -t, -k1,1n | awk -F, '!seen[$1]++'; } > loss_total_0_49.csv
```

```bash
python3 - <<'PY'
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("loss_total_0_49.csv")
plt.figure(figsize=(7,4))
plt.plot(df["epoch"], df["loss"], marker="o", linewidth=1.5)
plt.title("CTC training loss (Total loss) vs epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_total_0_49.png", dpi=200)
print("Wrote loss_total_0_49.png")
PY
```

### What the loss shows

- Epoch 0: **~95.47**
- Epoch 49: **~5.19**
- Absolute decrease: **~90.28** (≈ **94.6%** decrease)

## Dataset/task composition (important caveat)

My current reported numbers are **pooled across tasks** because `classify_tasks.py` does **not** split by `task`.

### German (`features_German_QCP_python.csv`, 765 rows)

- Tasks:
  - `German_vowel`: 351
  - `German_ddk`: 238
  - `German_read`: 176
- Speakers per task: **176** in each task

### Czech (`features_Czech_QCP_python.csv`, 599 rows)

- Tasks:
  - `Czech_ddk`: 200
  - `Czech_vowel`: 200
  - `Czech_read`: 199
- Speakers per task: **100** in each task

➡️ For the cleanest comparison to paper baselines (which are usually **task-specific**), classification should be rerun **per task** (e.g., only `*_read` or only `*_ddk`).

## My results (pooled across vowel + DDK + read)

Source: `classification_results/run_20260423_141721/global_summary_metrics.csv`

### German (n=765)

Best-performing subset overall: **glottal_plus_direct (63 features)**

- **RF** (speaker-level): acc **0.6699 ± 0.0932**, bal-acc **0.6848 ± 0.0753**, F1 **0.7128 ± 0.0652**, AUC **0.6840 ± 0.1290**
- **XGB** (speaker-level): acc **0.6582 ± 0.1421**, bal-acc **0.6682 ± 0.1259**, F1 **0.7090 ± 0.1105**, AUC **0.6689 ± 0.1427**

### Czech (n=599)

Best speaker-accuracy subset: **glottal_plus_direct (63 features)** (RF)

- **RF** (speaker-level): acc **0.7200 ± 0.1470**, bal-acc **0.7200 ± 0.1470**, F1 **0.7077 ± 0.1871**, AUC **0.7760 ± 0.1977**

Best speaker-AUC variant: **glottal_only (28 features)** (XGB)

- **XGB** (speaker-level): acc **0.7100 ± 0.0831**, bal-acc **0.7100 ± 0.0831**, F1 **0.7271 ± 0.0745**, AUC **0.7840 ± 0.1764**

## Paper baselines (same datasets) and comparison

### Bocklet et al., 2011 (Czech early-PD corpus; multiple speech tasks)

- Setup: **SVM**, **leave-one-speaker-out**, **task-specific** results (T1–T7).
- Reported task-dependent accuracies (examples):
  - Prosody: up to **~90.5%** (reading) and AUC up to **0.97** (monologue)
  - MFCC/GMM: typically **~80–88%**
  - Glottal two-mass model: **~50–79%**, best on some reading tasks

**Comparison to my pooled Czech results**:

- My pooled speaker accuracy (**~0.72**) is **lower** than their best task-specific numbers (~0.85–0.91), but this is **not directly comparable** because:
  - their results are **per task**, while mine are **pooled across vowel+DDK+read**
  - different feature sets/models (SVM vs RF/XGB) and different preprocessing
  - small cohort in 2011 paper → higher variance/optimism risk

### Pérez‑Toro et al., 2021 (PC‑GITA, Colombian Spanish, PD vs HC)

- Baseline: openSMILE ComParE + SVM, nested LOSO
- Baseline performance (PD vs HC): **F1 = 0.69**, **UAR = 70%**

**Comparison to my pooled results**:

- My best pooled speaker-level F1 is **~0.71** (German RF glottal+direct) and **~0.73** (Czech XGB glottal-only).
- Numerically similar/slightly higher, but this is **cross-dataset and cross-language**, so it should be stated as “same ballpark”, not a direct SOTA claim.

### Rios‑Urrego et al., 2024 (Czech; PD vs ET and PD/ET/HC)

- Primary task is **PD vs ET** (and tri-class PD/ET/HC), not PD vs HC.
- Headline accuracies (PD vs ET): up to **86.2%** (/pa-ta-ka) and **81.4%** (monologue), using fusion of speech dimensions.

**Comparison to my results**:

- Not directly comparable because I’m doing **PD vs HC** (binary) and currently **pooled across tasks**.

## Next step for a fairer baseline comparison

Rerun `classify_tasks.py` **separately per task**, e.g.:

- German: `German_read` (to compare with reading baselines), `German_ddk` (to compare with /pa-ta-ka), `German_vowel`
- Czech: `Czech_read`, `Czech_ddk`, `Czech_vowel`

## Future improvements (to better match paper baselines)

- Run **LOSO (leave-one-speaker-out)** CV in addition to 10-fold group CV.
- Enable and report an **SVM (RBF)** baseline (common in prior work), alongside RF/XGB.
- Report **task-specific** speaker-level metrics (accuracy/AUC; and balanced accuracy/UAR where needed).
- Add feature families closer to prior work (explicit **prosody** and **articulation** descriptors), beyond current glottal+direct features.

