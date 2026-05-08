# Zipformer 2-Layer Exits (100h) - Training and Inference Notes

## Scope
This note summarizes the commands and pipeline for:
- Zipformer 2-layer exits on LibriSpeech train-clean-100 (100h)
- Training with and without glottal features
- Inference for both setups
- How glottal features are extracted and fused in the Python pipeline

## Command Notes

### 1) Train baseline 2(no glottal), 100h, target 350 epochs
```bash
python3 train.py \
  --decoder_mode ctc \
  --model_type early_zipformer_2layer_exits \
  --train_split 100h \
  --n_enc_layers_per_exit 2 \
  --n_enc_exits 6 \
  --n_epochs 350 \
  --batch_size 8 \
  --n_workers 0 \
  --save_model_dir trained_model_zip2layer_100h_noglottal
```

### 2) Train with glottal features (strict normalization from train stats)
```bash
python3 train.py \
  --decoder_mode ctc \
  --model_type early_zipformer_2layer_exits \
  --train_split 100h \
  --n_enc_layers_per_exit 2 \
  --n_enc_exits 6 \
  --n_epochs 350 \
  --batch_size 8 \
  --n_workers 0 \
  --save_model_dir trained_model_zip2layer_100h_glottal \
  --append_glottal_features \
  --glottal_features_path glottal_features_100h_QCP_python_tuned.csv \
  --glottal_drop_mfcc \
  --glottal_standardize \
  --glottal_norm_stats_out trained_model_zip2layer_100h_glottal/glottal_norm_stats_train100h.npz
```

### 3) Inference baseline (no glottal)
```bash
python3 inference.py \
  --decoder_mode ctc \
  --model_type early_zipformer_2layer_exits \
  --n_enc_layers_per_exit 2 \
  --n_enc_exits 6 \
  --load_model_path trained_model_zip2layer_100h_noglottal/mod349-transformer \
  --batch_size 8 \
  --n_workers 0 \
  --results_file classification_results/zipformer2_100h_noglottal_ep350_wer.json
```

### 4) Inference with glottal (strict: reuse train normalization stats)
```bash
python3 inference.py \
  --decoder_mode ctc \
  --model_type early_zipformer_2layer_exits \
  --n_enc_layers_per_exit 2 \
  --n_enc_exits 6 \
  --load_model_path trained_model_zip2layer_100h_glottal/mod349-transformer \
  --append_glottal_features \
  --glottal_features_path glottal_features_train_test_QCP_python_tuned.csv \
  --glottal_drop_mfcc \
  --glottal_standardize \
  --glottal_norm_stats_in trained_model_zip2layer_100h_glottal/glottal_norm_stats_train100h.npz \
  --batch_size 8 \
  --n_workers 0 \
  --results_file classification_results/zipformer2_100h_glottal_ep350_wer.json
```

### 5) Find checkpoints if path is uncertain
```bash
find . -type f -name "mod349-transformer"
```

### 6) Extract LibriSpeech 100h glottal CSV (Python QCP tuned)
```bash
python3 extract_librispeech_100h_glottal_features.py \
  --librispeech_root LibriSpeech/train-clean-100 \
  --output_csv glottal_features_100h_QCP_python_tuned.csv \
  --num_workers 4 \
  --save_every 100 \
  --resume
```

## Important Training Detail
- In this codebase, checkpoint names are zero-indexed by training step in the save loop.
- `mod349-transformer` corresponds to the 350th epoch if training started from epoch 0 and reached step 349.
- Early stopping is enabled by default. If it triggers early, you may not get `mod349-transformer`.

## How the Glottal Pipeline Works (Code-Based Overview)

### Step A: Per-utterance glottal extraction
From `extract_glottal_features_qcp.py` (`extract_file_qcp`):
- Loads waveform, applies polarity fix and high-pass filtering.
- Frames signal (50 ms, 10 ms hop), estimates F0, runs QCP-based glottal analysis.
- Computes glottal descriptors per frame (NAQ, QOQ, HRF, H1H2, etc.).
- Keeps voiced frames, then summarizes each feature by statistics (mean/std/skew/kurtosis/quantiles).
- Writes one row per utterance.

Code excerpt:
```python
frames, _ = create_fixed_frames(x, frame_length, frame_shift)
pitch_info = estimate_pitch(x, fs)

for i, frame in enumerate(frames):
    g_flow, _, residual = qcp(frame, fs, options)
    metrics = compute_glottal_metrics(g_flow, fs, global_f0)
    NAQ_all[i] = metrics["NAQ"]
    QOQ_all[i] = metrics["QOQ"]
    HRF_all[i] = metrics["HRF"]
    H1H2_all[i] = metrics["H1H2"]
```

### Step B: MFCC block status in glottal extractor
MFCC computation is currently disabled in your extractor (commented out intentionally).

Code excerpt:
```python
# MFCC extraction disabled.
# mfcc_voiced = compute_matlab_like_mfcc(...)
# row[f"mfcc_{c}_mean"] = ...
# row[f"mfcc_{c}_std"] = ...
```

This means your current Python glottal CSV can be glottal-only features unless another CSV source already includes `mfcc_*` columns.

### Step C: Loader reads CSV, aggregates by utterance, standardizes
From `util/data_loader.py` (`load_glottal_feature_map`):
- Selects numeric feature columns.
- Optionally drops any `mfcc_*` columns when `--glottal_drop_mfcc` is used.
- Aggregates duplicate rows per utterance by averaging.
- Applies z-score standardization.
- Saves train stats (`--glottal_norm_stats_out`) or loads them (`--glottal_norm_stats_in`) for strict eval.

Code excerpt:
```python
cols = _select_numeric_feature_columns(reader.fieldnames, drop_mfcc)
...
per_utt_values.setdefault(uid, []).append(np.asarray(values, dtype=np.float32))
...
per_utt_matrix.append(np.mean(stacked, axis=0))
...
feature_matrix = _apply_standardization(feature_matrix, mean, std)
```

### Step D: Feature fusion in collate (mel + repeated glottal vector)
From `util/data_loader.py` (`CollatePaddingFn` and `CollateInferFn`):
- Acoustic features are built from waveform: spectrogram -> mel transform.
- A per-utterance glottal vector is fetched by utterance id.
- That vector is repeated across all time frames.
- Concatenation is channel-wise: `[mel; glottal_repeated]`.

Code excerpt:
```python
spec = spec_transform(waveform, self.args)
spec = melspec_transform(spec, self.args)

g = self.glottal_feat_map.get(uid)
g_rep = g.unsqueeze(1).repeat(1, spec.size(1))
spec = torch.cat([spec, g_rep], dim=0)
```

This is not frame-wise MFCC concatenation from the same acoustic frontend. It is mel features concatenated with an utterance-level glottal vector broadcast over time.

### Step E: Model input dimension update
From `train.py` and `inference.py`:
- Without glottal: `input_features_length = n_mels`
- With append mode: `input_features_length = n_mels + glottal_dim`

Code excerpt:
```python
input_features_length = args.n_mels
...
input_features_length = args.n_mels + glottal_dim
```

## New Architecture
- Base encoder-decoder setup is unchanged: Zipformer with early exits (`early_zipformer_2layer_exits`, 6 exits, 2 layers per exit).
- The architectural change is at the input feature level.
- Baseline path: mel-only acoustic frontend.
- New path: mel frontend plus appended glottal descriptors (utterance-level vector repeated across time).
- Feature hygiene is controlled with `--glottal_drop_mfcc` to avoid duplicating MFCC-like information if present in CSV.
- Strict evaluation is supported by train-only normalization stats reused at inference (`--glottal_norm_stats_out` in train, `--glottal_norm_stats_in` in inference).


