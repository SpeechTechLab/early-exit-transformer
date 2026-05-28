# Bridge2AI Zipformer ASR Pilot — 2-Week Summary

---

## Slide 1 — Goal, data & pipeline

### Objective
Adapt **early-exit Zipformer** (mel-only, CTC, LibriSpeech BPE) to **Bridge2AI** adult + pediatric speech for ASR pilot experiments.

### Data (local)
| Item | Amount |
|------|--------|
| Labeled adult + pediatric (all tasks) | **~163 h** |
| Read-speech subset (Harvard, passages) | **~24 h** |
| Speaker-disjoint splits | 15% test / 10% dev / rest train |

### Pipeline
1. **Build** — `build_bridge2ai_zipformer_subset.py` → mel CSVs (80-dim) + `train` / `dev` / `test` manifests  
2. **Train** — `train.py`, `early_zipformer_2layer_exits`, `--use_precomputed_features`, no glottal  
3. **Eval** — `inference.py`, WER (jiwer), early-exit layers 1–6 (report **exit 6**)

### Engineering fixes (during pilot)
| Fix | Purpose |
|-----|---------|
| BPE label uppercase (`normalize_label_for_bpe`) | Avoid `<unk>` / collapse on lowercase refs |
| `--min_mel_frames 100` | Skip very short clips (BatchNorm crash) |
| Collate: label before features | Skip empty-BPE utterances without crashing |
| Task filters + `new_*` manifests | Drop breath/DDK; normalize refs (lowercase, no punct) |

### Libri baseline (cluster)
| Checkpoint | Role |
|------------|------|
| `trained_model_zip2layer_100h_mels_only_50ep/mod049-transformer` | Libri 100 h mel-only pretrain (~16% WER on Libri test-clean) |

---

## Slide 2 — Experiments & results

**Model:** `early_zipformer_2layer_exits` · **Features:** 80 mel · **Decoder:** CTC · **Metric:** WER % (exit 6)

| # | Experiment | Train data | Init | Dev manifest | Dev WER | Test WER | Notes |
|---|------------|------------|------|--------------|---------|----------|--------|
| 1 | **Pilot v1 / v2 / v3** | ~1.5 h (1 h adult + 0.5 h ped), read-speech mix | Scratch | v3 `dev.txt` (136 utt) | **~97%** | — | Wrong task mix in v1; BPE fix in v2; v3 read-speech only |
| 2 | **Full corpus scratch** | **70 h** train (all tasks, 31k utt) | Scratch | `new_dev.txt` (2.4k utt) | **96.7%** | — | Collapsed hyps (`a`, `the`); train OK, ASR failed |
| 3 | **Filtered mixed + Libri finetune** | **30 h** `new_train.txt` (19k utt, language/cognitive + read + picture…) | **Libri mod049** | `new_dev.txt` | **81.2%** | — | Clear gain vs scratch; still poor on mixed tasks |
| 4 | **Read-speech + Libri finetune** ✓ | **6 h** `read_train.txt` (2.9k utt) | **Libri mod049** | `read_dev.txt` (396 utt) | **59.4%** | **57.4%** | Best pilot; test hyps e.g. caterpillar passage partly correct |

### Result trend (exit 6 WER, dev)
```
Scratch mixed (~70h train)     ████████████████████████████████  97%
Libri → mixed finetune         ██████████████████████████        81%
Libri → read-speech finetune   ███████████████████               59%
Libri 100h (reference)         ███                               ~16%  (Libri test-clean)
```

### Takeaways
- **Pipeline is valid** — splits, mels, training, and inference all work end-to-end.  
- **Libri init is required** — Bridge2AI-only scratch does not learn usable ASR at pilot scale.  
- **Task match matters** — read-speech finetune (**59% dev / 57% test**) vs mixed (**81% dev**).  
- **Next levers** — more read-speech hours, GPU training, adult-only Harvard slice, lower LR / more epochs.

### Artifacts (repo)
| Path | Content |
|------|---------|
| `bridge2ai_zipformer_full_all_tasks/` | Full build, mels, manifests |
| `trained_model_scratch/` | Scratch full-corpus run (`mod029`) |
| `trained_model_libri_init/` | Libri→mixed finetune (`mod015`) |
| `trained_model_libri_readspeech/` | Libri→read-speech finetune (`mod016`) |
| `wer_*_libri_readspeech.json` | Best dev/test metrics |
