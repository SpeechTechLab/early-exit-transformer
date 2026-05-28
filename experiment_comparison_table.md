# Experiment comparison (best per language)

Speaker-level **F1 / Spec / Sens** (mean±std over 10-fold outer CV). Threshold tuned for **sensitivity ≥ 0.90** ([arXiv:2603.22225v2](https://arxiv.org/abs/2603.22225)), same protocol as Hernandez et al. Table 3.

**Paper:** monolingual **HuBERT-Large**, DDK only (CZ, DE, ES). No published vowel baseline.

**Ours:** HuBERT-base-ls960 embeddings; QCP glottal + direct features; RF/XGB nested CV.

| Source runs | Feature setup |
|-------------|----------------|
| QCP glottal | `run_20260515_134932` (DDK), `run_20260515_125311` (vowels) |
| HuBERT | `run_20260520_122815` (DDK), `run_20260520_134052` (vowels) |
| HuBERT + glottal | `run_hubert_glottal_combined` |

---

## DDK (diadochokinetic)

| Lang | Paper (HuBERT-L) Spec / Sens / **F1** | QCP glottal Spec / Sens / **F1** | HuBERT Spec / Sens / **F1** | HuBERT+glottal Spec / Sens / **F1** | Best ours (ΔF1 vs paper) |
|------|--------------------------------------|----------------------------------|----------------------------|-------------------------------------|--------------------------|
| **CZ** | 0.53±0.12 / 0.89±0.07 / **0.76±0.05** | 0.18±0.19 / 0.88±0.18 / **0.65±0.11** | 0.54±0.16 / 0.94±0.09 / **0.79±0.09** | 0.54±0.16 / 0.94±0.09 / **0.79±0.09** | **0.79±0.09** hubert (RF) **+0.03** |
| **DE** | 0.47±0.06 / 0.91±0.03 / **0.75±0.02** | 0.44±0.13 / 0.91±0.10 / **0.73±0.08** | 0.54±0.20 / 0.90±0.08 / **0.76±0.09** | 0.58±0.16 / 0.91±0.10 / **0.78±0.09** | **0.78±0.09** hub+glot (XGB) **+0.03** |
| **ES** | 0.41±0.09 / 0.88±0.06 / **0.71±0.04** | 0.52±0.24 / 0.88±0.20 / **0.74±0.14** | 0.59±0.16 / 0.86±0.16 / **0.75±0.11** | 0.67±0.19 / 0.86±0.16 / **0.78±0.11** | **0.78±0.11** hub+glot (XGB) **+0.07** |

**DDK notes:** Czech does not gain from adding glottal (HuBERT alone is best). German and Colombian DDK improve with concatenated features (ES specificity +0.08 vs HuBERT-only). Paper uses HuBERT-**Large**; we use HuBERT-**base** — not directly comparable on architecture, same task/protocol.

---

## Vowels (no paper baseline)

| Lang | QCP glottal Spec / Sens / **F1** | HuBERT Spec / Sens / **F1** | HuBERT+glottal Spec / Sens / **F1** | Best ours |
|------|----------------------------------|----------------------------|-------------------------------------|-----------|
| **CZ** | 0.38±0.24 / 0.92±0.13 / **0.73±0.09** | 0.24±0.12 / 0.88±0.13 / **0.66±0.08** | 0.38±0.24 / 0.92±0.13 / **0.73±0.09** | **0.73±0.09** glottal (XGB) |
| **DE** | 0.19±0.13 / 0.91±0.07 / **0.67±0.06** | 0.16±0.15 / 0.93±0.08 / **0.67±0.05** | 0.16±0.15 / 0.93±0.08 / **0.67±0.05** | **0.67±0.06** glottal (XGB) |
| **CO** | 0.54±0.13 / 0.90±0.13 / **0.76±0.09** | 0.42±0.17 / 0.88±0.16 / **0.71±0.09** | 0.70±0.16 / 0.92±0.13 / **0.83±0.11** | **0.83±0.11** hub+glot (RF) |

**Vowels notes:** Largest gain from HuBERT+glottal on Colombian vowels (**0.83** vs **0.76** glottal-only, **0.71** HuBERT-only). Czech vowels: glottal ≈ combined; HuBERT alone weakest on specificity.

---

## Summary (best F1 only)

| Task | Lang | Paper F1 | Best ours | Features | Δ vs paper |
|------|------|----------|-----------|----------|------------|
| DDK | CZ | 0.76±0.05 | **0.79±0.09** | HuBERT | +0.03 |
| DDK | DE | 0.75±0.02 | **0.78±0.09** | HuBERT+glottal | +0.03 |
| DDK | ES | 0.71±0.04 | **0.78±0.11** | HuBERT+glottal | +0.07 |
| Vowels | CZ | — | **0.73±0.09** | QCP glottal | — |
| Vowels | DE | — | **0.67±0.06** | QCP glottal | — |
| Vowels | CO | — | **0.83±0.11** | HuBERT+glottal | — |

---

## Caveats

- **Model size:** Paper = HuBERT-Large; ours = `facebook/hubert-base-ls960`.
- **Metrics:** Paper-style speaker F1 with sens ≥ 0.9 threshold tuning (inner CV).
- **Glottal:** QCP Python pipeline (`glottal_plus_direct`, 63 dims on vowel tables).
- Detail per run: `classification_results/*/best_results_by_language.md`.
