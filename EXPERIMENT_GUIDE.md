# Experiment Guide

Complete guide for running baseline vs olfactory NER experiments — locally, on Google Colab, and with tuning variants.

---

## Table of Contents

1. [Experimental Setup](#experimental-setup)
2. [Running Locally](#running-locally)
3. [Running on Google Colab](#running-on-google-colab)
4. [Tuning Experiments](#tuning-experiments)
5. [Analyzing Results](#analyzing-results)
6. [Expected Results](#expected-results)
7. [Troubleshooting](#troubleshooting)

---

## Experimental Setup

### Hypothesis

> Does the biologically inspired olfactory feature extractor add value beyond a standard language model for NER?

### The Two Models

**Baseline (mBERT + CRF)**:
```
mBERT (frozen) → Dropout → Linear → CRF → NER Tags
```

**Olfactory (mBERT + Olfactory + CRF)**:
```
mBERT (frozen) → Receptors → Glomeruli → Linear → CRF → NER Tags
```

Both use **frozen mBERT** and differ **only** in the olfactory layers, ensuring a fair comparison.

### Design Rationale

| Decision | Why |
|----------|-----|
| **Frozen mBERT for both** | Isolates the olfactory contribution — any improvement is from the added layers, not BERT fine-tuning |
| **No BiLSTM** | Keeps comparison clean — we test olfactory feature transformation, not BiLSTM |
| **CRF for both** | Prevents attributing structured decoding gains to olfactory layers |

**Claim**: "Structured, sparse, convergent representations (Olfactory) provide better features for the CRF than raw BERT embeddings" — NOT "we beat transformers."

### Datasets

| Dataset | Language | Type |
|---------|----------|------|
| CoNLL-2003 | English | High resource |
| WikiANN | Hindi, Marathi, Tamil, Bangla, Telugu | Low resource |

---

## Running Locally

### Quick Test (~5-10 min)

Verify everything works with 1 epoch on CoNLL-2003:

```bash
python run_baseline_vs_olfactory.py --quick_test
```

### Full Experiment Suite (~2-4 hours on GPU)

Run on all 6 datasets:

```bash
python run_baseline_vs_olfactory.py --epochs 5
```

### Custom Experiments

```bash
# Specific datasets
python run_baseline_vs_olfactory.py --datasets conll2003,hindi --epochs 5

# Single dataset, more epochs
python run_baseline_vs_olfactory.py --datasets conll2003 --epochs 10

# Adjust hyperparameters
python run_baseline_vs_olfactory.py --epochs 5 --batch_size 8 --lr 5e-5
```

### Manual Training (individual models)

```bash
# Baseline
python src/train_bert.py --dataset conll2003 --experiment baseline --epochs 5 --batch_size 16 --lr 2e-5

# Olfactory
python src/train_bert.py --dataset conll2003 --experiment olfactory --epochs 5 --batch_size 16 --lr 2e-5
```

### GloVe-based Experiments

```bash
# Baseline BiLSTM-CRF
python src/train.py --config config/experiments.yaml --experiment baseline

# Olfactory model
python src/train.py --config config/experiments.yaml --experiment olfactory_full

# Force CPU
python src/train.py --config config/experiments.yaml --experiment baseline --device cpu
```

---

## Running on Google Colab

### Setup (5 min)

1. Go to [Google Colab](https://colab.research.google.com/)
2. **File → Upload notebook** → upload `baseline_vs_olfactory_experiments.ipynb`
3. **Runtime → Change runtime type → GPU** (T4 or better)
4. In Cell 2, update the repo URL:
   ```python
   !git clone https://github.com/bhushan1729/olfaction-inspired-ner.git
   ```

### Running

**Run All** (recommended): **Runtime → Run all** — runs all 12 experiments (~6-8 hours)

**Run Selected**: Run setup cells (1-5) first, then only the experiments you want:

| Cells | Dataset |
|-------|---------|
| 6-7   | CoNLL-2003 (English) |
| 8-9   | Hindi |
| 10-11 | Marathi |
| 12-13 | Tamil |
| 14-15 | Bangla |
| 16-17 | Telugu |
| 18-24 | Analysis & visualization |

### Configuration (Cell 5)

```python
EPOCHS = 5          # Number of training epochs
BATCH_SIZE = 16     # Batch size (reduce to 8 or 4 if OOM)
LEARNING_RATE = 2e-5
```

### Google Drive Storage

Results auto-save to:
```
/content/drive/MyDrive/olfaction_ner_experiments/
├── results/           # Per-dataset results
├── comparison_analysis/   # Charts, reports
└── quick_results_summary.csv
```

### Tips for Long Runs

- **Session disconnects**: Reconnect, re-run Drive mount (Cell 1), skip completed experiments
- **Check completed experiments**:
  ```python
  import os
  for root, dirs, files in os.walk("/content/drive/MyDrive/olfaction_ner_experiments/results"):
      if "results.json" in files:
          print(f"✓ {root}")
  ```

### Runtime Estimates

| Setup | Per Experiment | All 12 |
|-------|---------------|--------|
| GPU (T4) | ~30-40 min | ~6-8 hours |
| CPU | ~2-3 hours | Not recommended |

---

## Tuning Experiments

For detailed parameter tuning theory, see [docs/PARAMETER_TUNING_GUIDE.md](docs/PARAMETER_TUNING_GUIDE.md).

### Quick Commands

```bash
# GELU activation (recommended first try)
python src/train.py --config config/tuning_experiments.yaml \
  --experiment activation_gelu --save_dir results/tuning/gelu

# Swish activation
python src/train.py --config config/tuning_experiments.yaml \
  --experiment activation_swish --save_dir results/tuning/swish

# 256 receptors
python src/train.py --config config/tuning_experiments.yaml \
  --experiment more_receptors --save_dir results/tuning/receptors_256

# Strong diversity loss (λ=0.05)
python src/train.py --config config/tuning_experiments.yaml \
  --experiment strong_diversity --save_dir results/tuning/diversity_0.05

# Best combo: GELU + 256 receptors + strong diversity
python src/train.py --config config/tuning_experiments.yaml \
  --experiment gelu_more_receptors --save_dir results/tuning/gelu_256

# Minimal: GELU + no glomeruli
python src/train.py --config config/tuning_experiments.yaml \
  --experiment minimal_gelu --save_dir results/tuning/minimal_gelu
```

### Recommended Order

1. **GELU** — easiest, most promising
2. **minimal_gelu** — best architecture + best activation
3. **gelu_more_receptors** — if results are good

### Expected Tuning Outcomes

| Variant | Expected Effect |
|---------|----------------|
| GELU | +0.5-1% F1 over ReLU |
| 256 receptors | Better interpretability, similar F1 |
| Strong diversity | More specialized receptors (visible in heatmaps) |
| Each experiment | ~20-30 min on Colab GPU |

---

## Analyzing Results

After experiments complete:

```bash
python src/analysis/compare_results.py --results_dir ./results
```

This generates:

| Output | Description |
|--------|-------------|
| `comparison_table.csv` | Side-by-side metrics |
| `comparison_bars.png` | Bar chart visualization |
| `improvement_heatmap.png` | Heatmap of improvements |
| `entity_analysis.json` | Per-entity breakdown |
| `statistical_tests.json` | Significance tests |
| `COMPARISON_REPORT.md` | Comprehensive report |

### Results Directory Structure

```
results/
├── experiment_summary.json
├── conll2003/en/
│   ├── mbert_baseline/   (best_model.pt, results.json)
│   └── mbert_olfactory/  (best_model.pt, results.json)
└── wikiann/
    ├── hi/  ├── mr/  ├── ta/  ├── bn/  └── te/
```

---

## Expected Results

### Success Criteria

✅ **Hypothesis validated if**:
- Olfactory F1 > Baseline F1 on ≥67% of datasets (4/6)
- Improvement statistically significant (p < 0.05)

### Typical Outcomes

| Setting | Baseline F1 | Olfactory F1 | Improvement |
|---------|------------|-------------|-------------|
| CoNLL-2003 (high resource) | ~88-90% | ~89-91% | +0.5-1.5% |
| Indic languages (low resource) | ~70-75% | ~72-77% | +1.5-3% |

**Key finding**: Olfactory layers should help more on low-resource languages where structured inductive biases matter more.

---

## Troubleshooting

| Issue | Solution |
|-------|---------|
| **CUDA out of memory** | Reduce batch size: `--batch_size 8` (or 4) |
| **Slow dataset downloads** | First run downloads ~100MB. Subsequent runs use cache |
| **`No module named 'src'`** (Colab) | Ensure clone cell + `%cd` command ran successfully |
| **Drive not mounted** (Colab) | Re-run Cell 1 and authorize Google Drive access |
| **Results inconsistent** | Run with multiple seeds: `--seed 42`, `43`, `44`, `45`, `46` |
| **Poor performance** | Check GloVe loaded ("Found embeddings for X%"), verify data exists |
| **Receptors not specializing** | Increase `lambda_diverse` (0.01 → 0.05), try 256 receptors |

---

## References

- **Biological Inspiration**: Buck & Axel (1991) — Olfactory receptor discovery
- **CoNLL-2003**: Tjong Kim Sang & De Meulder (2003)
- **mBERT**: Devlin et al. (2019)
- **BiLSTM-CRF for NER**: Huang et al. (2015)
- **GELU**: Hendrycks & Gimpel (2016)
