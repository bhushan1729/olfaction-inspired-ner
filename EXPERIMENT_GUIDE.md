# Baseline vs Olfactory Experiment Guide

## Experimental Setup

This guide describes the minimal, correct experiment to validate:

**Hypothesis**: Does my biologically inspired olfactory feature extractor add value beyond a standard language model for NER?

## The Two Models

### Baseline (LLM-only)
```
mBERT (frozen) → Dropout → Linear Classifier → NER Tags
Loss: CrossEntropyLoss
```

**Purpose**: Measure how good a strong pretrained language model is by itself.

### Olfactory (LLM + Olfactory layers)
```
mBERT (frozen) → Receptors → Glomeruli → BiLSTM → CRF → NER Tags  
Loss: Negative Log Likelihood from CRF
```

**Purpose**: Does biologically-inspired preprocessing improve NER?

## Why This Design?

### 1. Both use frozen mBERT
- **Fair comparison**: Models differ ONLY in the olfactory layers
- **Proves contribution**: Any improvement comes from olfactory processing, not BERT fine-tuning

### 2. Why BiLSTM + CRF for olfactory but not baseline?
**Because NER is a structured sequence task, not just token classification.**

Without BiLSTM+CRF, you get:
- Illegal tag sequences like `O → I-PER → I-LOC`
- Reviewers saying: "Your gains come from decoding weakness, not representation quality"

The baseline uses simple token-wise classification because:
- It represents what a "pure LLM" baseline does
- The olfactory model's added value is in BOTH representations AND structured decoding

### 3. What exactly are you claiming?

**NOT:** "We model context better than transformers"  
**YES:** "Before sequence modeling, structured, sparse, convergent representations help NER"

The contribution is the olfactory preprocessing:
- **Receptors**: Specialized feature detectors (sparsity + specialization)
- **Glomeruli**: Convergent aggregation (many → few, denoising)
- **BiLSTM**: Context modeling (standard in NER)
- **CRF**: Tag transition constraints (standard in NER)

## Running Experiments

### Quick Test (Recommended First)

Test that everything works (1 epoch on CoNLL-2003):

```bash
python run_baseline_vs_olfactory.py --quick_test
```

Expected time: ~5-10 minutes (depending on GPU/CPU)

### Full Experiment Suite

Run on all datasets (CoNLL-2003 + 5 Indic languages):

```bash
python run_baseline_vs_olfactory.py --epochs 5
```

Expected time: ~2-4 hours (depending on hardware)

### Custom Experiments

Run specific datasets:

```bash
# Just CoNLL-2003 and Hindi
python run_baseline_vs_olfactory.py --datasets conll2003,hindi --epochs 5

# Single dataset
python run_baseline_vs_olfactory.py --datasets conll2003 --epochs 10
```

Available datasets:
- `conll2003` - English (CoNLL-2003)
- `hindi` - Hindi (WikiANN)
- `marathi` - Marathi (WikiANN)
- `tamil` - Tamil (WikiANN)
- `bangla` - Bangla (WikiANN)
- `telugu` - Telugu (WikiANN)

### Manual Training

Train baseline manually:

```bash
python src/train_bert.py \
  --dataset conll2003 \
  --experiment baseline \
  --epochs 5 \
  --batch_size 16 \
  --lr 2e-5
```

Train olfactory manually:

```bash
python src/train_bert.py \
  --dataset conll2003 \
  --experiment olfactory \
  --epochs 5 \
  --batch_size 16 \
  --lr 2e-5
```

## Analyzing Results

After experiments complete, run comparative analysis:

```bash
python compare_results.py --results_dir ./results
```

This generates:
- `comparison_table.csv` - Side-by-side metrics
- `comparison_bars.png` - Bar chart visualization
- `improvement_heatmap.png` - Heatmap of improvements
- `entity_analysis.json` - Per-entity breakdown
- `statistical_tests.json` - Significance tests
- `COMPARISON_REPORT.md` - Comprehensive report

## Expected Results

### High-Resource (CoNLL-2003 English)
- **Baseline**: ~88-90% F1
- **Olfactory**: ~89-91% F1
- **Improvement**: Modest (+0.5-1.5%)

### Low-Resource (Indic Languages)
- **Baseline**: ~70-75% F1
- **Olfactory**: ~72-77% F1
- **Improvement**: Larger (+1.5-3%)

**Key Finding**: Olfactory layers should help more on low-resource languages where structured inductive biases matter more.

## Success Criteria

**Primary (Hypothesis Validated if):**
- ✅ Olfactory F1 > Baseline F1 on ≥67% of datasets (4/6)
- ✅ Improvement is statistically significant (p < 0.05)

**Secondary (Nice to Have):**
- Olfactory shows larger gains on low-resource languages
- Per-entity analysis shows consistent improvements  
- CRF ablation proves value of structured decoding

## Mental Model

Think of it like this:

| Component | Baseline | Olfactory | Biological Analogy |
|-----------|----------|-----------|-------------------|
| **BERT** | Eyes (sees everything) | Eyes (sees everything) | Visual input |
| **Olfactory** | ❌ None | Nose (filters signals) | Olfactory processing |
| **BiLSTM** | ❌ None | Brain (context) | Higher cortex |
| **CRF** | ❌ None | Grammar (rules) | Linguistic rules |
| **Classifier** | Direct classification | ❌ None | - |

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
python run_baseline_vs_olfactory.py --batch_size 8
```

### Slow downloads
First time running downloads datasets (~100MB). Subsequent runs use cache.

### Results inconsistent
Try running with multiple random seeds:
```bash
for seed in 42 43 44 45 46; do
  python src/train_bert.py --dataset conll2003 --experiment baseline --seed $seed
done
```

## Files Structure

After running experiments:

```
results/
├── experiment_summary.json          # Overall summary
├── conll2003/
│   └── en/
│       ├── mbert_baseline/
│       │   ├── best_model.pt
│       │   └── results.json
│       └── mbert_olfactory/
│           ├── best_model.pt
│           └── results.json
└── wikiann/
    ├── hi/ (Hindi)
    ├── mr/ (Marathi)
    ├── ta/ (Tamil)
    ├── bn/ (Bangla)
    └── te/ (Telugu)

comparison_analysis/
├── comparison_table.csv
├── comparison_bars.png
├── improvement_heatmap.png
├── entity_analysis.json
├── statistical_tests.json
└── COMPARISON_REPORT.md
```

## References

- **Biological Inspiration**: Buck & Axel (1991) - Olfactory receptor discovery
- **CoNLL-2003**: Tjong Kim Sang & De Meulder (2003)
- **mBERT**: Devlin et al. (2019)
- **BiLSTM-CRF for NER**: Huang et al. (2015)
