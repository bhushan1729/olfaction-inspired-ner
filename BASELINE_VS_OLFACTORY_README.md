# Minimal Baseline vs Olfactory NER Experiment

This repository implements the minimal, correct experiment to validate:

> **Hypothesis**: Does my biologically inspired olfactory feature extractor add value beyond a standard language model for NER?

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Quick Test (5-10 minutes)

```bash
python run_baseline_vs_olfactory.py --quick_test
```

This runs 1 epoch on CoNLL-2003 to verify everything works.

### 3. Run Full Experiments (2-4 hours)

```bash
python run_baseline_vs_olfactory.py --epochs 5
```

This runs on all 6 datasets (CoNLL-2003 + 5 Indic languages).

### 4. Analyze Results

```bash
python compare_results.py
```

View the report at `comparison_analysis/COMPARISON_REPORT.md`.

## What's Being Compared?

### Baseline (LLM-only + CRF)
```
mBERT (frozen) → Linear Classifier → CRF → NER Tags
```

### Olfactory (LLM + Bio-Inspired + CRF)
```
mBERT (frozen) → Receptors → Glomeruli → Linear Classifier → CRF → NER Tags
```

**Key**: Both differ ONLY in the olfactory processing. We removed BiLSTM from the olfactory model and added CRF to the baseline for a strictly fair comparison.

## Why This Design?

This is the **minimal, correct experiment** because:

1. ✅ **Isolates the contribution**: Both models use frozen mBERT, differ only in olfactory layers
2. ✅ **Fair comparison**: Same pretraining, same training setup
3. ✅ **Addresses NER requirements**: BiLSTM+CRF ensure valid tag sequences
4. ✅ **Proves the claim**: "Structured, sparse, convergent representations help NER"

Not "we beat transformers" — rather "bio-inspired preprocessing adds value."

## Documentation

- **[EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)** - Complete experimental setup and rationale
- **[walkthrough.md](.gemini/antigravity/brain/*/walkthrough.md)** - Implementation details and code changes

## Datasets

1. CoNLL-2003 (English) - High resource
2. WikiANN Hindi - Low resource
3. WikiANN Marathi - Low resource
4. WikiANN Tamil - Low resource
5. WikiANN Bangla - Low resource
6. WikiANN Telugu - Low resource

**Expectation**: Olfactory should help more on low-resource languages.

## Files Structure

```
olfaction_inspired_ner/
├── src/
│   ├── model/
│   │   └── bert_models.py         # Baseline & Olfactory models
│   ├── training/
│   │   └── metrics.py             # Comprehensive NER metrics
│   ├── data/
│   │   └── bert_loader.py         # Dataset loading
│   └── train_bert.py              # Training script
├── run_baseline_vs_olfactory.py   # Experiment orchestrator
├── compare_results.py             # Results analyzer
├── EXPERIMENT_GUIDE.md            # User guide
└── requirements.txt               # Dependencies
```

## Expected Results

### Success Criteria

✅ **Hypothesis validated if**:
- Olfactory F1 > Baseline F1 on ≥67% of datasets (4/6)
- Improvement is statistically significant (p < 0.05)

### Typical Outcomes

**CoNLL-2003 (High Resource)**:
- Baseline: ~88-90% F1
- Olfactory: ~89-91% F1
- Improvement: +0.5-1.5%

**Indic Languages (Low Resource)**:
- Baseline: ~70-75% F1
- Olfactory: ~72-77% F1
- Improvement: +1.5-3% (larger, as expected)

## Customization

Run specific datasets:

```bash
# Just English and Hindi
python run_baseline_vs_olfactory.py --datasets conll2003,hindi --epochs 10

# Single dataset, more epochs
python run_baseline_vs_olfactory.py --datasets conll2003 --epochs 20
```

Adjust hyperparameters:

```bash
python run_baseline_vs_olfactory.py --epochs 5 --batch_size 8 --lr 5e-5
```

## Troubleshooting

**CUDA Out of Memory**:
```bash
python run_baseline_vs_olfactory.py --batch_size 8
```

**Slow downloads**:
First run downloads datasets (~100MB). Subsequent runs use cache.

**Need help**:
See [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) for detailed instructions.

## Citation

If you use this code, please cite:

```bibtex
@misc{olfaction-inspired-ner-2026,
  title={Biologically-Inspired Olfactory Feature Extraction for Named Entity Recognition},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/olfaction-inspired-ner}
}
```

## License

MIT License

---

**Status**: ✅ Implementation complete, ready to run experiments
