# Olfaction-Inspired NER

Biologically-inspired Named Entity Recognition using olfactory coding principles.

## Overview

This project implements an **olfaction-inspired neural architecture for NER** that models entity recognition as combinatorial activation of specialized feature detectors (receptors), aggregated through convergent pooling (glomeruli), before contextual processing.

**Core Hypothesis**: Olfactory-style combinatorial coding provides useful inductive biases for NER through:
- **Compositionality** — combining multiple weak signals
- **Interpretability** — explicit feature specialization
- **Robustness** — noise tolerance through aggregation

> We do not claim state-of-the-art performance. Our goal is to test whether olfactory-style combinatorial coding provides a useful inductive bias for NER.

---

## Architecture

Two model families exist — **GloVe-based** and **mBERT-based** — each with a baseline and olfactory variant.

### Baseline (without olfactory layers)

```
GloVe:  Embeddings → BiLSTM → CRF → NER Tags
mBERT:  mBERT (frozen) → Linear → CRF → NER Tags
```

### Olfactory-Enhanced

```
GloVe:  Embeddings → 🧬 Receptors → Glomeruli → BiLSTM → CRF → NER Tags
mBERT:  mBERT (frozen) → 🧬 Receptors → Glomeruli → Linear → CRF → NER Tags
```

The **only structural difference** is the insertion of **Receptor → Glomerular** layers between the embeddings and the sequence encoder.

> For a detailed architecture deep-dive with tensor shapes and math, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Quick Start

### Local Setup

```bash
# Clone repository
git clone https://github.com/bhushan1729/olfaction-inspired-ner.git
cd olfaction-inspired-ner

# Install dependencies
pip install -r requirements.txt

# Quick test (1 epoch, ~5-10 minutes)
python run_baseline_vs_olfactory.py --quick_test

# Full experiments (all 6 datasets, ~2-4 hours)
python run_baseline_vs_olfactory.py --epochs 5

# Analyze results
python src/analysis/compare_results.py
```

### Google Colab (Recommended for GPU)

1. Upload `baseline_vs_olfactory_experiments.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Enable GPU: **Runtime → Change runtime type → GPU**
3. Run all cells

> For detailed instructions on running experiments, Colab setup, and tuning, see [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md).

---

## Key Components

### Receptor Layer (`src/model/layers.py`)
- **Biological inspiration**: Olfactory receptors are highly specialized one-neuron-one-receptor detectors
- **Implementation**: Linear projections with ReLU/GELU activation → sparse feature activations
- **Regularization**: Diversity loss prevents redundant receptors

### Glomerular Layer (`src/model/layers.py`)
- **Biological inspiration**: Multiple neurons with same receptor converge to one glomerulus
- **Implementation**: Learnable aggregation (128 receptors → 32 glomeruli)
- **Purpose**: Denoising and feature abstraction through convergence

### CRF Decoder (`src/model/crf.py`)
- Enforces valid BIO tag sequences
- Training: Forward algorithm (negative log-likelihood)
- Inference: Viterbi decoding

---

## Experiments

### Main Comparison (mBERT-based)

| Model | What's Compared |
|-------|----------------|
| `BertBaseline` | mBERT (frozen) → Linear → CRF |
| `BertOlfactory` | mBERT (frozen) → Receptors → Glomeruli → Linear → CRF |

Both use **frozen mBERT** and differ **only** in the olfactory layers, isolating their contribution.

### GloVe-based Ablations

| Experiment | Description | Purpose |
|------------|-------------|---------|
| `baseline` | BiLSTM-CRF | Control |
| `olfactory_full` | Full model with regularization | Main hypothesis |
| `olfactory_no_sparse` | Without sparsity loss | Test sparsity importance |
| `olfactory_no_glomeruli` | Without aggregation | Test convergence importance |

### Datasets

| Dataset | Language | Type |
|---------|----------|------|
| CoNLL-2003 | English | High resource |
| WikiANN Hindi | Hindi | Low resource |
| WikiANN Marathi | Marathi | Low resource |
| WikiANN Tamil | Tamil | Low resource |
| WikiANN Bangla | Bangla | Low resource |
| WikiANN Telugu | Telugu | Low resource |

**Expectation**: Olfactory layers should help more on low-resource languages where structured inductive biases matter more.

---

## Success Criteria

We consider the hypothesis validated if **any** of:
- ✅ Olfactory F1 > Baseline F1 on ≥67% of datasets (4/6)
- ✅ Comparable F1 with fewer parameters
- ✅ Better low-resource performance
- ✅ Clear interpretable receptor patterns
- ✅ Lower variance across runs

### Key Visualizations

| Output | What It Shows |
|--------|--------------|
| `receptor_heatmap.png` | Receptor specialization by entity type |
| `glomeruli_tsne.png` | Feature clustering by entity type |
| `model_comparison.png` | Cross-model F1 comparison |
| `results.json` | Detailed metrics (F1, precision, recall, per-entity) |

---

## Project Structure

```
olfaction-inspired-ner/
├── src/
│   ├── model/
│   │   ├── layers.py              # Receptor & glomerular layers
│   │   ├── olfactory_ner.py       # OlfactoryNER (GloVe-based)
│   │   ├── baseline.py            # BaselineNER (GloVe-based)
│   │   ├── bert_models.py         # BertBaseline & BertOlfactory (mBERT-based)
│   │   └── crf.py                 # CRF decoder
│   ├── data/
│   │   ├── dataset.py             # CoNLL-2003 loading, GloVe embeddings
│   │   ├── bert_loader.py         # HuggingFace datasets, WordPiece alignment
│   │   └── unified_loader.py      # Unified loader for all datasets
│   ├── training/
│   │   └── metrics.py             # Comprehensive NER metrics
│   ├── analysis/
│   │   ├── visualize.py           # Receptor analysis & visualization
│   │   ├── compare_results.py     # Results comparison & statistical tests
│   │   ├── generate_heatmaps.py   # Receptor/glomeruli heatmap generation
│   │   └── final_analysis.py      # Comprehensive analysis
│   ├── utils/
│   │   ├── colab_git.py           # Colab Git integration
│   │   ├── save_results.py        # Results saving utilities
│   │   └── create_marathi_notebook.py  # Marathi notebook generator
│   ├── train.py                   # GloVe-based training script
│   └── train_bert.py              # mBERT-based training script
├── config/
│   └── experiments.yaml           # Experiment configurations
├── docs/
│   ├── ARCHITECTURE.md            # Detailed architecture deep-dive
│   ├── RESULTS.md                 # Experimental results
│   ├── PARAMETER_TUNING_GUIDE.md  # Hyperparameter tuning guide
│   └── GELU_COMPARISON.md         # ReLU vs GELU comparison
├── notebooks/
│   └── *.ipynb                    # Colab notebooks
├── run_baseline_vs_olfactory.py   # Experiment orchestrator
├── starting.md                    # Theoretical foundation (olfactory biology → NER)
├── EXPERIMENT_GUIDE.md            # How to run experiments
└── requirements.txt
```

---

## Interpreting Results

### ✅ Strong Evidence → Write Paper
- Receptors show clear specialization (different entities activate different receptors)
- F1 comparable to baseline (within 1 point)
- Ablations degrade performance

### ⚠️ Mixed Results → Tune & Iterate
- Adjust `lambda_diverse` (0.01 → 0.05), try different `num_receptors` (64, 128, 256)
- See [docs/PARAMETER_TUNING_GUIDE.md](docs/PARAMETER_TUNING_GUIDE.md)

### ❌ No Advantage → Pivot
- Re-examine hypothesis or try different architecture

---

## Citation

```bibtex
@misc{olfaction-inspired-ner-2026,
  title={Biologically-Inspired Olfactory Feature Extraction for Named Entity Recognition},
  author={Bhushan},
  year={2026},
  url={https://github.com/bhushan1729/olfaction-inspired-ner}
}
```

## License

MIT License

## Acknowledgments

- Biological inspiration: Buck & Axel (1991) — olfactory receptor discovery
- CoNLL-2003: Tjong Kim Sang & De Meulder (2003)
- GloVe: Pennington et al. (2014)
- mBERT: Devlin et al. (2019)
- BiLSTM-CRF for NER: Huang et al. (2015)
