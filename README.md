# Olfaction-Inspired NER

Biologically-inspired Named Entity Recognition using olfactory coding principles.

## Overview

This project implements an **olfaction-inspired neural architecture for NER** that models entity recognition as combinatorial activation of specialized feature detectors (receptors), aggregated through convergent pooling (glomeruli), before contextual processing.

**Core Hypothesis**: Olfactory-style combinatorial coding provides useful inductive biases for NER through improved:
- **Compositionality**: Combining multiple weak signals
- **Interpretability**: Explicit feature specialization  
- **Robustness**: Noise tolerance through aggregation

## Quick Start

### Local Setup

```bash
# Clone repository
git clone <repository-url>
cd olfaction_inspired_ner

# Install dependencies
pip install -r requirements.txt

# Download GloVe embeddings (optional but recommended)
# Visit: https://nlp.stanford.edu/projects/glove/
# Download glove.6B.zip and extract to data/

# Train baseline model
python src/train.py --config config/experiments.yaml --experiment baseline

# Train olfactory model
python src/train.py --config config/experiments.yaml --experiment olfactory_full

# Analyze results
python -m src.analysis.visualize
```

### Google Colab

For GPU training without local setup:

1. Open `notebooks/olfaction_ner_colab.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run all cells

Expected runtime: **2-3 hours** for all experiments

## Architecture

```
Tokens
  ↓
Embeddings (GloVe 300d)
  ↓
Receptor Layer (128 specialized detectors)
  ↓
Glomerular Layer (32 aggregated features)
  ↓
BiLSTM Encoder (256 hidden)
  ↓
CRF Decoder
  ↓
Entity Labels
```

## Experiments

| Experiment | Description | Purpose |
|------------|-------------|---------|
| **baseline** | BiLSTM-CRF | Control |
| **olfactory_full** | Full model with regularization | Main hypothesis |
| **olfactory_no_sparse** | Without sparsity loss | Test sparsity importance |
| **olfactory_no_glomeruli** | Without aggregation | Test convergence importance |

## Results

After training, check:
- `results/<experiment>/results.json` - Metrics
- `analysis_results/receptor_heatmap.png` - Receptor specialization
- `analysis_results/glomeruli_tsne.png` - Feature clustering
- `comparison/model_comparison.png` - Cross-model comparison

## Project Structure

```
olfaction_inspired_ner/
├── src/
│   ├── data/
│   │   └── dataset.py          # CoNLL-2003 data loading
│   ├── model/
│   │   ├── layers.py           # Receptor & glomerular layers
│   │   ├── crf.py              # CRF implementation
│   │   ├── olfactory_ner.py    # Main model
│   │   └── baseline.py         # Baseline BiLSTM-CRF
│   ├── training/
│   │   └── evaluate.py         # Evaluation metrics
│   ├── analysis/
│   │   └── visualize.py        # Receptor analysis
│   └── train.py                # Training script
├── config/
│   └── experiments.yaml        # Experiment configs
├── notebooks/
│   └── olfaction_ner_colab.ipynb  # Colab notebook
├── data/                       # Downloaded data
├── results/                    # Experiment results
└── requirements.txt
```

## Key Components

### Receptor Layer
- **What**: Specialized micro-feature detectors (e.g., capitalization, suffixes, patterns)
- **How**: Linear projections with ReLU for sparsity
- **Why**: Enforces feature specialization, inspired by "one neuron-one receptor" principle

### Glomerular Layer  
- **What**: Convergent aggregation of receptor activations
- **How**: Learnable weighted sum: many receptors → fewer glomeruli
- **Why**: Denoising and abstraction, inspired by OSN convergence

### Regularization
- **Sparsity Loss** (L1): Encourages selective activation
- **Diversity Loss**: Penalizes redundant receptors (cosine similarity)

## Success Criteria

We consider the hypothesis validated if **any** of:
- ✅ Comparable F1 with fewer parameters
- ✅ Better low-resource performance
- ✅ Clear interpretable receptor patterns
- ✅ Lower variance across runs

**Note**: We do NOT aim to beat BERT or other SOTA models. This is an architectural exploration.

## Analysis

Key visualizations in `src/analysis/visualize.py`:

1. **Receptor Heatmap**: Mean activations per entity type
2. **Top Tokens per Receptor**: Interpretability check
3. **t-SNE of Glomeruli**: Clustering by entity type
4. **Specialization Metrics**: Sparsity, diversity, variance

## Citation

If you use this code, please cite:

```bibtex
@misc{olfaction-inspired-ner,
  title={Olfaction-Inspired Neural Architecture for Named Entity Recognition},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/olfaction-inspired-ner}
}
```

## License

MIT License

## Acknowledgments

- Biological inspiration from Buck & Axel (1991) - olfactory receptor discovery
- CoNLL-2003 dataset
- GloVe embeddings (Pennington et al., 2014)

## Contact

For questions or collaboration: [your.email@example.com]

---

**Status**: Minimal proof-of-concept (2-3 days) ✅  
**Next**: Extended validation (low-resource, cross-domain, multilingual)
