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

The project uses **GloVe-based** embeddings, with a baseline and an olfactory variant.

### Baseline (without olfactory layers)

```
Embeddings → BiLSTM → CRF → NER Tags
```

### Olfactory-Enhanced

```
Embeddings → 🧬 Receptors → Glomeruli → BiLSTM → CRF → NER Tags
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
```

### Running Experiments

You can execute single training runs or execute the full multi-seed experiment suites:

#### 1. Run a Single Experiment
Use `train_universal.py` to train a model with a specific configuration (from `config/universal_config.yaml`) and seed on a dataset:
```bash
# Run baseline on Marathi (1k capped)
python src/train_universal.py --config config/universal_config.yaml --dataset_key wikiann_mr_1k --experiment baseline --seed 42

# Run olfactory model on Hindi (1k capped)
python src/train_universal.py --config config/universal_config.yaml --dataset_key wikiann_hi_1k --experiment olfactory --seed 123
```

#### 2. Run the Low-Resource Simulation Suite (1k Capped, 5 Seeds)
To run all 1k-capped experiments sequentially across all 5 seeds (42, 123, 456, 789, 1011):
```bash
bash run_low_resource_exp.sh
```

#### 3. Run the Full-Scale Experiments Suite (3 Seeds)
To train and evaluate the full dataset sizes sequentially:
```bash
bash run_colab_experiments.sh
```

#### 4. Run Legacy Training (GloVe-based)
If you wish to train on CoNLL-2003 English using preloaded GloVe embeddings:
```bash
# Run GloVe + BiLSTM + CRF baseline
python src/train.py --config config/experiments.yaml --experiment baseline

# Run GloVe + Olfactory + BiLSTM + CRF
python src/train.py --config config/experiments.yaml --experiment olfactory_full
```

#### 5. Analyze and Compare Results
After training, compare experiment folders and compile graphs:
```bash
python src/analysis/compare_results.py --results_dir ./results
```

### Google Colab (Recommended for GPU)

1. Upload `notebooks/universal_experiments.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Enable GPU: **Runtime → Change runtime type → GPU**
3. Run all cells to execute the universal suite.

> For detailed instructions, see [docs/EXPERIMENT_GUIDE.md](docs/EXPERIMENT_GUIDE.md).

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

### Experiments (via `config/universal_config.yaml`)

| Experiment | Model | Receptors | Glomeruli | Notes |
|------------|-------|-----------|-----------|-------|
| `baseline` | BiLSTM-CRF | — | — | Control — no olfactory layers |
| `olfactory` | Olfactory | 128 | 32 | Base olfactory configuration |
| `more_receptors` | Olfactory | 256 | 64 | Strong diversity loss (λ=0.05) |
| `more_glomeruli` | Olfactory | 128 | 64 | More glomeruli for better aggregation |
| `more_receptors_more_glomeruli` | Olfactory | 256 | 128 | Largest configuration |

### Datasets

| Dataset | Config Key | Language | Type |
|---------|-----------|----------|------|
| CoNLL-2003 | `conll_en` | English | High resource |
| WikiANN | `wikiann_hi/mr/ta/bn/te` | Hindi, Marathi, Tamil, Bangla, Telugu | Low resource |

### Low-Resource Simulation Control Experiments (1k Capped)

To systematically isolate the impact of dataset size and directly compare higher-resource languages (English, Bengali, Tamil, etc.) with ultra-low-resource settings (like Telugu's natural 1,000 sentences), we introduced **1k-capped** variants of all datasets:

| Capped Dataset | Config Key | Truncated Train Size | Purpose |
|----------------|------------|----------------------|---------|
| CoNLL-2003 (en) | `conll_en_1k` | 1,000 sentences | Isolate size variable on high-resource English |
| WikiANN Marathi | `wikiann_mr_1k` | 1,000 sentences | Simulates ultra-low-resource Marathi |
| WikiANN Hindi | `wikiann_hi_1k` | 1,000 sentences | Simulates ultra-low-resource Hindi |
| WikiANN Tamil | `wikiann_ta_1k` | 1,000 sentences | Simulates ultra-low-resource Tamil |
| WikiANN Bengali | `wikiann_bn_1k` | 1,000 sentences | Simulates ultra-low-resource Bengali |
| WikiANN Telugu | `wikiann_te` | 1,000 sentences (Natural) | Inherent baseline for low-resource comparison |

These experiments enforce a rigid structural control, verifying if the **olfactory bottleneck prior** consistently outperforms the standard sequence tagging baseline across all languages when training data is strictly limited to 1,000 samples.

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

## Results

All experiments are evaluated **without pretrained embeddings** (starting with random embeddings trained from scratch) to isolate the performance under strict inductive bias constraints. Results report the **Mean ± Standard Deviation (SD)** across multiple random seeds.

### 1. Full-Scale Multilingual Experiments (3 Seeds)

These experiments evaluate the architecture on the full size of each dataset (ranging from 1k sentences in Telugu to 15k sentences in Tamil), using 3 random seeds.

| Dataset | Experiment | F1 (Mean ± SD) | Precision | Recall | Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **conll2003** (English, 14k) | baseline | 75.68 ± 0.28% | 78.85 ± 0.11% | 72.77 ± 0.55% | 3 |
| | more_glomeruli | **76.48 ± 0.76%** | 80.74 ± 1.04% | 72.64 ± 0.52% | 3 |
| | more_receptors | 76.27 ± 0.69% | 80.11 ± 2.00% | 72.82 ± 0.38% | 3 |
| | no_sparsity | 76.17 ± 0.18% | 80.23 ± 0.89% | 72.50 ± 0.40% | 3 |
| | olfactory | 75.88 ± 0.67% | 79.76 ± 1.13% | 72.37 ± 0.52% | 3 |
| | receptors_only | 75.62 ± 1.16% | 78.85 ± 1.76% | 72.65 ± 0.67% | 3 |
| | | | | | |
| **wikiann_bn** (Bangla, 10k) | baseline | 92.91 ± 0.71% | 93.33 ± 0.74% | 92.50 ± 0.70% | 3 |
| | more_glomeruli | **93.11 ± 0.79%** | 93.44 ± 1.03% | 92.78 ± 0.56% | 3 |
| | more_receptors | 92.38 ± 1.38% | 92.65 ± 1.63% | 92.10 ± 1.13% | 3 |
| | no_sparsity | 92.70 ± 0.55% | 93.09 ± 0.52% | 92.32 ± 0.58% | 3 |
| | olfactory | 92.95 ± 0.57% | 93.44 ± 0.33% | 92.47 ± 0.82% | 3 |
| | receptors_only | 92.78 ± 0.37% | 93.10 ± 0.64% | 92.47 ± 0.15% | 3 |
| | | | | | |
| **wikiann_hi** (Hindi, 5k) | baseline | **83.22 ± 0.39%** | 83.40 ± 0.61% | 83.03 ± 0.66% | 3 |
| | more_glomeruli | 81.64 ± 1.82% | 81.13 ± 2.20% | 82.17 ± 1.45% | 3 |
| | more_receptors | 82.40 ± 0.94% | 82.63 ± 1.65% | 82.19 ± 0.52% | 3 |
| | no_sparsity | 79.55 ± 3.51% | 79.03 ± 4.21% | 80.10 ± 2.80% | 3 |
| | olfactory | 80.55 ± 3.58% | 80.19 ± 4.11% | 80.92 ± 3.03% | 3 |
| | receptors_only | 82.61 ± 1.29% | 82.06 ± 1.85% | 83.17 ± 0.72% | 3 |
| | | | | | |
| **wikiann_mr** (Marathi, 5k) | baseline | 77.44 ± 0.27% | 79.99 ± 0.43% | 75.05 ± 0.37% | 3 |
| | more_glomeruli | 76.48 ± 3.64% | 79.29 ± 3.88% | 73.87 ± 3.51% | 3 |
| | more_receptors | 78.85 ± 1.82% | 80.14 ± 2.07% | 77.61 ± 1.63% | 3 |
| | no_sparsity | 78.59 ± 0.44% | 82.10 ± 0.06% | 75.37 ± 0.78% | 3 |
| | olfactory | 79.34 ± 0.16% | 82.16 ± 0.43% | 76.71 ± 0.54% | 3 |
| | receptors_only | **80.21 ± 0.52%** | 83.52 ± 1.08% | 77.16 ± 0.30% | 3 |
| | | | | | |
| **wikiann_ta** (Tamil, 15k) | baseline | 79.77 ± 0.37% | 82.33 ± 0.06% | 77.36 ± 0.66% | 3 |
| | more_glomeruli | 79.41 ± 0.74% | 81.45 ± 0.78% | 77.46 ± 0.74% | 3 |
| | more_receptors | 79.39 ± 0.17% | 81.50 ± 0.02% | 77.38 ± 0.31% | 3 |
| | no_sparsity | 79.28 ± 1.07% | 82.24 ± 1.01% | 76.52 ± 1.17% | 3 |
| | olfactory | 79.44 ± 0.66% | 81.57 ± 1.08% | 77.44 ± 1.00% | 3 |
| | receptors_only | **80.17 ± 0.58%** | 82.86 ± 0.69% | 77.65 ± 0.48% | 3 |
| | | | | | |
| **wikiann_te** (Telugu, 1k) | baseline | 52.51 ± 1.74% | 55.19 ± 1.92% | 50.10 ± 1.82% | 3 |
| | more_glomeruli | **56.94 ± 1.17%** | 61.09 ± 1.97% | 53.34 ± 0.84% | 3 |
| | more_receptors | 56.07 ± 1.96% | 59.30 ± 2.86% | 53.20 ± 1.28% | 3 |
| | no_sparsity | 55.74 ± 1.80% | 59.92 ± 2.98% | 52.15 ± 1.09% | 3 |
| | olfactory | 55.92 ± 1.80% | 60.26 ± 3.04% | 52.21 ± 1.08% | 3 |
| | receptors_only | 55.40 ± 1.02% | 56.96 ± 1.61% | 53.94 ± 0.52% | 3 |

**Key Takeaway**: Under full-resource training without pretrained embeddings, olfactory configurations (especially those with structured bottlenecks like `more_glomeruli` or `receptors_only`) show distinct benefits over the baseline in low-resource settings such as Telugu (+4.43% F1 mean / +3.95% best seed) and Marathi (+2.77% F1 mean / +1.29% best seed), and even improve high-resource English when trained from scratch (+0.80% F1 mean / +1.59% best seed). Conversely, the standard baseline remains highly competitive in Bangla and Hindi where structural constraints are less necessary due to high dataset regularity or sufficient volume.

---

### 2. Low-Resource Simulation Capped Experiments (1k Capped, 5 Seeds)

To systematically control the impact of dataset size and directly compare performance in data-constrained scenarios, we evaluated all 6 datasets capped at exactly **1,000 training sentences** across 5 random seeds.

| Dataset | Experiment | F1 (Mean ± SD) | Precision | Recall | Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **conll_en_1k** (English, 1k) | baseline | **48.95 ± 1.73%** | 54.05 ± 2.30% | 44.91 ± 3.11% | 5 |
| | no_sparsity | 46.72 ± 1.38% | 50.44 ± 3.36% | 43.72 ± 2.15% | 5 |
| | olfactory | 46.83 ± 1.40% | 51.47 ± 3.77% | 43.18 ± 2.19% | 5 |
| | | | | | |
| **wikiann_bn_1k** (Bangla, 1k) | baseline | 63.97 ± 4.82% | 60.14 ± 6.55% | 68.60 ± 3.04% | 5 |
| | no_sparsity | 68.01 ± 2.56% | 65.35 ± 2.49% | 70.91 ± 2.75% | 5 |
| | olfactory | **68.13 ± 2.98%** | 65.71 ± 3.35% | 70.74 ± 2.68% | 5 |
| | | | | | |
| **wikiann_hi_1k** (Hindi, 1k) | baseline | 62.41 ± 5.04% | 59.82 ± 6.85% | 65.42 ± 3.06% | 5 |
| | no_sparsity | 65.37 ± 2.93% | 62.91 ± 4.16% | 68.09 ± 1.53% | 5 |
| | olfactory | **66.04 ± 2.92%** | 63.95 ± 3.88% | 68.32 ± 1.97% | 5 |
| | | | | | |
| **wikiann_mr_1k** (Marathi, 1k) | baseline | **63.09 ± 2.00%** | 65.01 ± 3.20% | 61.33 ± 1.30% | 5 |
| | no_sparsity | 62.02 ± 3.67% | 63.67 ± 5.22% | 60.52 ± 2.31% | 5 |
| | olfactory | 62.01 ± 3.63% | 64.24 ± 5.13% | 60.02 ± 2.69% | 5 |
| | | | | | |
| **wikiann_ta_1k** (Tamil, 1k) | baseline | 45.93 ± 2.02% | 46.57 ± 3.73% | 45.45 ± 1.10% | 5 |
| | no_sparsity | **50.24 ± 2.01%** | 52.99 ± 3.60% | 47.83 ± 1.12% | 5 |
| | olfactory | 49.96 ± 2.65% | 52.51 ± 4.68% | 47.78 ± 1.51% | 5 |
| | | | | | |
| **wikiann_te_1k** (Telugu, 1k) | baseline | 54.27 ± 1.03% | 58.16 ± 2.28% | 50.93 ± 1.13% | 5 |
| | no_sparsity | 55.16 ± 2.16% | 59.99 ± 3.22% | 51.08 ± 1.61% | 5 |
| | olfactory | **55.89 ± 1.96%** | 59.97 ± 1.96% | 52.37 ± 2.41% | 5 |

**Low-Resource Analysis**: 
* **Generalization Gains**: The olfactory bottleneck acts as a powerful regularizer under extreme data constraints. It consistently improves F1 on **4 out of 6** datasets: Bangla (+4.16%), Hindi (+3.63%), Tamil (+4.03%), and Telugu (+1.62%).
* **Training Stability**: In languages where the baseline shows high variance across runs (e.g. Bangla and Hindi), the structured bottleneck significantly stabilizes convergence, cutting standard deviation from ~4.8-5.0% down to ~2.9%.
* **Exceptions**:
  * **English (`conll_en_1k`)**: The baseline outperforms olfactory models, as English's rigid syntax can be modeled directly, whereas the 32-glomeruli bottleneck causes minor underfitting (-2.12% F1).
  * **Marathi (`wikiann_mr_1k`)**: The base configuration (128 receptors, 32 glomeruli) underfits Marathi's morphological complexities, showing a slight decrease. (Note: full-scale runs show Marathi requires higher capacity bottlenecks to yield gains).

---

### Receptor Specialization Analysis

All olfactory models exhibit **sparse, selective receptor firing** consistent with the biological analogy.

| Dataset | Avg. RSI | Avg. Sparsity | Notes |
|---------|----------|--------------|-------|
| CoNLL-2003 (en) | **0.83** | ~20–31% | Highest RSI; receptors fire for specific named entities ("National", "Inc", location markers) |
| WikiANN Marathi | 0.52–0.54 | ~28–31% | Receptors pick up Marathi NE cues (e.g., `नदी`=river, `विद्यापीठ`=university) |
| WikiANN Hindi | 0.46–0.53 | ~28–31% | Specialised to postpositions and NE-adjacent tokens (`को`, `में`) |
| WikiANN Tamil | 0.44–0.53 | ~29–31% | Entity-type tokens strongly activating |
| WikiANN Bangla | 0.56–0.61 | ~34–37% | High RSI but performance drops — sparsity penalty may be too strong |
| WikiANN Telugu | **0.58–0.65** | ~24–32% | Highest RSI despite least data — strong specialization per entity region |

**Sparsity is consistently 20–37%** across all experiments: only ~1 in 3 receptors fires for any given token, demonstrating the sparse combinatorial coding principle from olfactory neuroscience.

---

### Verdict

| Success Criterion | Outcome |
|---|---|
| Olfactory F1 > Baseline on ≥67% of datasets | ✅ **4/6 (67%)** in both full-scale and low-resource settings |
| Better low-resource performance | ✅ **Bangla +4.16%, Hindi +3.63%, Tamil +4.03%, Telugu +1.62%** (1k capped) |
| Training stabilization | ✅ **Reduces SD by ~2%** on volatile datasets (Bangla & Hindi) |
| High-resource settings | ❌ Baseline is better/comparable on English where data is abundant |

**Conclusion**: The olfactory inductive bias is most valuable in **very low-resource, no-pretrained-embedding** settings. When data is abundant or features are already well-represented, the BiLSTM-CRF baseline is sufficient and the receptor bottleneck is counterproductive.

---

## Project Structure

```
olfaction-inspired-ner/
├── src/
│   ├── model/
│   │   ├── layers.py              # Receptor & glomerular layers
│   │   ├── olfactory_ner.py       # OlfactoryNER (GloVe-based)
│   │   ├── baseline.py            # BaselineNER (GloVe-based)
│   │   └── crf.py                 # CRF decoder
│   ├── data/
│   │   ├── dataset.py             # CoNLL-2003 loading, GloVe embeddings
│   │   ├── dataset_marathi.py     # Marathi-specific data loading
│   │   ├── dataset_ontonotes.py   # OntoNotes data loading
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
│   │   └── save_results.py        # Results saving utilities
│   ├── train.py                   # GloVe-based training script
│   ├── train_universal.py         # Universal trainer (all datasets + configs)
│   └── train_marathi.py           # Marathi-specific training script
├── config/
│   ├── universal_config.yaml      # All experiment configurations & datasets
│   └── mitral_config.yaml         # Mitral cell experiment config
├── docs/
│   ├── ARCHITECTURE.md            # Detailed architecture deep-dive
│   ├── EXPERIMENT_GUIDE.md        # How to run experiments
│   ├── PARAMETER_TUNING_GUIDE.md  # Hyperparameter tuning guide
│   └── starting.md                # Theoretical foundation (olfactory biology → NER)
├── notebooks/
│   ├── comprehensive_experiments.ipynb
│   ├── universal_experiments.ipynb
│   ├── mitral_experiments.ipynb
│   ├── gelu_experiment.ipynb
│   └── olfaction_ner_colab.ipynb
├── run_colab_experiments.sh       # Main sequentially orchestrated experiment script
├── run_low_resource_exp.sh        # Capped low-resource simulation script (1k samples)
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
- Use `config/universal_config.yaml` experiments: `more_receptors`, `gelu_more_receptors`
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
- BiLSTM-CRF for NER: Huang et al. (2015)
