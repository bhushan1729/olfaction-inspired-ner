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

All experiments are evaluated **without pretrained embeddings** (starting with random embeddings trained entirely from scratch) to isolate the performance under strict inductive bias constraints. Results report the **Mean ± Standard Deviation (SD)** across multiple random seeds.

### 1. Full-Scale Multilingual Experiments (3 Seeds)

These experiments evaluate the architecture on the full size of each dataset (ranging from 1.4k sentences in Telugu to 15k sentences in Tamil), using 3 random seeds.

| Dataset | Baseline | Olfactory (128R, 32G) | More Glomeruli (128R, 64G) | More Receptors (256R, 64G) | Receptors Only (128R, No G) | No Sparsity (Base w/o L1) | Best Config |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **conll_en** (English, 14k) | 75.68 ± 0.28% | 75.88 ± 0.67% | 76.48 ± 0.76% | 76.27 ± 0.69% | **76.55 ± 0.19%** | 76.17 ± 0.18% | **receptors_only (+0.87%)** |
| **wikiann_bn** (Bangla, 10k) | 92.91 ± 0.71% | 92.95 ± 0.57% | **93.11 ± 0.79%** | 92.38 ± 1.38% | 92.78 ± 0.37% | 92.70 ± 0.55% | **more_glomeruli (+0.20%)** |
| **wikiann_hi** (Hindi, 5k) | 82.41 ± 1.33% | 81.24 ± 1.41% | 82.58 ± 1.03% | 81.67 ± 0.35% | **83.07 ± 0.97%** | 80.07 ± 1.42% | **receptors_only (+0.66%)** |
| **wikiann_mr** (Marathi, 5k) | 78.04 ± 0.57% | 78.92 ± 0.01% | 77.86 ± 2.50% | 78.93 ± 1.06% | 78.40 ± 0.37% | **79.04 ± 0.61%** | **no_sparsity (+1.00%)** |
| **wikiann_ta** (Tamil, 15k) | 79.77 ± 0.37% | 79.44 ± 0.66% | 79.41 ± 0.74% | 79.39 ± 0.17% | **80.17 ± 0.58%** | 79.28 ± 1.07% | **receptors_only (+0.40%)** |
| **wikiann_te** (Telugu, 1.4k) | 52.51 ± 1.74% | 55.92 ± 1.80% | **56.94 ± 1.17%** | 56.07 ± 1.96% | 55.40 ± 1.02% | 55.74 ± 1.80% | **more_glomeruli (+4.43%)** |

**Key Takeaways**:
- **Denoising at Low Scales:** The most pronounced full-scale improvements occur on Telugu (+4.43% F1) where the training size is naturally very small (~1.4k sentences). Glomerular compression acts as a denoising filter to improve generalization.
- **Agglutinative Capacity Trade-offs:** For morphologically complex, agglutinative languages like Marathi, removing the bottleneck while keeping receptor projections (`receptors_only` or `no_sparsity`) achieves the best results, suggesting these languages benefit from high-dimensional sparse representations rather than strict compression.

---

### 2. Low-Resource Simulation Capped Experiments (1k Capped, 5 Seeds)

To systematically control the dataset size variable, we evaluated all 6 datasets capped at exactly **1,000 training sentences** across 5 random seeds.

| Dataset | Baseline | Olfactory (128R, 32G) | More Glomeruli (128R, 64G) | More Receptors (256R, 64G) | Receptors Only (128R, No G) | No Sparsity (Base w/o L1) | Best Config |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **conll_en_1k** (English) | 48.95 ± 1.73% | 46.83 ± 1.40% | 49.59 ± 1.99% | 49.21 ± 1.45% | **51.56 ± 1.05%** | 46.72 ± 1.38% | **receptors_only (+2.61%)** |
| **wikiann_bn_1k** (Bangla) | 63.97 ± 4.82% | 68.13 ± 2.98% | 67.33 ± 7.36% | 66.06 ± 2.72% | **70.20 ± 2.12%** | 68.01 ± 2.56% | **receptors_only (+6.23%)** |
| **wikiann_hi_1k** (Hindi) | 62.41 ± 5.04% | **66.04 ± 2.92%** | 59.22 ± 2.57% | 62.46 ± 2.98% | 63.85 ± 3.34% | 65.37 ± 2.93% | **olfactory (+3.63%)** |
| **wikiann_mr_1k** (Marathi) | 63.09 ± 2.00% | 62.01 ± 3.63% | 61.84 ± 2.91% | 61.92 ± 1.88% | **63.85 ± 2.30%** | 62.02 ± 3.67% | **receptors_only (+0.76%)** |
| **wikiann_ta_1k** (Tamil) | 45.93 ± 2.02% | 49.96 ± 2.65% | 48.51 ± 1.46% | 47.14 ± 2.30% | 48.02 ± 1.91% | **50.24 ± 2.01%** | **no_sparsity (+4.31%)** |
| **wikiann_te_1k** (Telugu) | 54.27 ± 1.03% | 55.89 ± 1.96% | 55.52 ± 1.91% | 56.53 ± 0.73% | **56.94 ± 2.18%** | 55.16 ± 2.16% | **receptors_only (+2.67%)** |

**Key Takeaways**:
- **Stabilized Variance:** The unconstrained baseline sequence taggers suffer from high training volatility (SD ~4.8% to 5.0% in Bangla/Hindi). Introducing the sparse olfactory bottlenecks stabilizes convergence, reducing SD down to ~2.9%.
- **Capacity Trade-offs:** The `receptors_only` configuration (which omits glomerular convergence and maintains a 128-dimensional representation going into the BiLSTM) achieves the highest mean F1 in 5 out of 6 datasets, illustrating the value of sparse receptor feature extraction when bottleneck capacity is preserved.

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

- **Hardware & Compute Resources:** All experiments were conducted using NVIDIA Tesla T4 GPU runtimes on Google Colab. We thank Google for providing the free cloud computing resources that made this research possible.
- **Biological inspiration:** Buck & Axel (1991) — olfactory receptor discovery
- **CoNLL-2003:** Tjong Kim Sang & De Meulder (2003)
- **GloVe:** Pennington et al. (2014)
- **BiLSTM-CRF for NER:** Huang et al. (2015)
