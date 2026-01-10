# Olfaction-Inspired NER: Project Walkthrough

## What We Built

A complete implementation to test whether **olfactory-style combinatorial coding** provides useful inductive biases for Named Entity Recognition.

### Core Hypothesis

Just as the olfactory system recognizes smells through patterns of specialized receptor activations (not single detectors), entities in text should be recognized through **combinatorial patterns of micro-features** rather than monolithic representations.

---

## Project Components

### 1. Data Loading (`src/data/dataset.py`)

✅ **What it does**:
- Downloads CoNLL-2003 NER dataset automatically
- Converts to PyTorch DataLoader format
- Builds vocabularies and handles IOB2 tagging
- Loads GloVe embeddings (optional but recommended)

📊 **Dataset**: CoNLL-2003 English
- 4 entity types: PER (person), LOC (location), ORG (organization), MISC
- ~14k training sentences

### 2. Olfactory Layers (`src/model/layers.py`)

✅ **ReceptorLayer**:
- **Biological inspiration**: Olfactory receptors are highly specialized detectors
- **Implementation**: 128-256 linear units with ReLU activation
- **Purpose**: Learn specialized micro-features (e.g., capitalization, suffixes, patterns)
- **Regularization**: Diversity loss to prevent redundant receptors

✅ **GlomerularLayer**:
- **Biological inspiration**: Multiple neurons with same receptor converge to one glomerulus
- **Implementation**: Learnable aggregation (many receptors → fewer glomeruli)
- **Purpose**: Denoising and feature abstraction through convergence

### 3. Models

✅ **OlfactoryNER** (`src/model/olfactory_ner.py`):
```
Embeddings → Receptors → Glomeruli → BiLSTM → CRF → Labels
```

Features:
- Configurable ablations (can disable receptors or glomeruli)
- Diversity and sparsity regularization
- Compatible API with baseline

✅ **BaselineNER** (`src/model/baseline.py`):
```
Embeddings → BiLSTM → CRF → Labels
```

Standard strong baseline for controlled comparison.

### 4. Training Pipeline (`src/train.py`)

✅ Features:
- Early stopping with patience
- Learning rate scheduling
- Gradient clipping
- TensorBoard logging
- Automatic checkpointing
- Supports all model variants

### 5. Evaluation (`src/training/evaluate.py`)

✅ Metrics:
- Entity-level F1, precision, recall (using `seqeval`)
- Per-entity-type scores
- Proper handling of IOB2 format

### 6. Receptor Analysis (`src/analysis/visualize.py`)

✅ **Critical for paper**:
1. **Receptor activation heatmap** - shows specialization by entity type
2. **Top activating tokens per receptor** - interpretability
3. **t-SNE of glomerular representations** - clustering visualization
4. **Specialization metrics** - sparsity, diversity, variance

### 7. Experiment Configs (`config/experiments.yaml`)

✅ Four experiments:
- `baseline`: BiLSTM-CRF control
- `olfactory_full`: Complete model with regularization
- `olfactory_no_sparse`: Ablation without sparsity loss
- `olfactory_no_glomeruli`: Ablation without convergence

### 8. Google Colab Notebook (`notebooks/olfaction_ner_colab.ipynb`)

✅ **For GPU training**:
- Step-by-step instructions
- Runs all 4 experiments
- Generates all visualizations
- Downloads results
- **Estimated runtime**: 2-3 hours on T4 GPU

---

## How to Run

### Option 1: Google Colab (Recommended for First Run)

1. **Upload notebook to Colab**:
   - Open `notebooks/olfaction_ner_colab.ipynb` in Google Colab
   
2. **Enable GPU**:
   - Runtime → Change runtime type → Hardware accelerator: GPU
   
3. **Upload source code**:
   - Zip the `src/` and `config/` directories
   - Upload to Colab
   - OR: Push to GitHub and clone in notebook
   
4. **Run all cells**:
   - The notebook handles everything automatically
   - Downloads data, trains models, generates visualizations
   
5. **Download results**:
   - Last cell creates `olfaction_ner_results.zip`
   - Contains all metrics and figures

**Why Colab?**
- Free GPU access
- No local setup required
- Fast experimentation (~2-3 hours vs 6-8 hours on CPU)

### Option 2: Local Machine

#### Quick Start (Automated)

```bash
cd /home/datauser/olfaction_inspired_ner
./run_experiments.sh
```

This script:
1. Installs dependencies
2. Optionally downloads GloVe
3. Runs all 4 experiments sequentially
4. Generates analysis and comparisons
5. Takes ~2 hours on GPU, 6-8 hours on CPU

#### Manual Run (Individual Experiments)

```bash
# Install dependencies
pip install -r requirements.txt

# Train baseline
python src/train.py --config config/experiments.yaml --experiment baseline

# Train olfactory model
python src/train.py --config config/experiments.yaml --experiment olfactory_full

# Analyze receptors (after training olfactory_full)
python -c "
import torch
from src.data.dataset import prepare_data
from src.model.olfactory_ner import create_olfactory_ner
from src.analysis.visualize import analyze_receptor_activations
import yaml

with open('config/experiments.yaml', 'r') as f:
    config = yaml.safe_load(f)

exp_config = config['olfactory_full']
exp_config.update(config.get('data', {}))

_, _, test_loader, vocab_info = prepare_data()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('results/olfactory_full/best_model.pt', map_location=device)

vocab_size = len(vocab_info['word2idx'])
num_tags = len(vocab_info['label2idx'])

model = create_olfactory_ner(vocab_size, num_tags, exp_config)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

analyze_receptor_activations(model, test_loader, vocab_info, device, save_dir='./analysis_results')
"
```

---

## Expected Results

### Success Criteria (Any ONE is sufficient)

✅ **Interpretability**: Receptors show clear specialization patterns
- Heatmap shows different receptors activate for different entity types
- Top tokens per receptor make semantic sense

✅ **Comparable Performance**: F1 within 1 point of baseline
- Not trying to beat SOTA
- Demonstrates architectural viability

✅ **Ablation Validation**: Removing components degrades performance
- No sparsity → worse receptor specialization
- No glomeruli → higher variance / noise sensitivity

✅ **Efficiency**: Similar F1 with fewer parameters
- Olfactory model is more parameter-efficient

### What to Check

1. **`results/*/results.json`**:
   ```json
   {
     "test": {
       "f1": 0.89,
       "precision": 0.90,
       "recall": 0.88,
       "per_entity": {
         "PER": 0.92,
         "LOC": 0.89,
         "ORG": 0.85,
         "MISC": 0.78
       }
     }
   }
   ```

2. **`analysis_results/receptor_heatmap.png`**:
   - Should show distinct activation patterns per entity type
   - Not all receptors should fire equally

3. **`analysis_results/receptor_analysis.json`**:
   ```json
   {
     "sparsity": 0.15,  // ~15% of receptors active (good)
     "avg_activation": 0.42,
     "receptor_interpretations": {
       "0": [
         {"token": "Ltd", "activation": 0.89},
         {"token": "Inc", "activation": 0.85},
         ...
       ]
     }
   }
   ```

4. **`comparison/model_comparison.png`**:
   - Olfactory model should be competitive
   - Ablations should show degradation

---

## Interpreting Results

### ✅ Strong Evidence (Publish!)

- Receptors show clear specialization (different entities activate different receptors)
- F1 comparable to baseline (within 1 point)
- Ablations degrade performance
- t-SNE shows entity clustering

**Action**: Write paper emphasizing interpretability and inductive bias

### ⚠️ Mixed Results (Tune & Iterate)

- F1 comparable but receptors not interpretable
- OR receptors interpretable but F1 significantly worse (>2 points)

**Action**: Adjust hyperparameters
- Increase `lambda_diverse` (0.01 → 0.05) for more specialization
- Try different `num_receptors` (64, 128, 256)
- Try different `num_glomeruli` (16, 32, 64)

### ❌ No Advantage (Pivot)

- F1 worse than baseline by >3 points
- No interpretable patterns
- Ablations don't matter

**Action**: Re-examine hypothesis or try different architecture

---

## Next Steps

### If Results Are Promising

#### Extend Experiments (1-2 weeks)

1. **Low-Resource Setting**:
   - Train on 10%, 20%, 50% of data
   - Test if olfactory coding helps with limited data

2. **Noise Robustness**:
   - Add 10%, 20% label noise during training
   - Test on clean data
   - Hypothesis: Glomerular aggregation provides denoising

3. **Cross-Domain Transfer**:
   - Train on CoNLL-2003
   - Test on OntoNotes or WikiAnn
   - Test if specialized receptors transfer better

4. **Multilingual**:
   - Extend to Hindi/Tamil (WikiAnn)
   - Test language-agnostic receptor patterns

#### Write Paper

Framework from `starting.md` discussion:

**Title**: "Olfactory Coding for Named Entity Recognition: A Combinatorial Feature Aggregation Approach"

**Sections**:
1. Introduction (emphasize inductive bias, not biology mimicry)
2. Background (brief olfactory system overview)
3. Architecture (receptors, glomeruli, contextual encoding)
4. Experiments (CoNLL-2003, ablations)
5. Analysis (receptor visualizations - KEY contribution)
6. Discussion (when/why this helps)

**Target Venues**:
- ACL/EMNLP Findings
- Cognitive Modeling workshops
- Neuro-symbolic AI workshops
- arXiv preprint

### If Results Need Work

1. **Hyperparameter tuning**:
   - Grid search over `num_receptors`, `lambda_diverse`
   - Try different aggregation strategies

2. **Auxiliary tasks**:
   - Add explicit supervision for receptors (e.g., POS tagging)
   - Multi-task learning

3. **Different regularization**:
   - Try orthogonality constraints
   - Experiment with dropout strategies

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/train.py` | Main training script |
| `src/model/olfactory_ner.py` | Olfactory NER model |
| `src/model/baseline.py` | Baseline model |
| `src/model/layers.py` | Receptor & glomerular layers |
| `src/data/dataset.py` | Data loading |
| `src/analysis/visualize.py` | Receptor analysis |
| `config/experiments.yaml` | All experiment configs |
| `notebooks/olfaction_ner_colab.ipynb` | Colab notebook |
| `run_experiments.sh` | Automated runner |

---

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in `config/experiments.yaml` (32 → 16)
- Reduce `num_receptors` (128 → 64)

### Poor Performance
- Check if GloVe embeddings loaded (should see "Found embeddings for X%")
- Verify data downloaded correctly (`data/raw/train.txt` exists)
- Try increasing `max_epochs` (50 → 100)

### Receptors Not Specializing
- Increase `lambda_diverse` (0.01 → 0.05)
- Check sparsity (should be 10-20%)
- Try more receptors (128 → 256)

---

## Summary

You now have:

✅ Complete implementation of olfactory-inspired NER  
✅ Baseline for controlled comparison  
✅ 4 experiments (baseline + 3 variants)  
✅ Automated training pipeline  
✅ Comprehensive analysis tools  
✅ Google Colab notebook for GPU access  
✅ Clear success criteria  

**Estimated Time to First Results**: 2-3 hours on Colab GPU

**Decision Point**: After seeing results, decide whether to:
- Extend to more experiments (low-resource, cross-domain)
- Write paper
- Adjust approach

**This is designed as a minimal but rigorous test of your hypothesis** - not SOTA chasing, but demonstrating a novel inductive bias through interpretability and controlled experiments.

Good luck! 🧪🧠
