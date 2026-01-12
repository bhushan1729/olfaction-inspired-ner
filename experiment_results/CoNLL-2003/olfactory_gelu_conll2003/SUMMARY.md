# olfactory_gelu_conll2003

**Dataset**: CoNLL-2003
**Model**: olfactory_gelu
**Date**: 2026-01-12 10:31:01

## Configuration

```yaml
embed_dim: 300
num_receptors: 128
num_glomeruli: 32
receptor_activation: gelu
lstm_hidden: 256
dropout: 0.5
learning_rate: 0.001
batch_size: 32
```

## Results

### Test Performance

| Metric | Score |
|--------|-------|
| **F1** | **0.7306** |
| Precision | 0.7873 |
| Recall | 0.6815 |

### Per-Entity F1 Scores

| Entity | F1 |
|--------|-----|
| LOC | 0.8082 |
| MISC | 0.6761 |
| ORG | 0.7067 |
| PER | 0.6914 |

### Training

- Epochs trained: 1
- Best validation F1: 0.7306
