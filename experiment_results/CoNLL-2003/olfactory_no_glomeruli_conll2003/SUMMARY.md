# olfactory_no_glomeruli_conll2003

**Dataset**: CoNLL-2003
**Model**: olfactory_no_glomeruli
**Date**: 2026-01-12 10:31:01

## Configuration

```yaml
embed_dim: 300
num_receptors: 128
num_glomeruli: 32
receptor_activation: relu
lstm_hidden: 256
dropout: 0.5
learning_rate: 0.001
batch_size: 32
```

## Results

### Test Performance

| Metric | Score |
|--------|-------|
| **F1** | **0.7353** |
| Precision | 0.7878 |
| Recall | 0.6888 |

### Per-Entity F1 Scores

| Entity | F1 |
|--------|-----|
| LOC | 0.8198 |
| MISC | 0.6346 |
| ORG | 0.6863 |
| PER | 0.7212 |
