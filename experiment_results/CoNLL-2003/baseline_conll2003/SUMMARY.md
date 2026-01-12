# baseline_conll2003

**Dataset**: CoNLL-2003
**Model**: baseline
**Date**: 2026-01-12 10:31:01

## Configuration

```yaml
embed_dim: 300
lstm_hidden: 256
dropout: 0.5
learning_rate: 0.001
batch_size: 32
```

## Results

### Test Performance

| Metric | Score |
|--------|-------|
| **F1** | **0.7730** |
| Precision | 0.8008 |
| Recall | 0.7472 |

### Per-Entity F1 Scores

| Entity | F1 |
|--------|-----|
| LOC | 0.8528 |
| MISC | 0.6784 |
| ORG | 0.7340 |
| PER | 0.7697 |

### Training

- Epochs trained: 1
- Best validation F1: 0.8509
