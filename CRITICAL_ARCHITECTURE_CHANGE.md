# Critical Architecture Change - 2026-01-28

## Overview

Per your request, we have revised the model architectures to ensure a **scientifically fair comparison**.

## The New Architecture

### 1. Baseline Model (Updated)
**Before**: `mBERT → Linear → CrossEntropyLoss`  
**Now**: `mBERT → Linear → CRF`

**Why?**
- Adding CRF makes the baseline strictly stronger.
- Matches the structured decoding capability of the Olfactory model.

### 2. Olfactory Model (Updated)
**Before**: `mBERT → Receptors → Glomeruli → BiLSTM → CRF`  
**Now**: `mBERT → Receptors → Glomeruli → Linear → CRF`

**Why?**
- **Removed BiLSTM**: This isolates the contribution of the *Olfactory* layers.
- If we kept BiLSTM, gains could be attributed to "BERT+BiLSTM is better than BERT", which is trivial.
- Now, both models have the same depth: `BERT → [Features] → Linear → CRF`.
- **Hypothesis**: The *Olfactory Features* (Receptor+Glomeruli) are better than *Raw BERT Features* for the CRF.

## Comparison Table

| Component | Modified Baseline | Modified Olfactory |
|:---|:---|:---|
| **Backbone** | Frozen mBERT | Frozen mBERT |
| **Feature Extractor** | Dropout | **Receptor → Glomeruli** |
| **Context** | - | - |
| **Decoder** | **CRF** | **CRF** |
| **Loss** | Neg Log Likelihood | Neg Log Likelihood |

## Expected Impact

1. **Baseline Performance**: Will **increase** (CRF helps NER significantly).
2. **Olfactory Performance**: Might **decrease slightly** (BiLSTM is powerful), but the *comparison* is now much more meaningful.
3. **Training Time**: Slightly faster (no BiLSTM).

## How to Apply

```bash
git pull origin main
```

Then run your experiments as usual. No command line arguments need to change.

```bash
# Example
python src/train_bert.py --dataset wikiann --language hi --experiment olfactory --epochs 5
```

---
**Status**: ✅ Implemented and Pushed
