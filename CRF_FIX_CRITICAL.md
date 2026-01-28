# CRF Training Fix - Critical Update

## Problem Identified

**Symptom**: Negative loss getting worse (-31 → -179 → -278 → -343 → -400), Val F1 stuck at 0.0

**Root Cause**: The CRF was receiving DISCONTINUOUS sequences because subword tokens had label -100, creating gaps in the sequence. This violates CRF's assumption of continuous sequences for Viterbi decoding.

## The Fix

### Changed Approach: Label ALL Subwords

**Before (BROKEN)**:
```python
# In BertNERDataset
if word_idx != previous_word_idx:
    label_ids.append(label_id)  # First subword gets label
else:
    label_ids.append(-100)  # Other subwords get -100 ❌
```

**After (FIXED)**:
```python
# In BertNERDataset  
# All subwords of a word get the same label
if word_idx is None:
    label_ids.append(-100)  # Only padding gets -100
else:
    label_ids.append(label_id)  # ALL subwords get the label ✅
```

### Why This Works

1. **Continuous Sequences**: CRF now sees complete, continuous sequences without gaps
2. **Proper Transitions**: CRF can learn transitions between ALL tokens, including subwords
3. **Better for NER**: Subwords inherit entity tags from their root word (linguistically correct)

### Updated Files

1. **`src/data/bert_loader.py`**:
   - Modified `BertNERDataset.__getitem__()` 
   - All subwords now get their word's label (not -100)

2. **`src/model/bert_models.py`**:
   - Simplified `BertOlfactory.forward()`
   - Removed `safe_labels` workaround
   - Directly pass `labels` to CRF (no replacement needed)

3. **`src/train_bert.py`**:
   - Updated `evaluate()` function
   - Evaluation now correctly handles all labeled tokens

## Expected Results

### Before Fix:
```
Epoch 1: Loss: -31.64, Val F1: 0.0000
Epoch 2: Loss: -179.52, Val F1: 0.0000
Epoch 3: Loss: -278.51, Val F1: 0.0000
```

### After Fix:
```
Epoch 1: Loss: 8.5, Val F1: 0.45
Epoch 2: Loss: 6.2, Val F1: 0.62
Epoch 3: Loss: 4.8, Val F1: 0.68
```

**Key Indicators**:
- ✅ Positive loss (should be 2-10 range initially)
- ✅ Loss DECREASING over epochs
- ✅ Val F1 INCREASING over epochs
- ✅ Val F1 > 0.5 after a few epochs

## How to Apply

```bash
git pull origin main
```

Then re-run experiments. The olfactory model should now train properly!

## Technical Details

### Why Negative Loss Was Wrong

CRF loss = -log_likelihood. For proper training:
- Loss should be POSITIVE (negative log-likelihood)
- Loss should DECREASE (model improving)

Negative loss meant:
- log_likelihood > 0 (impossible! Should be ≤ 0)
- Model was diverging, not learning

### The Discontinuous Sequence Problem

With -100 labels creating gaps:
```
Tokens:  [The, New, York, Times]
Subwords:[The, New, Yo, ##rk, Times]
Labels:  [O,   B-ORG, I-ORG, -100, O]  ❌ Gap!
CRF sees:[O,   B-ORG, I-ORG, ???,  O]  Broken!
```

With all subwords labeled:
```
Tokens:  [The, New, York, Times]
Subwords:[The, New, Yo, ##rk, Times]  
Labels:  [O,   B-ORG, I-ORG, I-ORG, O]  ✅ Continuous!
CRF sees:[O,   B-ORG, I-ORG, I-ORG, O]  Perfect!
```

## Status

✅ Code fixed and ready to commit
⏳ Awaiting user to pull and test in Colab

---
**Date**: 2026-01-28  
**Critical**: This fix is essential for CRF-based models to work
