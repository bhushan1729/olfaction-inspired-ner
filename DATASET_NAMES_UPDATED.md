# Dataset Loading - Corrected Names

## ⚠️ Important: Dataset Names Have Changed

The HuggingFace datasets library no longer supports dataset scripts. You must use the full organization/dataset names.

## Correct Dataset Names

### CoNLL-2003 (English)

**OLD (NO LONGER WORKS)**:
```python
load_dataset("conll2003")  # ❌ Scripts not supported without trust_remote_code
```

**NEW (WORKS)**:
```python
load_dataset("tner/conll2003", trust_remote_code=True)  # ✅
# Note: Uses 'tags' field (not 'ner_tags')
```

### WikiANN (Multilingual)

**OLD (NO LONGER WORKS)**:
```python
load_dataset("wikiann", "hi")  # ❌ 404 Not Found
load_dataset("Babelscape/wikineural", "hi")  # ❌ Config not found
```

**NEW (WORKS)**:
```python
load_dataset("unimelb-nlp/wikiann", "hi")  # ✅ Hindi
load_dataset("unimelb-nlp/wikiann", "mr")  # ✅ Marathi
load_dataset("unimelb-nlp/wikiann", "ta")  # ✅ Tamil
load_dataset("unimelb-nlp/wikiann", "bn")  # ✅ Bangla
load_dataset("unimelb-nlp/wikiann", "te")  # ✅ Telugu
```

## Updated Files

The following files have been updated with correct dataset names:

1. **`src/data/bert_loader.py`** - BERT experiments
2. **`src/data/unified_loader.py`** - Universal loader

## Testing

Before running full experiments, test dataset loading:

```bash
python test_dataset_loading.py
```

Or in Colab:
```python
!python test_dataset_loading.py
```

## In train_bert.py

The command-line usage remains the same:

```bash
# CoNLL-2003
python src/train_bert.py --dataset conll2003 --experiment baseline

# Hindi
python src/train_bert.py --dataset wikiann --language hi --experiment baseline
```

The script internally maps `conll2003` → `eriktks/conll2003` and `wikiann` → `unimelb-nlp/wikiann`.

## In Colab Notebook

All experiment cells already use the correct commands. Just pull the latest code:

```python
!git pull origin main
```

## Summary

| Dataset | Old Name (Broken) | New Name (Working) | Field Name |
|---------|------------------|-------------------|------------|
| CoNLL-2003 | `conll2003` (no trust) | `tner/conll2003` + trust | `tags` |
| WikiANN Hindi | `wikiann` or `Babelscape/wikineural` | `unimelb-nlp/wikiann` | `ner_tags` |
| WikiANN Others | Same | Same pattern | `ner_tags` |

**Key Points**:
- CoNLL-2003: Uses field name `tags` (not `ner_tags`)
- WikiANN: Uses field name `ner_tags`
- Both require `trust_remote_code=True` or proper organization prefix

---

**Last Updated**: 2026-01-28  
**Reason**: HuggingFace datasets library deprecated dataset scripts
