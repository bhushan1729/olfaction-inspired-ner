# Baseline vs Olfactory Experiments - Notebook Overview

## 📓 Notebook: `baseline_vs_olfactory_experiments.ipynb`

### Structure

```
┌─────────────────────────────────────────────────────────────┐
│                         SETUP (5 cells)                      │
├─────────────────────────────────────────────────────────────┤
│ 1. Mount Google Drive                                       │
│ 2. Clone Repository                                         │
│ 3. Install Dependencies                                     │
│ 4. Check GPU                                                │
│ 5. Configuration (EPOCHS, BATCH_SIZE, LEARNING_RATE)       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   EXPERIMENTS (12 cells)                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  CoNLL-2003 (English)                                       │
│  ├─ 6.  Baseline   (~30 min)                                │
│  └─ 7.  Olfactory  (~40 min)                                │
│                                                              │
│  WikiANN Hindi                                              │
│  ├─ 8.  Baseline   (~30 min)                                │
│  └─ 9.  Olfactory  (~40 min)                                │
│                                                              │
│  WikiANN Marathi                                            │
│  ├─ 10. Baseline   (~30 min)                                │
│  └─ 11. Olfactory  (~40 min)                                │
│                                                              │
│  WikiANN Tamil                                              │
│  ├─ 12. Baseline   (~30 min)                                │
│  └─ 13. Olfactory  (~40 min)                                │
│                                                              │
│  WikiANN Bangla                                             │
│  ├─ 14. Baseline   (~30 min)                                │
│  └─ 15. Olfactory  (~40 min)                                │
│                                                              │
│  WikiANN Telugu                                             │
│  ├─ 16. Baseline   (~30 min)                                │
│  └─ 17. Olfactory  (~40 min)                                │
│                                                              │
│  Total: ~6-8 hours with GPU                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     ANALYSIS (7 cells)                       │
├─────────────────────────────────────────────────────────────┤
│ 18. Quick Results Summary (table)                          │
│ 19. Comprehensive Analysis (runs compare_results.py)       │
│ 20. View Comparison Report (markdown)                      │
│ 21. Visualizations (bar charts, heatmaps)                  │
│ 22. Statistical Significance (t-test, Cohen's d)           │
│ 23. Final Summary (hypothesis validation)                  │
│ 24. Download Results (optional zip download)               │
└─────────────────────────────────────────────────────────────┘
```

## 💾 Google Drive Structure

After running, your Drive will have:

```
📁 olfaction_ner_experiments/
├── 📁 results/
│   ├── 📁 conll2003/
│   │   └── 📁 en/
│   │       ├── 📁 mbert_baseline/
│   │       │   ├── best_model.pt
│   │       │   └── results.json
│   │       └── 📁 mbert_olfactory/
│   │           ├── best_model.pt
│   │           └── results.json
│   └── 📁 wikiann/
│       ├── 📁 hi/ (Hindi)
│       ├── 📁 mr/ (Marathi)
│       ├── 📁 ta/ (Tamil)
│       ├── 📁 bn/ (Bangla)
│       └── 📁 te/ (Telugu)
├── 📁 comparison_analysis/
│   ├── 📄 COMPARISON_REPORT.md
│   ├── 📊 comparison_bars.png
│   ├── 📈 improvement_heatmap.png
│   ├── 📄 comparison_table.csv
│   ├── 📄 entity_analysis.json
│   └── 📄 statistical_tests.json
└── 📄 quick_results_summary.csv
```

## 🎯 Usage Modes

### Mode 1: Run Everything (Batch)
```
Runtime → Run all
```
- Runs all 12 experiments
- Takes ~6-8 hours
- Best for: Getting all results in one go

### Mode 2: Run Selected Datasets
```
Run cells: 1-5 (setup)
Then run: Only cells for datasets you want
Finally: 18-23 (analysis)
```
- Example: Only English (cells 6-7)
- Example: Only Hindi + Marathi (cells 8-11)
- Best for: Testing or focused experiments

### Mode 3: Resume After Disconnect
```
1. Reconnect runtime
2. Run cell 1 (mount Drive)
3. Check what's already done
4. Skip completed cells
5. Continue from next experiment
```
- Best for: Recovering from session timeout

## 📊 What You Get

### Immediate (After Each Experiment)
- ✅ Trained model saved to Drive
- ✅ Metrics (F1, Precision, Recall) saved
- ✅ Per-entity breakdown

### After All Experiments
- ✅ Side-by-side comparison table
- ✅ Statistical significance tests
- ✅ Visualizations (bars + heatmaps)
- ✅ Comprehensive markdown report
- ✅ Hypothesis validation

## ⚙️ Customization

### Quick Test (1 epoch)
```python
# Cell 5
EPOCHS = 1
```

### Run on CPU (slow!)
```python
# Just don't enable GPU
# Warning: ~20x slower
```

### Reduce Memory Usage
```python
# Cell 5
BATCH_SIZE = 8  # or even 4
```

### Different Datasets
Edit cells to change `--language` parameter:
```bash
!python src/train_bert.py \
  --dataset wikiann \
  --language kn \  # Kannada
  --experiment baseline \
  ...
```

## 🔍 Monitoring Progress

Each experiment cell shows:
```
Epoch 1/5: 100%|██████████| 878/878 [05:23<00:00]
Avg Loss: 0.4567
Val F1: 0.8523
Saved Best Model
✓ CoNLL-2003 Baseline completed
```

## ✅ Success Criteria

After Cell 23 runs, you'll see:

**If hypothesis validated:**
```
✅ HYPOTHESIS VALIDATED
The olfactory feature extractor adds value beyond standard mBERT for NER!

Datasets where Olfactory wins: 5/6 (83.3%)
Average Baseline F1:   0.7823 (±0.0456)
Average Olfactory F1:  0.8012 (±0.0432)
Average Improvement:   +0.0189 (±0.0078)
```

**If results mixed:**
```
⚠️ RESULTS MIXED
Further investigation needed.
```

## 🚀 Quick Start Checklist

Before running:
- [ ] Open notebook in Google Colab
- [ ] Enable GPU (T4 recommended)
- [ ] Update Git repo URL in Cell 2
- [ ] Set desired epochs in Cell 5
- [ ] Run Cell 1 to mount Drive

To run all experiments:
- [ ] Runtime → Run all
- [ ] Wait 6-8 hours
- [ ] Check results in Cell 23

To run quick test:
- [ ] Set EPOCHS=1 in Cell 5
- [ ] Run cells 1-7 only
- [ ] Takes ~15-20 minutes

## 📁 Files Created

In your project directory:
1. `baseline_vs_olfactory_experiments.ipynb` - The Colab notebook
2. `COLAB_NOTEBOOK_GUIDE.md` - This guide

Upload the `.ipynb` file to Google Colab and follow the guide!

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| No GPU | Runtime → Change runtime type → GPU |
| Drive not mounted | Re-run Cell 1 |
| CUDA OOM | Reduce BATCH_SIZE in Cell 5 |
| Module not found | Check Cell 2 cloned successfully |
| Session timeout | Remount Drive, skip completed cells |

## 📞 Support

See `COLAB_NOTEBOOK_GUIDE.md` for detailed troubleshooting and tips.

---

**Ready to validate your hypothesis? Upload to Colab and run!** 🚀
