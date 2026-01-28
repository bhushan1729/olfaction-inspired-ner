# Google Colab Notebook Usage Guide

## Quick Start

### 1. Upload to Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Upload `baseline_vs_olfactory_experiments.ipynb`

**OR**

1. Upload the notebook to your Google Drive
2. Right-click → Open with → Google Colaboratory

### 2. Enable GPU (Important!)

1. Click **Runtime → Change runtime type**
2. Set **Hardware accelerator** to **GPU** (T4 or better)
3. Click **Save**

### 3. Update Repository URL

In the **Setup** section (Cell 2), replace:
```python
!git clone https://github.com/YOUR_USERNAME/olfaction-inspired-ner.git
```

With your actual repository URL:
```python
!git clone https://github.com/bhushan1729/olfaction-inspired-ner.git
```

### 4. Run Cells

**Option A: Run All (Recommended for batch processing)**
- Click **Runtime → Run all**
- This will run all 12 experiments sequentially (takes 6-8 hours)

**Option B: Run Selected Experiments**
- Run setup cells first (cells 1-5)
- Then run only the experiment cells you want
- For example, run only CoNLL-2003 experiments (cells 6-7)

## Notebook Structure

### Setup Cells (1-5)
1. **Mount Google Drive** - Connects to your Drive
2. **Clone Repository** - Downloads the code
3. **Install Dependencies** - Installs required packages
4. **Check GPU** - Verifies GPU is available
5. **Configuration** - Sets hyperparameters

### Experiment Cells (6-17)

Each dataset has 2 cells:

#### CoNLL-2003 (English)
- Cell 6: Baseline
- Cell 7: Olfactory

#### Hindi
- Cell 8: Baseline  
- Cell 9: Olfactory

#### Marathi
- Cell 10: Baseline
- Cell 11: Olfactory

#### Tamil
- Cell 12: Baseline
- Cell 13: Olfactory

#### Bangla
- Cell 14: Baseline
- Cell 15: Olfactory

#### Telugu
- Cell 16: Baseline
- Cell 17: Olfactory

### Analysis Cells (18-24)
- Cell 18: Quick results summary
- Cell 19: Comprehensive analysis
- Cell 20: View comparison report
- Cell 21: Visualizations
- Cell 22: Statistical tests
- Cell 23: Final summary
- Cell 24: Download results (optional)

## Google Drive Storage

All results are automatically saved to:
```
/content/drive/MyDrive/olfaction_ner_experiments/
├── results/
│   ├── conll2003/
│   ├── wikiann/ (multiple language subdirs)
│   └── ...
├── comparison_analysis/
│   ├── COMPARISON_REPORT.md
│   ├── comparison_bars.png
│   ├── improvement_heatmap.png
│   └── ...
└── quick_results_summary.csv
```

## Customization

### Change Hyperparameters

In Cell 5 (Configuration), modify:

```python
EPOCHS = 5          # Number of training epochs
BATCH_SIZE = 16     # Batch size (reduce if OOM)
LEARNING_RATE = 2e-5  # Learning rate
```

### Run Subset of Experiments

Just run setup cells (1-5), then run only the experiment cells you want.

For example, to run only English experiments:
1. Run cells 1-5 (setup)
2. Run cells 6-7 (CoNLL-2003 baseline + olfactory)
3. Run cells 18-23 (analysis)

### Adjust for Memory Constraints

If you get "CUDA out of memory" errors:

```python
BATCH_SIZE = 8   # Reduce batch size
```

## Expected Runtime

### With GPU (T4)
- **Per experiment**: ~30-40 minutes
- **All 12 experiments**: ~6-8 hours
- **Analysis**: ~5 minutes

### Without GPU (CPU only)
- **Per experiment**: ~2-3 hours
- **Not recommended** for full run

## Tips for Long Runs

### 1. Monitor Progress

Each cell prints progress:
```
Epoch 1/5: 100%|██████| 878/878 [05:23<00:00]
Avg Loss: 0.4567
Val F1: 0.8523
```

### 2. Save Checkpoints

Results are saved to Google Drive after EACH experiment, so you can:
- Stop at any time
- Resume later from where you left off
- Run experiments in multiple sessions

### 3. Session Disconnects

If Colab disconnects:
1. Reconnect to runtime
2. Re-run mount Drive cell (Cell 1)
3. Skip completed experiments (check Drive for saved results)
4. Continue from next experiment

### 4. Check What's Done

```python
# Add this cell to check completed experiments
import os
results_path = "/content/drive/MyDrive/olfaction_ner_experiments/results"
for root, dirs, files in os.walk(results_path):
    if "results.json" in files:
        print(f"✓ {root}")
```

## Troubleshooting

### Issue: "No module named 'src'"

**Solution**: Make sure you ran the clone cell and the `%cd` command executed successfully.

### Issue: "Drive not mounted"

**Solution**: Run Cell 1 again and authorize Google Drive access.

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size in Cell 5:
```python
BATCH_SIZE = 8  # or even 4
```

### Issue: "Dataset download stuck"

**Solution**: 
- First run downloads datasets (~100MB)
- Be patient, it can take 5-10 minutes
- Subsequent runs use cached data

### Issue: "Module 'compare_results' not found"

**Solution**: Make sure you cloned the repository correctly and all files are present:
```python
!ls -la /content/olfaction-inspired-ner/
```

## After Experiments Complete

### 1. View Results in Drive

Navigate to:
```
Google Drive → olfaction_ner_experiments → comparison_analysis → COMPARISON_REPORT.md
```

### 2. Share Results

The Google Drive folder can be shared:
1. Right-click folder in Drive
2. Get link → Share with collaborators

### 3. Download Everything

Run the last cell (Cell 24) to download a zip file with all results.

## Example: Quick Test Run

Want to test everything works? Modify Cell 5:

```python
EPOCHS = 1  # Just 1 epoch for testing
BATCH_SIZE = 16
```

Then run just CoNLL-2003:
- Cells 1-5 (setup)
- Cells 6-7 (CoNLL baseline + olfactory)
- Cells 18, 23 (quick summary)

This takes ~15-20 minutes total and validates everything works.

## Support

If you encounter issues:

1. Check the error message carefully
2. Make sure GPU is enabled
3. Verify Google Drive is mounted
4. Ensure repository cloned successfully
5. Check you have enough Drive storage (~2GB needed)

## Success Checklist

Before running all experiments:

- [ ] GPU is enabled (Runtime → Change runtime type)
- [ ] Google Drive mounted successfully (Cell 1)
- [ ] Repository cloned (Cell 2)
- [ ] Dependencies installed (Cell 3)
- [ ] GPU detected (Cell 4 shows GPU name)
- [ ] Configuration set (Cell 5)
- [ ] Updated GitHub URL (Cell 2)

Ready to go! 🚀
