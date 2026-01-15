# Experiment Execution Summary

## Status: ✅ RUNNING

All experiments are currently executing in the background.

**Process ID**: 1671690  
**Started**: 2026-01-15 07:00  
**Log File**: `/home/datauser/olfaction_inspired_ner/experiment_run.log`

## Experiments Queue (11 Total)

All experiments will run sequentially on CPU:

1. ✅ baseline
2. ✅ olfactory_full  
3. ✅ olfactory_no_sparse
4. ✅ olfactory_no_glomeruli
5. ✅ exp3_more_receptors
6. ✅ exp4_more_glomeruli
7. ✅ exp5_larger_lstm
8. ✅ exp6_lower_dropout
9. ✅ exp7_larger_batch
10. ✅ exp8_strong_reg
11. ✅ activation_gelu

## Monitoring

### Check Progress

```bash
cd /home/datauser/olfaction_inspired_ner
python monitor_experiments.py
```

### View Live Log

```bash
tail -f experiment_run.log
```

### Check Process

```bash
ps aux | grep "python run_all_experiments_cpu.py"
```

## What Happens Next

The script will automatically:

1. **Train each model** (30-60 min per experiment)
2. **Save best models** to `results/<experiment>/best_model.pt`
3. **Save results** to `results/<experiment>/results.json`
4. **Generate training curves** for each experiment
5. **Create comparison plots** across all experiments
6. **Save metadata** to `experiment_results/CoNLL-2003/`

## After Completion

Once all experiments finish, run:

```bash
python generate_heatmaps.py
```

This will generate:
- Receptor activation heatmaps
- Glomeruli activation heatmaps
- Visualization comparisons

## Files Created

### Scripts
- `run_all_experiments_cpu.py` - Main experiment runner
- `generate_heatmaps.py` - Heatmap generation
- `monitor_experiments.py` - Progress monitoring

### Outputs
- `results/` - Model checkpoints and results
- `experiment_results/CoNLL-2003/` - Organized experiment data
- `visualizations/` - Heatmaps and comparison plots (after heatmap generation)
- `experiment_run.log` - Execution log

## Estimated Completion

**Total Runtime**: 6-11 hours  
**Expected Completion**: ~2026-01-15 13:00-18:00

## Notes

- ✅ All experiments run on CPU only (no GPU/Ray cluster)
- ✅ Automatically resumes if interrupted
- ✅ Skips already completed experiments
- ✅ All results and plots saved automatically
- ✅ No manual intervention required

The system is fully automated and will complete all experiments while you sleep! 🌙
