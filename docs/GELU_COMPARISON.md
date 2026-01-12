# ReLU vs GELU Comparison

**Experiment Date**: January 12, 2026

---

## Performance Comparison

| Metric | Baseline | ReLU | GELU | Δ (GELU-ReLU) |
|--------|----------|------|------|---------------|
| **Test F1** | **77.30%** | 72.56% | **73.06%** | **+0.50%** ✅ |
| Precision | 80.08% | 76.79% | 78.73% | +1.94% |
| Recall | 74.72% | 68.77% | 68.15% | -0.62% |

## Per-Entity F1 Scores

| Entity | Baseline | ReLU | GELU | Δ (GELU-ReLU) |
|--------|----------|------|------|---------------|
| LOC | 85.28% | 81.38% | 80.82% | -0.56% |
| **MISC** | 67.84% | 62.27% | **67.61%** | **+5.34%** ✨ |
| ORG | 73.40% | 68.99% | 70.67% | +1.68% |
| PER | 76.97% | 71.20% | 69.14% | -2.06% |

---

## Visualizations

### Receptor Heatmap Comparison

**GELU Shows Clear Specialization**:

![GELU Receptor Heatmap](gelu_heatmap.png)

### t-SNE Clustering

**Better MISC Separation with GELU**:

![GELU t-SNE](gelu_tsne.png)

---

## Conclusion

✅ **GELU improves olfactory model** (+0.5% F1 over ReLU)  
✅ **Strong gains on MISC** (+5.34%) - matches baseline on this category!  
⚠️ **Baseline still best overall** (77.30% vs 73.06%)  
✅ **More biologically plausible** - smooth, stochastic activation  
✅ **Recommended for olfactory variants** - better than ReLU

---

**Full analysis**: See detailed comparison document for complete analysis and paper recommendations.
