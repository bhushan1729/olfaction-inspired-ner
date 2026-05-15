# Olfactory-Inspired Sparse Combinatorial Coding for Low-Resource Named Entity Recognition

## Abstract

Named Entity Recognition (NER) in low-resource languages suffers from limited supervision and a lack of high-quality pretrained embeddings. Biological olfaction, which relies on sparse combinatorial coding through receptor and glomerular organization, offers a compelling paradigm for learning robust representations under uncertainty. In this paper, we introduce a receptor-glomerular bottleneck—a loosely inspired olfactory architecture—between standard token embeddings and a BiLSTM-CRF sequence model. We evaluate our architecture across six multilingual datasets encompassing diverse resource regimes. Our results demonstrate that this structured inductive bias yields improvements on four out of six datasets, with the strongest gain observed in the ultra-low-resource Telugu setting (+9.2% F1). Conversely, high-resource settings like English and Bangla do not benefit from this bottleneck. Furthermore, we show that sparse specialization emerges naturally within the receptor layer, mirroring the biological properties of combinatorial coding. We conclude that structured sparse coding is a useful inductive bias specifically in low-resource regimes where data is scarce.

---

## 1. Introduction

Named Entity Recognition (NER) in low-resource languages is severely constrained by sparse supervision and the unavailability of rich pretrained embeddings. While high-capacity models—such as transformers or those reliant on dense contextual embeddings—excel in data-rich environments, they struggle when supervision is scarce and languages exhibit high morphological complexity.

Biological olfaction offers an intriguing alternative. The mammalian olfactory system detects odors via receptors that respond weakly to multiple odorants, which then converge onto glomeruli to aggregate signals. This many-to-many mapping yields sparse combinatorial representations that are both compositional and robust to noise.

We hypothesize that sparse combinatorial coding may provide a useful inductive bias for low-resource NER. To test this, we introduce an exploratory architecture that inserts a receptor-glomerular bottleneck into a standard BiLSTM-CRF model. 

Our contributions are as follows:
1. We introduce a receptor–glomerular bottleneck architecture for NER.
2. We evaluate this architecture across 6 multilingual datasets with varying resource levels.
3. We show strongest performance gains in very low-resource settings, validating our inductive bias hypothesis.
4. We demonstrate the natural emergence of sparse receptor specialization.
5. We analyze the conditions under which this inductive bias helps and fails.

---

## 2. Related Work

### 2.1 Named Entity Recognition
NER is traditionally modeled as sequence labeling. Standard architectures employ BiLSTM-CRF (Huang et al., 2015) and transformer-based methods (Devlin et al., 2019; Lample et al., 2016). Multilingual and low-resource NER typically focuses on cross-lingual transfer or data augmentation, as high-capacity models rely heavily on large-scale supervision. Recent low-resource NER work has also explored cross-lingual transfer from related languages (Sunna et al., 2023) and meta-learning for few-shot scenarios (Jia et al., 2021), emphasizing that inductive bias is a key component to fast generalization from limited examples.

### 2.2 Sparse Representations
Sparse coding, mixture-of-experts (MoE), and structured bottlenecks are widely used for feature disentanglement and capacity control. Classic neuroscience research has shown that enforcing sparsity yields interpretable codes, such as sparse coding of natural images producing edge detectors (Olshausen & Field, 1996). In deep learning, Shazeer et al. (2017) introduced a sparsely-gated MoE layer that routes inputs to a few expert sub-networks, echoing our use of receptors as fixed sparse experts. Furthermore, Yang et al. (2025) proposed a Structured Information Bottleneck to preserve relevant information under compression. Our work shares similarities with these methods by enforcing sparse activation, but it specifically targets a combinatorial feature aggregation step loosely inspired by olfactory wiring.

### 2.3 Neuroscience-Inspired AI
AI has frequently drawn from neuroscience, including attention mechanisms (cognition), predictive coding, and hippocampal memory systems. Previous olfactory computation literature has explored robustness and associative learning. A striking example of artificial systems converging on biological structures is the work by Wang et al. (2021), who demonstrated that a neural network trained on an odor classification task spontaneously developed a receptor-glomeruli architecture mirroring biological olfaction. This supports the notion that sparse, combinatorial layers can emerge naturally under pressure to compress features. We emphasize that our architecture is an abstract computational analogy to olfactory processing, not a biological simulation.

---

## 3. Biological Motivation

### 3.1 Olfactory Pipeline
In biological olfaction, odor molecules bind to olfactory receptor neurons (ORNs). These ORNs project to glomeruli in the olfactory bulb, which then transmit signals to the higher cortex.

### 3.2 Key Computational Properties
- **Sparse activation:** Only a subset of receptors fires for a given stimulus.
- **Combinatorial coding:** Meaning emerges from patterns across multiple receptors, not single units.
- **Robustness:** Convergence of many noisy neurons to fewer glomeruli tolerates noise.
- **Specialization:** Different receptors specialize in distinct molecular features.

### 3.3 Mapping to NLP
| Olfactory System | NER Model |
| --- | --- |
| Odor molecules | Token embeddings |
| Receptors | Sparse detectors |
| Glomeruli | Feature aggregators |
| Cortex | BiLSTM |

This mapping is an abstract computational analogy, not a biological simulation.

---

## 4. Methodology

### 4.1 Baseline Architecture
Our baseline is a standard sequence tagger: Embedding → BiLSTM → CRF.

### 4.2 Olfactory Architecture
The olfactory-enhanced architecture modifies the baseline by inserting a bottleneck:
Embedding → Receptor Layer → Glomerular Layer → BiLSTM → CRF.

**Receptor Layer:**
This layer comprises sparse nonlinear projections acting as weak feature detectors. Given an input embedding $x_t \in \mathbb{R}^d$, the receptor activation is:
$r_i(x_t) = \sigma(W_i x_t + b_i)$
where $\sigma$ is an activation function like ReLU to enforce sparsity.

**Glomerular Layer:**
Receptors aggregate their signals into a smaller number of glomeruli, acting as feature pooling and compression:
$g_j = \sum_{i \in \mathcal{G}_j} A_{ji} r_i$

**Sparsity Regularization:**
We optionally apply a sparsity penalty and a diversity loss to encourage sparse activation and prevent redundant detectors:
$L = L_{NER} + \lambda_{sparse} L_{sparse} + \lambda_{diverse} L_{diverse}$

### 4.3 Receptor Specialization Index (RSI)
To quantify interpretability, we compute a Receptor Specialization Index (RSI) that measures activation concentration and entity-conditioned activation, capturing how well individual receptors tune to specific named entities.

---

## 5. Experimental Setup

### 5.1 Datasets
We evaluate on CoNLL-2003 (English) and five languages from WikiANN (Marathi, Hindi, Tamil, Bangla, Telugu).

| Dataset | Language | Train Size | Resource Level | Embeddings |
| --- | --- | --- | --- | --- |
| CoNLL-2003 | English | ~14k | High | GloVe |
| WikiANN | Bangla | 10k | Higher | Random |
| WikiANN | Tamil | 15k | Low | Random |
| WikiANN | Hindi | 5k | Low | Random |
| WikiANN | Marathi | 5k | Low | Random |
| WikiANN | Telugu | 1k | Ultra-Low | Random |

### 5.2 Configurations
Models were trained with varying receptor counts (128, 256) and glomeruli counts (32, 64, 128). We optimized using standard hyperparameters and report precision, recall, and F1 score with CRF Viterbi decoding. Average results are reported from comprehensive testing.

---

## 6. Results

### 6.1 Main Results
We summarize the F1 scores across datasets. The olfactory architecture improves F1 on 4 out of 6 datasets, particularly in resource-constrained regimes.

- **Telugu (Ultra-Low):** +9.2% F1 (0.5038 to 0.5955)
- **Marathi (Low):** +1.3% F1 (0.7881 to 0.8010)
- **Hindi (Low):** +0.7% F1 (0.8367 to 0.8437)
- **Tamil (Low):** +0.3% F1 (0.7930 to 0.7962)
- **Bangla (Higher):** -0.4% F1
- **English (High):** -3.3% F1

### 6.2 Resource-Level Analysis
The central finding is the asymmetry of performance gains across resource levels. When supervision is scarce (e.g., Telugu with 1k sentences and no pretrained embeddings), the sparse combinatorial bottleneck provides a vital inductive bias (+9.2%). Conversely, in data-rich environments with high-quality embeddings (CoNLL-2003), the bottleneck acts as a capacity constraint, degrading performance (-3.3%).

### 6.3 Receptor Specialization
Analysis of receptor activations reveals emergent structured specialization. Receptors demonstrate sparsity ranging from 20% to 37% and high Receptor Selectivity Index (RSI) values (0.44–0.83). Receptors specifically trigger on morphological cues, suffixes, and contextual hints associated with named entities.

### 6.4 Failure Cases
The olfactory approach fails on English and Bangla. In English, GloVe embeddings already capture rich entity-relevant features; passing them through a sparse bottleneck discards information. On Bangla, abundant training data renders the structured prior unnecessary, highlighting that the bottleneck can lead to over-regularization.

---

## 7. Discussion

### 7.1 Key Insight
Sparse bottlenecks are effective under uncertainty—specifically when embeddings are weak or supervision is limited. They behave as a structured prior that encourages feature disentanglement and robust aggregation. However, they constrain model capacity in rich-data settings.

### 7.2 Connect to Inductive Biases
The receptor-glomerular setup operates as a structured prior and regularization mechanism, compelling the network to disentangle noisy inputs into a combinatorial array of distinct features before feeding them into the sequence model.

### 7.3 Limitations
This architecture is an exploratory ML study and does not claim state-of-the-art performance against large transformer models. The scale of the architecture is limited, evaluated primarily with BiLSTM-CRF on a constrained set of languages without incorporating modern transformer architectures yet.

---

## 8. Conclusion

We introduced an olfactory-inspired architecture for NER, utilizing a receptor-glomerular bottleneck. Our empirical evaluation confirms that sparse combinatorial coding provides an effective inductive bias for low-resource NER, demonstrating significant improvements (+9.2% F1) in ultra-low-resource settings such as Telugu. Specialization naturally emerges within the receptor layer, highlighting the utility of biologically inspired sparse coding. Future work could explore integrating these bottlenecks into transformer architectures, applying it to multilingual transfer, and investigating adaptive sparsity.

---

## References

- Buck, L., & Axel, R. (1991). A novel multigene family may encode odorant receptors: a molecular basis for odor recognition. *Cell*, 65(1), 175-187.
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL*.
- Huang, Z., Xu, W., & Yu, K. (2015). Bidirectional LSTM-CRF Models for Sequence Tagging. *arXiv preprint arXiv:1508.01991*.
- Jia, et al. (2021). Meta-Learning for Few-Shot Named Entity Recognition. *MetaNLP*.
- Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016). Neural Architectures for Named Entity Recognition. *NAACL*.
- Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature*.
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. *EMNLP*.
- Shazeer, N., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR*.
- Sunna, et al. (2023). Named Entity Recognition for Low-Resource Languages - Profiting from Language Families. *BSNLP*.
- Tjong Kim Sang, E. F., & De Meulder, F. (2003). Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition. *CoNLL*.
- Wang, P. Y., Sun, Y., Axel, R., Abbott, L. F., & Yang, G. R. (2021). Evolving the olfactory system with machine learning. *Neuron*, 109(24), 3879-3892.
- Yang, et al. (2025). Structured IB: Improving Information Bottleneck with Structured Feature Learning. *AAAI*.


## Appendix

### Receptor Selectivity Index (RSI)



Viewed visualize.py:1-496

Based on the code in `src/analysis/visualize.py` (specifically lines 343-356), the Receptor Selectivity Index (RSI) is calculated to measure how specialized a given unit (receptor, glomerulus, or mitral cell) is to specific entity types.

Here is the mathematical formulation of the RSI calculation:

$$ RSI(r) = \begin{cases} \frac{\max_{e}(\mu_{r, e}) - \min_{e}(\mu_{r, e})}{\max_{e}(\mu_{r, e})} & \text{if } \max_{e}(\mu_{r, e}) > 10^{-6} \\ 0 & \text{otherwise} \end{cases} $$

**Where:**
*   $r$ is the index of the specific unit (e.g., receptor, glomerulus, or mitral cell).
*   $e$ represents a specific entity type (e.g., PER, LOC, ORG).
*   $\mu_{r, e}$ is the mean activation of unit $r$ when exposed to tokens of entity type $e$.
*   $\max_{e}(\mu_{r, e})$ is the maximum mean activation for unit $r$ across all entity types.
*   $\min_{e}(\mu_{r, e})$ is the minimum mean activation for unit $r$ across all entity types.
*   $10^{-6}$ is a small threshold used to avoid division by zero for inactive receptors.

**Intuition:**
*   An **RSI near 1.0** indicates high selectivity: the unit is highly active for at least one entity type and nearly inactive for at least one other.
*   An **RSI near 0.0** indicates low selectivity: the unit responds similarly across all entity types (or is completely inactive).