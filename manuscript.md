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

### 3.1 The Olfactory Pathway
In biological olfactory systems (found in both vertebrates and insects), odor detection and processing follow a highly structured, conserved pathway that maps chemical stimuli to neural representations:
1. **Sensory Neurons (OSNs/ORNs):** Olfactory sensory neurons (OSNs) express exactly one type of olfactory receptor from a large multigene family. Each receptor responds selectively to specific chemical features (epitopes) of odor molecules.
2. **Glomerular Convergence:** All OSNs expressing the same specific receptor converge onto an anatomically distinct locus called a glomerulus (located in the antennal lobe of insects, or the olfactory bulb of vertebrates). This acts as a severe structural bottleneck, pooling many redundant inputs to filter noise and amplify signals.
3. **Projection & Sharpening (Mitral Cells / Projection Neurons):** Glomerular activations are processed and relayed by principal output neurons—mitral/tufted cells in vertebrates, or projection neurons in insects. These cells can refine and sharpen the combinatorial activation patterns (often through lateral inhibition).
4. **Higher Cortical Processing:** These output neurons project to higher brain regions (such as the piriform cortex in mammals or the mushroom body/Kenyon cells in insects) where sparse combinatorial codes are translated into associative memories, patterns, and behavioral decisions.

### 3.2 Key Computational Properties
- **Sparse Activation:** Only a small subset of olfactory receptors fires for any given odorant, leading to highly efficient energy and representation usage.
- **Combinatorial Coding:** Meaning is encoded combinatorially; individual receptors are broad/weak feature detectors, and the identity of an odor is determined by the specific combination of activated receptors rather than a single "labeled line."
- **Robustness and Noise Tolerance:** The convergent pooling of thousands of sensory neurons into a small number of glomeruli averages out stochastic noise, allowing the system to detect weak signals in complex backgrounds.
- **Emergent Specialization:** Different receptors develop sensitivity to distinct molecular features, establishing a distributed feature extraction system.

### 3.3 Mapping to NLP
This biological architecture provides an intuitive blueprint for sequence labeling tasks like Named Entity Recognition:

| Biological Olfactory System | NLP/NER Model Equivalent | Function |
| --- | --- | --- |
| Odor molecules / Chemical stimuli | Token embeddings | Raw sensory input |
| Olfactory Sensory Neurons (OSNs) | Receptor Layer | Sparse, localized feature detection |
| Glomeruli (Olfactory Bulb / Antennal Lobe) | Glomerular Layer | Convergent feature pooling & noise reduction |
| Mitral Cells / Projection Neurons | Mitral Layer (Optional) | Output projection & feature sharpening |
| Olfactory Cortex / Mushroom Body | BiLSTM Encoder | Contextual sequence encoding |
| Higher Behavioral Output | CRF Decoder | Sequence decoding and tag assignment |

This mapping is an abstract computational analogy inspired by biological olfactory wiring rather than a direct physiological simulation.

---

## 4. Methodology

### 4.1 Baseline Architecture
Our baseline is a standard sequence tagger:  Embedding → BiLSTM → CRF. The word embeddings have a dimensionality of $d=300$. These are fed into a 1-layer bidirectional LSTM with a hidden dimension of 256. The outputs are projected to the target label space and decoded using a Conditional Random Field (CRF) layer. Total parameter count for the baseline model is approximately 1.5 million (excluding the embedding matrix).

### 4.2 Olfactory Architecture
The olfactory-enhanced architecture introduces a biologically-inspired sparse bottleneck between the embeddings and the BiLSTM. The forward pass is defined as: Embedding → Receptor Layer → Glomerular Layer → BiLSTM → CRF.

**Receptor Layer:**
This layer comprises $N_r = 128$ (or 256) sparse nonlinear projections acting as weak feature detectors. Given an input embedding $x_t \in \mathbb{R}^{300}$, the receptor activation vector $r_t \in \mathbb{R}^{N_r}$ is:
$$r_t = \sigma(W_R x_t + b_R)$$
where $W_R \in \mathbb{R}^{N_r \times 300}$ is a dense weight matrix, $b_R \in \mathbb{R}^{N_r}$ is a bias vector, and $\sigma$ is the ReLU activation function. The use of ReLU is critical as it naturally enforces a non-negative, sparse firing pattern akin to biological olfactory receptors.

**Glomerular Layer:**
Receptors aggregate their signals into a smaller number of glomeruli ($N_g = 32$ or $64$), acting as convergent feature pooling and noise reduction. The glomerular activation vector $g_t \in \mathbb{R}^{N_g}$ is computed as:
$$g_t = \text{ReLU}(W_G r_t)$$
where $W_G \in \mathbb{R}^{N_g \times N_r}$ serves as the assignment matrix defining the connection strength from receptors to glomeruli. The output $g_t$ is then passed into the BiLSTM (hidden dimension 256).

**Sparsity and Diversity Regularization:**
To encourage distinct, specialized receptor functions and prevent redundant feature collapse, we optimize the network using a composite loss function:
$$L = L_{NER} + \lambda_{sparse} L_{sparse} + \lambda_{diverse} L_{diverse}$$
Here, $L_{NER}$ is the standard negative log-likelihood from the CRF. 
$L_{sparse}$ acts as an L1 penalty on the receptor activations to enforce population sparsity. 
$L_{diverse}$ penalizes the cosine similarity between the weight vectors of different receptors, ensuring maximum utilization of the receptor space. We set $\lambda_{sparse} = 0.001$.

**Training Procedure:**
Models are trained using the Adam optimizer with a learning rate of $0.001$. We use a batch size of 32 and train for up to 30 epochs, applying early stopping with a patience of 5 epochs based on validation F1-score. Dropout ($p=0.2$ to $0.5$) is applied after the embedding layer and before the BiLSTM.

### 4.3 Receptor Selectivity Index (RSI)
To quantify the interpretability of our learned sparse representations, we introduce the Receptor Selectivity Index (RSI). RSI measures the degree to which a specific unit (e.g., an individual receptor or glomerulus) is specialized to detect particular named entity classes rather than firing uniformly across all classes.

For a given unit $r$, let $\mu_{r, e}$ represent the mean activation of that unit when exposed to tokens belonging to entity type $e \in \mathcal{E}$ (e.g., PER, LOC, ORG). The RSI is formulated as the normalized range of its mean activations across all entity types:

$$ RSI(r) = \begin{cases} \frac{\max_{e}(\mu_{r, e}) - \min_{e}(\mu_{r, e})}{\max_{e}(\mu_{r, e})} & \text{if } \max_{e}(\mu_{r, e}) > 10^{-6} \\ 0 & \text{otherwise} \end{cases} $$

where $10^{-6}$ is a small threshold to avoid division by zero for inactive units. 

An RSI near $1.0$ indicates extreme specialization (the unit fires strongly for at least one entity type and is nearly silent for at least one other), while an RSI near $0.0$ implies a lack of selectivity (the unit fires uniformly regardless of the entity class).

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

### 7.1 The Asymmetry of Inductive Biases
Our central finding—that the olfactory-inspired architecture yields massive gains in the ultra-low-resource setting (+9.2% on Telugu) while degrading performance in high-resource settings (-3.3% on English)—highlights a fundamental principle of machine learning: **inductive biases matter most when data is scarce.** In rich-data environments, high-capacity networks can easily discover optimal manifolds directly from the data. However, when supervision is weak and pre-trained representations are poorly aligned, unconstrained networks often overfit to noise. 

### 7.2 Connection to Representation Learning and Structured Priors
By forcing the network to route information through a sparse, combinatorial receptor-glomerular layer, we inject a **structured prior** into the learning process. This forces the model to disentangle dense, noisy embeddings into isolated, discrete micro-features (receptors) before compressing them (glomeruli). This process closely mirrors the objectives of the **Information Bottleneck Method** (Tishby et al., 1999), where a network is forced to discard task-irrelevant noise while preserving predictive signals. The olfactory bottleneck acts as a structural regularizer that explicitly prevents the model from memorizing the limited training set.

### 7.3 Implications for Few-Shot Learning and Sparse Manifolds
The success of this sparse combinatorial coding extends beyond biological mimicry and has profound implications for **few-shot learning**. In low-resource scenarios, the underlying useful data often lies on a sparse manifold; dense, fully-connected architectures struggle to isolate these relevant dimensions without catastrophic overfitting. By explicitly enforcing sparsity via ReLU activations and an L1 penalty, the receptor layer creates a sparse, disentangled manifold where distinct morphological and contextual cues become easily separable for the sequence model. This perfectly aligns with regularization theory, which posits that sparsity-inducing constraints naturally perform feature selection—a critical requirement when the model capacity heavily outnumbers the available training examples.

### 7.4 Limitations
This architecture is an exploratory study of biological inductive biases and does not claim state-of-the-art performance against massively pre-trained large language models (LLMs) or large-scale Transformers. The scale of the architecture is currently limited, evaluated primarily with a BiLSTM-CRF backbone on a constrained set of languages. Furthermore, the hard structural bottleneck strictly limits the theoretical upper bound of the model's capacity, explaining the exact performance degradation observed in the data-rich English CoNLL-2003 setting.

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
- Tishby, N., Pereira, F. C., & Bialek, W. (1999). The information bottleneck method. *arXiv preprint physics/0004057*.
- Tjong Kim Sang, E. F., & De Meulder, F. (2003). Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition. *CoNLL*.
- Wang, P. Y., Sun, Y., Axel, R., Abbott, L. F., & Yang, G. R. (2021). Evolving the olfactory system with machine learning. *Neuron*, 109(24), 3879-3892.
- Yang, et al. (2025). Structured IB: Improving Information Bottleneck with Structured Feature Learning. *AAAI*.
