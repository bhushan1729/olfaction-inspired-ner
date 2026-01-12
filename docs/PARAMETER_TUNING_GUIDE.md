# Parameter Tuning Guide for Olfaction-Inspired NER

## 📊 Current Parameter Counts

### Baseline BiLSTM-CRF
- **Total**: ~4.74M parameters
- **Breakdown**:
  - Embeddings: `vocab_size × embed_dim` = 11,984 × 300 = 3.6M
  - BiLSTM: `4 × lstm_hidden × (embed_dim + lstm_hidden + 1)` ≈ 1.0M
  - CRF: `num_tags²` = 8 × 8 = 64 (negligible)

### Olfactory (Full)
- **Total**: ~5.1M parameters (estimated)
- **Breakdown**:
  - Embeddings: 3.6M (same)
  - **Receptors**: `embed_dim × num_receptors` = 300 × 128 = 38.4K
  - **Glomeruli**: `num_receptors × num_glomeruli` = 128 × 32 = 4.1K
  - BiLSTM: `4 × lstm_hidden × (num_glomeruli + lstm_hidden + 1)` ≈ 0.8M
  - CRF: 64 (same)

**Why More Parameters?**
1. **Receptor layer adds**: 38.4K parameters for specialized feature extraction
2. **Glomeruli layer adds**: 4.1K parameters for aggregation
3. **Total overhead**: ~42.5K (~0.9% increase) - actually quite small!

The BiLSTM in olfactory model has **fewer** parameters because it takes `num_glomeruli` (32) as input instead of `embed_dim` (300).

---

## 🎛️ Tunable Parameters

### Architecture Parameters

#### 1. **Receptor Layer**
```yaml
num_receptors: 128      # Current
# Try: 64, 96, 128, 192, 256
# Impact: More receptors = more specialization but more parameters
# Recommendation: Try 64 (faster) or 256 (more expressive)
```

**Trade-offs**:
- **64 receptors**: Faster training, fewer params, may lose specialization
- **256 receptors**: Better specialization, more interpretable, slower

#### 2. **Glomerular Layer**
```yaml
num_glomeruli: 32       # Current
# Try: 16, 24, 32, 48, 64
# Impact: Controls information bottleneck
# Recommendation: Try 16 (stronger bottleneck) or 64 (less compression)
```

**Trade-offs**:
- **16 glomeruli**: Strong compression, may lose information
- **64 glomeruli**: Less compression, preserves more features

#### 3. **Receptor-to-Glomeruli Ratio**
```yaml
# Current: 128:32 = 4:1
# Try:
- 64:16 (4:1 ratio, smaller)
- 128:16 (8:1 ratio, stronger convergence)
- 256:32 (8:1 ratio, more receptors)
- 128:64 (2:1 ratio, less compression)
```

#### 4. **BiLSTM Hidden Size**
```yaml
lstm_hidden: 256        # Current
# Try: 128, 192, 256, 384, 512
# Impact: Model capacity for sequence modeling
```

#### 5. **Dropout Rate**
```yaml
dropout: 0.5           # Current
# Try: 0.3, 0.4, 0.5, 0.6
# Impact: Regularization strength
```

---

### Training Hyperparameters

#### 1. **Learning Rate**
```yaml
learning_rate: 0.001   # Current
# Try: 0.0005, 0.001, 0.002, 0.005
# Impact: Convergence speed and stability
```

#### 2. **Batch Size**
```yaml
batch_size: 32         # Current
# Try: 16, 24, 32, 48, 64
# Impact: Training speed and gradient estimates
```

#### 3. **Regularization Weights**
```yaml
lambda_sparse: 0.001   # L1 sparsity on receptors
# Try: 0.0, 0.0005, 0.001, 0.005, 0.01

lambda_diverse: 0.01   # Diversity loss on receptors
# Try: 0.0, 0.005, 0.01, 0.02, 0.05
# Higher = stronger push for receptor specialization
```

**Note**: Our results showed sparsity didn't matter, but diversity might!

---

## 🧠 Activation Functions

### Current: ReLU
```python
self.activation = nn.ReLU()
```

**Characteristics**:
- Simple, fast
- Non-differentiable at 0
- Can cause "dead neurons"
- NOT biologically plausible

### Better Alternatives

#### 1. **GELU (Gaussian Error Linear Unit)** ✨ RECOMMENDED
```python
self.activation = nn.GELU()
```

**Why GELU?**
- ✅ **Smooth, differentiable** everywhere
- ✅ **Used in BERT, GPT** (proven effective)
- ✅ **More biologically plausible** than ReLU
- ✅ **Stochastic interpretation**: neuron activates with probability based on input
- ✅ **Better gradient flow**

**Mathematical form**:
```
GELU(x) = x × Φ(x)
where Φ(x) is cumulative distribution function of standard normal
```

**Biological justification**: Mimics stochastic neuron firing based on input strength!

#### 2. **SELU (Scaled Exponential Linear Unit)**
```python
self.activation = nn.SELU()
```

**Why SELU?**
- ✅ **Self-normalizing** (maintains mean=0, var=1)
- ✅ **Prevents vanishing/exploding gradients**
- ❌ Requires specific initialization (LeCun normal)

#### 3. **Swish / SiLU (Sigmoid Linear Unit)**
```python
self.activation = nn.SiLU()  # Same as Swish
```

**Why Swish?**
- ✅ **Smooth, non-monotonic**
- ✅ **Discovered by neural architecture search**
- ✅ **Better than ReLU on many tasks**

**Mathematical form**:
```
Swish(x) = x × sigmoid(x)
```

#### 4. **Mish**
```python
self.activation = nn.Mish()
```

**Why Mish?**
- ✅ **Smooth, unbounded above**
- ✅ **Strong performance** on vision tasks
- ✅ **Non-monotonic** like Swish

---

## 🔬 Spiking Neural Networks (Future)

You're right - spiking models are even MORE biologically plausible!

### What Are SNNs?
- Neurons communicate via **discrete spikes** (action potentials)
- **Temporal coding**: information in spike timing, not just rates
- **Energy efficient**: only compute when spikes occur

### SNN Libraries for PyTorch
1. **snnTorch** - easiest to use
2. **Norse** - research-grade
3. **SpykeTorch** - STDP learning

### How to Adapt Olfactory Model to SNN

**Receptor as Leaky Integrate-and-Fire (LIF) Neuron**:
```python
import snntorch as snn

class SpikingReceptorLayer(nn.Module):
    def __init__(self, input_dim, num_receptors):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_receptors)
        self.lif = snn.Leaky(beta=0.9)  # Leaky integrate-and-fire
        
    def forward(self, x, timesteps=10):
        # x: [batch, seq_len, input_dim]
        mem = self.lif.init_leaky()
        spikes = []
        
        for t in range(timesteps):
            cur = self.fc(x)
            spk, mem = self.lif(cur, mem)
            spikes.append(spk)
        
        # Aggregate spike counts
        spike_counts = torch.stack(spikes).sum(dim=0)
        return spike_counts  # [batch, seq_len, num_receptors]
```

**Benefits**:
- More biologically realistic
- Potential for neuromorphic hardware
- Temporal dynamics

**Challenges**:
- Harder to train (non-differentiable spikes)
- Requires surrogate gradients
- Slower inference

**My Recommendation**: Start with GELU, then try SNNs as an extension!

---

## 🎯 Recommended Tuning Experiments

### Experiment 1: Activation Functions
**Goal**: Test if biologically-plausible activations help

**Variants**:
1. Receptors: ReLU (baseline)
2. Receptors: GELU
3. Receptors: Swish/SiLU
4. Receptors: Mish

**Keep constant**: All other hyperparameters

**Expected**: GELU or Swish may improve F1 by 0.5-1%

---

### Experiment 2: Receptor Count
**Goal**: Find optimal number of receptors

**Variants**:
1. 64 receptors, 16 glomeruli (4:1)
2. 128 receptors, 32 glomeruli (4:1) - current
3. 256 receptors, 64 glomeruli (4:1)

**Expected**: More receptors = better specialization (visible in heatmaps)

---

### Experiment 3: Diversity Loss Strength
**Goal**: Test impact of forcing receptor specialization

**Variants**:
1. lambda_diverse = 0.0 (no diversity loss)
2. lambda_diverse = 0.01 (current)
3. lambda_diverse = 0.05 (strong)
4. lambda_diverse = 0.1 (very strong)

**Expected**: Higher diversity may improve interpretability

---

### Experiment 4: Architecture Simplification
**Goal**: Minimal viable olfactory model

**Variants**:
1. Embeddings → Receptors (64) → BiLSTM → CRF (no glomeruli)
2. Embeddings → Receptors (128) → BiLSTM → CRF

**Expected**: May match or beat full model (we saw this pattern!)

---

## 🛠️ How to Run Tuning Experiments

### 1. Create New Config
```yaml
# config/tuning_experiments.yaml

activation_gelu:
  model_type: olfactory
  embed_dim: 300
  num_receptors: 128
  num_glomeruli: 32
  receptor_activation: 'gelu'  # NEW parameter
  lstm_hidden: 256
  dropout: 0.5
  learning_rate: 0.001
  batch_size: 32
  lambda_sparse: 0.001
  lambda_diverse: 0.01

more_receptors:
  model_type: olfactory
  embed_dim: 300
  num_receptors: 256  # CHANGED
  num_glomeruli: 64   # CHANGED
  lstm_hidden: 256
  # ... rest same

strong_diversity:
  model_type: olfactory
  num_receptors: 128
  num_glomeruli: 32
  lambda_diverse: 0.05  # CHANGED
  # ... rest same
```

### 2. Modify Code to Support Activation Choice

I'll create a code update for you to add activation function selection!

### 3. Run Experiments
```bash
# GELU activation
python src/train.py --config config/tuning_experiments.yaml \
  --experiment activation_gelu --save_dir results/tuning/gelu

# More receptors
python src/train.py --config config/tuning_experiments.yaml \
  --experiment more_receptors --save_dir results/tuning/receptors_256

# Strong diversity
python src/train.py --config config/tuning_experiments.yaml \
  --experiment strong_diversity --save_dir results/tuning/diversity_0.05
```

---

## 📊 How to Analyze Results

### 1. Compare F1 Scores
```python
results = {
    'ReLU': 72.56,
    'GELU': 73.21,  # hypothetical
    'Swish': 72.98,
}
```

### 2. Analyze Receptor Heatmaps
- Do different activations lead to different specialization patterns?
- Are receptors more or less specialized?

### 3. Compare Training Dynamics
- Which activation converges faster?
- More stable training?

---

## 🎓 Further Reading

### Activation Functions
1. GELU paper: "Gaussian Error Linear Units" (Hendrycks & Gimpel, 2016)
2. Swish paper: "Searching for Activation Functions" (Ramachandran et al., 2017)
3. Mish paper: "A Self Regularized Non-Monotonic Activation Function" (Misra, 2019)

### Spiking Neural Networks
1. snnTorch tutorial: https://snntorch.readthedocs.io/
2. "Deep Learning with Spiking Neurons" (Tavanaei et al., 2019)
3. "Surrogate Gradient Learning in SNNs" (Neftci et al., 2019)

---

## ✅ Summary

**Best Next Steps**:
1. ✅ **Try GELU activation** - easy change, biologically motivated
2. ✅ **Experiment with 256 receptors** - test if more specialization helps
3. ✅ **Tune diversity loss** (0.05, 0.1) - force stronger specialization
4. ⏳ **Spiking models** - save for later (requires more infrastructure)

**Quick Win**: GELU activation - change 1 line, could improve results!

Want me to update the code to support activation function selection?
