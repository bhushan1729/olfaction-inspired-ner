#!/bin/bash
# Quick start script to run all experiments locally

set -e

echo "========================================="
echo "Olfaction-Inspired NER - Quick Start"
echo "========================================="

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.7+"
    exit 1
fi

echo "✓ Python found: $(python --version)"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Download CoNLL-2003 data (automatic in code)
echo ""
echo "📥 Preparing CoNLL-2003 dataset..."
echo "(Data will be downloaded automatically during training)"

# Optional: Download GloVe
echo ""
echo "📥 GloVe embeddings (optional but recommended)"
read -p "Download GloVe 6B embeddings? This takes ~5 minutes. (y/n): " download_glove

if [ "$download_glove" = "y" ] || [ "$download_glove" = "Y" ]; then
    if [ ! -f "data/glove.6B.300d.txt" ]; then
        echo "Downloading GloVe..."
        mkdir -p data
        wget -q --show-progress http://nlp.stanford.edu/data/glove.6B.zip -O data/glove.6B.zip
        unzip -q data/glove.6B.zip -d data/
        rm data/glove.6B.zip
        echo "✓ GloVe downloaded"
    else
        echo "✓ GloVe already exists"
    fi
else
    echo "⚠️  Skipping GloVe. Will use random embeddings (affects performance)."
fi

# Run experiments
echo ""
echo "========================================="
echo "Starting Experiments"
echo "========================================="
echo ""
echo "This will run 4 experiments:"
echo "  1. Baseline BiLSTM-CRF (~30 min)"
echo "  2. Olfactory-NER Full (~30 min)"
echo "  3. Ablation: No Sparsity (~30 min)"
echo "  4. Ablation: No Glomeruli (~30 min)"
echo ""
echo "Total time: ~2 hours on GPU, ~6-8 hours on CPU"
echo ""
read -p "Continue? (y/n): " continue_exp

if [ "$continue_exp" != "y" ] && [ "$continue_exp" != "Y" ]; then
    echo "Exiting. Run individual experiments with:"
    echo "  python src/train.py --config config/experiments.yaml --experiment <name>"
    exit 0
fi

# Experiment 1: Baseline
echo ""
echo "========================================="
echo "Experiment 1/4: Baseline"
echo "========================================="
python src/train.py --config config/experiments.yaml --experiment baseline --save_dir results/baseline

# Experiment 2: Full Olfactory
echo ""
echo "========================================="
echo "Experiment 2/4: Olfactory-NER (Full)"
echo "========================================="
python src/train.py --config config/experiments.yaml --experiment olfactory_full --save_dir results/olfactory_full

# Experiment 3: No Sparsity
echo ""
echo "========================================="
echo "Experiment 3/4: No Sparsity Regularization"
echo "========================================="
python src/train.py --config config/experiments.yaml --experiment olfactory_no_sparse --save_dir results/olfactory_no_sparse

# Experiment 4: No Glomeruli
echo ""
echo "========================================="
echo "Experiment 4/4: No Glomerular Layer"
echo "========================================="
python src/train.py --config config/experiments.yaml --experiment olfactory_no_glomeruli --save_dir results/olfactory_no_glomeruli

# Analysis
echo ""
echo "========================================="
echo "Running Analysis"
echo "========================================="

# Create analysis script
cat > run_analysis.py << 'EOF'
import torch
from src.data.dataset import prepare_data
from src.model.olfactory_ner import create_olfactory_ner
from src.analysis.visualize import analyze_receptor_activations, compare_models
import yaml

# Load config
with open('config/experiments.yaml', 'r') as f:
    config = yaml.safe_load(f)

exp_config = config['olfactory_full']
exp_config.update(config.get('data', {}))

# Load data
print("Loading data...")
train_loader, valid_loader, test_loader, vocab_info = prepare_data(
    data_dir='./data/raw', batch_size=32
)

# Load model
print("Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('results/olfactory_full/best_model.pt', map_location=device)

vocab_size = len(vocab_info['word2idx'])
num_tags = len(vocab_info['label2idx'])

model = create_olfactory_ner(vocab_size, num_tags, exp_config)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

# Analyze receptors
print("\nAnalyzing receptor activations...")
analyze_receptor_activations(model, test_loader, vocab_info, device, save_dir='./analysis_results')

# Compare models
print("\nComparing all models...")
results_dirs = {
    'Baseline': 'results/baseline',
    'Olfactory (Full)': 'results/olfactory_full',
    'No Sparsity': 'results/olfactory_no_sparse',
    'No Glomeruli': 'results/olfactory_no_glomeruli'
}
compare_models(results_dirs, save_dir='./comparison')

print("\n✓ Analysis complete!")
EOF

python run_analysis.py
rm run_analysis.py

echo ""
echo "========================================="
echo "✓ All Experiments Complete!"
echo "========================================="
echo ""
echo "Results saved in:"
echo "  - results/<experiment>/results.json"
echo "  - analysis_results/"
echo "  - comparison/"
echo ""
echo "Next steps:"
echo "  1. Review results.json files for metrics"
echo "  2. Check analysis_results/ for receptor visualizations"
echo "  3. See comparison/ for cross-model comparison"
echo ""
echo "If results look promising, consider:"
echo "  - Low-resource experiments (10%, 20% data)"
echo "  - Cross-domain evaluation"
echo "  - Writing up findings"
echo ""
