# @title [PATCH] Apply Code Fixes & Add Experiments
import os
import shutil

print("Applying patches to project files...")

# ==============================================================================
# 1. Fix ImportError in unified_loader.py (load_glove_embeddings)
# ==============================================================================
fn_loader = 'src/data/unified_loader.py'
if os.path.exists(fn_loader):
    with open(fn_loader, 'r') as f:
        content = f.read()
    
    if "load_glove_embeddings" not in content and "collate_fn" in content:
        print(f"Patching {fn_loader}...")
        new_content = content.replace("collate_fn", "collate_fn,\n    load_glove_embeddings")
        with open(fn_loader, 'w') as f:
            f.write(new_content)
        print("✓ Fixed unified_loader.py")
    else:
        print(f"✓ {fn_loader} already patched")
else:
    print(f"Warning: {fn_loader} not found!")

# ==============================================================================
# 2. Fix TypeError in train_universal.py (verbose=True)
# ==============================================================================
fn_train = 'src/train_universal.py'
if os.path.exists(fn_train):
    with open(fn_train, 'r') as f:
        content = f.read()

    if 'verbose=True' in content:
        print(f"Patching {fn_train}...")
        content = content.replace(', verbose=True', '')
        with open(fn_train, 'w') as f:
            f.write(content)
        print("✓ Fixed train_universal.py (removed verbose=True)")
    else:
        print(f"✓ {fn_train} already patched")
else:
    print(f"Warning: {fn_train} not found!")

# ==============================================================================
# 3. Add New Experiments to Config
# ==============================================================================
fn_config = 'config/universal_config.yaml'
new_experiments_yaml = """
  # 8. More Glomeruli (GELU)
  gelu_more_glomeruli:
    <<: *common
    model_type: "olfactory"
    num_receptors: 128
    num_glomeruli: 64
    activation: "gelu"

  # 9. More Receptors + More Glomeruli
  gelu_more_receptors_more_glomeruli:
    <<: *common
    model_type: "olfactory"
    num_receptors: 256
    num_glomeruli: 128
    activation: "gelu"
"""

if os.path.exists(fn_config):
    with open(fn_config, 'r') as f:
        content = f.read()
    
    if "gelu_more_glomeruli" not in content:
        print(f"Updates {fn_config}...")
        # Append to end of file
        with open(fn_config, 'a') as f:
            f.write(new_experiments_yaml)
        print("✓ Added new experiments to config")
    else:
        print(f"✓ {fn_config} already contains new experiments")

# ==============================================================================
# 4. Add Glomeruli Heatmap to visualize.py
# ==============================================================================
fn_viz = 'src/analysis/visualize.py'
glomeruli_plot_code = r"""
    # 1b. Glomeruli activation heatmap per entity type
    print("Creating glomeruli activation heatmap...")
    fig_g, ax_g = plt.subplots(figsize=(12, 6))
    
    mean_glomeruli_activations = []
    
    for entity in entity_types:
        acts = np.array(glomeruli_activations_by_entity[entity])
        mean_act = acts.mean(axis=0)
        mean_glomeruli_activations.append(mean_act)
    
    mean_glomeruli_activations = np.array(mean_glomeruli_activations)
    
    sns.heatmap(mean_glomeruli_activations, 
                xticklabels=range(0, mean_glomeruli_activations.shape[1], 10),
                yticklabels=entity_types,
                cmap='YlOrRd',
                cbar_kws={'label': 'Mean Activation'},
                ax=ax_g)
    ax_g.set_xlabel('Glomerulus Index')
    ax_g.set_ylabel('Entity Type')
    
    title_g = 'Mean Glomeruli Activations by Entity Type'
    if experiment_name:
        title_g += f'\n({experiment_name})'
    ax_g.set_title(title_g)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'glomeruli_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    results['mean_glomeruli_activations'] = mean_glomeruli_activations
"""

if os.path.exists(fn_viz):
    with open(fn_viz, 'r') as f:
        content = f.read()
        
    if "mean_glomeruli_activations" not in content:
        print(f"Patching {fn_viz}...")
        # Insert after receptor heatmap section
        target = "results['entity_types'] = entity_types"
        if target in content:
            new_content = content.replace(target, target + "\n" + glomeruli_plot_code)
            with open(fn_viz, 'w') as f:
                f.write(new_content)
            print("✓ Added Glomeruli heatmap code")
        else:
            print("Warning: Could not find insertion point in visualize.py")
    else:
        print(f"✓ {fn_viz} already patched")
else:
    print(f"Warning: {fn_viz} not found!")

print("\nAll Patches Applied Successfully.")
