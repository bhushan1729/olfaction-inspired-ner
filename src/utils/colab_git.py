"""
Colab helper to push results to GitHub.
Run this at the end of your Colab notebook.
"""

import os
import subprocess
from pathlib import Path


def setup_git_in_colab(repo_path='/content/olfaction-inspired-ner'):
    """Configure git in Colab environment."""
    os.chdir(repo_path)
    
    # Set git config (use generic for Colab)
    subprocess.run(['git', 'config', 'user.name', 'Colab Experiment'])
    subprocess.run(['git', 'config', 'user.email', 'colab@experiment.local'])
    
    print("✓ Git configured")


def push_results_to_github(
    repo_path='/content/olfaction-inspired-ner',
    commit_message=None,
    github_token=None
):
    """
    Push experiment results to GitHub from Colab.
    
    Args:
        repo_path: Path to cloned repository
        commit_message: Custom commit message
        github_token: GitHub personal access token (for private repos)
    
    Usage in Colab:
        from src.utils.colab_git import push_results_to_github
        
        # For public repo:
        push_results_to_github()
        
        # For private repo:
        push_results_to_github(github_token='your_token_here')
    """
    
    os.chdir(repo_path)
    
    # Setup git
    setup_git_in_colab(repo_path)
    
    # Add all results
    subprocess.run(['git', 'add', 'experiment_results/'])
    subprocess.run(['git', 'add', 'docs/'])
    
    # Check if there are changes
    status = subprocess.run(
        ['git', 'status', '--porcelain'],
        capture_output=True,
        text=True
    )
    
    if not status.stdout.strip():
        print("✓ No changes to commit")
        return
    
    # Commit
    if commit_message is None:
        from datetime import datetime
        commit_message = f"Add experiment results from Colab - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    subprocess.run(['git', 'commit', '-m', commit_message])
    
    # Push
    if github_token:
        # For private repos, update remote URL with token
        result = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            capture_output=True,
            text=True
        )
        
        origin_url = result.stdout.strip()
        
        if origin_url.startswith('https://github.com/'):
            # Insert token
            auth_url = origin_url.replace(
                'https://github.com/',
                f'https://{github_token}@github.com/'
            )
            subprocess.run(['git', 'remote', 'set-url', 'origin', auth_url])
    
    # Push to main
    push_result = subprocess.run(
        ['git', 'push', 'origin', 'main'],
        capture_output=True,
        text=True
    )
    
    if push_result.returncode == 0:
        print("\n✅ Results pushed to GitHub successfully!")
        print("View at: https://github.com/YOUR_USERNAME/olfaction-inspired-ner/tree/main/experiment_results")
    else:
        print("\n⚠️ Push failed. Error:")
        print(push_result.stderr)
        print("\nTry:")
        print("1. Make sure repo is public, OR")
        print("2. Provide GitHub token for private repo")


def save_and_push_experiment(
    experiment_name,
    dataset_name,
    model_type,
    config,
    results,
    visualization_dir=None,
    github_token=None
):
    """
    One-shot function: save results and push to GitHub.
    
    Usage in Colab (add at end of training):
        from src.utils.colab_git import save_and_push_experiment
        
        save_and_push_experiment(
            experiment_name='baseline_ontonotes',
            dataset_name='OntoNotes5',
            model_type='baseline',
            config=config,
            results=results,
            visualization_dir='./analysis_results',
            github_token='YOUR_TOKEN'  # Optional, for private repos
        )
    """
    
    # Save results
    from src.utils.save_results import save_experiment_results, generate_results_index
    
    save_experiment_results(
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        model_type=model_type,
        config=config,
        results=results,
        visualization_dir=visualization_dir
    )
    
    # Generate index
    generate_results_index()
    
    # Push to GitHub
    push_results_to_github(github_token=github_token)


if __name__ == '__main__':
    print("Usage in Colab:")
    print("  from src.utils.colab_git import save_and_push_experiment")
    print("  save_and_push_experiment(...)")
