"""
Olfactory-inspired layers: Receptors and Glomeruli.

Receptors: Specialized micro-feature detectors
Glomeruli: Convergent aggregation with denoising
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReceptorLayer(nn.Module):
    """
    Receptor layer - specialized feature detectors.
    
    Biological inspiration:
    - Each olfactory receptor responds to specific molecular features
    - Receptors are highly specialized (one neuron - one receptor rule)
    - Responses are sparse and selective
    
    Implementation:
    - Each receptor is a small-capacity detector
    - ReLU activation for sparsity
    - Optional diversity regularization to prevent redundancy
    """
    
    def __init__(self, input_dim: int, num_receptors: int):
        """
        Args:
            input_dim: Input embedding dimension
            num_receptors: Number of receptor units (e.g., 128-256)
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_receptors = num_receptors
        
        # Receptor weights - each receptor is a linear projection
        self.W = nn.Parameter(torch.randn(num_receptors, input_dim))
        self.b = nn.Parameter(torch.zeros(num_receptors))
        
        # Initialize with small weights for specialization
        nn.init.xavier_uniform_(self.W, gain=0.5)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]
        
        Returns:
            r: [batch, seq_len, num_receptors] - receptor activations
        """
        # Apply linear transformation: r = xW^T + b
        # Using einsum for clarity: batch-seq-dim, receptor-dim -> batch-seq-receptor
        r = torch.einsum('bsd,rd->bsr', x, self.W) + self.b
        
        # ReLU for sparse activations
        r = F.relu(r)
        
        return r
    
    def get_diversity_loss(self):
        """
        Diversity loss to encourage receptor specialization.
        Penalizes high cosine similarity between receptor weight vectors.
        
        Returns:
            loss: Scalar diversity loss
        """
        # Normalize receptor weights
        W_norm = F.normalize(self.W, p=2, dim=1)
        
        # Compute pairwise cosine similarities
        similarity_matrix = torch.mm(W_norm, W_norm.t())
        
        # Penalize off-diagonal elements (high similarity between different receptors)
        # Create mask to exclude diagonal
        mask = 1 - torch.eye(self.num_receptors, device=self.W.device)
        
        # Mean absolute similarity (excluding self-similarity)
        diversity_loss = (similarity_matrix.abs() * mask).sum() / (self.num_receptors * (self.num_receptors - 1))
        
        return diversity_loss


class GlomerularLayer(nn.Module):
    """
    Glomerular layer - convergent aggregation.
    
    Biological inspiration:
    - Multiple OSNs with same receptor converge to one glomerulus
    - Provides signal amplification and noise reduction
    - Abstracts receptor patterns into stable features
    
    Implementation:
    - Learnable aggregation of receptor activations
    - Typically: many receptors -> fewer glomeruli
    - Acts as feature pooling/grouping
    """
    
    def __init__(self, num_receptors: int, num_glomeruli: int):
        """
        Args:
            num_receptors: Number of input receptors
            num_glomeruli: Number of glomerular units (should be < num_receptors)
        """
        super().__init__()
        self.num_receptors = num_receptors
        self.num_glomeruli = num_glomeruli
        
        # Aggregation weights: how receptors contribute to each glomerulus
        self.assignment = nn.Parameter(torch.randn(num_glomeruli, num_receptors))
        
        # Initialize with small positive weights
        nn.init.xavier_uniform_(self.assignment, gain=0.5)
    
    def forward(self, r):
        """
        Args:
            r: [batch, seq_len, num_receptors] - receptor activations
        
        Returns:
            g: [batch, seq_len, num_glomeruli] - glomerular activations
        """
        # Aggregate receptors into glomeruli
        # glomerulus-receptor, batch-seq-receptor -> batch-seq-glomerulus
        g = torch.einsum('gr,bsr->bsg', self.assignment, r)
        
        # ReLU to keep activations positive
        g = F.relu(g)
        
        return g


class OlfactoryEncoder(nn.Module):
    """
    Complete olfactory-inspired encoder.
    
    Architecture:
        Input embeddings
        -> Receptor layer (specialized detectors)
        -> Glomerular layer (convergent aggregation)
        -> Output features
    """
    
    def __init__(self, input_dim: int, num_receptors: int, num_glomeruli: int):
        """
        Args:
            input_dim: Input embedding dimension
            num_receptors: Number of receptor units
            num_glomeruli: Number of glomerular units
        """
        super().__init__()
        
        self.receptors = ReceptorLayer(input_dim, num_receptors)
        self.glomeruli = GlomerularLayer(num_receptors, num_glomeruli)
        
        self.output_dim = num_glomeruli
    
    def forward(self, x, return_receptors=False):
        """
        Args:
            x: [batch, seq_len, input_dim]
            return_receptors: If True, return receptor activations for analysis
        
        Returns:
            g: [batch, seq_len, num_glomeruli]
            r (optional): [batch, seq_len, num_receptors]
        """
        r = self.receptors(x)
        g = self.glomeruli(r)
        
        if return_receptors:
            return g, r
        return g
    
    def get_diversity_loss(self):
        """Get receptor diversity loss."""
        return self.receptors.get_diversity_loss()


if __name__ == '__main__':
    # Test layers
    batch_size = 4
    seq_len = 10
    input_dim = 300
    num_receptors = 128
    num_glomeruli = 32
    
    # Create random input
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Test receptor layer
    print("Testing ReceptorLayer...")
    receptor_layer = ReceptorLayer(input_dim, num_receptors)
    r = receptor_layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Receptor output shape: {r.shape}")
    print(f"Receptor diversity loss: {receptor_layer.get_diversity_loss().item():.4f}")
    
    # Test glomerular layer
    print("\nTesting GlomerularLayer...")
    glomerular_layer = GlomerularLayer(num_receptors, num_glomeruli)
    g = glomerular_layer(r)
    print(f"Receptor input shape: {r.shape}")
    print(f"Glomerular output shape: {g.shape}")
    
    # Test full encoder
    print("\nTesting OlfactoryEncoder...")
    encoder = OlfactoryEncoder(input_dim, num_receptors, num_glomeruli)
    g, r = encoder(x, return_receptors=True)
    print(f"Input shape: {x.shape}")
    print(f"Glomerular output shape: {g.shape}")
    print(f"Receptor output shape: {r.shape}")
    print(f"Diversity loss: {encoder.get_diversity_loss().item():.4f}")
    
    # Check sparsity
    sparsity = (r > 0).float().mean()
    print(f"Receptor activation sparsity: {sparsity.item():.2%}")
