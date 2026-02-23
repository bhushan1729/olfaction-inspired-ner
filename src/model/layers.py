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
    
    def __init__(self, input_dim: int, num_receptors: int, activation: str = 'relu'):
        """
        Args:
            input_dim: Input embedding dimension
            num_receptors: Number of receptor units (e.g., 128-256)
            activation: Activation function ('relu', 'gelu', 'swish', 'mish')
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_receptors = num_receptors
        
        # Receptor weights - each receptor is a linear projection
        self.W = nn.Parameter(torch.randn(num_receptors, input_dim))
        self.b = nn.Parameter(torch.zeros(num_receptors))
        
        # Initialize with small weights for specialization
        nn.init.xavier_uniform_(self.W, gain=0.5)
        
        # Activation function selection
        activation_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),  # SiLU is PyTorch's name for Swish
            'silu': nn.SiLU(),
            'mish': nn.Mish(),
        }
        
        if activation.lower() not in activation_map:
            raise ValueError(f"Unknown activation: {activation}. Choose from {list(activation_map.keys())}")
        
        self.activation = activation_map[activation.lower()]
    
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
        
        # Apply activation function
        r = self.activation(r)
        
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


class MitralLayer(nn.Module):
    """
    Mitral layer - principal output neurons of the olfactory bulb.
    
    Biological inspiration:
    - Receive input from a single glomerulus.
    - Act as the principal output neurons of the olfactory bulb.
    - Lateral inhibition via granule cells sharpens patterns.
    
    Implementation:
    - Linear projection from glomeruli to mitral cells.
    - Optional dropout/inhibition to simulate sharpening.
    - Activation function maps to output space.
    """
    
    def __init__(self, num_glomeruli: int, num_mitral: int, activation: str = 'relu'):
        """
        Args:
            num_glomeruli: Number of input glomerular units
            num_mitral: Number of output mitral units
            activation: Activation function ('relu', 'gelu', etc.)
        """
        super().__init__()
        self.num_glomeruli = num_glomeruli
        self.num_mitral = num_mitral
        
        # Projection weights
        self.W = nn.Parameter(torch.randn(num_mitral, num_glomeruli))
        self.b = nn.Parameter(torch.zeros(num_mitral))
        
        nn.init.xavier_uniform_(self.W, gain=0.5)
        
        # Activation function selection (same as receptors)
        activation_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'silu': nn.SiLU(),
            'mish': nn.Mish(),
        }
        
        if activation.lower() not in activation_map:
            raise ValueError(f"Unknown activation: {activation}. Choose from {list(activation_map.keys())}")
        
        self.activation = activation_map[activation.lower()]
    
    def forward(self, g):
        """
        Args:
            g: [batch, seq_len, num_glomeruli] - glomerular activations
        
        Returns:
            m: [batch, seq_len, num_mitral] - mitral activations
        """
        # m = gW^T + b
        m = torch.einsum('bsg,mg->bsm', g, self.W) + self.b
        m = self.activation(m)
        return m


class OlfactoryEncoder(nn.Module):
    """
    Complete olfactory-inspired encoder.
    
    Architecture:
        Input embeddings
        -> Receptor layer (specialized detectors)
        -> Glomerular layer (convergent aggregation)
        -> Output features
    """
    
    def __init__(self, input_dim: int, num_receptors: int, num_glomeruli: int, 
                 num_mitral: int = None, activation: str = 'relu', mitral_activation: str = 'relu'):
        """
        Args:
            input_dim: Input embedding dimension
            num_receptors: Number of receptor units
            num_glomeruli: Number of glomerular units
            num_mitral: Number of mitral units (optional)
            activation: Activation function for receptors ('relu', 'gelu', 'swish', 'mish')
            mitral_activation: Activation function for mitral cells
        """
        super().__init__()
        
        self.receptors = ReceptorLayer(input_dim, num_receptors, activation=activation)
        self.glomeruli = GlomerularLayer(num_receptors, num_glomeruli)
        
        if num_mitral is not None and num_mitral > 0:
            self.mitral = MitralLayer(num_glomeruli, num_mitral, activation=mitral_activation)
            self.output_dim = num_mitral
            self.use_mitral = True
        else:
            self.mitral = None
            self.output_dim = num_glomeruli
            self.use_mitral = False
    
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
        
        out = g
        if self.use_mitral:
            out = self.mitral(g)
            
        if return_receptors:
            return out, r
        return out
    
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
    print("\nTesting OlfactoryEncoder (No Mitral)...")
    encoder = OlfactoryEncoder(input_dim, num_receptors, num_glomeruli)
    out, r = encoder(x, return_receptors=True)
    print(f"Input shape: {x.shape}")
    print(f"Encoder output shape: {out.shape}")
    print(f"Receptor output shape: {r.shape}")
    print(f"Diversity loss: {encoder.get_diversity_loss().item():.4f}")
    
    # Test full encoder with mitral
    num_mitral = 8
    print("\nTesting OlfactoryEncoder (With Mitral)...")
    encoder_mitral = OlfactoryEncoder(input_dim, num_receptors, num_glomeruli, num_mitral=num_mitral)
    out_mitral, r_mitral = encoder_mitral(x, return_receptors=True)
    print(f"Input shape: {x.shape}")
    print(f"Encoder output shape (Mitral): {out_mitral.shape}")
    print(f"Receptor output shape: {r_mitral.shape}")
    
    # Check sparsity
    sparsity = (r > 0).float().mean()
    print(f"Receptor activation sparsity: {sparsity.item():.2%}")
