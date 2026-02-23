"""
Olfaction-inspired NER model.

Architecture:
    Embeddings -> Receptors -> Glomeruli -> BiLSTM -> CRF
"""

import torch
import torch.nn as nn
from .layers import OlfactoryEncoder
from .crf import CRF


class OlfactoryNER(nn.Module):
    """
    Olfactory-inspired Named Entity Recognition model.
    
    Key components:
    1. Embedding layer (GloVe pre-trained)
    2. Receptor layer (specialized feature detectors)
    3. Glomerular layer (convergent aggregation)
    4. Mitral layer (optional sharpening and output mapping)
    5. BiLSTM encoder (contextual modeling)
    6. CRF decoder (sequence constraints)
    """
    
    def __init__(self, 
                 vocab_size: int,
                 num_tags: int,
                 embed_dim: int = 300,
                 num_receptors: int = 128,
                 num_glomeruli: int = 32,
                 lstm_hidden: int = 256,
                 lstm_layers: int = 1,
                 dropout: float = 0.5,
                 pretrained_embeddings=None,
                 use_receptors: bool = True,
                 use_glomeruli: bool = True,
                 use_mitral: bool = False,
                 num_mitral: int = None,
                 receptor_activation: str = 'relu',
                 mitral_activation: str = 'relu',
                 use_crf: bool = True):
        """
        Args:
            vocab_size: Size of vocabulary
            num_tags: Number of NER tags
            embed_dim: Embedding dimension (default 300 for GloVe)
            num_receptors: Number of receptor units
            num_glomeruli: Number of glomerular units
            lstm_hidden: LSTM hidden dimension
            lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            pretrained_embeddings: Pre-trained embedding matrix (numpy array)
            use_receptors: If False, skip receptor layer (ablation)
            use_glomeruli: If False, skip glomerular layer (ablation)
            use_mitral: If True, include mitral layer
            num_mitral: Number of mitral units (required if use_mitral=True)
            receptor_activation: Activation function ('relu', 'gelu', 'swish', 'mish')
            mitral_activation: Activation function for mitral layer
            use_crf: If False, skip CRF layer and use CrossEntropyLoss (ablation)
        """
        super().__init__()
        
        self.use_receptors = use_receptors
        self.use_glomeruli = use_glomeruli
        self.use_mitral = use_mitral
        self.use_crf_layer = use_crf
        self.num_tags = num_tags
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        
        # Olfactory encoder (receptors + glomeruli + mitral)
        if use_receptors:
            self.olfactory_encoder = OlfactoryEncoder(
                input_dim=embed_dim,
                num_receptors=num_receptors,
                num_glomeruli=num_glomeruli if use_glomeruli else num_receptors,
                num_mitral=num_mitral if use_mitral else None,
                activation=receptor_activation,
                mitral_activation=mitral_activation
            )
            # Determine lstm input dim based on highest active layer
            if use_mitral and num_mitral:
                lstm_input_dim = num_mitral
            elif use_glomeruli:
                lstm_input_dim = num_glomeruli
            else:
                lstm_input_dim = num_receptors
        else:
            self.olfactory_encoder = None
            lstm_input_dim = embed_dim
        
        self.dropout = nn.Dropout(dropout)
        
        # BiLSTM for contextual encoding
        self.lstm = nn.LSTM(
            lstm_input_dim,
            lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Linear projection to tag space
        self.hidden2tag = nn.Linear(lstm_hidden * 2, num_tags)
        
        # CRF layer or CrossEntropyLoss
        if self.use_crf_layer:
            self.crf = CRF(num_tags, batch_first=True)
        else:
            self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(self, sentences, tags=None, lengths=None):
        """
        Forward pass.
        
        Args:
            sentences: [batch, seq_len] - token indices
            tags: [batch, seq_len] - tag indices (required for training)
            lengths: [batch] - actual lengths (for packing)
        
        Returns:
            If tags provided: loss
            If tags None: predicted tags
        """
        batch_size, seq_len = sentences.shape
        
        # Get embeddings
        embeds = self.embedding(sentences)  # [batch, seq, embed_dim]
        embeds = self.dropout(embeds)
        
        # Apply olfactory encoder if enabled
        if self.use_receptors:
            features = self.olfactory_encoder(embeds)  # [batch, seq, num_glomeruli or num_receptors]
        else:
            features = embeds
        
        features = self.dropout(features)
        
        # Pack sequences for efficient LSTM processing
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                features, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(features)
        
        # Project to tag space
        emissions = self.hidden2tag(lstm_out)  # [batch, seq, num_tags]
        
        # Create mask
        if lengths is not None:
            mask = torch.arange(seq_len, device=sentences.device).expand(batch_size, seq_len) < lengths.unsqueeze(1)
        else:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=sentences.device)
        
        # Training: compute loss
        if tags is not None:
            if self.use_crf_layer:
                return self.crf(emissions, tags, mask)
            else:
                # Use CrossEntropyLoss
                # Only calculate loss for masked positions
                active_loss = mask.view(-1)
                active_logits = emissions.view(-1, self.num_tags)[active_loss]
                active_labels = tags.view(-1)[active_loss]
                return self.loss_fct(active_logits, active_labels)
        
        # Inference: decode
        if self.use_crf_layer:
            return self.crf.decode(emissions, mask)
        else:
            # Simple argmax decoding
            return torch.argmax(emissions, dim=-1)
    
    def get_diversity_loss(self):
        """Get receptor diversity regularization loss."""
        if self.use_receptors:
            return self.olfactory_encoder.get_diversity_loss()
        return torch.tensor(0.0, device=next(self.parameters()).device)
    
    def get_receptor_activations(self, sentences):
        """
        Get receptor activations for analysis.
        
        Args:
            sentences: [batch, seq_len]
        
        Returns:
            receptors: [batch, seq_len, num_receptors]
            glomeruli: [batch, seq_len, num_glomeruli]
            mitral: [batch, seq_len, num_mitral] (or None)
        """
        with torch.no_grad():
            embeds = self.embedding(sentences)
            
            if self.use_receptors:
                out, receptors, glomeruli, mitral = self.olfactory_encoder(embeds, return_activations=True)
                return receptors, glomeruli, mitral
            else:
                return None, None, None


def create_olfactory_ner(vocab_size, num_tags, config, pretrained_embeddings=None):
    """
    Factory function to create OlfactoryNER model with config.
    
    Args:
        vocab_size: Vocabulary size
        num_tags: Number of NER tags
        config: Dictionary with model hyperparameters
        pretrained_embeddings: Optional pre-trained embeddings
    
    Returns:
        model: OlfactoryNER instance
    """
    return OlfactoryNER(
        vocab_size=vocab_size,
        num_tags=num_tags,
        embed_dim=config.get('embed_dim', 300),
        num_receptors=config.get('num_receptors', 128),
        num_glomeruli=config.get('num_glomeruli', 32),
        lstm_hidden=config.get('lstm_hidden', 256),
        lstm_layers=config.get('lstm_layers', 1),
        dropout=config.get('dropout', 0.5),
        pretrained_embeddings=pretrained_embeddings,
        use_receptors=config.get('use_receptors', True),
        use_glomeruli=config.get('use_glomeruli', True),
        use_mitral=config.get('use_mitral', False),
        num_mitral=config.get('num_mitral', None),
        receptor_activation=config.get('receptor_activation', 'relu'),
        mitral_activation=config.get('mitral_activation', 'relu'),
        use_crf=config.get('use_crf', True)
    )


if __name__ == '__main__':
    # Test model
    vocab_size = 1000
    num_tags = 9
    batch_size = 2
    seq_len = 10
    
    config = {
        'embed_dim': 300,
        'num_receptors': 128,
        'num_glomeruli': 32,
        'lstm_hidden': 256,
        'dropout': 0.5
    }
    
    config_mitral = {
        'embed_dim': 300,
        'num_receptors': 128,
        'num_glomeruli': 32,
        'use_mitral': True,
        'num_mitral': 8,
        'lstm_hidden': 256,
        'dropout': 0.5
    }
    
    model = create_olfactory_ner(vocab_size, num_tags, config)
    model_mitral = create_olfactory_ner(vocab_size, num_tags, config_mitral)
    
    # Random input
    sentences = torch.randint(0, vocab_size, (batch_size, seq_len))
    tags = torch.randint(0, num_tags, (batch_size, seq_len))
    lengths = torch.tensor([10, 7])
    
    print("Testing OlfactoryNER...")
    print(f"Input shape: {sentences.shape}")
    
    # Training mode
    loss = model(sentences, tags, lengths)
    print(f"Loss: {loss.item():.4f}")
    
    # Inference mode
    predictions = model(sentences, lengths=lengths)
    print(f"Predictions shape: {predictions.shape}")
    
    # Diversity loss
    div_loss = model.get_diversity_loss()
    print(f"Diversity loss: {div_loss.item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nTesting OlfactoryNER (With Mitral)...")
    loss_mitral = model_mitral(sentences, tags, lengths)
    preds_mitral = model_mitral(sentences, lengths=lengths)
    print(f"Loss (Mitral): {loss_mitral.item():.4f}")
    print(f"Predictions shape (Mitral): {preds_mitral.shape}")
    print(f"Total parameters (Mitral): {sum(p.numel() for p in model_mitral.parameters()):,}")
