"""
Baseline BiLSTM-CRF model for NER.
Standard architecture without olfactory components.
"""

import torch
import torch.nn as nn
from .crf import CRF


class BaselineNER(nn.Module):
    """
    Baseline BiLSTM-CRF model for Named Entity Recognition.
    
    Architecture:
        Embeddings -> BiLSTM -> CRF
    
    This serves as the control in our experiments.
    """
    
    def __init__(self,
                 vocab_size: int,
                 num_tags: int,
                 embed_dim: int = 300,
                 lstm_hidden: int = 256,
                 lstm_layers: int = 1,
                 dropout: float = 0.5,
                 pretrained_embeddings=None):
        """
        Args:
            vocab_size: Size of vocabulary
            num_tags: Number of NER tags
            embed_dim: Embedding dimension
            lstm_hidden: LSTM hidden dimension
            lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            pretrained_embeddings: Pre-trained embedding matrix (numpy array)
        """
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        
        self.dropout = nn.Dropout(dropout)
        
        # BiLSTM encoder
        self.lstm = nn.LSTM(
            embed_dim,
            lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Linear projection to tag space
        self.hidden2tag = nn.Linear(lstm_hidden * 2, num_tags)
        
        # CRF layer
        self.crf = CRF(num_tags, batch_first=True)
    
    def forward(self, sentences, tags=None, lengths=None):
        """
        Forward pass.
        
        Args:
            sentences: [batch, seq_len] - token indices
            tags: [batch, seq_len] - tag indices (for training)
            lengths: [batch] - actual sequence lengths
        
        Returns:
            If tags provided: loss
            If tags None: predicted tags
        """
        batch_size, seq_len = sentences.shape
        
        # Get embeddings
        embeds = self.embedding(sentences)
        embeds = self.dropout(embeds)
        
        # Pack sequences
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embeds, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embeds)
        
        # Project to tag space
        emissions = self.hidden2tag(lstm_out)
        
        # Create mask
        if lengths is not None:
            mask = torch.arange(seq_len, device=sentences.device).expand(batch_size, seq_len) < lengths.unsqueeze(1)
        else:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=sentences.device)
        
        # Training: compute loss
        if tags is not None:
            return self.crf(emissions, tags, mask)
        
        # Inference: decode
        return self.crf.decode(emissions, mask)
    
    def get_diversity_loss(self):
        """Dummy method for API compatibility with OlfactoryNER."""
        return torch.tensor(0.0, device=next(self.parameters()).device)


def create_baseline_ner(vocab_size, num_tags, config, pretrained_embeddings=None):
    """
    Factory function to create baseline model.
    
    Args:
        vocab_size: Vocabulary size
        num_tags: Number of NER tags
        config: Dictionary with model hyperparameters
        pretrained_embeddings: Optional pre-trained embeddings
    
    Returns:
        model: BaselineNER instance
    """
    return BaselineNER(
        vocab_size=vocab_size,
        num_tags=num_tags,
        embed_dim=config.get('embed_dim', 300),
        lstm_hidden=config.get('lstm_hidden', 256),
        lstm_layers=config.get('lstm_layers', 1),
        dropout=config.get('dropout', 0.5),
        pretrained_embeddings=pretrained_embeddings
    )


if __name__ == '__main__':
    # Test baseline model
    vocab_size = 1000
    num_tags = 9
    batch_size = 2
    seq_len = 10
    
    config = {
        'embed_dim': 300,
        'lstm_hidden': 256,
        'dropout': 0.5
    }
    
    model = create_baseline_ner(vocab_size, num_tags, config)
    
    # Random input
    sentences = torch.randint(0, vocab_size, (batch_size, seq_len))
    tags = torch.randint(0, num_tags, (batch_size, seq_len))
    lengths = torch.tensor([10, 7])
    
    print("Testing BaselineNER...")
    print(f"Input shape: {sentences.shape}")
    
    # Training
    loss = model(sentences, tags, lengths)
    print(f"Loss: {loss.item():.4f}")
    
    # Inference
    predictions = model(sentences, lengths=lengths)
    print(f"Predictions shape: {predictions.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
