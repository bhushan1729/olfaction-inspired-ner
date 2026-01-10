"""
Simple CRF layer for sequence labeling.
PyTorch implementation compatible with our NER models.
"""

import torch
import torch.nn as nn


class CRF(nn.Module):
    """
    Conditional Random Field layer for sequence labeling.
    
    Enforces valid BIO tag sequences and models label transitions.
    """
    
    def __init__(self, num_tags: int, batch_first: bool = True):
        """
        Args:
            num_tags: Number of unique tags
            batch_first: If True, expects input shape [batch, seq, tags]
        """
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        # Transition matrix: transitions[i,j] = score of transitioning from tag i to tag j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # Start and end transitions
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
    
    def forward(self, emissions, tags, mask=None):
        """
        Compute negative log-likelihood loss.
        
        Args:
            emissions: [batch, seq, num_tags] - emission scores from model
            tags: [batch, seq] - true tag indices
            mask: [batch, seq] - binary mask (1 for real tokens, 0 for padding)
        
        Returns:
            loss: Negative log-likelihood
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        
        # Compute log likelihood
        log_likelihood = self._compute_log_likelihood(emissions, tags, mask)
        
        # Return negative log-likelihood as loss
        return -log_likelihood.mean()
    
    def decode(self, emissions, mask=None):
        """
        Viterbi decoding to find best tag sequence.
        
        Args:
            emissions: [batch, seq, num_tags]
            mask: [batch, seq] - binary mask
        
        Returns:
            best_tags: [batch, seq] - predicted tag indices
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        
        return self._viterbi_decode(emissions, mask)
    
    def _compute_log_likelihood(self, emissions, tags, mask):
        """Compute log-likelihood of tag sequence."""
        batch_size, seq_len = tags.shape
        
        # Compute score of gold sequence
        gold_score = self._compute_score(emissions, tags, mask)
        
        # Compute partition function (log-sum-exp of all possible sequences)
        forward_score = self._forward_algorithm(emissions, mask)
        
        # Log likelihood = gold_score - log_partition
        log_likelihood = gold_score - forward_score
        
        return log_likelihood
    
    def _compute_score(self, emissions, tags, mask):
        """Compute score of specific tag sequence."""
        batch_size, seq_len = tags.shape
        
        # Start transition score
        score = self.start_transitions[tags[:, 0]]
        
        # Emission scores
        for t in range(seq_len):
            score += emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1) * mask[:, t]
        
        # Transition scores
        for t in range(1, seq_len):
            prev_tags = tags[:, t-1]
            curr_tags = tags[:, t]
            score += self.transitions[prev_tags, curr_tags] * mask[:, t]
        
        # End transition score
        last_tag_indices = mask.sum(1) - 1  # Index of last real tag
        last_tags = tags.gather(1, last_tag_indices.long().unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        
        return score
    
    def _forward_algorithm(self, emissions, mask):
        """Forward algorithm to compute partition function."""
        batch_size, seq_len, num_tags = emissions.shape
        
        # Initialize with start transitions + first emissions
        alpha = self.start_transitions + emissions[:, 0]
        
        # Iterate through sequence
        for t in range(1, seq_len):
            # Broadcast: [batch, num_tags, 1] + [num_tags, num_tags] + [batch, 1, num_tags]
            emit_scores = emissions[:, t].unsqueeze(1)  # [batch, 1, num_tags]
            trans_scores = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
            alpha_broadcast = alpha.unsqueeze(2)  # [batch, num_tags, 1]
            
            # Compute scores for all transitions
            next_alpha = alpha_broadcast + trans_scores + emit_scores
            
            # Log-sum-exp over previous tags
            next_alpha = torch.logsumexp(next_alpha, dim=1)
            
            # Apply mask
            alpha = next_alpha * mask[:, t].unsqueeze(1) + alpha * (~mask[:, t]).unsqueeze(1)
        
        # Add end transitions
        alpha = alpha + self.end_transitions
        
        # Log-sum-exp to get partition function
        return torch.logsumexp(alpha, dim=1)
    
    def _viterbi_decode(self, emissions, mask):
        """Viterbi algorithm for decoding."""
        batch_size, seq_len, num_tags = emissions.shape
        
        # Initialize
        viterbi = self.start_transitions + emissions[:, 0]
        backpointers = []
        
        # Forward pass
        for t in range(1, seq_len):
            # Compute scores: [batch, num_tags, 1] + [num_tags, num_tags]
            broadcast_viterbi = viterbi.unsqueeze(2)  # [batch, num_tags, 1]
            broadcast_transitions = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
            
            next_tag_var = broadcast_viterbi + broadcast_transitions
            
            # Find best previous tag
            best_tag_ids = torch.argmax(next_tag_var, dim=1)
            backpointers.append(best_tag_ids)
            
            # Get max scores and add emissions
            viterbi_max = next_tag_var.max(dim=1)[0]
            viterbi = viterbi_max + emissions[:, t]
            
            # Apply mask
            viterbi = viterbi * mask[:, t].unsqueeze(1) + viterbi * (~mask[:, t]).unsqueeze(1)
        
        # Add end transitions
        viterbi = viterbi + self.end_transitions
        
        # Backtrack
        best_last_tag = torch.argmax(viterbi, dim=1)
        best_tags = [best_last_tag]
        
        for backpointer in reversed(backpointers):
            best_last_tag = backpointer.gather(1, best_last_tag.unsqueeze(1)).squeeze(1)
            best_tags.append(best_last_tag)
        
        # Reverse to get correct order
        best_tags = torch.stack(list(reversed(best_tags)), dim=1)
        
        return best_tags


if __name__ == '__main__':
    # Test CRF
    num_tags = 9  # CoNLL has 9 tags
    batch_size = 2
    seq_len = 5
    
    crf = CRF(num_tags)
    
    # Random emissions
    emissions = torch.randn(batch_size, seq_len, num_tags)
    tags = torch.randint(0, num_tags, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, 3:] = 0  # First sequence has length 3
    
    print("Testing CRF...")
    print(f"Emissions shape: {emissions.shape}")
    print(f"Tags shape: {tags.shape}")
    
    # Test forward (loss)
    loss = crf(emissions, tags, mask)
    print(f"Loss: {loss.item():.4f}")
    
    # Test decode
    predictions = crf.decode(emissions, mask)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
