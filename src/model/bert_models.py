import torch
import torch.nn as nn
from transformers import AutoModel
# Corrected Imports
from src.model.layers import ReceptorLayer, GlomerularLayer
from src.model.crf import CRF

class BertBaseline(nn.Module):
    def __init__(self, num_labels, model_name='bert-base-multilingual-cased', dropout=0.1, freeze_bert=True):
        super(BertBaseline, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze BERT if requested (for fair comparison with olfactory model)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_labels) # mBERT hidden is 768
        
        # CRF for structured prediction (matching olfactory model)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        # Use no_grad for frozen BERT to save memory
        if not next(self.bert.parameters()).requires_grad:
            with torch.no_grad():
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                sequence_output = outputs.last_hidden_state
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state
            
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)  # Emission scores for CRF
        
        if labels is not None:
            # Training: CRF Negative Log Likelihood
            mask = attention_mask.bool()
            
            # Replace -100 (padding) with 0 for CRF
            safe_labels = labels.clone()
            safe_labels[labels == -100] = 0
            
            # CRF Loss
            loss = -self.crf(emissions, safe_labels, mask=mask)
            return None, loss
        else:
            # Inference: Viterbi Decoding
            mask = attention_mask.bool()
            predictions = self.crf.decode(emissions, mask=mask)
            return predictions, None

class BertOlfactory(nn.Module):
    def __init__(self, num_labels, config, model_name='bert-base-multilingual-cased'):
        super(BertOlfactory, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freezing BERT is crucial for the "Feature Extractor" argument
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Olfactory Layers
        # Input dim is 768 (mBERT) -> Receptor Dim
        self.receptor_layer = ReceptorLayer(
            input_dim=768, 
            num_receptors=config.get('num_receptors', 128),
            activation='relu' # Default to ReLU
        )
        
        self.glomeruli_layer = GlomerularLayer(
            num_receptors=config.get('num_receptors', 128),
            num_glomeruli=config.get('num_glomeruli', 32)
        )
        
        # Linear Projection to Tags (directly from Glomeruli, no BiLSTM)
        num_glomeruli = config.get('num_glomeruli', 32)
        self.hidden2tag = nn.Linear(num_glomeruli, num_labels)
        
        # CRF
        self.crf = CRF(num_labels, batch_first=True)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # 1. BERT Features
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            bert_feats = outputs.last_hidden_state # (Batch, Seq, 768)
            
        # 2. Receptors
        receptor_out = self.receptor_layer(bert_feats) # (Batch, Seq, Num_Rec)
        
        # 3. Glomeruli
        glomeruli_out = self.glomeruli_layer(receptor_out) # (Batch, Seq, Num_Glom)
        
        # 4. Emission Scores (no BiLSTM - direct from Glomeruli)
        emissions = self.hidden2tag(glomeruli_out) # (Batch, Seq, Num_Tags)
        
        if labels is not None:
            # Training: Negative Log Likelihood
            # Now labels are properly assigned to all subwords (not -100)
            # Only padding tokens have -100
            mask = attention_mask.bool()
            
            # Replace -100 (padding) with 0 for CRF (CRF can't handle -100 as index)
            safe_labels = labels.clone()
            safe_labels[labels == -100] = 0
            
            # CRF Loss
            loss = -self.crf(emissions, safe_labels, mask=mask)
            
            return None, loss
        else:
            # Inference: Viterbi Decoding
            mask = attention_mask.bool()
            predictions = self.crf.decode(emissions, mask=mask)
            return predictions, None
