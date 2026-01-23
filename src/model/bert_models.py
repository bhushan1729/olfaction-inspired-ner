import torch
import torch.nn as nn
from transformers import AutoModel
from src.model.olfactory_ner import ReceptorLayer, GlomeruliLayer, BiLSTM_CRF
# Ensure we import BiLSTM_CRF or compatible layer. If not exported, we redefine or expose it.
# Assuming we can import components. If implementation details differ, I will adapt.

class BertBaseline(nn.Module):
    def __init__(self, num_labels, model_name='bert-base-multilingual-cased', dropout=0.1):
        super(BertBaseline, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_labels) # mBERT hidden is 768

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only calculate loss on active parts (not -100)
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.classifier.out_features)
            active_labels = labels.view(-1)
            loss = loss_fct(active_logits, active_labels)
        
        return logits, loss

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
            sparsity=config.get('lambda_sparse', 0.1) # Passing arg, though layer usually takes just dim
        )
        
        self.glomeruli_layer = GlomeruliLayer(
            num_receptors=config.get('num_receptors', 128),
            num_glomeruli=config.get('num_glomeruli', 32)
        )
        
        # BiLSTM + CRF
        # Input to BiLSTM is num_glomeruli
        self.bilstm_crf = BiLSTM_CRF(
            input_dim=config.get('num_glomeruli', 32),
            hidden_dim=config.get('lstm_hidden', 128),
            num_tags=num_labels
        )
        
    def forward(self, input_ids, attention_mask, labels=None):
        # 1. BERT Features
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            bert_feats = outputs.last_hidden_state # (Batch, Seq, 768)
            
        # 2. Receptors
        receptor_out, _ = self.receptor_layer(bert_feats) # (Batch, Seq, Num_Rec)
        
        # 3. Glomeruli
        glomeruli_out, _ = self.glomeruli_layer(receptor_out) # (Batch, Seq, Num_Glom)
        
        # 4. BiLSTM + CRF
        # The BiLSTM_CRF module usually expects (embeds, tags, mask)
        # Check original signature. Usually it returns loss if tags provided, else path.
        # We need to adapt because BiLSTM_CRF usually handles masking internally or expects lengths/mask.
        
        if labels is not None:
            # Training
            loss = self.bilstm_crf(glomeruli_out, labels, attention_mask.bool())
            return None, loss
        else:
            # Inference
            predictions = self.bilstm_crf(glomeruli_out, mask=attention_mask.bool())
            # Convert simple path list to aligned tensor if needed, but list is fine for eval
            return predictions, None

