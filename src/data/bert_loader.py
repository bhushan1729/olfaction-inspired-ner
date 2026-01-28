import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np

class BertNERDataset(Dataset):
    def __init__(self, examples, tokenizer, label2idx, max_len=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        tokens = item['tokens']
        
        # Handle both 'tags' and 'ner_tags' field names
        ner_tags = item.get('ner_tags', item.get('tags', []))

        # Tokenize and align labels
        tokenized_inputs = self.tokenizer(
            tokens, 
            truncation=True, 
            is_split_into_words=True, 
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )

        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subtoken of the word gets the label
                # Use label map if provided, else raw int
                label = ner_tags[word_idx]
                # If label is string, map to int; if int, verify
                if isinstance(label, str):
                    label_id = self.label2idx.get(label, 0) # Default to O (0)
                else:
                    label_id = label # Already int from HF dataset
                
                label_ids.append(label_id)
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        return {
            'input_ids': tokenized_inputs['input_ids'].squeeze(0),
            'attention_mask': tokenized_inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

def get_bert_dataset(dataset_name, language=None, model_name='bert-base-multilingual-cased', batch_size=32, max_len=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load Dataset
    if dataset_name == 'conll2003':
        # Use tner/conll2003 with trust_remote_code=True
        print(f"Loading CoNLL-2003 from tner/conll2003...")
        ds = load_dataset("tner/conll2003", trust_remote_code=True)
        tag_field = 'tags'  # tner uses 'tags' not 'ner_tags'
    elif dataset_name == 'wikiann':
        # Use wikiann dataset - full name with organization
        print(f"Loading WikiANN ({language}) from unimelb-nlp/wikiann...")
        ds = load_dataset("unimelb-nlp/wikiann", language)
        tag_field = 'ner_tags'
    else: # Fallback or custom
        raise ValueError(f"Dataset {dataset_name} not supported yet in bert_loader.")

    # Create Label Map
    # Extract unique labels from train set - handle both 'tags' and 'ner_tags'
    if tag_field in ds['train'].features:
        if hasattr(ds['train'].features[tag_field], 'feature'):
            # ClassLabel inside Sequence
            train_tags = ds['train'].features[tag_field].feature.names
        else:
            # Direct ClassLabel
            train_tags = ds['train'].features[tag_field].names
    else:
        raise ValueError(f"Tag field '{tag_field}' not found in dataset features")
    
    label2idx = {tag: i for i, tag in enumerate(train_tags)}
    idx2label = {i: tag for tag, i in label2idx.items()}
    
    # Ensure consistent field name is 'ner_tags'
    # (Most datasets already use this, but adding for robustness)

    train_dataset = BertNERDataset(ds['train'], tokenizer, label2idx, max_len)
    valid_dataset = BertNERDataset(ds['validation'], tokenizer, label2idx, max_len)
    test_dataset = BertNERDataset(ds['test'], tokenizer, label2idx, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, valid_loader, test_loader, {'label2idx': label2idx, 'idx2label': idx2label}


