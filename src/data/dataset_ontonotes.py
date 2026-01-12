"""
OntoNotes5 dataset loader for NER.

OntoNotes5 has 18 entity types (vs 4 in CoNLL-2003):
CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, 
ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, WORK_OF_ART
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from collections import Counter


def download_ontonotes5(cache_dir='./data/ontonotes'):
    """
    Download OntoNotes5 dataset from Hugging Face.
    
    Returns:
        dict: {'train': dataset, 'validation': dataset, 'test': dataset}
    """
    print("Downloading OntoNotes5 from Hugging Face...")
    
    dataset = load_dataset('tner/ontonotes5', cache_dir=cache_dir)
    
    print(f"✓ OntoNotes5 downloaded")
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Valid: {len(dataset['validation'])} samples")
    print(f"  Test: {len(dataset['test'])} samples")
    
    return dataset


class OntoNotes5Dataset(Dataset):
    """PyTorch Dataset for OntoNotes5."""
    
    def __init__(self, data, word2idx, label2idx):
        """
        Args:
            data: HuggingFace dataset split
            word2idx: Word to index mapping
            label2idx: Label to index mapping
        """
        self.data = data
        self.word2idx = word2idx
        self.label2idx = label2idx
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        tokens = sample['tokens']
        tags = sample['tags']
        
        # Convert to indices
        token_ids = [self.word2idx.get(token.lower(), self.word2idx['<UNK>']) 
                     for token in tokens]
        tag_ids = tags  # Already numeric in OntoNotes5
        
        return {
            'tokens': torch.LongTensor(token_ids),
            'tags': torch.LongTensor(tag_ids),
            'length': len(tokens)
        }


def build_vocab_ontonotes(dataset, min_freq=2):
    """
    Build vocabulary from OntoNotes5 dataset.
    
    Returns:
        dict: {
            'word2idx': word to index mapping,
            'idx2word': index to word mapping,
            'label2idx': label to index mapping,
            'idx2label': index to label mapping
        }
    """
    # Count words
    word_counter = Counter()
    for split in ['train', 'validation', 'test']:
        for sample in dataset[split]:
            for token in sample['tokens']:
                word_counter[token.lower()] += 1
    
    # Build word vocab
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counter.items():
        if count >= min_freq:
            word2idx[word] = len(word2idx)
    
    idx2word = {v: k for k, v in word2idx.items()}
    
    # OntoNotes5 label mapping (from dataset schema)
    # Tags are: O=0, B-TYPE=odd, I-TYPE=even
    label_names = [
        'O',  # 0
        'B-CARDINAL', 'I-CARDINAL',
        'B-DATE', 'I-DATE',
        'B-EVENT', 'I-EVENT',
        'B-FAC', 'I-FAC',
        'B-GPE', 'I-GPE',
        'B-LANGUAGE', 'I-LANGUAGE',
        'B-LAW', 'I-LAW',
        'B-LOC', 'I-LOC',
        'B-MONEY', 'I-MONEY',
        'B-NORP', 'I-NORP',
        'B-ORDINAL', 'I-ORDINAL',
        'B-ORG', 'I-ORG',
        'B-PERCENT', 'I-PERCENT',
        'B-PERSON', 'I-PERSON',
        'B-PRODUCT', 'I-PRODUCT',
        'B-QUANTITY', 'I-QUANTITY',
        'B-TIME', 'I-TIME',
        'B-WORK_OF_ART', 'I-WORK_OF_ART'
    ]
    
    label2idx = {label: idx for idx, label in enumerate(label_names)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    
    return {
        'word2idx': word2idx,
        'idx2word': idx2word,
        'label2idx': label2idx,
        'idx2label': idx2label
    }


def load_glove_embeddings_ontonotes(glove_path, word2idx, embed_dim=300):
    """Load GloVe embeddings (reuse from original)."""
    from .dataset import load_glove_embeddings
    return load_glove_embeddings(glove_path, word2idx, embed_dim)


def collate_fn_ontonotes(batch):
    """Collate function for DataLoader."""
    # Get max length in batch
    max_len = max([item['length'] for item in batch])
    
    # Pad sequences
    tokens_padded = []
    tags_padded = []
    lengths = []
    
    for item in batch:
        length = item['length']
        tokens = item['tokens']
        tags = item['tags']
        
        # Pad
        padding = max_len - length
        tokens_padded.append(torch.cat([tokens, torch.zeros(padding, dtype=torch.long)]))
        tags_padded.append(torch.cat([tags, torch.zeros(padding, dtype=torch.long)]))
        lengths.append(length)
    
    return (
        torch.stack(tokens_padded),
        torch.stack(tags_padded),
        torch.LongTensor(lengths)
    )


def prepare_ontonotes_data(cache_dir='./data/ontonotes', 
                           batch_size=32, 
                           min_freq=2):
    """
    Prepare OntoNotes5 data for training.
    
    Returns:
        train_loader, valid_loader, test_loader, vocab_info
    """
    # Download dataset
    dataset = download_ontonotes5(cache_dir)
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    vocab_info = build_vocab_ontonotes(dataset, min_freq=min_freq)
    
    print(f"Vocabulary size: {len(vocab_info['word2idx'])}")
    print(f"Number of labels: {len(vocab_info['label2idx'])}")
    print(f"Labels: {list(vocab_info['idx2label'].values())[:10]}... (showing first 10)")
    
    # Create datasets
    train_dataset = OntoNotes5Dataset(
        dataset['train'],
        vocab_info['word2idx'],
        vocab_info['label2idx']
    )
    
    valid_dataset = OntoNotes5Dataset(
        dataset['validation'],
        vocab_info['word2idx'],
        vocab_info['label2idx']
    )
    
    test_dataset = OntoNotes5Dataset(
        dataset['test'],
        vocab_info['word2idx'],
        vocab_info['label2idx']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_ontonotes
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_ontonotes
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_ontonotes
    )
    
    return train_loader, valid_loader, test_loader, vocab_info


if __name__ == '__main__':
    # Test
    print("Testing OntoNotes5 data loading...")
    train_loader, valid_loader, test_loader, vocab_info = prepare_ontonotes_data()
    
    # Check first batch
    for tokens, tags, lengths in train_loader:
        print(f"\nBatch shape:")
        print(f"  Tokens: {tokens.shape}")
        print(f"  Tags: {tags.shape}")
        print(f"  Lengths: {lengths.shape}")
        break
    
    print("\n✓ OntoNotes5 data loading successful!")
