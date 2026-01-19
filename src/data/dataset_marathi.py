"""
Data loading utilities for Naamapadam (Marathi) NER dataset from HuggingFace.
Handles the ai4bharat/naamapadam dataset and prepares data for PyTorch training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
from collections import Counter
import os


class NaampadamDataset(Dataset):
    """Naamapadam NER dataset (Marathi config)."""
    
    def __init__(self, sentences: List[List[str]], labels: List[List[int]], 
                 word2idx: Dict[str, int], pad_idx: int = 0):
        """
        Args:
            sentences: List of tokenized sentences
            labels: List of label sequences (numerical indices)
            word2idx: Word to index mapping
            pad_idx: Padding index for labels
        """
        self.sentences = sentences
        self.labels = labels
        self.word2idx = word2idx
        self.pad_idx = pad_idx
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        # Convert words and labels to indices
        word_ids = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in self.sentences[idx]]
        label_ids = self.labels[idx]
        
        return torch.tensor(word_ids, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)


def load_naamapadam_data(cache_dir: str = './data/naamapadam'):
    """
    Load Naamapadam dataset from HuggingFace.
    
    Returns:
        train_data: (sentences, labels) for training
        valid_data: (sentences, labels) for validation (using test split)
        test_data: (sentences, labels) for test
        label_names: List of NER tag names
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets library: pip install datasets")
    
    print("Loading Naamapadam dataset from HuggingFace...")
    print("Dataset: ai4bharat/naamapadam")
    print("Configuration: mr (Marathi)")
    
    # Load dataset with Marathi configuration
    dataset = load_dataset("ai4bharat/naamapadam", "mr", cache_dir=cache_dir)
    
    # The Naamapadam dataset has 'train' split only
    # We'll split it into train/valid/test
    train_dataset = dataset['train']
    
    print(f"\nTotal samples: {len(train_dataset)}")
    
    # Get label names from dataset features
    label_feature = train_dataset.features['ner_tags']
    label_names = label_feature.feature.names
    
    print(f"Number of labels: {len(label_names)}")
    print(f"Label names: {label_names}")
    
    # Extract sentences and labels
    all_sentences = []
    all_labels = []
    
    for item in train_dataset:
        tokens = item['tokens']
        ner_tags = item['ner_tags']
        
        # Filter out empty sentences
        if len(tokens) > 0:
            all_sentences.append(tokens)
            all_labels.append(ner_tags)
    
    print(f"\nProcessed {len(all_sentences)} sentences")
    
    # Split data: 80% train, 10% valid, 10% test
    total = len(all_sentences)
    train_size = int(0.8 * total)
    valid_size = int(0.1 * total)
    
    train_sentences = all_sentences[:train_size]
    train_labels = all_labels[:train_size]
    
    valid_sentences = all_sentences[train_size:train_size + valid_size]
    valid_labels = all_labels[train_size:train_size + valid_size]
    
    test_sentences = all_sentences[train_size + valid_size:]
    test_labels = all_labels[train_size + valid_size:]
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_sentences)} sentences")
    print(f"  Valid: {len(valid_sentences)} sentences")
    print(f"  Test: {len(test_sentences)} sentences")
    
    return (train_sentences, train_labels), (valid_sentences, valid_labels), (test_sentences, test_labels), label_names


def build_vocab(sentences: List[List[str]], min_freq: int = 2) -> Dict[str, int]:
    """Build vocabulary from sentences."""
    counter = Counter()
    for sentence in sentences:
        counter.update(sentence)
    
    # Add special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1}
    
    # Add words that appear at least min_freq times
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab


def load_glove_embeddings(glove_path: str, word2idx: Dict[str, int], embed_dim: int = 300):
    """
    Load pre-trained GloVe embeddings.
    
    Args:
        glove_path: Path to GloVe file (e.g., glove.6B.300d.txt)
        word2idx: Vocabulary mapping
        embed_dim: Embedding dimension
    
    Returns:
        embedding_matrix: numpy array of shape (vocab_size, embed_dim)
    """
    import numpy as np
    
    vocab_size = len(word2idx)
    embedding_matrix = np.random.randn(vocab_size, embed_dim) * 0.01
    
    # Set padding to zeros
    embedding_matrix[word2idx['<PAD>']] = np.zeros(embed_dim)
    
    if not os.path.exists(glove_path):
        print(f"Warning: GloVe file not found at {glove_path}")
        print("Will use random embeddings. Download GloVe from: https://nlp.stanford.edu/projects/glove/")
        return embedding_matrix
    
    print(f"Loading GloVe embeddings from {glove_path}...")
    found = 0
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in word2idx:
                embedding = np.array([float(x) for x in parts[1:]])
                embedding_matrix[word2idx[word]] = embedding
                found += 1
    
    print(f"Found embeddings for {found}/{vocab_size} words ({100*found/vocab_size:.1f}%)")
    
    return embedding_matrix


def collate_fn(batch):
    """Collate function for DataLoader - handles variable length sequences."""
    sentences, labels = zip(*batch)
    
    # Get lengths
    lengths = torch.tensor([len(s) for s in sentences])
    
    # Pad sequences
    max_len = lengths.max().item()
    padded_sentences = torch.zeros(len(sentences), max_len, dtype=torch.long)
    padded_labels = torch.zeros(len(labels), max_len, dtype=torch.long)
    
    for i, (sent, label) in enumerate(zip(sentences, labels)):
        length = len(sent)
        padded_sentences[i, :length] = sent
        padded_labels[i, :length] = label
    
    return padded_sentences, padded_labels, lengths


def prepare_marathi_data(cache_dir: str = './data/naamapadam',
                         batch_size: int = 32,
                         min_freq: int = 2):
    """
    Prepare Naamapadam (Marathi) data for training.
    
    Returns:
        train_loader, valid_loader, test_loader: DataLoaders
        word2idx, idx2word: Word vocabulary
        label2idx, idx2label: Label vocabulary
    """
    # Load data from HuggingFace
    train_data, valid_data, test_data, label_names = load_naamapadam_data(cache_dir)
    
    train_sentences, train_labels = train_data
    valid_sentences, valid_labels = valid_data
    test_sentences, test_labels = test_data
    
    # Build vocabularies
    print("\nBuilding vocabularies...")
    word2idx = build_vocab(train_sentences, min_freq=min_freq)
    
    # Create label mapping from label names
    label2idx = {label: idx for idx, label in enumerate(label_names)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    idx2label = {idx: label for label, idx in label2idx.items()}
    
    print(f"Vocabulary size: {len(word2idx)}")
    print(f"Number of labels: {len(label2idx)}")
    print(f"Labels: {list(label2idx.keys())}")
    
    # Create datasets
    train_dataset = NaampadamDataset(train_sentences, train_labels, word2idx)
    valid_dataset = NaampadamDataset(valid_sentences, valid_labels, word2idx)
    test_dataset = NaampadamDataset(test_sentences, test_labels, word2idx)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, 
                              shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             shuffle=False, collate_fn=collate_fn)
    
    vocab_info = {
        'word2idx': word2idx,
        'idx2word': idx2word,
        'label2idx': label2idx,
        'idx2label': idx2label
    }
    
    return train_loader, valid_loader, test_loader, vocab_info


if __name__ == '__main__':
    # Test data loading
    train_loader, valid_loader, test_loader, vocab_info = prepare_marathi_data()
    
    # Print sample batch
    for sentences, labels, lengths in train_loader:
        print(f"\nBatch shape: {sentences.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Lengths: {lengths[:5]}")
        break
