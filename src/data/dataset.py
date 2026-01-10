"""
Data loading utilities for CoNLL-2003 NER dataset.
Handles IOB2 format and prepares data for PyTorch training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
from collections import Counter
import urllib.request
import os


class CoNLL2003Dataset(Dataset):
    """CoNLL-2003 NER dataset in IOB2 format."""
    
    def __init__(self, sentences: List[List[str]], labels: List[List[str]], 
                 word2idx: Dict[str, int], label2idx: Dict[str, int]):
        """
        Args:
            sentences: List of tokenized sentences
            labels: List of label sequences (IOB2 format)
            word2idx: Word to index mapping
            label2idx: Label to index mapping
        """
        self.sentences = sentences
        self.labels = labels
        self.word2idx = word2idx
        self.label2idx = label2idx
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        # Convert words and labels to indices
        word_ids = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in self.sentences[idx]]
        label_ids = [self.label2idx[l] for l in self.labels[idx]]
        
        return torch.tensor(word_ids, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)


def download_conll2003(data_dir: str = './data/raw'):
    """
    Download CoNLL-2003 dataset from Hugging Face.
    Uses the datasets library for reliable access.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.check_call(['pip', 'install', '-q', 'datasets'])
        from datasets import load_dataset
    
    os.makedirs(data_dir, exist_ok=True)
    
    print("Downloading CoNLL-2003 from Hugging Face...")
    dataset = load_dataset("eriktks/conll2003")
    
    # Convert to CoNLL format and save
    for split_name, hf_split in [('train', 'train'), ('valid', 'validation'), ('test', 'test')]:
        filepath = os.path.join(data_dir, f'{split_name}.txt')
        
        if not os.path.exists(filepath):
            print(f"Creating {split_name}.txt...")
            with open(filepath, 'w', encoding='utf-8') as f:
                for example in dataset[hf_split]:
                    # Write each token with its NER tag
                    for token, ner_tag in zip(example['tokens'], example['ner_tags']):
                        tag_name = dataset[hf_split].features['ner_tags'].feature.int2str(ner_tag)
                        # CoNLL format: token pos chunk ner_tag
                        f.write(f"{token} X X {tag_name}\n")
                    f.write("\n")  # Empty line between sentences
            print(f"✓ Created {split_name}.txt")
        else:
            print(f"✓ {split_name}.txt already exists")
    
    return data_dir


def read_conll_file(filepath: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Read CoNLL format file.
    
    Format:
    word pos chunk tag
    -DOCSTART- -X- -X- O
    
    EU NNP B-NP B-ORG
    rejects VBZ B-VP O
    ...
    
    Returns:
        sentences: List of token lists
        labels: List of label lists (NER tags only)
    """
    sentences = []
    labels = []
    
    current_sentence = []
    current_labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Empty line = sentence boundary
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
                continue
            
            # Skip -DOCSTART- lines
            if line.startswith('-DOCSTART-'):
                continue
            
            # Parse token line
            parts = line.split()
            if len(parts) >= 4:
                token = parts[0]
                label = parts[3]  # NER tag is 4th column
                
                current_sentence.append(token)
                current_labels.append(label)
    
    # Add last sentence if exists
    if current_sentence:
        sentences.append(current_sentence)
        labels.append(current_labels)
    
    return sentences, labels


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


def build_label_vocab(labels: List[List[str]]) -> Dict[str, int]:
    """Build label vocabulary."""
    unique_labels = set()
    for label_seq in labels:
        unique_labels.update(label_seq)
    
    # Sort for consistency
    label_list = sorted(unique_labels)
    label2idx = {label: idx for idx, label in enumerate(label_list)}
    
    return label2idx


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


def prepare_data(data_dir: str = './data/raw', 
                 batch_size: int = 32,
                 min_freq: int = 2):
    """
    Prepare CoNLL-2003 data for training.
    
    Returns:
        train_loader, valid_loader, test_loader: DataLoaders
        word2idx, idx2word: Word vocabulary
        label2idx, idx2label: Label vocabulary
    """
    # Download if needed
    if not os.path.exists(os.path.join(data_dir, 'train.txt')):
        download_conll2003(data_dir)
    
    # Read data
    print("Loading training data...")
    train_sentences, train_labels = read_conll_file(os.path.join(data_dir, 'train.txt'))
    print("Loading validation data...")
    valid_sentences, valid_labels = read_conll_file(os.path.join(data_dir, 'valid.txt'))
    print("Loading test data...")
    test_sentences, test_labels = read_conll_file(os.path.join(data_dir, 'test.txt'))
    
    print(f"Train: {len(train_sentences)} sentences")
    print(f"Valid: {len(valid_sentences)} sentences")
    print(f"Test: {len(test_sentences)} sentences")
    
    # Build vocabularies
    print("\nBuilding vocabularies...")
    word2idx = build_vocab(train_sentences, min_freq=min_freq)
    label2idx = build_label_vocab(train_labels)
    
    idx2word = {idx: word for word, idx in word2idx.items()}
    idx2label = {idx: label for label, idx in label2idx.items()}
    
    print(f"Vocabulary size: {len(word2idx)}")
    print(f"Number of labels: {len(label2idx)}")
    print(f"Labels: {list(label2idx.keys())}")
    
    # Create datasets
    train_dataset = CoNLL2003Dataset(train_sentences, train_labels, word2idx, label2idx)
    valid_dataset = CoNLL2003Dataset(valid_sentences, valid_labels, word2idx, label2idx)
    test_dataset = CoNLL2003Dataset(test_sentences, test_labels, word2idx, label2idx)
    
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
    train_loader, valid_loader, test_loader, vocab_info = prepare_data()
    
    # Print sample batch
    for sentences, labels, lengths in train_loader:
        print(f"Batch shape: {sentences.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Lengths: {lengths[:5]}")
        break
