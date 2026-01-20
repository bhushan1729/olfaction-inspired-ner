"""
Unified Data Loader for Olfactory NER.
Handles loading of multiple datasets (CoNLL, WikiANN, etc.) and languages through a single interface.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter

# Import utilities from existing dataset modules to reuse code
from src.data.dataset import (
    CoNLL2003Dataset, 
    read_conll_file, 
    download_conll2003, 
    build_vocab, 
    collate_fn
)
# We can reuse NaampadamDataset as a generic HF Token Classification Dataset wrapper
from src.data.dataset_marathi import NaampadamDataset as GenericHFDataset


def load_huggingface_dataset(dataset_name: str, config_name: Optional[str] = None, cache_dir: str = './data'):
    """
    Generic loader for HuggingFace Token Classification datasets.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'wikiann', 'masakhaner')
        config_name: Configuration/Language code (e.g., 'mr', 'yo', 'en'). Can be None.
        cache_dir: Directory to cache the dataset
        
    Returns:
        train_data, valid_data, test_data, label_names
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets library: pip install datasets")
    
    print(f"Loading {dataset_name} dataset from HuggingFace...")
    print(f"Configuration: {config_name if config_name else 'default'}")
    
    # Load dataset
    if config_name:
        dataset = load_dataset(dataset_name, config_name, cache_dir=cache_dir)
    else:
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        
    print(f"\nData splits available: {list(dataset.keys())}")
    
    # Standardize split names (HF datasets use 'validation' or 'valid')
    train_split = dataset['train']
    valid_split = dataset['validation'] if 'validation' in dataset else dataset['valid'] if 'valid' in dataset else dataset['test'] # Fallback
    test_split = dataset['test']
    
    # Get label names
    # Try common column names for tags
    tag_column = 'ner_tags' if 'ner_tags' in train_split.features else 'tags' if 'tags' in train_split.features else None
    
    if not tag_column:
        raise ValueError(f"Could not find tag column in dataset. Features: {train_split.features.keys()}")
        
    label_feature = train_split.features[tag_column]
    label_names = label_feature.feature.names
    
    print(f"Number of labels: {len(label_names)}")
    print(f"Label names: {label_names}")
    
    # Extract function
    def extract_data(split_dataset):
        sentences = []
        labels = []
        token_column = 'tokens' if 'tokens' in split_dataset.features else 'sentence' # Some datasets might behave differently
        
        for item in split_dataset:
            # Handle token column
            if token_column in item:
                tokens = item[token_column]
            else:
                 # Fallback logic if needed
                 tokens = []
            
            tags = item[tag_column]
            
            if len(tokens) > 0:
                sentences.append(tokens)
                labels.append(tags)
        return sentences, labels
    
    train_sentences, train_labels = extract_data(train_split)
    valid_sentences, valid_labels = extract_data(valid_split)
    test_sentences, test_labels = extract_data(test_split)
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_sentences)} sentences")
    print(f"  Valid: {len(valid_sentences)} sentences")
    print(f"  Test: {len(test_sentences)} sentences")
    
    return (train_sentences, train_labels), (valid_sentences, valid_labels), (test_sentences, test_labels), label_names


def get_dataset(dataset_name: str, language: Optional[str] = None, 
                cache_dir: str = './data', 
                batch_size: int = 32,
                min_freq: int = 2):
    """
    Universal factory function to get data loaders for any supported dataset.
    
    Args:
        dataset_name: 'conll2003', 'wikiann', 'masakhaner', etc.
        language: Language code (e.g., 'mr', 'en'). Required for multilingual datasets.
        cache_dir: Directory to store data
        batch_size: Batch size for loaders
        min_freq: Minimum frequency for vocabulary
        
    Returns:
        train_loader, valid_loader, test_loader, vocab_info
    """
    
    dataset_name = dataset_name.lower()
    
    # 1. Handle CoNLL-2003 (English)
    if dataset_name == 'conll2003' or dataset_name == 'conll':
        print(f"Initialize CoNLL-2003 dataset loading...")
        raw_dir = os.path.join(cache_dir, 'conll2003')
        
        # Download if needed
        if not os.path.exists(os.path.join(raw_dir, 'train.txt')):
            download_conll2003(raw_dir)
            
        print("Loading training data...")
        train_sentences, train_labels_str = read_conll_file(os.path.join(raw_dir, 'train.txt'))
        valid_sentences, valid_labels_str = read_conll_file(os.path.join(raw_dir, 'valid.txt'))
        test_sentences, test_labels_str = read_conll_file(os.path.join(raw_dir, 'test.txt'))
        
        # Build vocab
        print("\nBuilding vocabularies...")
        word2idx = build_vocab(train_sentences, min_freq=min_freq)
        
        # Build label vocab (strings to ints)
        unique_labels = set()
        for label_seq in train_labels_str:
            unique_labels.update(label_seq)
        label_list = sorted(unique_labels)
        label2idx = {label: idx for idx, label in enumerate(label_list)}
        
        # Convert string labels to indices for CoNLL
        def convert_labels(label_seqs):
            return [[label2idx[l] for l in seq] for seq in label_seqs]
            
        train_labels = convert_labels(train_labels_str)
        valid_labels = convert_labels(valid_labels_str)
        test_labels = convert_labels(test_labels_str)
        
        # Create datasets (Reuse GenericHFDataset as it takes indices)
        # Note: CoNLL2003Dataset in dataset.py took string labels and converted them.
        # GenericHFDataset takes integer labels. We converted them above.
        train_dataset = GenericHFDataset(train_sentences, train_labels, word2idx)
        valid_dataset = GenericHFDataset(valid_sentences, valid_labels, word2idx)
        test_dataset = GenericHFDataset(test_sentences, test_labels, word2idx)


    # 2. Handle HuggingFace Datasets (WikiANN, etc.)
    elif dataset_name in ['wikiann', 'masakhaner', 'euronews']:
        # Ensure language is provided if required by the dataset (Wikiann requires it)
        # But we pass whatever is given. 'wikiann' needs a config name.
        
        if dataset_name == 'wikiann' and not language:
             print("Warning: WikiANN usually requires a language code (e.g., 'mr', 'en'). Using default if available or erroring.")
             
        hf_cache_dir = os.path.join(cache_dir, dataset_name)
        
        train_data, valid_data, test_data, label_names = load_huggingface_dataset(
            dataset_name, language, cache_dir=hf_cache_dir
        )
        
        train_sentences, train_labels = train_data
        valid_sentences, valid_labels = valid_data
        test_sentences, test_labels = test_data
        
        # Build vocab
        print("\nBuilding vocabularies...")
        word2idx = build_vocab(train_sentences, min_freq=min_freq)
        
        # Create label map
        label2idx = {label: idx for idx, label in enumerate(label_names)}
        
        # Create datasets
        train_dataset = GenericHFDataset(train_sentences, train_labels, word2idx)
        valid_dataset = GenericHFDataset(valid_sentences, valid_labels, word2idx)
        test_dataset = GenericHFDataset(test_sentences, test_labels, word2idx)
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Common: Create DataLoaders and Return
    
    # Create reverse mappings
    idx2word = {idx: word for word, idx in word2idx.items()}
    idx2label = {idx: label for label, idx in label2idx.items()}
    
    print(f"Vocabulary size: {len(word2idx)}")
    print(f"Number of labels: {len(label2idx)}")
    
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
