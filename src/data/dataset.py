import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import os

from utils.text_processing import syllabify_ipa

class CuedSpeechDataset(Dataset):
    """Dataset for Cued Speech Recognition."""
    
    def __init__(self, features_dir: str, labels_dir: str, mode: str = 'train', vocab: Dict[str, int] = None):
        """
        Initialize the dataset.
        
        Args:
            features_dir: Directory containing feature files
            labels_dir: Directory containing label files
            mode: 'train' or 'test'
            vocab: Vocabulary dictionary mapping tokens to indices
        """
        if not isinstance(features_dir, str):
            raise TypeError(f"features_dir must be a string, got {type(features_dir)}")
        if not isinstance(labels_dir, str):
            raise TypeError(f"labels_dir must be a string, got {type(labels_dir)}")
            
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.mode = mode
        self.vocab = vocab
        
        # Load data
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        """Load and preprocess data."""
        data = []
        
        # Get all feature files
        if not os.path.exists(self.features_dir):
            raise FileNotFoundError(f"Features directory not found: {self.features_dir}")
        if not os.path.exists(self.labels_dir):
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")
            
        feature_files = [f for f in os.listdir(self.features_dir) 
                        if f.endswith('_features.csv') and 'sent_' not in f]
        for feature_file in sorted(feature_files):
            base_name = feature_file.replace('_features.csv', '')
            
            # Load features
            features_df = pd.read_csv(os.path.join(self.features_dir, feature_file))
            features_df.dropna(inplace=True)
            
            # Separate features into different modalities
            hand_shape_cols = [col for col in features_df.columns 
                             if "hand" in col and "face" not in col]
            hand_pos_cols = [col for col in features_df.columns 
                           if "face" in col]
            lip_cols = [col for col in features_df.columns 
                       if "lip" in col]
            
            # Extract features
            hand_shape = features_df[hand_shape_cols].to_numpy()
            hand_pos = features_df[hand_pos_cols].to_numpy()
            lips = features_df[lip_cols].to_numpy()
            
            # Load labels
            label_file = os.path.join(self.labels_dir, f"{base_name}.csv")
            if os.path.exists(label_file):
                labels_df = pd.read_csv(label_file, header=None)
                labels = syllabify_ipa(" ".join(labels_df.squeeze().tolist()[1:-1]))  # Skip header and footer
                # Convert string labels to numeric indices if vocab is provided
                if self.vocab is not None:
                    numeric_labels = [self.vocab['<SOS>']]
                    for label in labels:
                        if isinstance(label, str):
                            if label in self.vocab:
                                numeric_labels.append(self.vocab[label])
                            else:
                                # Use a default index for unknown tokens
                                numeric_labels.append(self.vocab.get('<UNK>', 0))
                                print("should not happen: ", label, " , ", base_name)
                        else:
                            numeric_labels.append(label)
                    numeric_labels.append(self.vocab['<EOS>'])
                    labels = numeric_labels
                
                data.append({
                    'hand_shape': hand_shape,
                    'hand_pos': hand_pos,
                    'lips': lips,
                    'labels': labels
                })
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, 
                                            torch.Tensor, torch.Tensor]:
        """Get a data sample."""
        sample = self.data[idx]
        
        # Ensure labels are numeric
        labels = sample['labels']
        if not all(isinstance(label, (int, float)) for label in labels):
            # Convert any remaining string labels to numeric
            if self.vocab is not None:
                numeric_labels = []
                for label in labels:
                    if isinstance(label, str):
                        if label in self.vocab:
                            numeric_labels.append(self.vocab[label])
                        else:
                            numeric_labels.append(self.vocab.get('<UNK>', 0))
                    else:
                        numeric_labels.append(label)
                labels = numeric_labels
            else:
                # If no vocab provided, use a simple mapping
                label_map = {label: i for i, label in enumerate(set(labels))}
                labels = [label_map[label] for label in labels]
        
        return (
            torch.FloatTensor(sample['hand_shape']),
            torch.FloatTensor(sample['hand_pos']),
            torch.FloatTensor(sample['lips']),
            torch.LongTensor(labels)
        )

def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """
    Custom collate function for batching.
    
    Args:
        batch: List of (hand_shape, hand_pos, lips, labels) tuples
        
    Returns:
        Tuple of padded tensors
    """
    hand_shapes, hand_positions, lips, labels = zip(*batch)
    
    # Pad sequences
    hand_shapes_padded = pad_sequence(
        hand_shapes,
        batch_first=True,
        padding_value=0
    )
    
    hand_positions_padded = pad_sequence(
        hand_positions,
        batch_first=True,
        padding_value=0
    )
    
    lips_padded = pad_sequence(
        lips,
        batch_first=True,
        padding_value=0
    )
    
    labels_padded = pad_sequence(
        labels,
        batch_first=True,
        padding_value=1  # Use 0 as padding for labels
    )
    
    return hand_shapes_padded, hand_positions_padded, lips_padded, labels_padded

def create_dataloader(features_dir: str, labels_dir: str, mode: str = 'train',
                     batch_size: int = 32, shuffle: bool = True,
                     num_workers: int = 4, vocab: Dict[str, int] = None) -> DataLoader:
    """
    Create a DataLoader for the dataset.
    
    Args:
        features_dir: Directory containing feature files
        labels_dir: Directory containing label files
        mode: 'train' or 'test'
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        vocab: Vocabulary dictionary mapping tokens to indices
        
    Returns:
        DataLoader instance
    """
    dataset = CuedSpeechDataset(features_dir, labels_dir, mode, vocab)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    ) 