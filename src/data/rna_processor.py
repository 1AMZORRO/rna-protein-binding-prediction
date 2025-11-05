"""
RNA sequence processor with one-hot encoding.
"""

import torch
import numpy as np
from typing import List, Union


class RNAProcessor:
    """Process RNA sequences with one-hot encoding."""
    
    def __init__(self, seq_length: int = 101):
        """
        Initialize RNA processor.
        
        Args:
            seq_length: Fixed length for RNA sequences (default: 101)
        """
        self.seq_length = seq_length
        # Nucleotide to index mapping
        self.nucleotide_to_idx = {
            'A': 0, 'C': 1, 'G': 2, 'U': 3,
            'T': 3,  # Treat T as U for RNA
            'N': 4   # Unknown nucleotide
        }
        self.num_nucleotides = 4  # A, C, G, U
        
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """
        Convert RNA sequence to one-hot encoding.
        
        Args:
            sequence: RNA sequence string
            
        Returns:
            One-hot encoded tensor of shape (seq_length, 4)
        """
        # Convert to uppercase
        sequence = sequence.upper().replace('T', 'U')
        
        # Pad or truncate to fixed length
        if len(sequence) < self.seq_length:
            # Pad with 'N' (which will be encoded as zeros)
            sequence = sequence + 'N' * (self.seq_length - len(sequence))
        elif len(sequence) > self.seq_length:
            # Truncate from center
            start = (len(sequence) - self.seq_length) // 2
            sequence = sequence[start:start + self.seq_length]
        
        # Create one-hot encoding
        encoding = np.zeros((self.seq_length, self.num_nucleotides), dtype=np.float32)
        
        for i, nucleotide in enumerate(sequence):
            idx = self.nucleotide_to_idx.get(nucleotide, 4)  # Default to 4 (N) for unknown
            if idx < 4:  # Only encode known nucleotides
                encoding[i, idx] = 1.0
        
        return torch.from_numpy(encoding)
    
    def encode_batch(self, sequences: List[str]) -> torch.Tensor:
        """
        Encode a batch of RNA sequences.
        
        Args:
            sequences: List of RNA sequence strings
            
        Returns:
            Batch of one-hot encoded tensors of shape (batch_size, seq_length, 4)
        """
        encoded = [self.encode_sequence(seq) for seq in sequences]
        return torch.stack(encoded)
    
    def decode_sequence(self, encoding: torch.Tensor) -> str:
        """
        Decode one-hot encoding back to RNA sequence.
        
        Args:
            encoding: One-hot encoded tensor of shape (seq_length, 4)
            
        Returns:
            RNA sequence string
        """
        idx_to_nucleotide = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
        
        if encoding.dim() == 3:  # (batch, seq_length, 4)
            encoding = encoding[0]  # Take first in batch
            
        sequence = []
        for i in range(encoding.shape[0]):
            idx = torch.argmax(encoding[i]).item()
            if idx < 4:
                sequence.append(idx_to_nucleotide[idx])
            else:
                sequence.append('N')
                
        return ''.join(sequence)
