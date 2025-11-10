"""
Dataset class for RNA-Protein binding prediction.
"""

import torch
from torch.utils.data import Dataset
from Bio import SeqIO
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path


class RNAProteinDataset(Dataset):
    """Dataset for RNA-Protein binding pairs."""
    
    def __init__(
        self,
        rna_sequences: List[str],
        protein_sequences: List[str],
        labels: Optional[List[int]] = None,
        rna_processor=None,
        protein_processor=None,
        protein_ids: Optional[List[str]] = None
    ):
        """
        Initialize dataset.
        
        Args:
            rna_sequences: List of RNA sequence strings
            protein_sequences: List of protein sequence strings
            labels: List of binary labels (1 for binding, 0 for non-binding)
            rna_processor: RNAProcessor instance
            protein_processor: ProteinProcessor instance
            protein_ids: Optional list of protein IDs for precomputed embeddings
        """
        self.rna_sequences = rna_sequences
        self.protein_sequences = protein_sequences
        self.labels = labels if labels is not None else [1] * len(rna_sequences)
        self.protein_ids = protein_ids
        
        # Check that all lists have the same length
        assert len(self.rna_sequences) == len(self.protein_sequences), \
            "RNA and protein sequences must have the same length"
        assert len(self.rna_sequences) == len(self.labels), \
            "Labels must have the same length as sequences"
        if protein_ids is not None:
            assert len(self.rna_sequences) == len(self.protein_ids), \
                "Protein IDs must have the same length as sequences"
        
        self.rna_processor = rna_processor
        self.protein_processor = protein_processor
        
    def __len__(self) -> int:
        return len(self.rna_sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (rna_embedding, protein_embedding, label)
        """
        rna_seq = self.rna_sequences[idx]
        protein_seq = self.protein_sequences[idx]
        label = self.labels[idx]
        protein_id = self.protein_ids[idx] if self.protein_ids else None
        
        # Encode sequences
        if self.rna_processor is not None:
            rna_emb = self.rna_processor.encode_sequence(rna_seq)
        else:
            rna_emb = torch.zeros(101, 4)  # Placeholder
            
        if self.protein_processor is not None:
            protein_emb = self.protein_processor.encode_sequence(
                protein_seq, 
                protein_id=protein_id
            )
        else:
            protein_emb = torch.zeros(100, 1280)  # Placeholder
        
        label = torch.tensor(label, dtype=torch.float32)
        
        return rna_emb, protein_emb, label
    
    @staticmethod
    def load_fasta(fasta_file: str) -> List[str]:
        """
        Load sequences from FASTA file.
        
        Args:
            fasta_file: Path to FASTA file
            
        Returns:
            List of sequence strings
        """
        sequences = []
        with open(fasta_file, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                sequences.append(str(record.seq))
        return sequences
    
    @staticmethod
    def load_fasta_with_ids(fasta_file: str) -> Tuple[List[str], List[str]]:
        """
        Load sequences and IDs from FASTA file.
        
        Args:
            fasta_file: Path to FASTA file
            
        Returns:
            Tuple of (sequences, ids)
        """
        sequences = []
        ids = []
        with open(fasta_file, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                sequences.append(str(record.seq))
                ids.append(record.id)
        return sequences, ids
    
    @staticmethod
    def from_fasta_files(
        rna_fasta: str,
        protein_fasta: str,
        labels: Optional[List[int]] = None,
        rna_processor=None,
        protein_processor=None,
        use_protein_ids: bool = False
    ):
        """
        Create dataset from FASTA files.
        
        Args:
            rna_fasta: Path to RNA FASTA file
            protein_fasta: Path to protein FASTA file
            labels: Optional list of labels
            rna_processor: RNAProcessor instance
            protein_processor: ProteinProcessor instance
            use_protein_ids: Whether to extract and use protein IDs for precomputed embeddings
            
        Returns:
            RNAProteinDataset instance
        """
        rna_sequences = RNAProteinDataset.load_fasta(rna_fasta)
        
        if use_protein_ids:
            protein_sequences, protein_ids = RNAProteinDataset.load_fasta_with_ids(protein_fasta)
        else:
            protein_sequences = RNAProteinDataset.load_fasta(protein_fasta)
            protein_ids = None
        
        return RNAProteinDataset(
            rna_sequences,
            protein_sequences,
            labels,
            rna_processor,
            protein_processor,
            protein_ids
        )


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    
    Args:
        batch: List of (rna_emb, protein_emb, label) tuples
        
    Returns:
        Batched tensors with padding
    """
    rna_embs, protein_embs, labels = zip(*batch)
    
    # RNA sequences are already fixed length, just stack
    rna_batch = torch.stack(rna_embs)
    
    # Protein sequences may have variable length, need padding
    max_protein_len = max(emb.shape[0] for emb in protein_embs)
    protein_dim = protein_embs[0].shape[1]
    
    padded_proteins = []
    protein_masks = []
    
    for emb in protein_embs:
        seq_len = emb.shape[0]
        if seq_len < max_protein_len:
            # Pad with zeros
            padding = torch.zeros(max_protein_len - seq_len, protein_dim)
            padded_emb = torch.cat([emb, padding], dim=0)
        else:
            padded_emb = emb
        
        # Create mask (1 for real tokens, 0 for padding)
        mask = torch.zeros(max_protein_len)
        mask[:seq_len] = 1
        
        padded_proteins.append(padded_emb)
        protein_masks.append(mask)
    
    protein_batch = torch.stack(padded_proteins)
    protein_mask_batch = torch.stack(protein_masks)
    labels_batch = torch.stack(labels)
    
    return rna_batch, protein_batch, protein_mask_batch, labels_batch
