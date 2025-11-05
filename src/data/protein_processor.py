"""
Protein sequence processor with ESM2 embedding.
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional
import numpy as np


class ProteinProcessor:
    """Process protein sequences using ESM2 pre-trained model."""
    
    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D", device: str = "cuda"):
        """
        Initialize protein processor with ESM2 model.
        
        Args:
            model_name: Name of the ESM2 model to use
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # Load ESM2 tokenizer and model
        print(f"Loading ESM2 model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        
    def encode_sequence(self, sequence: str, max_length: Optional[int] = None) -> torch.Tensor:
        """
        Convert protein sequence to ESM2 embedding.
        
        Args:
            sequence: Protein sequence string
            max_length: Maximum sequence length (truncate if longer)
            
        Returns:
            Embedding tensor of shape (seq_length, embedding_dim)
        """
        # Convert to uppercase
        sequence = sequence.upper()
        
        # Tokenize
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use last hidden state as embeddings
            embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_length, embedding_dim)
            
            # Remove special tokens (CLS and EOS)
            embeddings = embeddings[1:-1, :]
            
        return embeddings.cpu()
    
    def encode_batch(self, sequences: List[str], max_length: Optional[int] = None) -> torch.Tensor:
        """
        Encode a batch of protein sequences.
        
        Args:
            sequences: List of protein sequence strings
            max_length: Maximum sequence length
            
        Returns:
            Batch of embeddings with padding to the longest sequence
        """
        # Encode each sequence
        embeddings_list = [self.encode_sequence(seq, max_length) for seq in sequences]
        
        # Find max length in batch
        max_len = max(emb.shape[0] for emb in embeddings_list)
        
        # Pad sequences to same length
        padded_embeddings = []
        for emb in embeddings_list:
            if emb.shape[0] < max_len:
                # Pad with zeros
                padding = torch.zeros(max_len - emb.shape[0], emb.shape[1])
                emb = torch.cat([emb, padding], dim=0)
            padded_embeddings.append(emb)
        
        return torch.stack(padded_embeddings)
    
    def get_mean_embedding(self, sequence: str) -> torch.Tensor:
        """
        Get mean-pooled embedding for a protein sequence.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Mean-pooled embedding tensor of shape (embedding_dim,)
        """
        embeddings = self.encode_sequence(sequence)
        return embeddings.mean(dim=0)
