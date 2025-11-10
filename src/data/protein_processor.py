"""
Protein sequence processor with ESM2 embedding.
"""

import torch
import esm
from esm import pretrained
from typing import List, Optional, Dict
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ProteinProcessor:
    """Process protein sequences using ESM2 pre-trained model."""
    
    def __init__(
        self, 
        model_name: str = "esm2_t33_650M_UR50D", 
        device: str = "cuda",
        precomputed_embeddings_path: Optional[str] = None
    ):
        """
        Initialize protein processor with ESM2 model.
        
        Args:
            model_name: Name of the ESM2 model to use (e.g., 'esm2_t33_650M_UR50D')
            device: Device to run the model on ('cuda' or 'cpu')
            precomputed_embeddings_path: Path to precomputed embeddings file (.pt)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.precomputed_embeddings = {}
        
        # Load precomputed embeddings if available
        if precomputed_embeddings_path and Path(precomputed_embeddings_path).exists():
            logger.info(f"Loading precomputed embeddings from {precomputed_embeddings_path}")
            self.precomputed_embeddings = torch.load(precomputed_embeddings_path, map_location='cpu')
            logger.info(f"Loaded {len(self.precomputed_embeddings)} precomputed embeddings")
            # Get embedding dimension from precomputed embeddings
            first_key = next(iter(self.precomputed_embeddings))
            self.embedding_dim = self.precomputed_embeddings[first_key].shape[-1]
            self.model = None
            self.alphabet = None
            self.batch_converter = None
        else:
            # Load ESM2 model using fair-esm library
            logger.info(f"Loading ESM2 model: {model_name}")
            logger.info("Successfully imported ESM library")
            
            # Load model based on name
            if model_name == "esm2_t33_650M_UR50D":
                self.model, self.alphabet = pretrained.esm2_t33_650M_UR50D()
            elif model_name == "esm2_t36_3B_UR50D":
                self.model, self.alphabet = pretrained.esm2_t36_3B_UR50D()
            elif model_name == "esm2_t30_150M_UR50D":
                self.model, self.alphabet = pretrained.esm2_t30_150M_UR50D()
            elif model_name == "esm2_t12_35M_UR50D":
                self.model, self.alphabet = pretrained.esm2_t12_35M_UR50D()
            elif model_name == "esm2_t6_8M_UR50D":
                self.model, self.alphabet = pretrained.esm2_t6_8M_UR50D()
            else:
                # Default to t33_650M
                logger.warning(f"Unknown model {model_name}, using esm2_t33_650M_UR50D")
                self.model, self.alphabet = pretrained.esm2_t33_650M_UR50D()
            
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            self.batch_converter = self.alphabet.get_batch_converter()
            
            # Get embedding dimension
            self.embedding_dim = self.model.embed_dim
            
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
    def encode_sequence(
        self, 
        sequence: str, 
        max_length: Optional[int] = None,
        protein_id: Optional[str] = None
    ) -> torch.Tensor:
        """
        Convert protein sequence to ESM2 embedding.
        
        Args:
            sequence: Protein sequence string
            max_length: Maximum sequence length (truncate if longer)
            protein_id: Protein identifier for precomputed embeddings lookup
            
        Returns:
            Embedding tensor of shape (seq_length, embedding_dim)
        """
        # Check if we have precomputed embedding for this protein
        if protein_id and protein_id in self.precomputed_embeddings:
            return self.precomputed_embeddings[protein_id].cpu()
        
        # If no model loaded (using only precomputed embeddings), raise error
        if self.model is None:
            raise ValueError(
                f"No precomputed embedding found for protein '{protein_id}' "
                f"and no model is loaded. Please provide a model or precompute embeddings."
            )
        
        # Convert to uppercase
        sequence = sequence.upper()
        
        # Truncate if needed
        if max_length and len(sequence) > max_length:
            sequence = sequence[:max_length]
        
        # Prepare batch
        data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            # Get the number of layers
            num_layers = self.model.num_layers
            results = self.model(batch_tokens, repr_layers=[num_layers], return_contacts=False)
            token_representations = results["representations"][num_layers]
            
            # Remove BOS and EOS tokens (first and last)
            embeddings = token_representations[0, 1:-1, :]
            
        return embeddings.cpu()
    
    def encode_batch(
        self, 
        sequences: List[str], 
        max_length: Optional[int] = None,
        protein_ids: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Encode a batch of protein sequences.
        
        Args:
            sequences: List of protein sequence strings
            max_length: Maximum sequence length
            protein_ids: Optional list of protein IDs for precomputed embeddings lookup
            
        Returns:
            Batch of embeddings with padding to the longest sequence
        """
        # Encode each sequence
        embeddings_list = []
        for i, seq in enumerate(sequences):
            protein_id = protein_ids[i] if protein_ids else None
            emb = self.encode_sequence(seq, max_length, protein_id)
            embeddings_list.append(emb)
        
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
    
    def get_mean_embedding(self, sequence: str, protein_id: Optional[str] = None) -> torch.Tensor:
        """
        Get mean-pooled embedding for a protein sequence.
        
        Args:
            sequence: Protein sequence string
            protein_id: Optional protein ID for precomputed embeddings lookup
            
        Returns:
            Mean-pooled embedding tensor of shape (embedding_dim,)
        """
        embeddings = self.encode_sequence(sequence, protein_id=protein_id)
        return embeddings.mean(dim=0)
    
    def save_precomputed_embeddings(self, embeddings_dict: Dict[str, torch.Tensor], save_path: str):
        """
        Save precomputed embeddings to disk.
        
        Args:
            embeddings_dict: Dictionary mapping protein IDs to embeddings
            save_path: Path to save the embeddings
        """
        torch.save(embeddings_dict, save_path)
        logger.info(f"Saved {len(embeddings_dict)} embeddings to {save_path}")
    
    def precompute_embeddings_from_fasta(
        self, 
        fasta_path: str, 
        save_path: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Precompute embeddings for all sequences in a FASTA file.
        
        Args:
            fasta_path: Path to FASTA file
            save_path: Optional path to save the embeddings
            
        Returns:
            Dictionary mapping protein IDs to embeddings
        """
        from Bio import SeqIO
        
        if self.model is None:
            raise ValueError("Cannot precompute embeddings without a model loaded")
        
        embeddings_dict = {}
        
        logger.info(f"Precomputing embeddings from {fasta_path}")
        
        with open(fasta_path, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                protein_id = record.id
                sequence = str(record.seq)
                
                # Encode sequence
                embedding = self.encode_sequence(sequence)
                embeddings_dict[protein_id] = embedding
                
                if len(embeddings_dict) % 10 == 0:
                    logger.info(f"Processed {len(embeddings_dict)} proteins...")
        
        logger.info(f"Completed precomputing {len(embeddings_dict)} embeddings")
        
        if save_path:
            self.save_precomputed_embeddings(embeddings_dict, save_path)
        
        return embeddings_dict
