"""
RNA-Protein binding prediction model with cross-attention mechanism.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer where RNA is Query and Protein is Key/Value.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Multi-head attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query: RNA embeddings (batch_size, rna_len, hidden_dim)
            key_value: Protein embeddings (batch_size, protein_len, hidden_dim)
            key_padding_mask: Mask for padded positions (batch_size, protein_len)
            
        Returns:
            Output tensor and attention weights
        """
        # Cross-attention: RNA queries attend to protein keys/values
        attn_output, attn_weights = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask if key_padding_mask is not None else None,
            need_weights=True,
            average_attn_weights=False
        )
        
        # Add & Norm
        query = self.norm1(query + self.dropout(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(query)
        
        # Add & Norm
        output = self.norm2(query + ffn_output)
        
        return output, attn_weights


class RNAProteinBindingModel(nn.Module):
    """
    RNA-Protein binding prediction model with cross-attention.
    RNA sequences serve as Query, Protein sequences as Key/Value.
    """
    
    def __init__(
        self,
        rna_input_dim: int = 4,
        protein_input_dim: int = 1280,
        hidden_dim: int = 256,
        num_attention_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        rna_seq_length: int = 101
    ):
        super().__init__()
        
        self.rna_seq_length = rna_seq_length
        self.hidden_dim = hidden_dim
        
        # Projection layers to map inputs to hidden dimension
        self.rna_projection = nn.Linear(rna_input_dim, hidden_dim)
        self.protein_projection = nn.Linear(protein_input_dim, hidden_dim)
        
        # Positional encodings
        self.rna_pos_encoding = PositionalEncoding(hidden_dim, max_len=rna_seq_length)
        self.protein_pos_encoding = PositionalEncoding(hidden_dim, max_len=1000)
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_attention_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Global pooling and classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        rna_emb: torch.Tensor,
        protein_emb: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass.
        
        Args:
            rna_emb: RNA embeddings (batch_size, rna_len, 4)
            protein_emb: Protein embeddings (batch_size, protein_len, 1280)
            protein_mask: Padding mask for protein (batch_size, protein_len)
                         1 for real tokens, 0 for padding
            
        Returns:
            Binding probability predictions and attention weights
        """
        batch_size = rna_emb.size(0)
        
        # Project to hidden dimension
        rna_hidden = self.rna_projection(rna_emb)  # (batch, rna_len, hidden_dim)
        protein_hidden = self.protein_projection(protein_emb)  # (batch, protein_len, hidden_dim)
        
        # Add positional encodings
        rna_hidden = self.rna_pos_encoding(rna_hidden)
        protein_hidden = self.protein_pos_encoding(protein_hidden)
        
        # Apply dropout
        rna_hidden = self.dropout(rna_hidden)
        protein_hidden = self.dropout(protein_hidden)
        
        # Convert mask: 1->False (not masked), 0->True (masked) for PyTorch attention
        if protein_mask is not None:
            key_padding_mask = (protein_mask == 0)
        else:
            key_padding_mask = None
        
        # Apply cross-attention layers
        attention_weights_list = []
        for layer in self.cross_attention_layers:
            rna_hidden, attn_weights = layer(rna_hidden, protein_hidden, key_padding_mask)
            attention_weights_list.append(attn_weights)
        
        # Global pooling over RNA sequence dimension
        # rna_hidden: (batch, rna_len, hidden_dim) -> (batch, hidden_dim, rna_len)
        rna_hidden = rna_hidden.transpose(1, 2)
        pooled = self.global_pool(rna_hidden).squeeze(-1)  # (batch, hidden_dim)
        
        # Classification
        output = self.classifier(pooled)  # (batch, 1)
        output = output.squeeze(-1)  # (batch,)
        
        return output, attention_weights_list
    
    def predict(
        self,
        rna_emb: torch.Tensor,
        protein_emb: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict binding probabilities.
        
        Returns:
            Binding probabilities (batch_size,)
        """
        with torch.no_grad():
            output, _ = self.forward(rna_emb, protein_emb, protein_mask)
        return output
