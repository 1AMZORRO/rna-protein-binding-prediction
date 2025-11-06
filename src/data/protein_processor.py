"""
Protein sequence processor with ESM2 embedding.
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional
import numpy as np


class ProteinProcessor:
    """Process protein sequences using ESM2 pre-trained model."""
    
    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D", device: str = "cuda", local_model_path: Optional[str] = None):
        """
        Initialize protein processor with ESM2 model.
        
        Args:
            model_name: Name of the ESM2 model to use (e.g., 'facebook/esm2_t33_650M_UR50D')
            device: Device to run the model on ('cuda' or 'cpu')
            local_model_path: Path to local model directory (for offline mode). If provided,
                            the model will be loaded from this path instead of downloading
                            from Hugging Face. The directory should contain the model files
                            downloaded from Hugging Face.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # Determine the model path to use
        model_path = local_model_path if local_model_path else model_name
        
        # Load ESM2 tokenizer and model
        if local_model_path:
            print(f"Loading ESM2 model from local path: {local_model_path}")
        else:
            print(f"Loading ESM2 model: {model_name}")
        
        try:
            # Try to load from local path or download
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=bool(local_model_path))
            self.model = AutoModel.from_pretrained(model_path, local_files_only=bool(local_model_path))
        except Exception as e:
            if local_model_path:
                raise RuntimeError(
                    f"无法从本地路径加载模型: {local_model_path}\n"
                    f"请确保该路径包含完整的模型文件（config.json, pytorch_model.bin等）\n"
                    f"错误信息: {str(e)}"
                )
            else:
                raise RuntimeError(
                    f"无法下载模型: {model_name}\n"
                    f"如果您无法访问 huggingface.co，请:\n"
                    f"1. 在可以访问的环境下载模型到本地:\n"
                    f"   from transformers import AutoModel, AutoTokenizer\n"
                    f"   model = AutoModel.from_pretrained('{model_name}')\n"
                    f"   tokenizer = AutoTokenizer.from_pretrained('{model_name}')\n"
                    f"   model.save_pretrained('./local_esm2_model')\n"
                    f"   tokenizer.save_pretrained('./local_esm2_model')\n"
                    f"2. 将模型文件复制到无法联网的环境\n"
                    f"3. 在配置文件中设置 local_esm_model_path 为本地模型路径\n"
                    f"原始错误: {str(e)}"
                )
        
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
