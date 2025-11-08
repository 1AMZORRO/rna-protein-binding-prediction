"""
Script to precompute protein embeddings for all proteins in prot_seqs.fasta
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import logging
from src.data import ProteinProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Precompute protein embeddings to avoid repeated computation'
    )
    parser.add_argument(
        '--protein-fasta',
        type=str,
        default='prot_seqs.fasta',
        help='Path to protein FASTA file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='precomputed_protein_embeddings.pt',
        help='Output path for precomputed embeddings'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='esm2_t33_650M_UR50D',
        help='ESM2 model name to use'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    # Check if protein fasta exists
    if not Path(args.protein_fasta).exists():
        logger.error(f"Protein FASTA file not found: {args.protein_fasta}")
        sys.exit(1)
    
    # Initialize protein processor
    logger.info(f"Initializing ProteinProcessor with model: {args.model_name}")
    logger.info(f"Using device: {args.device}")
    
    processor = ProteinProcessor(
        model_name=args.model_name,
        device=args.device
    )
    
    # Precompute embeddings
    logger.info(f"Starting to precompute embeddings from {args.protein_fasta}")
    
    embeddings_dict = processor.precompute_embeddings_from_fasta(
        fasta_path=args.protein_fasta,
        save_path=args.output
    )
    
    # Print statistics
    logger.info("\nPrecomputation completed!")
    logger.info(f"Total proteins processed: {len(embeddings_dict)}")
    logger.info(f"Embeddings saved to: {args.output}")
    
    # Print some sample information
    sample_protein = next(iter(embeddings_dict))
    sample_embedding = embeddings_dict[sample_protein]
    logger.info(f"\nSample protein: {sample_protein}")
    logger.info(f"Embedding shape: {sample_embedding.shape}")
    logger.info(f"Embedding dimension: {sample_embedding.shape[-1]}")
    
    # Calculate file size
    file_size = Path(args.output).stat().st_size / (1024 * 1024)  # MB
    logger.info(f"File size: {file_size:.2f} MB")


if __name__ == '__main__':
    main()
