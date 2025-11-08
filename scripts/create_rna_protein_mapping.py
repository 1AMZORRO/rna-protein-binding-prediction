"""
Script to create mapping between RNA sequences and protein embeddings.

This script analyzes the 168_train.fasta and prot_seqs.fasta files to establish
the correspondence between RNA sequences and their associated proteins.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from Bio import SeqIO
import logging
import re
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_protein_name_from_rna_id(rna_id: str) -> str:
    """
    Extract protein name from RNA sequence ID.
    
    Example:
        '12_AARS_K562_ENCSR825SVO_pos' -> 'AARS_K562_ENCSR825SVO'
    """
    # Remove the leading number and underscore
    parts = rna_id.split('_')
    if len(parts) >= 4:
        # Skip first part (number), join the rest except last part (pos/neg)
        # But keep ENCSR ID
        protein_name = '_'.join(parts[1:-1])
        return protein_name
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Create mapping between RNA sequences and proteins'
    )
    parser.add_argument(
        '--rna-fasta',
        type=str,
        default='168_train.fasta',
        help='Path to RNA FASTA file (168_train.fasta)'
    )
    parser.add_argument(
        '--protein-fasta',
        type=str,
        default='prot_seqs.fasta',
        help='Path to protein FASTA file (prot_seqs.fasta)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='rna_protein_mapping.txt',
        help='Output file for RNA-protein mapping'
    )
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.rna_fasta).exists():
        logger.error(f"RNA FASTA file not found: {args.rna_fasta}")
        sys.exit(1)
    
    if not Path(args.protein_fasta).exists():
        logger.error(f"Protein FASTA file not found: {args.protein_fasta}")
        sys.exit(1)
    
    # Load protein IDs
    logger.info(f"Loading protein IDs from {args.protein_fasta}")
    protein_ids = set()
    with open(args.protein_fasta, 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            protein_ids.add(record.id)
    
    logger.info(f"Found {len(protein_ids)} unique proteins")
    logger.info(f"Sample protein IDs: {list(protein_ids)[:5]}")
    
    # Analyze RNA sequences and extract protein associations
    logger.info(f"\nAnalyzing RNA sequences from {args.rna_fasta}")
    rna_protein_map = {}
    protein_counts = defaultdict(int)
    unmatched_count = 0
    
    with open(args.rna_fasta, 'r') as f:
        for i, record in enumerate(SeqIO.parse(f, 'fasta')):
            rna_id = record.id
            
            # Extract protein name from RNA ID
            protein_name = extract_protein_name_from_rna_id(rna_id)
            
            if protein_name and protein_name in protein_ids:
                rna_protein_map[rna_id] = protein_name
                protein_counts[protein_name] += 1
            else:
                unmatched_count += 1
                if i < 10:  # Show first few unmatched
                    logger.warning(f"Could not match RNA '{rna_id}' to a protein")
            
            if (i + 1) % 100000 == 0:
                logger.info(f"Processed {i + 1} RNA sequences...")
    
    # Print statistics
    logger.info(f"\n{'='*60}")
    logger.info("Mapping Statistics:")
    logger.info(f"{'='*60}")
    logger.info(f"Total RNA sequences: {len(rna_protein_map) + unmatched_count}")
    logger.info(f"Successfully mapped: {len(rna_protein_map)}")
    logger.info(f"Unmatched: {unmatched_count}")
    logger.info(f"Unique proteins used: {len(protein_counts)}")
    
    logger.info(f"\nRNA sequences per protein:")
    for protein, count in sorted(protein_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  {protein}: {count}")
    
    # Save mapping
    logger.info(f"\nSaving mapping to {args.output}")
    with open(args.output, 'w') as f:
        f.write("# RNA_ID\tProtein_ID\n")
        for rna_id, protein_id in sorted(rna_protein_map.items()):
            f.write(f"{rna_id}\t{protein_id}\n")
    
    logger.info(f"Mapping saved successfully!")
    
    # Also print information about label distribution
    logger.info(f"\nAnalyzing label distribution...")
    pos_count = 0
    neg_count = 0
    
    for rna_id in rna_protein_map.keys():
        if '_pos;' in rna_id or 'class:1' in rna_id:
            pos_count += 1
        elif '_neg;' in rna_id or 'class:0' in rna_id:
            neg_count += 1
    
    logger.info(f"Positive samples (binding): {pos_count}")
    logger.info(f"Negative samples (non-binding): {neg_count}")
    logger.info(f"Ratio (pos/neg): {pos_count/neg_count if neg_count > 0 else 'N/A'}")


if __name__ == '__main__':
    main()
