"""
Script to create comprehensive mapping between RNA sequences, labels, and protein embeddings.

This script analyzes the 168_train.fasta and prot_seqs.fasta files to establish
the correspondence between:
1. RNA sequences
2. Labels (binding vs non-binding)
3. Protein embeddings (via protein IDs)
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
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_protein_name_from_rna_id(rna_id: str) -> str:
    """
    Extract protein name from RNA sequence ID.
    
    Example:
        '12_AARS_K562_ENCSR825SVO_pos; chr21; class:1' -> 'AARS_K562_ENCSR825SVO'
    """
    # Split by semicolon first to handle the full format
    parts = rna_id.split(';')[0].strip()
    
    # Split by underscore
    name_parts = parts.split('_')
    if len(name_parts) >= 4:
        # Skip first part (number), join the middle parts (protein name), exclude last (pos/neg)
        protein_name = '_'.join(name_parts[1:-1])
        return protein_name
    return None


def extract_label_from_rna_id(rna_id: str) -> int:
    """
    Extract label from RNA sequence ID.
    
    Returns:
        1 for binding (positive), 0 for non-binding (negative)
    """
    # Check for class: indicator
    if 'class:1' in rna_id:
        return 1
    elif 'class:0' in rna_id:
        return 0
    
    # Check for pos/neg indicator
    if '_pos;' in rna_id or '_pos' in rna_id.split(';')[0]:
        return 1
    elif '_neg;' in rna_id or '_neg' in rna_id.split(';')[0]:
        return 0
    
    # Default to unknown
    return -1


def main():
    parser = argparse.ArgumentParser(
        description='Create comprehensive mapping between RNA sequences, labels, and proteins'
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
        '--output-mapping',
        type=str,
        default='rna_protein_mapping.txt',
        help='Output file for RNA-protein-label mapping (TSV format)'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default='rna_protein_mapping.json',
        help='Output file for mapping in JSON format'
    )
    parser.add_argument(
        '--output-labels',
        type=str,
        default='168_train_labels.txt',
        help='Output file for labels only (one per line)'
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
    
    # Analyze RNA sequences and extract complete mapping
    logger.info(f"\nAnalyzing RNA sequences from {args.rna_fasta}")
    
    mapping_data = []
    labels_only = []
    protein_counts = defaultdict(int)
    label_counts = defaultdict(int)
    unmatched_count = 0
    unknown_label_count = 0
    
    with open(args.rna_fasta, 'r') as f:
        for i, record in enumerate(SeqIO.parse(f, 'fasta')):
            rna_id = record.id
            rna_seq = str(record.seq)
            
            # Extract protein name from RNA ID
            protein_name = extract_protein_name_from_rna_id(rna_id)
            
            # Extract label
            label = extract_label_from_rna_id(rna_id)
            
            if label == -1:
                unknown_label_count += 1
                if unknown_label_count <= 5:
                    logger.warning(f"Unknown label for RNA '{rna_id}'")
            
            if protein_name and protein_name in protein_ids:
                mapping_data.append({
                    'rna_id': rna_id,
                    'rna_sequence': rna_seq,
                    'protein_id': protein_name,
                    'label': label
                })
                labels_only.append(label)
                protein_counts[protein_name] += 1
                label_counts[label] += 1
            else:
                unmatched_count += 1
                if unmatched_count <= 10:  # Show first few unmatched
                    logger.warning(f"Could not match RNA '{rna_id}' to a protein (extracted: '{protein_name}')")
            
            if (i + 1) % 100000 == 0:
                logger.info(f"Processed {i + 1} RNA sequences...")
    
    # Print statistics
    logger.info(f"\n{'='*70}")
    logger.info("Mapping Statistics:")
    logger.info(f"{'='*70}")
    logger.info(f"Total RNA sequences processed: {len(mapping_data) + unmatched_count}")
    logger.info(f"Successfully mapped: {len(mapping_data)}")
    logger.info(f"Unmatched (no protein found): {unmatched_count}")
    logger.info(f"Unknown labels: {unknown_label_count}")
    logger.info(f"Unique proteins used: {len(protein_counts)}")
    
    logger.info(f"\nLabel distribution:")
    logger.info(f"  Positive (binding, label=1): {label_counts.get(1, 0)}")
    logger.info(f"  Negative (non-binding, label=0): {label_counts.get(0, 0)}")
    logger.info(f"  Unknown (label=-1): {label_counts.get(-1, 0)}")
    if label_counts[0] > 0:
        logger.info(f"  Ratio (pos/neg): {label_counts[1]/label_counts[0]:.3f}")
    
    logger.info(f"\nTop 10 proteins by RNA sequence count:")
    for protein, count in sorted(protein_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  {protein}: {count}")
    
    # Save mapping in TSV format
    logger.info(f"\nSaving mapping to {args.output_mapping}")
    with open(args.output_mapping, 'w') as f:
        f.write("RNA_ID\tRNA_Sequence\tProtein_ID\tLabel\n")
        for item in mapping_data:
            f.write(f"{item['rna_id']}\t{item['rna_sequence']}\t{item['protein_id']}\t{item['label']}\n")
    
    logger.info(f"TSV mapping saved: {len(mapping_data)} entries")
    
    # Save mapping in JSON format
    logger.info(f"Saving mapping to {args.output_json}")
    with open(args.output_json, 'w') as f:
        json.dump({
            'metadata': {
                'total_sequences': len(mapping_data),
                'unique_proteins': len(protein_counts),
                'positive_samples': label_counts.get(1, 0),
                'negative_samples': label_counts.get(0, 0),
                'unknown_labels': label_counts.get(-1, 0)
            },
            'mappings': mapping_data
        }, f, indent=2)
    
    logger.info(f"JSON mapping saved")
    
    # Save labels only
    logger.info(f"Saving labels to {args.output_labels}")
    with open(args.output_labels, 'w') as f:
        for label in labels_only:
            f.write(f"{label}\n")
    
    logger.info(f"Labels file saved: {len(labels_only)} labels")
    
    logger.info(f"\n{'='*70}")
    logger.info("Summary:")
    logger.info(f"{'='*70}")
    logger.info(f"Created 3 output files:")
    logger.info(f"  1. {args.output_mapping} - TSV format with RNA_ID, RNA_Sequence, Protein_ID, Label")
    logger.info(f"  2. {args.output_json} - JSON format with metadata and complete mapping")
    logger.info(f"  3. {args.output_labels} - Text file with labels only (one per line)")
    logger.info(f"\nThese files establish the correspondence between:")
    logger.info(f"  - RNA sequences (from 168_train.fasta)")
    logger.info(f"  - Protein IDs (for precomputed embeddings lookup)")
    logger.info(f"  - Labels (1=binding, 0=non-binding)")


if __name__ == '__main__':
    main()
