"""
Generate example data for testing the model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


def generate_rna_sequences(num_sequences: int, length: int = 101):
    """Generate random RNA sequences."""
    nucleotides = ['A', 'C', 'G', 'U']
    sequences = []
    
    for i in range(num_sequences):
        seq = ''.join(random.choice(nucleotides) for _ in range(length))
        record = SeqRecord(
            Seq(seq),
            id=f"RNA_{i+1}",
            description=f"Example RNA sequence {i+1}"
        )
        sequences.append(record)
    
    return sequences


def generate_protein_sequences(num_sequences: int, min_length: int = 50, max_length: int = 200):
    """Generate random protein sequences."""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    sequences = []
    
    for i in range(num_sequences):
        length = random.randint(min_length, max_length)
        seq = ''.join(random.choice(amino_acids) for _ in range(length))
        record = SeqRecord(
            Seq(seq),
            id=f"PROTEIN_{i+1}",
            description=f"Example protein sequence {i+1}"
        )
        sequences.append(record)
    
    return sequences


def generate_labels(num_samples: int, positive_ratio: float = 0.5):
    """Generate random binary labels."""
    num_positive = int(num_samples * positive_ratio)
    labels = [1] * num_positive + [0] * (num_samples - num_positive)
    random.shuffle(labels)
    return labels


def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create data directory
    data_dir = Path('data/examples')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate example data
    num_samples = 100
    print(f"Generating {num_samples} example samples...")
    
    # Generate RNA sequences
    rna_sequences = generate_rna_sequences(num_samples, length=101)
    rna_file = data_dir / 'rna_sequences.fasta'
    with open(rna_file, 'w') as f:
        SeqIO.write(rna_sequences, f, 'fasta')
    print(f"Saved RNA sequences to {rna_file}")
    
    # Generate protein sequences
    protein_sequences = generate_protein_sequences(num_samples, min_length=50, max_length=200)
    protein_file = data_dir / 'protein_sequences.fasta'
    with open(protein_file, 'w') as f:
        SeqIO.write(protein_sequences, f, 'fasta')
    print(f"Saved protein sequences to {protein_file}")
    
    # Generate labels
    labels = generate_labels(num_samples, positive_ratio=0.5)
    labels_file = data_dir / 'labels.txt'
    with open(labels_file, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
    print(f"Saved labels to {labels_file}")
    
    print(f"\nExample data generated successfully!")
    print(f"Positive samples: {sum(labels)}/{num_samples}")
    print(f"Negative samples: {num_samples - sum(labels)}/{num_samples}")


if __name__ == '__main__':
    main()
