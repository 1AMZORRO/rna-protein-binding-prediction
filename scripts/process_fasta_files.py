#!/usr/bin/env python3
"""
处理FASTA文件以生成训练数据
从prot_seqs.fasta和part1-8.fasta生成三个文件：
- rna_sequences.fasta: RNA序列（101bp）
- protein_sequences.fasta: 蛋白质序列
- labels.txt: 结合标签（1表示结合，0表示不结合）
"""

import os
import re
from collections import defaultdict

def read_fasta(filepath):
    """读取FASTA文件并返回序列字典"""
    sequences = {}
    current_header = None
    current_seq = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_header:
                    sequences[current_header] = ''.join(current_seq)
                current_header = line[1:]  # 去掉'>'
                current_seq = []
            else:
                current_seq.append(line)
        
        # 保存最后一条序列
        if current_header:
            sequences[current_header] = ''.join(current_seq)
    
    return sequences

def extract_protein_name(header):
    """从RNA序列header中提取蛋白质名称"""
    # Header格式: "12_AARS_K562_ENCSR825SVO_pos; chr21; class:1"
    # 提取蛋白质名称: AARS_K562_ENCSR825SVO
    
    # 先按分号分割，取第一部分
    first_part = header.split(';')[0].strip()
    
    # 然后从第一部分中提取蛋白质名称
    # 格式：数字_蛋白质名称_pos或neg
    match = re.match(r'^\d+_(.+?)_(pos|neg)$', first_part)
    if match:
        return match.group(1)
    return None

def extract_label(header):
    """从header中提取标签"""
    # 查找"class:1"或"class:0"
    match = re.search(r'class:(\d+)', header)
    if match:
        return match.group(1)
    return None

def main():
    base_dir = '/home/runner/work/rna-protein-binding-prediction/rna-protein-binding-prediction'
    output_dir = os.path.join(base_dir, 'data', 'trains')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print("开始处理FASTA文件...")
    
    # 1. 读取蛋白质序列
    print("\n1. 读取蛋白质序列文件...")
    protein_file = os.path.join(base_dir, 'prot_seqs.fasta')
    protein_seqs = read_fasta(protein_file)
    print(f"   读取了 {len(protein_seqs)} 个蛋白质序列")
    
    # 2. 处理8个part文件
    print("\n2. 处理RNA序列文件（part1-8.fasta）...")
    all_rna_data = []  # 存储(protein_name, rna_seq, label)元组
    
    for part_num in range(1, 9):
        part_file = os.path.join(base_dir, f'part{part_num}.fasta')
        print(f"   处理 part{part_num}.fasta...")
        
        rna_seqs = read_fasta(part_file)
        
        for header, seq in rna_seqs.items():
            protein_name = extract_protein_name(header)
            label = extract_label(header)
            
            if protein_name and label is not None:
                # 验证蛋白质名称是否在蛋白质序列文件中
                if protein_name in protein_seqs:
                    all_rna_data.append((protein_name, seq, label))
                else:
                    # 打印警告但继续处理
                    pass  # 静默处理未找到的蛋白质
        
        print(f"      完成，当前总计 {len(all_rna_data)} 条RNA序列")
    
    print(f"\n   总计处理了 {len(all_rna_data)} 条RNA序列")
    
    # 3. 写入输出文件
    print("\n3. 生成输出文件...")
    
    # 3a. 写入RNA序列文件
    rna_output = os.path.join(output_dir, 'rna_sequences.fasta')
    with open(rna_output, 'w') as f:
        for idx, (protein_name, rna_seq, label) in enumerate(all_rna_data, 1):
            f.write(f'>RNA_{idx} {protein_name}\n')
            f.write(f'{rna_seq}\n')
    print(f"   ✓ 生成 rna_sequences.fasta ({len(all_rna_data)} 条序列)")
    
    # 3b. 写入蛋白质序列文件（按照RNA数据的顺序）
    protein_output = os.path.join(output_dir, 'protein_sequences.fasta')
    with open(protein_output, 'w') as f:
        for idx, (protein_name, rna_seq, label) in enumerate(all_rna_data, 1):
            prot_seq = protein_seqs[protein_name]
            f.write(f'>PROTEIN_{idx} {protein_name}\n')
            f.write(f'{prot_seq}\n')
    print(f"   ✓ 生成 protein_sequences.fasta ({len(all_rna_data)} 条序列)")
    
    # 3c. 写入标签文件
    labels_output = os.path.join(output_dir, 'labels.txt')
    with open(labels_output, 'w') as f:
        for idx, (protein_name, rna_seq, label) in enumerate(all_rna_data, 1):
            f.write(f'{label}\n')
    print(f"   ✓ 生成 labels.txt ({len(all_rna_data)} 个标签)")
    
    # 4. 统计信息
    print("\n4. 数据统计:")
    positive_count = sum(1 for _, _, label in all_rna_data if label == '1')
    negative_count = sum(1 for _, _, label in all_rna_data if label == '0')
    print(f"   总序列数: {len(all_rna_data)}")
    if len(all_rna_data) > 0:
        print(f"   正样本(class=1): {positive_count} ({positive_count/len(all_rna_data)*100:.2f}%)")
        print(f"   负样本(class=0): {negative_count} ({negative_count/len(all_rna_data)*100:.2f}%)")
    else:
        print("   警告: 没有找到匹配的RNA序列数据！")
    
    # 统计涉及的蛋白质数量
    unique_proteins = set(protein_name for protein_name, _, _ in all_rna_data)
    print(f"   涉及的蛋白质种类: {len(unique_proteins)}")
    
    print("\n✓ 处理完成！")
    print(f"\n输出文件位置: {output_dir}")
    print("  - rna_sequences.fasta")
    print("  - protein_sequences.fasta")
    print("  - labels.txt")

if __name__ == '__main__':
    main()
