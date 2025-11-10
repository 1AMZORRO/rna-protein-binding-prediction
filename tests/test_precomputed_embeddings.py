"""
测试预计算embeddings功能
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import tempfile
import argparse
from unittest.mock import Mock, patch
import yaml


def test_train_script_argument_parsing():
    """测试 train.py 参数解析"""
    print("\n=== 测试 train.py 参数解析 ===")
    
    # 模拟 train.py 的参数解析器
    parser = argparse.ArgumentParser(description='Train RNA-Protein binding prediction model')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--rna-fasta', type=str, default=None)
    parser.add_argument('--protein-fasta', type=str, default=None)
    parser.add_argument('--labels', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--use-precomputed-embeddings', action='store_true',
                        help='Use precomputed protein embeddings from config file')
    
    # 测试1: 使用 --use-precomputed-embeddings
    args = parser.parse_args(['--use-precomputed-embeddings'])
    assert args.use_precomputed_embeddings == True, "应该解析到 --use-precomputed-embeddings 参数"
    print("✓ 正确解析 --use-precomputed-embeddings 参数")
    
    # 测试2: 不使用该参数
    args = parser.parse_args([])
    assert args.use_precomputed_embeddings == False, "默认应该是 False"
    print("✓ 默认值正确 (False)")
    
    # 测试3: 与其他参数一起使用
    args = parser.parse_args([
        '--config', 'test.yaml',
        '--rna-fasta', 'rna.fasta',
        '--protein-fasta', 'protein.fasta',
        '--labels', 'labels.txt',
        '--use-precomputed-embeddings'
    ])
    assert args.use_precomputed_embeddings == True
    assert args.config == 'test.yaml'
    assert args.rna_fasta == 'rna.fasta'
    assert args.protein_fasta == 'protein.fasta'
    assert args.labels == 'labels.txt'
    print("✓ 与其他参数配合正确")
    
    print("✓ train.py 参数解析测试通过!\n")


def test_predict_script_argument_parsing():
    """测试 predict.py 参数解析"""
    print("\n=== 测试 predict.py 参数解析 ===")
    
    # 模拟 predict.py 的参数解析器
    parser = argparse.ArgumentParser(description='Predict RNA-Protein binding')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--model', type=str, required=False)  # 测试时不要求
    parser.add_argument('--rna-fasta', type=str, required=False)
    parser.add_argument('--protein-fasta', type=str, required=False)
    parser.add_argument('--labels', type=str, default=None)
    parser.add_argument('--output', type=str, default='predictions.txt')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--use-precomputed-embeddings', action='store_true',
                        help='Use precomputed protein embeddings from config file')
    
    # 测试1: 使用 --use-precomputed-embeddings
    args = parser.parse_args(['--use-precomputed-embeddings'])
    assert args.use_precomputed_embeddings == True, "应该解析到 --use-precomputed-embeddings 参数"
    print("✓ 正确解析 --use-precomputed-embeddings 参数")
    
    # 测试2: 不使用该参数
    args = parser.parse_args([])
    assert args.use_precomputed_embeddings == False, "默认应该是 False"
    print("✓ 默认值正确 (False)")
    
    print("✓ predict.py 参数解析测试通过!\n")


def test_precomputed_embeddings_logic():
    """测试预计算embeddings的逻辑流程"""
    print("\n=== 测试预计算embeddings逻辑 ===")
    
    # 创建临时配置文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config = {
            'model': {
                'rna_seq_length': 101,
                'esm_model_name': 'esm2_t6_8M_UR50D',
                'precomputed_embeddings': 'test_embeddings.pt'
            }
        }
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        # 读取配置
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        # 模拟 train.py 中的逻辑
        use_precomputed = True  # 模拟 args.use_precomputed_embeddings
        
        precomputed_embeddings_path = None
        if use_precomputed:
            precomputed_embeddings_path = loaded_config['model'].get('precomputed_embeddings')
        
        assert precomputed_embeddings_path == 'test_embeddings.pt', \
            f"应该读取到正确的路径，得到: {precomputed_embeddings_path}"
        print("✓ 正确从配置文件读取预计算embeddings路径")
        
        # 测试没有指定路径的情况
        config_no_path = {
            'model': {
                'rna_seq_length': 101,
                'esm_model_name': 'esm2_t6_8M_UR50D',
                'precomputed_embeddings': None
            }
        }
        
        precomputed_embeddings_path = None
        if use_precomputed:
            precomputed_embeddings_path = config_no_path['model'].get('precomputed_embeddings')
        
        assert precomputed_embeddings_path is None, "没有路径时应该是 None"
        print("✓ 正确处理没有预计算embeddings路径的情况")
        
        # 测试 use_protein_ids 逻辑
        use_protein_ids = use_precomputed and precomputed_embeddings_path is not None
        assert use_protein_ids == False, "没有路径时 use_protein_ids 应该是 False"
        print("✓ 正确设置 use_protein_ids 标志")
        
        # 测试有路径的情况
        precomputed_embeddings_path = 'test_embeddings.pt'
        use_protein_ids = use_precomputed and precomputed_embeddings_path is not None
        assert use_protein_ids == True, "有路径时 use_protein_ids 应该是 True"
        print("✓ 有预计算embeddings路径时正确设置 use_protein_ids")
        
    finally:
        # 清理临时文件
        Path(config_path).unlink()
    
    print("✓ 预计算embeddings逻辑测试通过!\n")


def test_protein_processor_initialization():
    """测试 ProteinProcessor 的初始化参数"""
    print("\n=== 测试 ProteinProcessor 初始化 ===")
    
    from src.data.protein_processor import ProteinProcessor
    
    # 测试1: 创建临时的预计算embeddings文件
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        # 创建一个模拟的embeddings字典
        embeddings = {
            'PROTEIN_1': torch.randn(50, 1280),
            'PROTEIN_2': torch.randn(75, 1280),
        }
        torch.save(embeddings, f.name)
        embeddings_path = f.name
    
    try:
        # 测试使用预计算embeddings初始化
        processor = ProteinProcessor(
            model_name='esm2_t6_8M_UR50D',
            device='cpu',
            precomputed_embeddings_path=embeddings_path
        )
        
        assert processor.model is None, "使用预计算embeddings时不应该加载模型"
        assert len(processor.precomputed_embeddings) == 2, "应该加载2个预计算的embeddings"
        assert 'PROTEIN_1' in processor.precomputed_embeddings, "应该包含 PROTEIN_1"
        assert 'PROTEIN_2' in processor.precomputed_embeddings, "应该包含 PROTEIN_2"
        print("✓ 正确加载预计算embeddings")
        print(f"✓ 加载了 {len(processor.precomputed_embeddings)} 个蛋白质embeddings")
        
        # 测试使用protein ID查找
        emb = processor.encode_sequence('ACDEFGHIKL', protein_id='PROTEIN_1')
        assert emb.shape == (50, 1280), f"应该返回正确的embedding形状，得到: {emb.shape}"
        print("✓ 正确通过protein ID查找预计算的embedding")
        
    finally:
        # 清理临时文件
        Path(embeddings_path).unlink()
    
    print("✓ ProteinProcessor 初始化测试通过!\n")


def test_dataset_with_protein_ids():
    """测试 Dataset 使用 protein IDs"""
    print("\n=== 测试 Dataset 使用 protein IDs ===")
    
    from src.data.dataset import RNAProteinDataset
    from src.data.rna_processor import RNAProcessor
    from src.data.protein_processor import ProteinProcessor
    
    # 创建临时的预计算embeddings
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        embeddings = {
            'PROT1': torch.randn(30, 1280),
            'PROT2': torch.randn(40, 1280),
        }
        torch.save(embeddings, f.name)
        embeddings_path = f.name
    
    try:
        # 创建processors
        rna_processor = RNAProcessor(seq_length=101)
        protein_processor = ProteinProcessor(
            model_name='esm2_t6_8M_UR50D',
            device='cpu',
            precomputed_embeddings_path=embeddings_path
        )
        
        # 创建dataset，使用protein IDs
        rna_seqs = ['ACGU' * 25 + 'A', 'UGCA' * 25 + 'U']
        protein_seqs = ['ACDEFGHIKL' * 3, 'LMNPQRSTVW' * 4]  # 这些序列不会被用到
        labels = [1, 0]
        protein_ids = ['PROT1', 'PROT2']
        
        dataset = RNAProteinDataset(
            rna_sequences=rna_seqs,
            protein_sequences=protein_seqs,
            labels=labels,
            rna_processor=rna_processor,
            protein_processor=protein_processor,
            protein_ids=protein_ids
        )
        
        assert len(dataset) == 2, "Dataset应该有2个样本"
        print("✓ Dataset创建成功")
        
        # 测试获取样本
        rna_emb, protein_emb, label = dataset[0]
        assert rna_emb.shape == (101, 4), f"RNA embedding形状应该是(101, 4)，得到: {rna_emb.shape}"
        assert protein_emb.shape == (30, 1280), f"Protein embedding应该来自预计算，形状应该是(30, 1280)，得到: {protein_emb.shape}"
        assert label == 1, "标签应该是1"
        print("✓ 正确从预计算embeddings获取蛋白质embedding")
        
        # 测试第二个样本
        rna_emb, protein_emb, label = dataset[1]
        assert protein_emb.shape == (40, 1280), f"第二个样本的embedding形状应该是(40, 1280)，得到: {protein_emb.shape}"
        print("✓ 正确处理多个样本")
        
    finally:
        # 清理临时文件
        Path(embeddings_path).unlink()
    
    print("✓ Dataset使用protein IDs测试通过!\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("运行预计算Embeddings功能测试")
    print("="*70)
    
    try:
        test_train_script_argument_parsing()
        test_predict_script_argument_parsing()
        test_precomputed_embeddings_logic()
        test_protein_processor_initialization()
        test_dataset_with_protein_ids()
        
        print("\n" + "="*70)
        print("✓ 所有测试通过!")
        print("="*70 + "\n")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
