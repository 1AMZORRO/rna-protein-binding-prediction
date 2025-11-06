#!/usr/bin/env python3
"""
下载 ESM2 模型到本地，用于离线环境使用

使用方法:
    # 下载默认模型 (650M 参数)
    python scripts/download_esm2_model.py
    
    # 下载小型模型 (35M 参数)
    python scripts/download_esm2_model.py --model esm2_t12_35M_UR50D --save-path ./esm2_t12_model
    
    # 下载并查看模型信息
    python scripts/download_esm2_model.py --info
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    print("错误: 未安装 transformers 库")
    print("请运行: pip install transformers")
    sys.exit(1)


# 可用的 ESM2 模型
AVAILABLE_MODELS = {
    "esm2_t33_650M_UR50D": {
        "name": "facebook/esm2_t33_650M_UR50D",
        "params": "650M",
        "embedding_dim": 1280,
        "size": "~2.5GB",
        "description": "最大模型，最佳性能"
    },
    "esm2_t30_150M_UR50D": {
        "name": "facebook/esm2_t30_150M_UR50D",
        "params": "150M",
        "embedding_dim": 640,
        "size": "~600MB",
        "description": "中等模型，平衡性能和速度"
    },
    "esm2_t12_35M_UR50D": {
        "name": "facebook/esm2_t12_35M_UR50D",
        "params": "35M",
        "embedding_dim": 480,
        "size": "~140MB",
        "description": "小型模型，快速训练"
    },
    "esm2_t6_8M_UR50D": {
        "name": "facebook/esm2_t6_8M_UR50D",
        "params": "8M",
        "embedding_dim": 320,
        "size": "~35MB",
        "description": "最小模型，仅用于测试"
    }
}


def print_model_info():
    """打印所有可用模型的信息"""
    print("\n可用的 ESM2 模型:\n")
    print(f"{'模型名称':<25} {'参数量':<10} {'嵌入维度':<10} {'大小':<10} {'说明'}")
    print("-" * 85)
    
    for model_key, info in AVAILABLE_MODELS.items():
        print(f"{model_key:<25} {info['params']:<10} {info['embedding_dim']:<10} {info['size']:<10} {info['description']}")
    
    print("\n推荐:")
    print("  - 生产环境/论文实验: esm2_t33_650M_UR50D")
    print("  - 显存受限: esm2_t30_150M_UR50D")
    print("  - 快速实验: esm2_t12_35M_UR50D")
    print()


def download_model(model_key, save_path, use_mirror=False):
    """
    下载指定的 ESM2 模型
    
    Args:
        model_key: 模型键名 (如 'esm2_t33_650M_UR50D')
        save_path: 保存路径
        use_mirror: 是否使用镜像站
    """
    if model_key not in AVAILABLE_MODELS:
        print(f"错误: 未知的模型 '{model_key}'")
        print_model_info()
        sys.exit(1)
    
    model_info = AVAILABLE_MODELS[model_key]
    model_name = model_info["name"]
    
    print("=" * 80)
    print(f"下载 ESM2 模型")
    print("=" * 80)
    print(f"模型: {model_name}")
    print(f"参数量: {model_info['params']}")
    print(f"嵌入维度: {model_info['embedding_dim']}")
    print(f"大小: {model_info['size']}")
    print(f"保存路径: {save_path}")
    print("-" * 80)
    
    # 设置镜像
    if use_mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print("使用 Hugging Face 镜像站: https://hf-mirror.com")
    
    # 创建保存目录
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # 下载模型
        print("\n[1/2] 下载模型权重...")
        model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(save_path)
        print("✓ 模型权重下载完成")
        
        # 下载 tokenizer
        print("\n[2/2] 下载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_path)
        print("✓ Tokenizer 下载完成")
        
        print("\n" + "=" * 80)
        print("✓ 模型下载成功!")
        print("=" * 80)
        print(f"\n模型已保存到: {save_path.absolute()}")
        print(f"\n下一步:")
        print(f"1. 修改 config/config.yaml:")
        print(f"   model:")
        print(f"     esm_model_name: \"{model_name}\"")
        print(f"     protein_embedding_dim: {model_info['embedding_dim']}")
        print(f"     local_esm_model_path: \"{save_path}\"")
        print(f"\n2. 正常运行训练或预测命令")
        print()
        
    except Exception as e:
        print(f"\n✗ 下载失败: {str(e)}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 使用镜像站: 添加 --mirror 参数")
        print("3. 设置代理: export HTTP_PROXY=...")
        print("4. 手动下载: 参考 docs/OFFLINE_MODE.md")
        sys.exit(1)


def verify_model(model_path):
    """验证模型文件是否完整"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"✗ 路径不存在: {model_path}")
        return False
    
    required_files = [
        "config.json", 
        "pytorch_model.bin",
        "tokenizer_config.json",
        "vocab.txt"
    ]
    # Special tokens map is optional in some models
    optional_files = ["special_tokens_map.json"]
    
    missing_files = []
    
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ 模型文件不完整，缺少必需文件: {', '.join(missing_files)}")
        return False
    
    # Check optional files and warn
    missing_optional = []
    for file in optional_files:
        if not (model_path / file).exists():
            missing_optional.append(file)
    
    if missing_optional:
        print(f"⚠ 警告: 缺少可选文件: {', '.join(missing_optional)}")
    
    print(f"✓ 模型文件完整: {model_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='下载 ESM2 模型到本地用于离线环境',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载默认模型 (650M)
  python scripts/download_esm2_model.py
  
  # 下载小型模型
  python scripts/download_esm2_model.py --model esm2_t12_35M_UR50D --save-path ./esm2_t12_model
  
  # 使用镜像站下载 (中国大陆推荐)
  python scripts/download_esm2_model.py --mirror
  
  # 查看可用模型
  python scripts/download_esm2_model.py --info
  
  # 验证已下载的模型
  python scripts/download_esm2_model.py --verify ./esm2_model
        """
    )
    
    parser.add_argument(
        "--model",
        default="esm2_t33_650M_UR50D",
        choices=list(AVAILABLE_MODELS.keys()),
        help="要下载的模型名称"
    )
    
    parser.add_argument(
        "--save-path",
        default="./esm2_model",
        help="模型保存路径"
    )
    
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="使用 Hugging Face 镜像站 (推荐中国大陆用户使用)"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="显示所有可用模型的信息"
    )
    
    parser.add_argument(
        "--verify",
        metavar="PATH",
        help="验证指定路径的模型文件是否完整"
    )
    
    args = parser.parse_args()
    
    # 显示模型信息
    if args.info:
        print_model_info()
        return
    
    # 验证模型
    if args.verify:
        verify_model(args.verify)
        return
    
    # 下载模型
    download_model(args.model, args.save_path, args.mirror)


if __name__ == "__main__":
    main()
