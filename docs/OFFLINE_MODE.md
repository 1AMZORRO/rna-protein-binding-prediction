# 离线模式使用指南

本文档详细说明如何在无法访问 Hugging Face 的网络环境中使用本项目。

## 问题说明

当您看到以下错误时，说明您的网络环境无法访问 huggingface.co：

```
OSError: We couldn't connect to 'https://huggingface.co' to load this file...
NameResolutionError: Failed to resolve 'huggingface.co'
```

这在以下情况中很常见：
- 中国大陆的网络环境
- 企业内网环境
- 防火墙限制的服务器
- 没有互联网连接的离线环境

## 解决方案

### 方案 1: 使用 Hugging Face 镜像站（最简单）

如果您在中国大陆，可以使用官方镜像站：

```bash
# 临时使用（当前终端有效）
export HF_ENDPOINT=https://hf-mirror.com

# 永久使用（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

然后正常运行训练或预测命令：
```bash
python scripts/train.py --config config/config.yaml \
    --rna-fasta data/trains/rna_sequences.fasta \
    --protein-fasta data/trains/protein_sequences.fasta \
    --labels data/trains/labels.txt
```

### 方案 2: 完全离线模式（推荐）

如果完全无法访问外网，需要提前下载模型文件。

#### 步骤 1: 在有网络的环境下载模型

创建一个 Python 脚本 `download_model.py`：

```python
#!/usr/bin/env python3
"""下载 ESM2 模型到本地"""

from transformers import AutoModel, AutoTokenizer
import argparse

def download_model(model_name, save_path):
    print(f"正在下载模型: {model_name}")
    print(f"保存路径: {save_path}")
    
    # 下载模型
    print("下载模型权重...")
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(save_path)
    
    # 下载 tokenizer
    print("下载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_path)
    
    print(f"✓ 模型已成功保存到: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D",
                       help="模型名称")
    parser.add_argument("--save-path", default="./esm2_model",
                       help="保存路径")
    
    args = parser.parse_args()
    download_model(args.model, args.save_path)
```

运行脚本下载模型：

```bash
# 下载默认模型 (650M 参数，约 2.5GB)
python download_model.py

# 或下载更小的模型 (35M 参数，约 140MB)
python download_model.py \
    --model facebook/esm2_t12_35M_UR50D \
    --save-path ./esm2_t12_model
```

#### 步骤 2: 将模型文件复制到离线环境

使用 scp、rsync 或其他方式将下载的模型目录复制到无网络的服务器：

```bash
# 使用 scp 复制到远程服务器
scp -r esm2_model user@remote-server:/path/to/rna-protein-binding-prediction/

# 或使用 rsync
rsync -avz esm2_model/ user@remote-server:/path/to/rna-protein-binding-prediction/esm2_model/
```

#### 步骤 3: 配置使用本地模型

修改 `config/config.yaml` 文件：

```yaml
model:
  esm_model_name: "facebook/esm2_t33_650M_UR50D"
  protein_embedding_dim: 1280
  local_esm_model_path: "./esm2_model"  # 相对路径
  # 或使用绝对路径:
  # local_esm_model_path: "/data/models/esm2_model"
```

#### 步骤 4: 正常运行

现在可以在离线环境中正常运行：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train.py \
    --config config/config.yaml \
    --rna-fasta data/trains/rna_sequences.fasta \
    --protein-fasta data/trains/protein_sequences.fasta \
    --labels data/trains/labels.txt
```

您应该看到：
```
Loading ESM2 model from local path: ./esm2_model
```

而不是尝试从 Hugging Face 下载。

### 方案 3: 手动下载模型文件

如果无法使用 Python 下载，也可以手动下载：

1. 访问模型页面（需要能访问外网）：
   - ESM2-650M: https://huggingface.co/facebook/esm2_t33_650M_UR50D/tree/main
   - ESM2-35M: https://huggingface.co/facebook/esm2_t12_35M_UR50D/tree/main

2. 下载所有文件到同一个目录：
   - `config.json`
   - `pytorch_model.bin` (最大的文件)
   - `tokenizer_config.json`
   - `vocab.txt`
   - `special_tokens_map.json`
   - 其他 `.json` 文件

3. 将所有文件放在一个目录下（如 `esm2_model/`）

4. 配置 `local_esm_model_path` 指向该目录

## 推荐的 ESM2 模型选择

根据您的需求选择合适的模型：

| 模型名称 | 参数量 | 嵌入维度 | 大小 | 性能 | 适用场景 |
|---------|--------|---------|------|------|---------|
| esm2_t33_650M_UR50D | 650M | 1280 | 2.5GB | 最佳 | 生产环境、论文实验 |
| esm2_t30_150M_UR50D | 150M | 640 | 600MB | 较好 | 显存受限的GPU |
| esm2_t12_35M_UR50D | 35M | 480 | 140MB | 一般 | 快速实验、CPU训练 |
| esm2_t6_8M_UR50D | 8M | 320 | 35MB | 较差 | 仅用于快速测试 |

### 使用不同模型的配置示例

#### 使用 ESM2-650M（默认，最佳性能）
```yaml
model:
  esm_model_name: "facebook/esm2_t33_650M_UR50D"
  protein_embedding_dim: 1280
  local_esm_model_path: "./esm2_model"
```

#### 使用 ESM2-35M（快速、小型）
```yaml
model:
  esm_model_name: "facebook/esm2_t12_35M_UR50D"
  protein_embedding_dim: 480  # 注意修改嵌入维度
  local_esm_model_path: "./esm2_t12_model"
```

#### 使用 ESM2-150M（平衡）
```yaml
model:
  esm_model_name: "facebook/esm2_t30_150M_UR50D"
  protein_embedding_dim: 640  # 注意修改嵌入维度
  local_esm_model_path: "./esm2_t30_model"
```

## 验证离线模式是否正常工作

运行以下简单测试：

```python
#!/usr/bin/env python3
"""测试离线模式是否正常工作"""

import sys
sys.path.insert(0, '.')

from src.data import ProteinProcessor

# 测试加载本地模型
print("测试离线模式...")
processor = ProteinProcessor(
    model_name="facebook/esm2_t33_650M_UR50D",
    device="cpu",
    local_model_path="./esm2_model"
)

# 测试编码
test_seq = "ACDEFGHIKLMNPQRSTVWY"
embedding = processor.encode_sequence(test_seq)
print(f"✓ 成功生成蛋白质嵌入，维度: {embedding.shape}")
print("离线模式工作正常！")
```

## 常见问题排查

### 问题 1: 仍然尝试连接网络

**症状**：即使设置了 `local_esm_model_path`，程序仍然尝试连接 huggingface.co

**解决**：
- 检查配置文件路径是否正确
- 确认模型文件完整（包含所有必需的文件）
- 尝试使用绝对路径

### 问题 2: 加载本地模型失败

**错误信息**：
```
无法从本地路径加载模型: ./esm2_model
```

**解决**：
1. 检查目录是否存在：`ls -la ./esm2_model`
2. 检查是否包含必需文件：
   ```bash
   ls ./esm2_model/
   # 应该包含: config.json, pytorch_model.bin, tokenizer_config.json 等
   ```
3. 检查文件权限：`chmod -R 755 ./esm2_model`

### 问题 3: 嵌入维度不匹配

**错误信息**：
```
RuntimeError: size mismatch...
```

**解决**：确保配置文件中的 `protein_embedding_dim` 与模型匹配：
- esm2_t33_650M: 1280
- esm2_t30_150M: 640
- esm2_t12_35M: 480
- esm2_t6_8M: 320

### 问题 4: 显存不足

**症状**：
```
RuntimeError: CUDA out of memory
```

**解决**：
1. 使用更小的模型（如 esm2_t12_35M）
2. 减小 batch_size
3. 使用 CPU：在 config.yaml 中设置 `device: "cpu"`

## 网络代理设置（备选方案）

如果您有代理服务器，可以设置代理：

```bash
# HTTP 代理
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080

# SOCKS5 代理
export ALL_PROXY=socks5://127.0.0.1:1080
```

## 总结

推荐的工作流程：

1. **开发阶段**：使用小模型（esm2_t12_35M）快速迭代
2. **验证阶段**：使用中等模型（esm2_t30_150M）
3. **生产阶段**：使用大模型（esm2_t33_650M）获得最佳性能

所有模型都支持离线模式，只需提前下载并配置即可。

## 需要帮助？

如果遇到问题：
1. 检查本文档的常见问题排查部分
2. 确认模型文件完整性
3. 在 GitHub 提交 issue 并附上完整错误信息
