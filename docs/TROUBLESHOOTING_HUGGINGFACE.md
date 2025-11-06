# 解决 "无法访问 huggingface.co" 错误

## 问题描述

当您运行训练命令时出现以下错误：

```
OSError: We couldn't connect to 'https://huggingface.co' to load this file...
Failed to resolve 'huggingface.co' ([Errno -2] Name or service not known)
```

这是因为您的网络环境无法访问 Hugging Face 网站。

## 快速解决方案

### 方案 1: 使用镜像站（最简单，推荐中国大陆用户）

```bash
# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 然后正常运行训练
CUDA_VISIBLE_DEVICES=2 python scripts/train.py \
    --config config/config.yaml \
    --rna-fasta data/trains/rna_sequences.fasta \
    --protein-fasta data/trains/protein_sequences.fasta \
    --labels data/trains/labels.txt
```

### 方案 2: 离线模式（完全无法联网）

#### 第一步：在有网络的机器上下载模型

```bash
# 方法 A: 使用我们提供的下载脚本（推荐）
python scripts/download_esm2_model.py --mirror

# 方法 B: 手动下载
python -c "
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('facebook/esm2_t33_650M_UR50D')
tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
model.save_pretrained('./esm2_model')
tokenizer.save_pretrained('./esm2_model')
"
```

#### 第二步：复制模型到无网络的服务器

```bash
# 使用 scp
scp -r esm2_model zawang@sjtu1:/data1/zawang/rna-protein-binding-prediction/

# 或使用 rsync
rsync -avz esm2_model/ zawang@sjtu1:/data1/zawang/rna-protein-binding-prediction/esm2_model/
```

#### 第三步：修改配置文件

编辑 `config/config.yaml`：

```yaml
model:
  esm_model_name: "facebook/esm2_t33_650M_UR50D"
  protein_embedding_dim: 1280
  local_esm_model_path: "./esm2_model"  # 设置本地模型路径
```

或者使用绝对路径：

```yaml
model:
  local_esm_model_path: "/data1/zawang/rna-protein-binding-prediction/esm2_model"
```

#### 第四步：正常运行训练

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train.py \
    --config config/config.yaml \
    --rna-fasta data/trains/rna_sequences.fasta \
    --protein-fasta data/trains/protein_sequences.fasta \
    --labels data/trains/labels.txt
```

现在您应该看到：

```
Loading ESM2 model from local path: ./esm2_model
✓ 模型加载成功
```

而不是尝试连接 huggingface.co。

## 使用更小的模型（可选）

如果您希望更快的下载和训练速度，可以使用更小的 ESM2 模型：

### 下载小型模型

```bash
# 使用下载脚本
python scripts/download_esm2_model.py \
    --model esm2_t12_35M_UR50D \
    --save-path ./esm2_t12_model \
    --mirror

# 或手动下载
python -c "
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('facebook/esm2_t12_35M_UR50D')
tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
model.save_pretrained('./esm2_t12_model')
tokenizer.save_pretrained('./esm2_t12_model')
"
```

### 修改配置

编辑 `config/config.yaml`：

```yaml
model:
  esm_model_name: "facebook/esm2_t12_35M_UR50D"
  protein_embedding_dim: 480  # 注意：t12 模型的嵌入维度是 480
  local_esm_model_path: "./esm2_t12_model"
```

### 模型对比

| 模型 | 大小 | 嵌入维度 | 性能 | 速度 |
|------|------|---------|------|------|
| esm2_t33_650M | 2.5GB | 1280 | 最佳 | 慢 |
| esm2_t30_150M | 600MB | 640 | 较好 | 中等 |
| esm2_t12_35M | 140MB | 480 | 一般 | 快 |

## 验证设置

创建测试脚本 `test_setup.py`：

```python
#!/usr/bin/env python3
import yaml
from src.data import ProteinProcessor

# 加载配置
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 测试加载模型
print(f"模型名称: {config['model']['esm_model_name']}")
print(f"本地路径: {config['model']['local_esm_model_path']}")

try:
    processor = ProteinProcessor(
        model_name=config['model']['esm_model_name'],
        device='cpu',
        local_model_path=config['model'].get('local_esm_model_path')
    )
    print("✓ 模型加载成功!")
    
    # 测试编码
    test_seq = "ACDEFGHIKLMNPQRSTVWY"
    embedding = processor.encode_sequence(test_seq)
    print(f"✓ 编码成功，嵌入维度: {embedding.shape}")
    
except Exception as e:
    print(f"✗ 错误: {e}")
```

运行测试：

```bash
python test_setup.py
```

## 常见问题

### Q: 仍然尝试连接网络

**检查配置文件**：
```bash
grep local_esm_model_path config/config.yaml
```

确保路径设置正确且不为 `null`。

### Q: 找不到模型文件

**检查目录内容**：
```bash
ls -la ./esm2_model/
```

应该包含：
- config.json
- pytorch_model.bin
- tokenizer_config.json
- vocab.txt

### Q: 嵌入维度不匹配

确保 `protein_embedding_dim` 与模型匹配：
- esm2_t33_650M → 1280
- esm2_t30_150M → 640
- esm2_t12_35M → 480

## 获取更多帮助

- 详细文档：`docs/OFFLINE_MODE.md`
- 下载工具：`python scripts/download_esm2_model.py --help`
- 测试脚本：`python tests/test_offline_mode.py`
- 快速入门：`QUICKSTART.md`

## 总结

1. **中国大陆用户**：使用镜像站（方案1）
2. **完全离线环境**：下载模型到本地（方案2）
3. **显存受限**：使用更小的模型
4. **遇到问题**：运行测试脚本验证配置
