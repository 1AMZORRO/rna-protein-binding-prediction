# 使用指南

本文档详细介绍如何使用RNA-蛋白质结合位点预测模型。

## 目录

- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型预测](#模型预测)
- [结果解读](#结果解读)
- [参数调优](#参数调优)

## 环境配置

### 1. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python tests/test_basic.py
```

如果看到 "ALL TESTS PASSED!"，说明安装成功。

## 数据准备

### 方案1: 使用示例数据

生成100个示例样本用于测试：

```bash
python scripts/generate_example_data.py
```

### 方案2: 使用自己的数据

准备三个文件：

1. **RNA序列文件** (FASTA格式)
```
>RNA_ID_1
ACGUACGUACGUACGU...
>RNA_ID_2
UGCAUGCAUGCAUGCA...
```

2. **蛋白质序列文件** (FASTA格式)
```
>PROTEIN_ID_1
ACDEFGHIKLMNPQRS...
>PROTEIN_ID_2
LMNPQRSTVWYACDEF...
```

3. **标签文件** (每行一个数字)
```
1
0
```

**注意事项**：
- RNA和蛋白质序列数量必须一致
- 标签顺序必须与序列顺序对应
- RNA序列建议101 bp长度（会自动调整）
- 蛋白质序列可以是任意长度

### 方案3: 从eCLIP数据准备

如果您有eCLIP-seq数据：

1. 提取101 bp的RNA片段
2. 获取对应的蛋白质序列
3. 根据实验结果标注结合/非结合（1/0）

## 模型训练

### 基础训练

使用默认配置训练：

```bash
python scripts/train.py --config config/config.yaml
```

### 使用自定义数据训练

```bash
python scripts/train.py \
    --config config/config.yaml \
    --rna-fasta /path/to/rna_sequences.fasta \
    --protein-fasta /path/to/protein_sequences.fasta \
    --labels /path/to/labels.txt \
    --output-dir models/my_model
```

### 训练参数说明

在 `config/config.yaml` 中可以调整：

```yaml
training:
  epochs: 100                    # 训练轮数
  learning_rate: 0.001           # 学习率
  weight_decay: 0.0001           # 权重衰减（正则化）
  early_stopping_patience: 15    # 早停容忍度
  scheduler_patience: 5          # 学习率调整容忍度
  grad_clip: 1.0                 # 梯度裁剪
```

### 训练输出

训练完成后会生成：
- `models/checkpoints/best_model.pth` - 最佳模型
- `models/checkpoints/training_history.json` - 训练历史
- `output/training_history.png` - 训练曲线图

### 监控训练过程

训练过程中会显示：
```
Epoch 1/100
Training: 100%|████████| 7/7 [00:15<00:00, 0.46it/s, loss=0.6891]
Train Loss: 0.6891
Val Loss: 0.6823
Val Metrics: {'accuracy': 0.55, 'precision': 0.58, 'recall': 0.52, 'f1': 0.55, 'auc': 0.60}
```

## 模型预测

### 基础预测

```bash
python scripts/predict.py \
    --config config/config.yaml \
    --model models/checkpoints/best_model.pth \
    --rna-fasta data/examples/rna_sequences.fasta \
    --protein-fasta data/examples/protein_sequences.fasta \
    --output predictions.txt
```

### 带标签的预测（用于评估）

```bash
python scripts/predict.py \
    --config config/config.yaml \
    --model models/checkpoints/best_model.pth \
    --rna-fasta data/examples/rna_sequences.fasta \
    --protein-fasta data/examples/protein_sequences.fasta \
    --labels data/examples/labels.txt \
    --output predictions.txt
```

### 预测并生成可视化

```bash
python scripts/predict.py \
    --config config/config.yaml \
    --model models/checkpoints/best_model.pth \
    --rna-fasta data/examples/rna_sequences.fasta \
    --protein-fasta data/examples/protein_sequences.fasta \
    --output predictions.txt \
    --visualize \
    --output-dir visualizations
```

## 结果解读

### 预测结果文件

`predictions.txt` 包含：
```
prediction      label   correct
0.7234         1       True
0.3421         0       True
0.8901         1       True
```

- `prediction`: 结合概率 (0-1)
- `label`: 真实标签（如果提供）
- `correct`: 预测是否正确

### 评估指标

如果提供了标签，会输出：
```
==================================================
Evaluation Metrics
==================================================
Accuracy: 0.8500
Precision: 0.8200
Recall: 0.8700
F1: 0.8444
Auc: 0.9100
==================================================
```

- **Accuracy**: 整体准确率
- **Precision**: 预测为结合的样本中，真实结合的比例
- **Recall**: 真实结合的样本中，被正确预测的比例
- **F1**: Precision和Recall的调和平均
- **AUC**: ROC曲线下面积，越接近1越好

### 注意力可视化

生成的可视化图包括：

1. **attention_sample_*.png** - 注意力热力图
   - 显示RNA每个位置对蛋白质每个位置的关注程度
   - 颜色越深表示注意力越高
   - 帮助理解模型如何学习序列间的关系

2. **binding_sites_sample_*.png** - 结合位点分析
   - Top-10 RNA结合位点
   - Top-10 蛋白质结合位点
   - 基于注意力权重计算

## 参数调优

### 模型参数

在 `config/config.yaml` 中调整：

```yaml
model:
  hidden_dim: 256              # 增大提升表达能力，但增加计算量
  num_attention_heads: 8       # 通常是hidden_dim的因子
  num_layers: 3                # 增加层数可能提升性能
  dropout: 0.1                 # 0.1-0.3之间，防止过拟合
```

### 数据参数

```yaml
data:
  batch_size: 32               # GPU显存不足时减小
  train_split: 0.7             # 训练集比例
  val_split: 0.15              # 验证集比例
```

### 训练策略

```yaml
training:
  learning_rate: 0.001         # 常用范围：0.0001-0.01
  weight_decay: 0.0001         # L2正则化强度
  early_stopping_patience: 15  # 容忍多少轮不改善
```

### 显存优化

如果遇到显存不足（CUDA out of memory）：

1. 减小batch_size（如32→16→8）
2. 减小hidden_dim（如256→128）
3. 使用更小的ESM2模型：
   ```yaml
   model:
     esm_model_name: "facebook/esm2_t12_35M_UR50D"  # 更小的模型
     protein_embedding_dim: 480  # 对应的embedding维度
   ```

### 提升性能建议

1. **数据质量**
   - 确保标签准确
   - 增加数据量（建议>1000样本）
   - 保持正负样本平衡

2. **模型调整**
   - 先用小模型快速迭代
   - 逐步增加模型复杂度
   - 观察训练曲线，避免过拟合

3. **训练技巧**
   - 使用early stopping
   - 尝试不同的学习率
   - 增加训练轮数

## 常见问题

### Q1: 训练很慢怎么办？
- 使用GPU（CUDA）
- 减小batch_size
- 使用更小的ESM2模型

### Q2: 模型过拟合了？
- 增大dropout
- 增大weight_decay
- 减少模型参数（hidden_dim, num_layers）
- 增加训练数据

### Q3: 预测结果都接近0.5？
- 可能需要更多训练轮数
- 检查数据质量
- 调整学习率
- 增加模型容量

### Q4: 如何选择最优模型？
- 基于验证集AUC选择
- 考虑Precision和Recall的平衡
- 使用训练历史图分析

## 进阶使用

### 自定义模型架构

修改 `src/models/rna_protein_model.py` 可以：
- 调整attention机制
- 添加更多层
- 改变池化方式

### 自定义数据处理

修改 `src/data/` 下的处理器可以：
- 改变RNA编码方式
- 使用不同的蛋白质embedding
- 添加数据增强

### 批量预测

创建循环脚本批量处理多个文件：

```python
import subprocess

files = [
    ('rna1.fasta', 'protein1.fasta'),
    ('rna2.fasta', 'protein2.fasta'),
]

for rna_file, protein_file in files:
    subprocess.run([
        'python', 'scripts/predict.py',
        '--config', 'config/config.yaml',
        '--model', 'models/checkpoints/best_model.pth',
        '--rna-fasta', rna_file,
        '--protein-fasta', protein_file,
        '--output', f'predictions_{rna_file}.txt'
    ])
```

## 支持与反馈

遇到问题请：
1. 查看错误信息
2. 检查数据格式
3. 验证配置文件
4. 在GitHub提交issue
