# RNA-蛋白质结合位点预测模型

本项目实现了一个基于深度学习的RNA-蛋白质结合位点预测模型，采用cross-attention机制来学习RNA序列与蛋白质序列之间的相互作用。

## 项目特点

- **多模态输入处理**: RNA序列使用one-hot编码，蛋白质序列使用ESM2预训练模型生成embedding
- **Cross-attention机制**: RNA序列作为Query，蛋白质序列作为Key/Value，学习序列间的相互作用
- **端到端训练**: 支持完整的训练、验证和测试流程
- **可视化分析**: 提供注意力矩阵可视化和结合位点分析
- **灵活配置**: 通过YAML配置文件管理所有超参数

## 安装要求

```bash
pip install -r requirements.txt
```

主要依赖包括：
- torch>=1.9.0
- transformers>=4.20.0
- biopython>=1.79
- scikit-learn>=1.0.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- pyyaml>=6.0
- tqdm>=4.64.0

## 快速开始

### 1. 数据准备

准备RNA和蛋白质序列的FASTA文件：

```
data/examples/rna_sequences.fasta
data/examples/protein_sequences.fasta
```

### 2. 配置设置

修改 `config/config.yaml` 中的配置参数：

```yaml
model:
  rna_seq_length: 101
  hidden_dim: 256
  num_attention_heads: 8
  num_layers: 3

data:
  batch_size: 32
  train_split: 0.7

training:
  epochs: 100
  learning_rate: 0.001
```

### 3. 数据预处理

```bash
python scripts/train.py --config config/config.yaml --preprocess
```

### 4. 模型训练

```bash
python scripts/train.py --config config/config.yaml
```

### 5. 模型预测

```bash
python scripts/predict.py \
    --config config/config.yaml \
    --model models/checkpoints/best_model.pth \
    --rna_fasta data/examples/rna_sequences.fasta \
    --protein_fasta data/examples/protein_sequences.fasta \
    --output predictions.txt \
    --visualize
```

## 模型架构

### 输入处理
- **RNA序列**: 使用one-hot编码转换为4维向量表示
- **蛋白质序列**: 使用ESM2预训练模型生成1280维embedding

### 核心架构
1. **投影层**: 将输入embedding投影到统一的隐藏维度
2. **位置编码**: 为序列添加位置信息
3. **Multi-head Cross-attention**: RNA作为Query，蛋白质作为Key/Value
4. **前馈网络**: 进一步处理注意力输出
5. **输出层**: 预测结合概率

### 损失函数
使用二元交叉熵损失函数(BCELoss)进行优化。

## 评估指标

模型使用以下指标进行评估：
- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC

## 文件结构

```
rna-protein-binding-prediction/
├── config/                 # 配置文件
├── src/                    # 源代码
│   ├── data/              # 数据处理模块
│   ├── models/            # 模型定义
│   ├── training/          # 训练相关
│   └── utils/             # 工具函数
├── scripts/               # 执行脚本
├── data/                  # 数据目录
└── models/                # 模型保存目录
```

## 高级功能

### 注意力可视化
使用 `--visualize` 参数可以生成注意力矩阵热力图，帮助理解模型学到的RNA-蛋白质相互作用模式。

### 结合位点预测
模型不仅能预测结合概率，还能通过注意力权重识别具体的结合位点。

### 模型检查点
支持训练过程中的模型保存和恢复，便于长时间训练和模型调优。

## 引用

如果使用本代码，请引用：

```bibtex
@software{rna_protein_binding_prediction,
  title={RNA-Protein Binding Site Prediction with Cross-Attention},
  author={Your Name},
  year={2024},
  url={https://github.com/username/rna-protein-binding-prediction}
}
```

## 许可证

MIT License# rna-protein-binding-prediction
