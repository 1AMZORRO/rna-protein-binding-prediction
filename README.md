# RNA-蛋白质结合位点预测模型

本项目实现了一个基于深度学习的RNA-蛋白质结合位点预测模型，采用cross-attention机制来学习RNA序列与蛋白质序列之间的相互作用。该实现参考了iDeepG的设计思想，使用RNA序列作为Query，蛋白质序列作为Key/Value进行交叉注意力计算。

## 项目特点

- **多模态输入处理**: RNA序列使用one-hot编码（4维向量），蛋白质序列使用ESM2预训练模型生成embedding（1280维）
- **Cross-attention机制**: RNA序列作为Query，蛋白质序列作为Key/Value，学习序列间的相互作用关系
- **端到端训练**: 支持完整的训练、验证和测试流程
- **注意力可视化**: 提供注意力矩阵热力图和结合位点分析
- **灵活配置**: 通过YAML配置文件管理所有超参数

## 安装要求

### 环境要求
- Python >= 3.8
- CUDA (可选，用于GPU加速)

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包括：
- `torch>=1.9.0` - PyTorch深度学习框架
- `transformers>=4.20.0` - Hugging Face Transformers库，用于ESM2模型
- `biopython>=1.79` - 生物信息学工具，用于处理FASTA文件
- `scikit-learn>=1.0.0` - 评估指标计算
- `matplotlib>=3.5.0` - 绘图工具
- `seaborn>=0.11.0` - 数据可视化
- `pyyaml>=6.0` - YAML配置文件解析
- `tqdm>=4.64.0` - 进度条显示
- `numpy>=1.21.0` - 数值计算
- `pandas>=1.3.0` - 数据处理

## 快速开始

### 1. 生成示例数据

如果您还没有数据，可以使用提供的脚本生成示例数据：

```bash
python scripts/generate_example_data.py
```

这会在 `data/examples/` 目录下生成：
- `rna_sequences.fasta` - RNA序列文件（101 bp长度）
- `protein_sequences.fasta` - 蛋白质序列文件
- `labels.txt` - 结合标签文件（1表示结合，0表示不结合）

### 2. 配置设置

配置文件位于 `config/config.yaml`，主要参数包括：

```yaml
model:
  rna_seq_length: 101           # RNA序列固定长度
  rna_input_dim: 4              # One-hot编码维度
  protein_embedding_dim: 1280   # ESM2 embedding维度
  hidden_dim: 256               # 隐藏层维度
  num_attention_heads: 8        # 注意力头数
  num_layers: 3                 # Transformer层数

data:
  batch_size: 32                # 批次大小
  train_split: 0.7              # 训练集比例

training:
  epochs: 100                   # 训练轮数
  learning_rate: 0.001          # 学习率
```

### 3. 模型训练

使用示例数据训练（模型会自动生成模拟数据）：

```bash
python scripts/train.py --config config/config.yaml
```

使用自己的FASTA文件训练：

```bash
python scripts/train.py \
    --config config/config.yaml \
    --rna-fasta data/examples/rna_sequences.fasta \
    --protein-fasta data/examples/protein_sequences.fasta \
    --labels data/examples/labels.txt
```

训练过程中会：
- 保存最佳模型到 `models/checkpoints/best_model.pth`
- 保存训练历史到 `models/checkpoints/training_history.json`
- 生成训练曲线图到 `output/training_history.png`

### 4. 模型预测

使用训练好的模型进行预测：

```bash
python scripts/predict.py \
    --config config/config.yaml \
    --model models/checkpoints/best_model.pth \
    --rna-fasta data/examples/rna_sequences.fasta \
    --protein-fasta data/examples/protein_sequences.fasta \
    --output predictions.txt \
    --visualize
```

参数说明：
- `--model`: 训练好的模型检查点路径
- `--rna-fasta`: RNA序列FASTA文件
- `--protein-fasta`: 蛋白质序列FASTA文件
- `--labels`: （可选）真实标签文件，用于评估
- `--output`: 预测结果输出文件
- `--visualize`: 生成注意力可视化图
- `--output-dir`: 可视化图保存目录

预测输出：
- `predictions.txt` - 包含每个样本的预测概率
- `output/attention_sample_*.png` - 注意力热力图
- `output/binding_sites_sample_*.png` - 结合位点分析图
- `output/metrics.txt` - 评估指标（如果提供了标签）

### 5. 运行测试

验证安装和实现：

```bash
python tests/test_basic.py
```

## 模型架构

### 输入处理

1. **RNA序列处理** (`RNAProcessor`)
   - 输入：RNA序列字符串（如 "ACGUACGU..."）
   - 处理：
     - 将序列转换为大写，T转换为U
     - 填充或截断到固定长度（101 bp）
     - One-hot编码：A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], U=[0,0,0,1]
   - 输出：形状为 (101, 4) 的张量

2. **蛋白质序列处理** (`ProteinProcessor`)
   - 输入：蛋白质序列字符串（如 "ACDEFGH..."）
   - 处理：
     - 使用ESM2预训练模型 (`facebook/esm2_t33_650M_UR50D`) 生成embedding
     - 自动移除特殊token（CLS和EOS）
   - 输出：形状为 (seq_length, 1280) 的张量

### 核心架构

模型 (`RNAProteinBindingModel`) 包含以下组件：

1. **投影层**
   - RNA投影：将 (101, 4) 投影到 (101, hidden_dim)
   - 蛋白质投影：将 (seq_len, 1280) 投影到 (seq_len, hidden_dim)

2. **位置编码** (`PositionalEncoding`)
   - 为序列添加位置信息
   - 使用正弦/余弦位置编码

3. **Cross-Attention层** (`CrossAttentionLayer`)
   - RNA作为Query
   - 蛋白质作为Key和Value
   - 多头注意力机制（默认8个头）
   - 前馈神经网络（FFN）
   - 层归一化和残差连接

4. **输出层**
   - 全局平均池化
   - 两层全连接网络
   - Sigmoid激活函数输出结合概率 [0, 1]

### 损失函数和优化

- **损失函数**: 二元交叉熵损失（BCELoss）
- **优化器**: Adam优化器（带权重衰减）
- **学习率调度**: ReduceLROnPlateau（验证损失不降时减少学习率）
- **早停**: 验证损失连续15个epoch不改善时停止训练
- **梯度裁剪**: 防止梯度爆炸

## 评估指标

模型使用以下指标进行评估：
- **Accuracy**: 分类准确率
- **Precision**: 精确率
- **Recall**: 召回率
- **F1-score**: F1分数
- **AUC-ROC**: ROC曲线下面积

## 项目结构

```
rna-protein-binding-prediction/
├── config/
│   └── config.yaml              # 配置文件
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── rna_processor.py     # RNA序列处理
│   │   ├── protein_processor.py # 蛋白质序列处理
│   │   └── dataset.py           # 数据集类
│   ├── models/
│   │   ├── __init__.py
│   │   └── rna_protein_model.py # 模型定义
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py           # 训练器
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py           # 评估指标
│       └── visualization.py     # 可视化工具
├── scripts/
│   ├── train.py                 # 训练脚本
│   ├── predict.py               # 预测脚本
│   └── generate_example_data.py # 生成示例数据
├── tests/
│   └── test_basic.py            # 基础测试
├── data/
│   └── examples/                # 示例数据目录
├── models/
│   └── checkpoints/             # 模型检查点
├── output/                      # 输出文件
├── requirements.txt             # 依赖列表
└── README.md                    # 本文件
```

## 注意力可视化

模型支持多种可视化方式：

### 1. 注意力热力图
显示RNA序列的每个位置对蛋白质序列每个位置的注意力权重：
- X轴：蛋白质序列位置
- Y轴：RNA序列位置
- 颜色深度：注意力权重大小

### 2. 结合位点分析
识别并显示RNA和蛋白质中最重要的结合位点：
- 基于注意力权重总和
- 显示Top-K最可能的结合位点
- 包括具体的核苷酸/氨基酸信息

## 数据格式

### RNA FASTA格式
```
>RNA_1
ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUA
>RNA_2
UGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAU
```

### 蛋白质FASTA格式
```
>PROTEIN_1
ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY
>PROTEIN_2
LMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKL
```

### 标签格式
```
1
0
```
每行一个标签，1表示结合，0表示不结合，顺序与FASTA文件一致。

## 使用建议

1. **数据准备**
   - RNA序列应为101 bp长度（会自动填充或截断）
   - 蛋白质序列可以是任意长度
   - 确保RNA和蛋白质序列数量一致

2. **训练优化**
   - 建议使用GPU进行训练（ESM2模型较大）
   - 如果显存不足，可以：
     - 减小batch_size
     - 减小hidden_dim
     - 使用更小的ESM2模型（如 `facebook/esm2_t12_35M_UR50D`）

3. **模型调优**
   - 调整num_attention_heads和num_layers
   - 调整learning_rate和weight_decay
   - 使用early_stopping避免过拟合

## 常见问题

**Q: 训练时显存不足怎么办？**
A: 减小batch_size，或使用更小的ESM2模型，或使用CPU训练（速度会较慢）。

**Q: 如何使用自己的eCLIP-seq数据？**
A: 将RNA序列和蛋白质序列分别保存为FASTA格式，标签保存为文本文件，每行一个0或1。

**Q: 预测速度很慢怎么办？**
A: ESM2模型较大，首次运行需要下载。后续运行会使用缓存。建议使用GPU加速。

**Q: 可以只预测不训练吗？**
A: 需要先训练模型或使用预训练的检查点，然后使用predict.py进行预测。

## 引用

如果使用本代码，请引用：

```bibtex
@software{rna_protein_binding_prediction,
  title={RNA-Protein Binding Site Prediction with Cross-Attention},
  year={2024},
  url={https://github.com/1AMZORRO/rna-protein-binding-prediction}
}
```

相关工作：
- iDeepG: https://github.com/userscy/iDeepG
- ESM2: https://github.com/facebookresearch/esm

## 许可证

MIT License
