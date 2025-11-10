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
- `fair-esm>=2.0.0` - Facebook ESM库，直接使用ESM2预训练模型
- `biopython>=1.79` - 生物信息学工具，用于处理FASTA文件
- `scikit-learn>=1.0.0` - 评估指标计算
- `matplotlib>=3.5.0` - 绘图工具
- `seaborn>=0.11.0` - 数据可视化
- `pyyaml>=6.0` - YAML配置文件解析
- `tqdm>=4.64.0` - 进度条显示
- `numpy>=1.21.0` - 数值计算
- `pandas>=1.3.0` - 数据处理

## 数据准备

### 1. 合并训练数据

本项目提供了8个部分的RNA序列文件（part1.fasta到part8.fasta）。首先需要将它们合并为一个完整的训练文件：

```bash
cat part1.fasta part2.fasta part3.fasta part4.fasta part5.fasta part6.fasta part7.fasta part8.fasta > 168_train.fasta
```

合并后的`168_train.fasta`包含1,245,616个RNA序列，对应168个不同的蛋白质。

### 2. 建立RNA序列、标签与蛋白质的对应关系

运行以下脚本来创建RNA序列、标签信息与蛋白质embeddings之间的映射：

```bash
python scripts/create_rna_protein_mapping.py \
    --rna-fasta 168_train.fasta \
    --protein-fasta prot_seqs.fasta \
    --output-mapping rna_protein_mapping.txt \
    --output-json rna_protein_mapping.json \
    --output-labels 168_train_labels.txt
```

这将生成三个文件：
- `rna_protein_mapping.txt` - TSV格式，包含RNA_ID、RNA序列、Protein_ID和Label
- `rna_protein_mapping.json` - JSON格式，包含元数据和完整映射
- `168_train_labels.txt` - 纯标签文件，每行一个标签（1=结合，0=不结合）

### 3. 预计算蛋白质Embeddings（推荐）

为了避免每次训练都重复计算蛋白质embeddings，建议预先计算所有168个蛋白质的embeddings：

```bash
python scripts/precompute_protein_embeddings.py \
    --protein-fasta prot_seqs.fasta \
    --output precomputed_protein_embeddings.pt \
    --model-name esm2_t33_650M_UR50D \
    --device cuda
```

参数说明：
- `--protein-fasta`: 蛋白质序列文件（prot_seqs.fasta包含168个蛋白质）
- `--output`: 输出的embeddings文件路径
- `--model-name`: ESM2模型名称，可选：
  - `esm2_t6_8M_UR50D` - 最小模型，8M参数
  - `esm2_t12_35M_UR50D` - 小模型，35M参数
  - `esm2_t30_150M_UR50D` - 中型模型，150M参数
  - `esm2_t33_650M_UR50D` - 大模型，650M参数（默认）
  - `esm2_t36_3B_UR50D` - 超大模型，3B参数
- `--device`: 使用cuda（GPU）或cpu

**注意**: 首次运行会自动下载ESM2模型（约2.5GB），后续运行将使用缓存。预计算完成后，训练时将直接加载预计算的embeddings，大幅提升速度。

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

### 2. 配置设置（使用预计算embeddings）

如果已经预计算了蛋白质embeddings，可以在配置文件中指定：

```yaml
model:
  rna_seq_length: 101           # RNA序列固定长度
  rna_input_dim: 4              # One-hot编码维度
  protein_embedding_dim: 1280   # ESM2 embedding维度
  hidden_dim: 256               # 隐藏层维度
  num_attention_heads: 8        # 注意力头数
  num_layers: 3                 # Transformer层数
  esm_model_name: esm2_t33_650M_UR50D  # ESM2模型名称
  precomputed_embeddings: precomputed_protein_embeddings.pt  # 预计算的embeddings路径

data:
  batch_size: 32                # 批次大小
  train_split: 0.7              # 训练集比例

training:
  epochs: 100                   # 训练轮数
  learning_rate: 0.001          # 学习率
```

### 3. 模型训练

**使用预计算embeddings训练（推荐）**：

```bash
python scripts/train.py \
    --config config/config.yaml \
    --rna-fasta 168_train.fasta \
    --protein-fasta prot_seqs.fasta \
    --labels 168_train_labels.txt \
    --use-precomputed-embeddings
```

**不使用预计算embeddings训练**：

```bash
python scripts/train.py \
    --config config/config.yaml \
    --rna-fasta 168_train.fasta \
    --protein-fasta prot_seqs.fasta \
    --labels 168_train_labels.txt
```

注意：不使用预计算embeddings时，每个epoch都需要重新计算蛋白质embeddings，训练速度会很慢。

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
   - 处理方式：
     - **使用fair-esm库直接加载ESM2预训练模型**（不再使用Hugging Face Transformers）
     - 支持多种ESM2模型变体（8M到3B参数）
     - 自动移除特殊token（BOS和EOS）
     - **支持预计算embeddings**：可以预先计算所有蛋白质的embeddings并保存，训练时直接加载
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
   - **强烈推荐**：使用预计算的蛋白质embeddings以加速训练

2. **训练优化**
   - **使用预计算embeddings**：避免每次训练都重新计算蛋白质embeddings（本项目168个蛋白质对应124万个RNA序列）
   - 建议使用GPU进行训练（ESM2模型较大）
   - 如果显存不足，可以：
     - 减小batch_size
     - 减小hidden_dim
     - 使用更小的ESM2模型（如 `esm2_t12_35M_UR50D` 或 `esm2_t6_8M_UR50D`）
   - 预计算embeddings后，训练时直接从内存加载，速度提升数百倍

3. **模型调优**
   - 调整num_attention_heads和num_layers
   - 调整learning_rate和weight_decay
   - 使用early_stopping避免过拟合

## 常见问题

**Q: 训练速度很慢怎么办？**
A: 
1. **最重要**：使用预计算的蛋白质embeddings（见"数据准备"部分）
2. 使用GPU加速
3. 使用更小的ESM2模型（如 `esm2_t12_35M_UR50D`）
4. 减小batch_size

**Q: 训练时显存不足怎么办？**
A: 减小batch_size，或使用更小的ESM2模型，或使用CPU训练（速度会较慢），或使用预计算embeddings（不需要在训练时加载ESM2模型）。

**Q: 如何使用自己的eCLIP-seq数据？**
A: 
1. 将RNA序列保存为FASTA格式
2. 将蛋白质序列保存为FASTA格式
3. 标签保存为文本文件，每行一个0或1
4. 运行`create_rna_protein_mapping.py`建立映射关系
5. 运行`precompute_protein_embeddings.py`预计算蛋白质embeddings（推荐）

**Q: 预测速度很慢怎么办？**
A: 使用预计算的蛋白质embeddings。首次运行ESM2需要下载模型（约2.5GB），后续运行会使用缓存。建议使用GPU加速。

**Q: 可以只预测不训练吗？**
A: 需要先训练模型或使用预训练的检查点，然后使用predict.py进行预测。

## ESM模型相关

### ESM模型加载方式变更

本项目已从使用 Hugging Face Transformers 库切换到直接使用 Facebook 的 fair-esm 库：

```python
import esm
from esm import pretrained

# 加载模型
model, alphabet = pretrained.esm2_t33_650M_UR50D()
```

这种方式的优点：
- 直接使用官方实现，更稳定
- 支持更多ESM模型变体
- 更好的性能和兼容性

### 网络问题

ESM2模型文件会从 Facebook 服务器下载（约2.5GB）。如果遇到网络问题：

1. **使用代理**：设置HTTP/HTTPS代理
2. **离线使用**：提前下载模型文件到 `~/.cache/torch/hub/checkpoints/`
3. **使用预计算embeddings**：一旦embeddings计算完成，后续训练不需要再加载ESM2模型

注意：本项目不再使用Hugging Face服务，因此不需要配置HF_ENDPOINT镜像站。

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
