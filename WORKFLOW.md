# RNA-蛋白质结合预测 - 完整工作流程

本文档详细说明了从数据准备到模型训练的完整工作流程。

## 数据准备流程

### 第一步：合并RNA序列文件

将8个部分的RNA序列文件合并为一个完整的训练文件：

```bash
cat part1.fasta part2.fasta part3.fasta part4.fasta \
    part5.fasta part6.fasta part7.fasta part8.fasta > 168_train.fasta
```

**结果**：
- 输出文件：`168_train.fasta`
- 大小：181 MB
- 序列数量：1,245,616个RNA序列
- 包含168个不同蛋白质的数据

### 第二步：建立RNA-蛋白质-标签映射关系

运行映射脚本来建立RNA序列、标签和蛋白质之间的对应关系：

```bash
python scripts/create_rna_protein_mapping.py \
    --rna-fasta 168_train.fasta \
    --protein-fasta prot_seqs.fasta \
    --output-mapping rna_protein_mapping.txt \
    --output-json rna_protein_mapping.json \
    --output-labels 168_train_labels.txt
```

**生成的文件**：

1. **rna_protein_mapping.txt** (TSV格式)
   - 包含：RNA_ID、RNA_Sequence、Protein_ID、Label
   - 大小：约192 MB
   - 用途：完整的数据对应关系

2. **rna_protein_mapping.json** (JSON格式)
   - 包含：元数据和完整映射
   - 大小：约305 MB
   - 用途：结构化数据访问

3. **168_train_labels.txt** (纯文本)
   - 每行一个标签（1或0）
   - 大小：约2.4 MB
   - 用途：训练时的标签文件

**数据统计**：
```
总RNA序列: 1,245,616
成功映射: 1,245,616 (100%)
唯一蛋白质: 168
正样本 (binding): 622,808
负样本 (non-binding): 622,808
正负比例: 1:1 (完美平衡)
```

### 第三步：预计算蛋白质Embeddings（推荐）

为了大幅提升训练速度，建议预先计算所有蛋白质的embeddings：

```bash
python scripts/precompute_protein_embeddings.py \
    --protein-fasta prot_seqs.fasta \
    --output precomputed_protein_embeddings.pt \
    --model-name esm2_t33_650M_UR50D \
    --device cuda
```

**参数说明**：
- `--protein-fasta`: 蛋白质序列文件（prot_seqs.fasta含168个蛋白质）
- `--output`: 输出的embeddings文件
- `--model-name`: ESM2模型名称
  - `esm2_t6_8M_UR50D` - 最小，8M参数，适合测试
  - `esm2_t12_35M_UR50D` - 小型，35M参数
  - `esm2_t30_150M_UR50D` - 中型，150M参数
  - `esm2_t33_650M_UR50D` - 大型，650M参数（推荐）
  - `esm2_t36_3B_UR50D` - 超大，3B参数
- `--device`: cuda（GPU）或cpu

**注意事项**：
- 首次运行会自动下载ESM2模型（约2.5GB）
- 下载后的模型缓存在`~/.cache/torch/hub/checkpoints/`
- 预计算时间：使用GPU约5-30分钟（取决于模型大小）
- 输出文件大小：约100-500 MB（取决于embedding维度）

**为什么要预计算？**
- 168个蛋白质对应124万个RNA序列
- 如果每次训练都计算embeddings，一个epoch可能需要数小时
- 预计算后，训练时直接从内存加载，一个epoch可能只需要几分钟

## 训练流程

### 配置文件设置

编辑`config/config.yaml`，启用预计算embeddings：

```yaml
model:
  esm_model_name: "esm2_t33_650M_UR50D"
  precomputed_embeddings: "precomputed_protein_embeddings.pt"  # 启用预计算
```

### 开始训练

**使用预计算embeddings训练**（推荐）：

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

## 速度对比

| 方式 | 每个epoch时间 | 总训练时间(100 epochs) |
|------|---------------|------------------------|
| 不使用预计算 | 2-4小时 | 8-16天 |
| 使用预计算 | 5-15分钟 | 8-25小时 |

**速度提升**: 约 **20-50倍**

## 文件说明

### 数据文件（不在git仓库中，需要本地生成）

- `168_train.fasta` - 合并后的RNA序列
- `rna_protein_mapping.txt` - TSV格式映射
- `rna_protein_mapping.json` - JSON格式映射
- `168_train_labels.txt` - 标签文件
- `precomputed_protein_embeddings.pt` - 预计算的embeddings

### 原始数据文件（已在仓库中）

- `part1.fasta` ~ `part8.fasta` - 分割的RNA序列
- `prot_seqs.fasta` - 168个蛋白质序列

### 脚本文件

- `scripts/create_rna_protein_mapping.py` - 建立映射关系
- `scripts/precompute_protein_embeddings.py` - 预计算embeddings
- `scripts/train.py` - 训练模型
- `scripts/predict.py` - 预测

## 常见问题

### Q1: 为什么要分三步准备数据？

A: 
1. 合并FASTA文件 - 将分散的数据整合
2. 建立映射关系 - 明确RNA序列、蛋白质和标签的对应
3. 预计算embeddings - 避免训练时重复计算

### Q2: 可以跳过预计算embeddings吗？

A: 可以，但不推荐。跳过会导致：
- 训练速度极慢（每个epoch数小时）
- 每次训练都需要加载大型ESM2模型
- GPU显存占用更高

### Q3: 映射文件太大怎么办？

A: 这些文件仅在数据准备阶段需要，可以在训练前生成，训练后删除。实际训练只需要：
- `168_train.fasta`
- `168_train_labels.txt`
- `precomputed_protein_embeddings.pt`

### Q4: 如何验证数据准备是否成功？

A: 检查以下统计信息：
```bash
# 检查RNA序列数量
grep -c "^>" 168_train.fasta
# 应该输出: 1245616

# 检查标签数量
wc -l 168_train_labels.txt
# 应该输出: 1245616

# 检查蛋白质数量
grep -c "^>" prot_seqs.fasta
# 应该输出: 168
```

## 技术改进说明

### ESM模型加载方式变更

**之前（使用Hugging Face）**：
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
```

**现在（使用fair-esm）**：
```python
import esm
from esm import pretrained
model, alphabet = pretrained.esm2_t33_650M_UR50D()
```

**优势**：
- 直接使用官方实现
- 更稳定和高效
- 支持更多模型变体
- 更好的兼容性

## 总结

通过以上三步数据准备和预计算优化，可以将训练速度提升20-50倍，使得在大规模数据集（124万样本）上的训练变得可行。
