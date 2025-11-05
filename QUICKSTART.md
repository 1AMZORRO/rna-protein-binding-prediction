# 快速入门指南

5分钟快速开始使用RNA-蛋白质结合位点预测模型。

## 1. 安装依赖 (2分钟)

```bash
# 克隆仓库
git clone https://github.com/1AMZORRO/rna-protein-binding-prediction.git
cd rna-protein-binding-prediction

# 安装依赖
pip install -r requirements.txt
```

## 2. 验证安装 (1分钟)

```bash
# 运行测试
python tests/test_basic.py
```

看到 "ALL TESTS PASSED!" 说明安装成功。

## 3. 生成示例数据 (30秒)

```bash
# 生成100个示例样本
python scripts/generate_example_data.py
```

这会在 `data/examples/` 创建：
- `rna_sequences.fasta` - 100条RNA序列
- `protein_sequences.fasta` - 100条蛋白质序列  
- `labels.txt` - 结合标签

## 4. 训练模型 (可选，需要GPU)

```bash
# 使用示例数据训练（会自动生成模拟数据）
python scripts/train.py --config config/config.yaml
```

或使用自己的数据：

```bash
python scripts/train.py \
    --config config/config.yaml \
    --rna-fasta data/examples/rna_sequences.fasta \
    --protein-fasta data/examples/protein_sequences.fasta \
    --labels data/examples/labels.txt
```

**注意**: 
- 首次运行会下载ESM2模型（约2.5GB）
- 建议使用GPU，CPU训练会很慢
- 可在 `config/config.yaml` 中调整参数

## 5. 预测 (需要已训练的模型)

```bash
python scripts/predict.py \
    --config config/config.yaml \
    --model models/checkpoints/best_model.pth \
    --rna-fasta data/examples/rna_sequences.fasta \
    --protein-fasta data/examples/protein_sequences.fasta \
    --output predictions.txt \
    --visualize
```

输出：
- `predictions.txt` - 预测结果
- `output/attention_*.png` - 注意力热力图
- `output/binding_sites_*.png` - 结合位点分析

## 最简单的测试流程

如果只想快速测试代码能否运行：

```bash
# 1. 安装依赖
pip install torch biopython scikit-learn matplotlib seaborn pyyaml tqdm pandas numpy transformers

# 2. 运行测试
python tests/test_basic.py

# 3. 生成示例数据
python scripts/generate_example_data.py
```

## 自定义数据格式

### RNA FASTA 文件
```
>RNA_1
ACGUACGUACGU... (101 bp)
>RNA_2
UGCAUGCAUGCA...
```

### 蛋白质 FASTA 文件
```
>PROTEIN_1
ACDEFGHIKLMNPQRS...
>PROTEIN_2
LMNPQRSTVWYACDEF...
```

### 标签文件
```
1
0
1
```

每行一个数字（1=结合，0=不结合），顺序与FASTA一致。

## 常见问题

**Q: 显存不足怎么办？**
```yaml
# 修改 config/config.yaml
data:
  batch_size: 8  # 减小批次

model:
  hidden_dim: 128  # 减小模型
  esm_model_name: "facebook/esm2_t12_35M_UR50D"  # 用更小的ESM2
```

**Q: 没有GPU可以用吗？**

可以，但训练会很慢。在 `config/config.yaml` 中设置：
```yaml
device: "cpu"
```

**Q: 预测需要多久？**

取决于：
- 样本数量
- 蛋白质序列长度
- 是否使用GPU

通常100个样本在GPU上约1-2分钟。

## 下一步

- 阅读 [README.md](README.md) 了解详细信息
- 查看 [docs/USAGE.md](docs/USAGE.md) 学习高级用法
- 查看 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 理解模型结构

## 帮助

遇到问题？
1. 查看错误信息
2. 检查数据格式
3. 阅读文档
4. 提交 GitHub Issue
