# 模型架构说明

本文档详细说明RNA-蛋白质结合位点预测模型的架构设计。

## 整体架构

```
输入: RNA序列 + 蛋白质序列
  ↓
[RNA处理] → One-hot编码 (101, 4)
[蛋白质处理] → ESM2 Embedding (seq_len, 1280)
  ↓
[投影层] → 统一维度 (hidden_dim)
  ↓
[位置编码] → 添加位置信息
  ↓
[Cross-Attention层 × N] → RNA查询蛋白质
  ↓
[全局池化] → 聚合序列信息
  ↓
[分类头] → 结合概率 (0-1)
```

## 详细组件说明

### 1. 输入处理

#### RNA序列处理 (RNAProcessor)

**输入**: RNA序列字符串，如 `"ACGUACGUACGU..."`

**处理流程**:
1. 转换为大写，T→U
2. 长度标准化：
   - 短于101: 用'N'填充
   - 长于101: 从中心截断
3. One-hot编码：
   - A → [1, 0, 0, 0]
   - C → [0, 1, 0, 0]
   - G → [0, 0, 1, 0]
   - U → [0, 0, 0, 1]
   - N → [0, 0, 0, 0]

**输出**: 张量形状 `(101, 4)`

#### 蛋白质序列处理 (ProteinProcessor)

**输入**: 蛋白质序列字符串，如 `"ACDEFGHIKL..."`

**处理流程**:
1. 使用ESM2模型tokenize
2. 通过预训练的ESM2模型生成embedding
3. 移除特殊token (CLS, EOS)

**输出**: 张量形状 `(protein_length, 1280)`

### 2. 投影层 (Projection Layers)

将不同维度的输入映射到统一的隐藏空间：

```
RNA: (batch, 101, 4) → Linear → (batch, 101, hidden_dim)
Protein: (batch, seq_len, 1280) → Linear → (batch, seq_len, hidden_dim)
```

**参数**:
- `hidden_dim`: 默认256，可配置

### 3. 位置编码 (Positional Encoding)

使用正弦余弦位置编码为序列添加位置信息：

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**作用**: 让模型感知序列中的位置信息

### 4. Cross-Attention层 (CrossAttentionLayer)

核心组件，实现RNA和蛋白质序列的交互学习。

#### 多头注意力 (Multi-Head Attention)

```
Query (Q): RNA embeddings
Key (K): Protein embeddings  
Value (V): Protein embeddings

Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

**参数**:
- `num_attention_heads`: 注意力头数（默认8）
- 每个头的维度 = `hidden_dim / num_attention_heads`

**功能**:
- RNA的每个位置查询蛋白质的所有位置
- 学习RNA-蛋白质的相互作用模式
- 多头机制捕获不同类型的交互

#### 前馈网络 (Feed-Forward Network)

```
FFN(x) = ReLU(Linear(x)) → Dropout → Linear → Dropout
```

**结构**:
- 输入维度: `hidden_dim`
- 隐藏维度: `hidden_dim * 4`
- 输出维度: `hidden_dim`

#### 残差连接和层归一化

```
output = LayerNorm(input + Attention(input))
output = LayerNorm(output + FFN(output))
```

**作用**:
- 缓解梯度消失
- 加速训练收敛
- 提升模型性能

### 5. 多层堆叠

Cross-Attention层可以堆叠多层（默认3层）：

```
Layer 1: RNA_0 → Attention → RNA_1
Layer 2: RNA_1 → Attention → RNA_2
Layer 3: RNA_2 → Attention → RNA_3
```

每一层都从RNA查询蛋白质，逐步提取更高级的特征。

### 6. 输出层

#### 全局池化

```
Global Average Pooling over RNA sequence dimension:
(batch, 101, hidden_dim) → (batch, hidden_dim)
```

**作用**: 将变长序列信息聚合为固定长度向量

#### 分类头

```
hidden_dim → 128 → ReLU → Dropout → 1 → Sigmoid
```

**输出**: 结合概率 [0, 1]

## 模型参数量

以默认配置为例：

```yaml
rna_input_dim: 4
protein_input_dim: 1280
hidden_dim: 256
num_attention_heads: 8
num_layers: 3
```

**主要参数分布**:
1. 投影层: ~330K 参数
2. 位置编码: 0 参数（固定）
3. Cross-Attention层 × 3: ~2.3M 参数
4. 分类头: ~33K 参数

**总计**: 约 2.7M 可训练参数

## 前向传播流程

```python
# 伪代码
def forward(rna_seq, protein_seq):
    # 1. 输入处理
    rna_emb = one_hot_encode(rna_seq)        # (batch, 101, 4)
    protein_emb = esm2_encode(protein_seq)   # (batch, prot_len, 1280)
    
    # 2. 投影
    rna_hidden = rna_projection(rna_emb)          # (batch, 101, 256)
    protein_hidden = protein_projection(protein_emb)  # (batch, prot_len, 256)
    
    # 3. 位置编码
    rna_hidden = add_pos_encoding(rna_hidden)
    protein_hidden = add_pos_encoding(protein_hidden)
    
    # 4. Cross-Attention（多层）
    for layer in cross_attention_layers:
        rna_hidden, attn_weights = layer(
            query=rna_hidden,
            key_value=protein_hidden
        )
    
    # 5. 全局池化
    pooled = global_avg_pool(rna_hidden)     # (batch, 256)
    
    # 6. 分类
    prob = classifier(pooled)                 # (batch, 1) → (batch,)
    
    return prob  # 结合概率
```

## 注意力机制详解

### 注意力计算

对于RNA序列的第i个位置，计算其对蛋白质所有位置的注意力：

```
attention_i = softmax([
    similarity(rna_i, protein_1),
    similarity(rna_i, protein_2),
    ...
    similarity(rna_i, protein_n)
])
```

### 注意力输出

```
output_i = Σ(attention_i[j] * protein_j)
```

RNA的每个位置得到一个加权的蛋白质表示。

### 多头机制

不同的头可以学习不同类型的相互作用：
- 头1: 可能关注碱基配对
- 头2: 可能关注疏水作用
- 头3: 可能关注静电作用
- ...

## 训练策略

### 损失函数

```python
loss = BCE_Loss(predicted_prob, true_label)
```

二元交叉熵损失，适合二分类问题。

### 优化器

```python
Adam(
    params=model.parameters(),
    lr=0.001,
    weight_decay=0.0001  # L2正则化
)
```

### 学习率调度

```python
ReduceLROnPlateau(
    optimizer,
    patience=5,           # 5个epoch不改善
    factor=0.5            # 学习率减半
)
```

### 早停

验证损失连续15个epoch不改善时停止训练。

### 梯度裁剪

```python
clip_grad_norm_(model.parameters(), max_norm=1.0)
```

防止梯度爆炸。

## 可视化与解释

### 注意力热力图

通过可视化注意力权重矩阵，可以看到：
- RNA的哪些位置关注蛋白质的哪些位置
- 可能的结合位点
- 序列间的相互作用模式

### 结合位点分析

对注意力权重求和：
- RNA维度求和 → 识别重要的RNA位点
- 蛋白质维度求和 → 识别重要的蛋白质位点

## 模型优势

1. **多模态融合**: 结合RNA和蛋白质的不同表示
2. **注意力机制**: 显式建模序列间交互
3. **可解释性**: 注意力权重提供生物学洞察
4. **端到端**: 从序列直接预测，无需手工特征
5. **迁移学习**: 利用ESM2的预训练知识

## 模型限制

1. **固定RNA长度**: 当前限制为101 bp
2. **计算成本**: ESM2模型较大，需要GPU
3. **数据需求**: 需要足够的标注数据进行训练
4. **序列长度**: 蛋白质序列过长会增加显存消耗

## 扩展方向

1. **变长RNA**: 支持不同长度的RNA序列
2. **结构信息**: 引入RNA二级结构或蛋白质结构
3. **多任务学习**: 同时预测结合位点和亲和力
4. **图神经网络**: 利用分子图表示
5. **注意力变体**: 尝试其他注意力机制（如局部注意力）
