# 离线模式解决方案总结

本文档总结了为解决"无法访问 huggingface.co"问题所做的改进。

## 问题

用户在运行训练时遇到以下错误：
```
OSError: We couldn't connect to 'https://huggingface.co' to load this file...
Failed to resolve 'huggingface.co' ([Errno -2] Name or service not known)
```

## 解决方案

### 1. 代码改进

**修改的文件：**
- `src/data/protein_processor.py` - 添加 `local_model_path` 参数支持离线模式
- `config/config.yaml` - 添加 `local_esm_model_path` 配置项
- `scripts/train.py` - 支持从配置读取本地模型路径
- `scripts/predict.py` - 支持从配置读取本地模型路径

**核心改动：**
```python
class ProteinProcessor:
    def __init__(self, model_name: str, device: str, local_model_path: Optional[str] = None):
        # 如果提供了本地路径，从本地加载
        model_path = local_model_path if local_model_path else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            local_files_only=bool(local_model_path)
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=bool(local_model_path)
        )
```

### 2. 新增工具

**`scripts/download_esm2_model.py`**
- 便捷的模型下载工具
- 支持所有 ESM2 模型版本
- 支持镜像站下载
- 提供模型验证功能

**使用示例：**
```bash
# 查看可用模型
python scripts/download_esm2_model.py --info

# 下载默认模型
python scripts/download_esm2_model.py

# 使用镜像站下载
python scripts/download_esm2_model.py --mirror

# 下载小型模型
python scripts/download_esm2_model.py --model esm2_t12_35M_UR50D --save-path ./esm2_t12_model
```

### 3. 文档更新

**新增文档：**
- `docs/OFFLINE_MODE.md` - 离线模式详细指南（5971字）
- `docs/TROUBLESHOOTING_HUGGINGFACE.md` - 快速故障排除指南（4244字）

**更新文档：**
- `README.md` - 添加离线模式FAQ和详细说明
- `QUICKSTART.md` - 添加离线模式快速指南

### 4. 测试

**`tests/test_offline_mode.py`**
- 测试配置文件加载
- 测试 ProteinProcessor 类签名
- 测试离线模式错误消息
- 测试在线模式错误消息

所有测试通过 ✓

## 使用方法

### 方法 1: 使用镜像站（推荐中国大陆用户）

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/train.py --config config/config.yaml ...
```

### 方法 2: 完全离线模式

**步骤 1：下载模型（在有网络的机器上）**
```bash
python scripts/download_esm2_model.py --mirror
```

**步骤 2：复制模型到目标服务器**
```bash
scp -r esm2_model user@server:/path/to/project/
```

**步骤 3：修改配置文件**
```yaml
model:
  local_esm_model_path: "./esm2_model"
```

**步骤 4：正常运行**
```bash
python scripts/train.py --config config/config.yaml ...
```

## 特性

✅ 支持完全离线模式
✅ 友好的中文错误提示
✅ 提供详细的解决方案
✅ 支持多个 ESM2 模型版本
✅ 包含便捷的下载工具
✅ 完整的测试覆盖
✅ 详细的中英文文档

## 相关文档

- 详细离线模式指南: [docs/OFFLINE_MODE.md](docs/OFFLINE_MODE.md)
- 快速故障排除: [docs/TROUBLESHOOTING_HUGGINGFACE.md](docs/TROUBLESHOOTING_HUGGINGFACE.md)
- 快速入门: [QUICKSTART.md](QUICKSTART.md)
- 主文档: [README.md](README.md)

## 模型选择

| 模型 | 参数 | 嵌入维度 | 大小 | 用途 |
|------|------|---------|------|------|
| esm2_t33_650M_UR50D | 650M | 1280 | 2.5GB | 生产/论文 |
| esm2_t30_150M_UR50D | 150M | 640 | 600MB | 显存受限 |
| esm2_t12_35M_UR50D | 35M | 480 | 140MB | 快速实验 |
| esm2_t6_8M_UR50D | 8M | 320 | 35MB | 测试 |

## 验证

运行测试确认一切正常：
```bash
python tests/test_offline_mode.py
```

预期输出：
```
✓ 所有测试通过!
离线模式功能正常工作。
```
