# GEMORNA API 部署与使用说明

本文档说明如何部署并调用 GEMORNA REST API。API 基于 FastAPI 实现，入口为 `src.api:app`，对外提供 CDS 生成、UTR 生成和 UTR 表达/稳定性评分能力。

## 1. 目录与运行前提

请在仓库根目录运行所有命令，避免 checkpoint、vocab 和共享库路径解析失败。

API 运行依赖以下文件和目录：

- `src/api.py`：FastAPI 应用入口。
- `src/gemorna_services.py`：模型加载、生成和评分服务层。
- `checkpoints/`：CDS、5UTR、3UTR 生成与预测 checkpoint。
- `vocab/`：CDS 生成需要的蛋白和 CDS vocabulary。
- `src/shared/*.so`：闭源/编译扩展模块。
- `environment.yaml`：conda 环境定义，包含 FastAPI 和 Uvicorn。

服务启动后会按需懒加载模型。首次请求通常较慢，后续请求会复用进程内缓存的模型对象。设备选择沿用 PyTorch 逻辑：如果 CUDA 可用则使用 GPU，否则使用 CPU。

## 2. 本地 Conda 部署

创建环境：

```bash
conda env create -f environment.yaml
conda activate gemorna
```

如果环境已经存在，更新依赖：

```bash
conda env update -n gemorna -f environment.yaml
conda activate gemorna
```

启动 API：

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

也可以不激活环境，直接通过 `conda run` 启动：

```bash
conda run -n gemorna uvicorn src.api:app --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://localhost:8000/health
```

预期返回：

```json
{"status":"ok"}
```

FastAPI 自动生成的交互式接口文档地址：

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## 3. Docker 部署

从仓库根目录构建镜像：

```bash
docker build -t gemorna-api .
```

CPU 运行：

```bash
docker run --rm -p 8000:8000 gemorna-api
```

如果宿主机配置了 NVIDIA 驱动和 NVIDIA Container Toolkit，可以启用 GPU：

```bash
docker run --rm --gpus all -p 8000:8000 gemorna-api
```

容器默认监听 `0.0.0.0:8000`，宿主机通过 `http://localhost:8000` 访问。

## 4. API 端点总览

| 方法 | 路径 | 功能 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `POST` | `/api/v1/cds/open/generate` | 使用开放 Python 实现生成 CDS |
| `POST` | `/api/v1/cds/closed/generate` | 使用闭源扩展实现生成 CDS |
| `POST` | `/api/v1/utr/5/generate` | 生成 5UTR，并自动调用 5UTR predictor 评分 |
| `POST` | `/api/v1/utr/3/generate` | 生成 3UTR，并自动调用 3UTR predictor 评分 |
| `POST` | `/api/v1/utr/5/score` | 对输入 5UTR 序列评分 |
| `POST` | `/api/v1/utr/3/score` | 对输入 3UTR 序列评分 |

所有请求体均为 JSON。`seed` 为可选整数，用于控制采样随机性。

## 5. CDS 生成

### 5.1 开放实现

请求：

```bash
curl -X POST http://localhost:8000/api/v1/cds/open/generate \
  -H 'Content-Type: application/json' \
  -d '{"protein_sequence":"MVSKGEELFTGVVPILVE","seed":0}'
```

请求字段：

- `protein_sequence`：必填，氨基酸序列，只允许标准氨基酸字符。
- `seed`：可选，整数。传入后采样结果更便于复现。

响应示例：

```json
{
  "implementation": "open",
  "protein_sequence": "MVSKGEELFTGVVPILVE",
  "dna_sequence": "ATG...",
  "rna_sequence": "AUG...",
  "naturalness": 0.83,
  "sampling_seed": 99,
  "device": "cpu"
}
```

### 5.2 闭源扩展实现

请求：

```bash
curl -X POST http://localhost:8000/api/v1/cds/closed/generate \
  -H 'Content-Type: application/json' \
  -d '{"protein_sequence":"MVSKGEELFTGVVPILVE","seed":0}'
```

响应字段与开放实现相同，`implementation` 为 `closed`。该端点依赖 `src/shared/mod_xzr01.so`。

## 6. UTR 生成

### 6.1 生成 5UTR

请求：

```bash
curl -X POST http://localhost:8000/api/v1/utr/5/generate \
  -H 'Content-Type: application/json' \
  -d '{"length":"short","seed":0}'
```

### 6.2 生成 3UTR

请求：

```bash
curl -X POST http://localhost:8000/api/v1/utr/3/generate \
  -H 'Content-Type: application/json' \
  -d '{"length":"long","seed":0}'
```

请求字段：

- `length`：必填，只能是 `short`、`medium` 或 `long`。
- `seed`：可选，整数。

响应示例：

```json
{
  "utr_type": "5utr",
  "length": "short",
  "sequence": "ACGU...",
  "score": 7.25,
  "sampling_seed": 99,
  "device": "cpu"
}
```

生成端点会先生成 UTR 序列，再使用对应的 5UTR 或 3UTR predictor 自动评分。

## 7. UTR 评分

### 7.1 5UTR 评分

请求：

```bash
curl -X POST http://localhost:8000/api/v1/utr/5/score \
  -H 'Content-Type: application/json' \
  -d '{"sequence":"TACGTTTTGACCTTCGTTCATTTTG"}'
```

### 7.2 3UTR 评分

请求：

```bash
curl -X POST http://localhost:8000/api/v1/utr/3/score \
  -H 'Content-Type: application/json' \
  -d '{"sequence":"TGTCCCCGGGTCTTCCAACGGACTGGCGTTGCCCCGGTTCACTGGGGACTGCCCTTGGGGTCTCGCTCACCTTCAGCACACATTATCGGGAGCAGTGTCTTCCATAATGT"}'
```

请求字段：

- `sequence`：必填，UTR 序列。服务会转为大写，并将 `T` 归一化为 `U`；允许字符为 `A`、`C`、`G`、`U`、`N`。

响应示例：

```json
{
  "utr_type": "3utr",
  "sequence": "UGUCCCCGGG...",
  "score": 6.5,
  "device": "cpu"
}
```

## 8. Python 调用示例

```python
import requests

base_url = "http://localhost:8000"

response = requests.post(
    f"{base_url}/api/v1/cds/open/generate",
    json={
        "protein_sequence": "MVSKGEELFTGVVPILVE",
        "seed": 0,
    },
    timeout=300,
)
response.raise_for_status()
result = response.json()

print(result["rna_sequence"])
print(result["naturalness"])
```

模型推理可能耗时较长，客户端应设置足够长的超时时间。

## 9. 错误响应

请求体格式错误或字段类型不匹配时，FastAPI 会返回 `422 Unprocessable Entity`。

服务层校验失败时会返回 `400 Bad Request`，例如：

- CDS 输入为空。
- CDS 输入包含非标准氨基酸字符。
- UTR `length` 不是 `short`、`medium` 或 `long`。
- UTR 序列包含非法字符。

示例：

```json
{
  "detail": "UTR length must be short, medium, or long."
}
```

未预期的模型加载、checkpoint 缺失、共享库导入失败等问题通常会表现为 `500 Internal Server Error`，需要检查服务端日志。

## 10. 验证与排障

可先运行 API 合约测试确认路由和基本请求模型正常：

```bash
python -m unittest tests/test_api_contract.py -v
```

常见问题：

- `ModuleNotFoundError: fastapi`：环境未安装或未更新，执行 `conda env update -n gemorna -f environment.yaml`。
- checkpoint 找不到：确认从仓库根目录启动，且 `checkpoints/` 文件完整。
- vocabulary 找不到：确认 `vocab/prot_vocab.pkl` 和 `vocab/cds_vocab.pkl` 存在。
- `src/shared/*.so` 导入失败：确认运行平台与编译扩展兼容，或使用项目提供的 Docker 镜像环境。
- 首次请求很慢：模型会在首次请求时加载到 CPU/GPU，属于预期行为。

