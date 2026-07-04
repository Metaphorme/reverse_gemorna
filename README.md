# GEMORNA CDS 闭源模块逆向分析与开源接口说明

## 目的

本文说明 `src/shared/mod_xzr01.so` 中 CDS 生成逻辑的逆向分析流程、关键证据、开源复现逻辑，以及如何验证开源实现与闭源实现输出一致。

本次分析只覆盖 CDS 生成路径，即 `src/generate.py --mode cds` 调用的 `CDS.gen(...)`。UTR 生成和 UTR 预测流程不在本文范围内。

## 涉及文件

- `src/shared/mod_xzr01.so`：Linux 下的闭源 Cython 运行库。
- `src/generate.py`：原始闭源 CDS/UTR 命令行入口。
- `src/models/gemorna_cds.py`：开源 Encoder/Decoder 模型定义。
- `src/utils/utils_cds.py`：开源 `CDS_` 基类和 CDS 采样逻辑。
- `src/open_cds_generator.py`：开源 CDS 调用接口和 CLI。
- `src/test_open_closed_cds_equivalence.py`：开源/闭源等价性验证脚本。
- `ghidra_output/mod_xzr01.asm.txt`：Ghidra 反汇编输出。

## 逆向分析流程

### 1. 确认二进制性质

运行：

```bash
file src/shared/mod_xzr01.so
nm -D src/shared/mod_xzr01.so
strings -a src/shared/mod_xzr01.so
```

结论：

- `mod_xzr01.so` 是 x86-64 ELF shared object。
- 动态符号只暴露 Python 扩展模块入口 `PyInit_mod_xzr01` 等 Cython 运行时符号。
- 字符串表保留了大量 Cython/Python 层名称，例如 `CDS.gen`、`CDS.sampling`、`trunc`、`utils.utils_cds`、`codon_dict`、`codon_freq`、`manual_seed`、`multinomial`、`sharpened_probs`、`Generated CDS & Naturalness`。

### 2. 静态反汇编定位

使用 Ghidra 反汇编 `mod_xzr01.so`，在 `ghidra_output/mod_xzr01.asm.txt` 中定位到：

- `mod_xzr01.CDS.gen`
- `mod_xzr01.CDS.sampling`
- `mod_xzr01.trunc`
- `src/shared/mod_xzr01.pyx`

这些符号和字符串说明 `.so` 是 Cython 3.1.2 编译产物，且内部保留了 Python 源级变量名。反汇编中也能看到模块从 `utils.utils_cds`、`config`、`tokenization` 等开源模块导入对象。

### 3. Python 反射确认类关系

在 `gemorna` 环境下运行反射脚本后得到：

```text
CDS bases: (<class 'utils.utils_cds.CDS_'>,)
CDS mro: (<class 'mod_xzr01.CDS'>, <class 'utils.utils_cds.CDS_'>, ...)
is subclass CDS_: True
```

并且：

- `CDS.__init__` 来自 `utils.utils_cds.CDS_`
- `CDS.forward` 来自 `utils.utils_cds.CDS_`
- `CDS.make_prot_mask` 来自 `utils.utils_cds.CDS_`
- `CDS.make_cds_mask` 来自 `utils.utils_cds.CDS_`
- 闭源 Cython 只覆盖 `CDS.gen` 和 `CDS.sampling`

这说明闭源 CDS 类不是独立模型实现，而是继承开源 `CDS_` 容器，只把生成和采样方法编译进 `.so`。

### 4. 运行时行为对比

验证脚本使用同一 checkpoint、同一词汇表、同一 Encoder/Decoder 结构，分别实例化：

- 闭源 `shared.mod_xzr01.CDS`
- 开源 `utils.utils_cds.CDS_`

然后比较：

- checkpoint key 集合
- 参数量
- `forward()` 输出
- `gen()` 最终序列和 naturalness
- 每一步 decoder `fc_out` 的完整 logits

当前验证结果：

```bash
conda run -n gemorna python src/test_open_closed_cds_equivalence.py --seeds 0,1,199
```

输出：

```text
PASS: 27 open/closed CDS generation comparisons matched
device=cpu
```

测试覆盖短序列、终止氨基酸、论文示例序列、全部标准氨基酸、随机序列、170 分块边界、171 跨块、185/186 位置边界。每个组合都比较最终 DNA、naturalness 文本、RNA 转换、函数 API 与类 API、完整 logits。

## CDS 生成运行逻辑

开源实现位于 `src/open_cds_generator.py` 和 `src/utils/utils_cds.py`。

### 入口

命令行：

```bash
conda run -n gemorna python src/open_cds_generator.py \
  --mode cds \
  --ckpt_path checkpoints/gemorna_cds.pt \
  --protein_seq MVSKGEELFTGVVPILVE \
  --seed 0 \
  --output both
```

函数接口：

```python
from open_cds_generator import translate_protein_to_rna

result = translate_protein_to_rna(
    "MVSKGEELFTGVVPILVE",
    ckpt_path="checkpoints/gemorna_cds.pt",
    seed=0,
)

print(result.rna_sequence)
print(result.naturalness)
```

可复用类接口：

```python
from open_cds_generator import OpenCDSGenerator

generator = OpenCDSGenerator("checkpoints/gemorna_cds.pt")
result = generator.translate_protein_to_rna("MVSKGEELFTGVVPILVE", seed=0)
```

返回对象 `CDSGenerationResult` 包含：

- `protein_sequence`
- `dna_sequence`
- `rna_sequence`
- `naturalness`
- `sampling_seed`
- `device`

### 模型加载

1. 读取 `vocab/prot_vocab.pkl` 和 `vocab/cds_vocab.pkl`。
2. 使用 `GEMORNA_CDS_Config` 构建 Encoder/Decoder。
3. 加载 `checkpoints/gemorna_cds.pt`。
4. 设置 `model.eval()`。

`src/models/gemorna_cds.py` 现在显式声明 `max_length = 187`，不再依赖闭源 `.so` 通过星号导入提供该常量。导入开源入口时不会加载 `shared.mod_xzr01`。

### 输入处理

1. 检查蛋白质序列是否为空参数。
2. 检查是否含非标准氨基酸字符，允许 `ACDEFGHIKLMNPQRSTVWY*`。
3. 将蛋白质序列按 170 个氨基酸分块。
4. 每个分块转小写，再构造：

```python
[init_token] + tokenize_aa(seq) + [eos_token]
```

### 采样流程

每个分块执行同一 `sampling_seed`：

1. `torch.manual_seed(SEED)` 固定 PyTorch 采样。
2. 初始 decoder 输入为 CDS `<sos>`。
3. 对每个源氨基酸位置：
   - 构造因果 CDS mask。
   - 调用 decoder 得到当前所有位置 logits。
   - 取最后位置 `logits_last`。
   - `softmax(logits_last)` 得到 `normalized_probs`。
   - `normalized_probs.pow(2.3)` 得到 sharpened distribution。
   - 对 sharpened distribution 调用 `torch.multinomial(..., 1)`。
   - 将采样 token 转回密码子。
   - 如果该密码子不属于当前氨基酸的合法密码子集合，则回退到 `codon_freq[current_source_token][0]`。
   - 用未 sharpen 的 `normalized_probs[pred_token]` 累加 log probability。
   - 把最终 token 拼接回 decoder 输入，进入下一步。
4. 遇到源 token `<eos>` 时停止。

### 输出与评分

所有分块生成的 DNA codon 拼接并转大写：

```python
dna_sequence = "".join(generated_seqs).upper()
rna_sequence = dna_sequence.replace("T", "U")
naturalness = math.exp(final_modelscore / len(generated_seqs))
```

因此 naturalness 不是原始平均对数概率，而是平均 token 概率的几何平均形式。CLI 默认输出 RNA，也可通过 `--output dna` 或 `--output both` 输出 DNA。

## 与闭源实现的一致性证据

当前证据链包括：

1. `.so` 字符串和 Ghidra 反汇编中的函数名、变量名、源文件名与开源逻辑一致。
2. Python 反射证明闭源 `CDS` 继承开源 `CDS_`，只覆盖 `gen` 和 `sampling`。
3. checkpoint key 集合、参数量和 `forward()` 输出完全一致。
4. 在受控随机种子下，闭源和开源生成的 DNA 序列、naturalness 文本、decoder 调用次数、每步完整 logits 都一致。
5. 开源入口导入时不加载 `shared.mod_xzr01`，可以作为不依赖闭源运行库的 CDS/RNA 生成接口。

## 边界说明

由于 `.so` 是闭源二进制，无法在形式化意义上证明所有潜在未触发分支都与开源实现完全等价。本文结论基于静态符号证据、Python 反射证据和覆盖边界输入的运行时对比。对正常 CDS 生成路径，当前证据支持“开源实现与闭源实现逐步等价”。
