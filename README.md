# 新闻结构化摘要微调系统

基于 **Qwen3 + LLaMA-Factory** 的参数高效微调（PEFT）流水线，面向消费级 GPU 的结构化新闻摘要生成。

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.8+](https://img.shields.io/badge/CUDA-12.8%2B-green.svg)](https://developer.nvidia.com/cuda-downloads)

---

## 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [系统要求](#系统要求)
- [快速开始](#快速开始)
- [数据构建流程](#数据构建流程)
- [模型训练](#模型训练)
- [推理与应用](#推理与应用)
- [评测方法](#评测方法)
- [实验结果](#实验结果)
- [项目结构](#项目结构)
- [技术细节](#技术细节)
- [常见问题](#常见问题)
- [参考文献](#参考文献)

---

## 项目概述

本项目实现了一个**端到端的新闻结构化摘要微调系统**，通过 QLoRA（Quantized Low-Rank Adaptation）技术在消费级 GPU 上高效训练大语言模型，使其能够将新闻文本转化为固定 6 字段格式的结构化摘要，适用于客户终端、边缘设备的信息流展示。

**核心技术栈**：
- **基座模型**：Qwen3-4B / Qwen3-8B
- **微调框架**：LLaMA-Factory（支持 QLoRA、4-bit NF4 量化）
- **数据来源**：XL-Sum 多语言新闻数据集
- **标注方案**：DeepSeek API

**输出格式示例**：
```
【一句话摘要】事件核心描述
【核心要点】1. ... 2. ... 3. ...
【事件类别】科技/财经/社会/国际/...
【主要主体】相关组织或个人
【时间信息】事件发生时间
【潜在影响】对行业、市场、社会的影响分析
```

---

## 核心特性

- **固定字段结构化输出**：固定格式输出，零后处理直接适配 UI 渲染
- **QLoRA 参数高效微调**：4-bit 量化 + LoRA 低秩适配，消费级显卡 16GB 显存可训练
- **异步数据标注流水线**：DeepSeek API 异步并发打标（5 并发 ~50 条/分钟），成本 <￥15/1000 条
- **多维度评测体系**：ROUGE-L、格式合规率、推理延迟三维评测
- **基座/微调对比工具**：内置并排对比模式，量化微调收益

---

## 系统要求

| 组件 | 规格要求 |
|------|---------|
| **GPU** | NVIDIA RTX 3090 / 4090 / 5060 Ti 及以上（≥16GB 显存） |
| **CUDA** | 12.8+ |
| **Python** | 3.11 |
| **磁盘空间** | ~30GB（模型 + 数据集 + checkpoints） |

---

## 快速开始

### 1. 环境配置

```bash
# 创建 conda 环境
conda create -n my_sft python=3.11 -y
conda activate my_sft

# 安装 PyTorch（以CUDA 12.8为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 验证 GPU 可用性
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

### 2. 安装 LLaMA-Factory

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e ".[torch,metrics]"
```

### 3. 安装项目依赖

```bash
pip install -U datasets pandas tqdm pydantic python-dotenv openai \
               rouge-score jieba bitsandbytes pyyaml jsonlines
```

### 4. 配置 API 凭据（以deepseek为例）

```bash
cp projects/edge_news_summarizer/.env.example projects/edge_news_summarizer/.env
# 编辑 .env 填入：
# OPENAI_API_KEY=sk-xxxx          # DeepSeek API Key
# OPENAI_API_BASE=https://api.deepseek.com  
# OPENAI_MODEL=deepseek-chat  
```

### 5. 下载基座模型

```python
from huggingface_hub import snapshot_download

# Qwen3-4B
snapshot_download("Qwen/Qwen3-4B", local_dir=r"D:\LLM\models\Qwen3-4B")

# Qwen3-8B
snapshot_download("Qwen/Qwen3-8B", local_dir=r"D:\LLM\models\Qwen3-8B")
```

---

## 数据构建流程

构建高质量训练数据分为 5 个步骤，每个步骤对应一个独立脚本。所有脚本可在项目任意目录执行，路径已自动处理。

### Step 1: 原始数据采集

从 XL-Sum 数据集（BBC 多语言新闻）采集中英文混合数据，过滤涉政内容。

```bash
python projects/edge_news_summarizer/scripts/01_collect_news.py \
  --source xlsum --lang mixed --max_samples 6000
```

**输出**：`data/raw/news_raw.jsonl` (过滤后约4,800 条记录)

### Step 2: 结构化标注

使用 DeepSeek API 异步生成六字段结构化标签。

```bash
python projects/edge_news_summarizer/scripts/02_generate_labels_api.py \
  --max_samples 0 --concurrency 5
```

**输出**：`data/labeled/news_labeled_v1.jsonl` 

### Step 3: 校验与清洗

校验字段完整性、去重、检查格式合规性。

```bash
python projects/edge_news_summarizer/scripts/03_validate_and_clean.py
```

**输出**：`data/cleaned/cleaned_all.jsonl`, `data/reports/data_quality_report.md`

### Step 4: 数据集划分

按 8:1:1 划分训练集/验证集/测试集。

```bash
python projects/edge_news_summarizer/scripts/04_split_dataset.py \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

**输出**：`train.json` (3,843) / `val.json` (480) / `test.json` (481) / `test_manual_eval.json` (100)

### Step 5: 注册到 LLaMA-Factory

将数据集注册到 LLaMA-Factory 的配置文件。

```bash
# 在 LlamaFactory 根目录执行
python projects/edge_news_summarizer/scripts/05_register_dataset_info.py
```

**输出**：数据集 `news_structured_summary` 已注册至 `LlamaFactory/data/dataset_info.json`

**数据流示意**：
```
XL-Sum (BBC) → 筛选+内容过滤 → DeepSeek API → 校验清洗 → 数据划分 → Alpaca 格式
  ~200k           ~4,800      ~4,800        ~4,804  3,843/480/481   train.json
```

---

## 模型训练

本项目采用 **QLoRA（Quantized Low-Rank Adaptation）** 微调方法，在 16GB 显存上高效训练 4B/8B 模型。技术细节见 [Technical Details](#技术细节) 章节。

### 1. 训练命令

在 **LlamaFactory 根目录**执行：

```bash
conda activate my_sft

# Qwen3-4B（推荐先验证，训练约 2.5 小时）
llamafactory-cli train projects/edge_news_summarizer/configs/train_qwen3_4b_qlora_news.yaml

# Qwen3-8B（生产级质量，需调整配置）
llamafactory-cli train projects/edge_news_summarizer/configs/train_qwen3_8b_qlora_news.yaml
```

**训练信息**（4B 模型，RTX 5060 Ti 16GB）：
- **总步数**：651 步（3 epochs × 217 steps/epoch）
- **训练时长**：~2.5 小时（~14 秒/步）
- **显存占用**：~6.6 GB（batch=1 + 梯度累积）
- **Checkpoints**：每 50 步保存一次，保留最近 5 个

### 2. 合并 LoRA 权重（可选）

训练完成后，可将 LoRA adapter 合并回基座模型用于独立部署：

```bash
llamafactory-cli export \
  --model_name_or_path D:/LLM/models/Qwen3-4B \
  --adapter_name_or_path projects/edge_news_summarizer/outputs/checkpoints/qwen3-4b-qlora-news \
  --template qwen3_nothink \
  --finetuning_type lora \
  --export_dir projects/edge_news_summarizer/outputs/merged/qwen3-4b-news
```



---

## 推理与应用

### 1. 交互式 CLI

```bash
# 仅微调模型
python projects/edge_news_summarizer/scripts/08_demo_cli.py \
  --model_path D:/LLM/models/Qwen3-4B \
  --adapter_path projects/edge_news_summarizer/outputs/checkpoints/qwen3-4b-qlora-news

# 对比模式（基座 vs 微调，推荐）
python projects/edge_news_summarizer/scripts/08_demo_cli.py \
  --model_path D:/LLM/models/Qwen3-4B \
  --adapter_path projects/edge_news_summarizer/outputs/checkpoints/qwen3-4b-qlora-news \
  --compare
```

### 2. 批量对比推理

从测试集提取样本，生成基座/微调并排对比结果：

```bash
# 交互式对比
python projects/edge_news_summarizer/scripts/08_demo_cli.py \
  --model_path D:/LLM/models/Qwen3-4B \
  --adapter_path projects/edge_news_summarizer/outputs/checkpoints/qwen3-4b-qlora-news \
  --compare

# 批量对比（从测试集取 50 条，保存对比结果）
python projects/edge_news_summarizer/scripts/08_demo_cli.py \
  --model_path D:/LLM/models/Qwen3-4B \
  --adapter_path projects/edge_news_summarizer/outputs/checkpoints/qwen3-4b-qlora-news \
  --compare \
  --input_file projects/edge_news_summarizer/data/cleaned/test.json \
  --output_file projects/edge_news_summarizer/outputs/eval/compare_outputs.jsonl \
  --num_samples 50
```

两个模型均使用**相同的系统提示词**，区别仅在于有无 LoRA adapter，因此对比结果可以纯粹反映微调带来的格式约束与摘要质量提升。

对比输出格式示例：
```
======================================================================
新闻标题：苹果发布 iPhone 16 系列
======================================================================
【基座模型（Base + 系统提示词）】  耗时: 2.341s
----------------------------------------------------------------------
（无固定格式，通常为自由文本）这是一款非常好的手机...
======================================================================
【微调模型（Fine-tuned + 系统提示词）】  耗时: 2.158s
----------------------------------------------------------------------
【一句话摘要】苹果发布 iPhone 16，全系搭载 A18 芯片并支持 Apple Intelligence。
【核心要点】1. ...  2. ...  3. ...
【事件类别】科技
...
======================================================================
```

对比结果 JSONL 中每条记录包含 `base_prediction` 和 `ft_prediction` 两个字段，可直接传入 `06_eval_rouge_and_format.py` 分别评测两组结果。

---

## 评测方法

本项目主要从**文本生成质量**和**推理性能**两个维度进行量化评估。

### 1. 文本质量评测 (ROUGE + 格式合规)

执行以下命令生成测试集预测结果并计算指标：

```bash
# 生成预测结果
llamafactory-cli eval projects/edge_news_summarizer/configs/infer_news.yaml

# 计算 ROUGE 分数与格式合规率
python projects/edge_news_summarizer/scripts/06_eval_rouge_and_format.py \
  --test projects/edge_news_summarizer/data/cleaned/test.json \
  --predictions projects/edge_news_summarizer/outputs/eval/generated_predictions.jsonl
```

**评估指标说明**：

| 维度 | 指标 | 目标值 | 说明 |
|------|------|--------|------|
| **内容重叠** | ROUGE-1 | > 0.40 | 单词级重叠 (jieba 分词) |
| | ROUGE-2 | > 0.20 | 双词组重叠 (Bigram) |
| | ROUGE-L | > 0.30 | 最长公共子序列 |
| **格式合规** | 字段完整率 | 100% | 6 个预定义字段全部存在 |
| | 类别合规率 | 100% | 事件类别在白名单内 |
| | 要点格式率 | > 95% | 核心要点包含编号列表 |

### 2. 推理延迟基准测试

在目标部署硬件上评估模型推理速度：

```bash
python projects/edge_news_summarizer/scripts/07_benchmark_latency.py \
  --model_path D:/LLM/models/Qwen3-4B \
  --adapter_path projects/edge_news_summarizer/outputs/checkpoints/qwen3-4b-qlora-news \
  --num_samples 20
```

---

## 实验结果

本节记录 Qwen3-4B QLoRA SFT 的完整量化评测结果，包含**自动评测指标**、**格式合规性**及**基座/微调对比**，遵循业界主流 SFT 评测规范（参考 RLHF Survey, InstructGPT, LLaMA2 等工作的评测方法）。

### 实验配置

| 项目 | 配置 |
|------|------|
| **基座模型** | Qwen3-4B |
| **微调方法** | QLoRA（4-bit NF4，LoRA rank=8，alpha=16）|
| **训练数据** | 3,843 条（DeepSeek API 标注，来源 XL-Sum）|
| **验证集** | 480 条 |
| **测试集** | 481 条 |
| **训练步数** | 651 steps（3 epochs）|
| **训练时长** | 3h 06m 45s（RTX 5060 Ti 16GB）|
| **评测框架** | LLaMA-Factory predict + rouge-score + jieba 分词 |

### 训练过程指标

| 指标 | 数值 |
|------|------|
| 最终 Train Loss | 0.948 |
| 最终 Eval Loss | 0.9494 |
| 困惑度（Perplexity）| exp(0.9494) ≈ **2.58** |
| 最优 Checkpoint | Step 400（eval_loss = 0.9299）|

> Eval Loss 在 Step 400 后轻微回升（0.9299 → 0.9494），训练-验证 gap 从 0.038 增至 0.145，Epoch 3 存在轻微过拟合信号，对最终指标影响可忽略。

### 评测结果（测试集 481 条）

#### 文本质量（ROUGE，jieba 分词）

| 指标 | 基座模型（Qwen3-4B） | 微调模型（SFT）| 绝对提升 | 相对提升 |
|------|---------------------|--------------|----------|--------|
| **ROUGE-1** | 0.255 | **0.769** | +0.514 | +202% |
| **ROUGE-2** | 0.099 | **0.532** | +0.433 | +439% |
| **ROUGE-L** | 0.209 | **0.737** | +0.528 | +252% |

#### 格式合规性

| 指标 | 基座模型 | 微调模型 | 提升 |
|------|---------|---------|------|
| **必需字段完整率** | 0.00% | **99.58%** | +99.58pp |
| **类别合规率** | 0.00% | **100.00%** | +100pp |
| **要点格式率**（≥3条编号）| 0.00% | **99.79%** | +99.79pp |
| **时间信息完整率** | 0.00% | **100.00%** | +100pp |
| **平均要点数** | 0 条 | **3.53 条** | +3.53 |
| **格式错误样本** | 481 / 481 | **2 / 481** | -99.6% |

### 关键观察

1. **格式遵循能力是核心收益**：基座模型对 481 条测试样本均未输出任何符合规范的结构化字段（格式合规率 0%），微调后提升至 99.58%，验证了 SFT 在指令遵循方面的核心价值
2. **ROUGE 提升显著**：ROUGE-L 从 0.209 提升至 0.737（+252%），ROUGE-2 相对提升最大（+439%），说明微调模型不仅格式对齐，内容与参考摘要的信息重叠度也大幅改善
3. **推理效率提升**：微调模型平均输出 ~351 字符/条（约 150-200 tokens），基座模型因无格式约束频繁触达 512 token 上限，实测推理速度提升约 **3.4×**（8.27s vs 28.37s per batch）
4. **Bad cases 极少**：微调后仅 2 条格式缺失（0.4%），需后续人工核查是否为极短新闻等边界输入

---

## 项目结构

```
projects/edge_news_summarizer/
├── configs/               # LLaMA-Factory 配置文件 (Train/Infer)
├── data/                  # 数据流水线目录
│   ├── raw/               # 原始采集数据
│   ├── labeled/           # 标注中间产物
│   ├── cleaned/           # 最终训练集 (Train/Val/Test)
│   └── reports/           # 数据质量报告
├── scripts/               # 自动化脚本 (01-08)
├── outputs/               # 训练产物 (Checkpoints, Logs, Evaluations)
├── docs/                  # 开发文档与指南
└── README.md              # 项目主文档
```

---

## 技术细节

### 微调参数配置

#### LoRA 与量化设置
| 参数 | 4B 配置 | 8B 配置 | 设计意图 |
|------|---------|---------|----------|
| `quantization_bit` | 4 | 4 | 使用 NF4 格式量化，4B 权重仅约 2.5GB，大幅降低显存门槛 |
| `lora_target` | all | all | 覆盖所有线性层 (Attention + MLP)，最大化适应能力 |
| `lora_rank` | 8 | 16 | 8B 模型参数空间更大，提高 Rank 以保证表达能力 |
| `lora_alpha` | 16 | 32 | 保持 `alpha/rank = 2` 的缩放比例 |

#### 训练动态与显存优化
| 参数 | 4B 配置 | 8B 配置 | 设计意图 |
|------|---------|---------|----------|
| `batch_size` | 1 | 1 | 配合梯度累积使用，将单步显存占用降至最低 |
| `grad_accum` | 16 | 16 | 模拟有效 Batch Size = 16，稳定梯度下降方向 |
| `cutoff_len` | 1024 | 512 | 8B 模型显存紧张，限制序列长度以防 OOM |
| `learning_rate` | 2e-4 | 1e-4 | 8B 模型训练稳定性要求更高，降低 LR |

### 梯度累积原理

为在 16GB 显存上模拟大 Batch 训练，采用了 `batch_size=1` + `gradient_accumulation_steps=16` 策略：

1. **前向/反向传播**：每次只计算 1 条数据，累加梯度但不更新权重。
2. **权重更新**：每累积 16 次后，进行一次 Optimizer Step 并清空梯度。
3. **效果**：数学上等价于 Batch Size = 16，但显存峰值仅为 Batch Size = 1 的水平。

### VRAM 占用分析 (Qwen3-4B)

| 显存占用项 | 预估大小 | 说明 |
|------------|----------|------|
| **Base Model (4-bit)** | ~2.5 GB | 冻结权重 |
| **LoRA Adapters** | ~0.1 GB | 可训练参数 (BF16) |
| **Gradients/Optimizer**| ~1.5 GB | 仅针对 LoRA 参数 |
| **Activation** | ~1.5 GB | 激活值 (Batch=1) |
| **KV Cache/Overhead** | ~1.0 GB | 框架开销 |
| **Total** | **~6.6 GB** | 安全运行于 8GB+ 显卡 |

> **注意**：若增加 Batch Size 至 2，激活值显存占用将翻倍，导致总占用超出 16GB 物理显存（溢出至共享内存），训练速度将下降 ~40%。

---

## 常见问题

Q: 为什么生成的摘要有时候不包含【核心要点】？
A: 可能是训练步数不足或 `cutoff_len` 截断了输出。建议检查 ROUGE 报告中的 format rate。

Q: 如何解决 Windows 下的编码错误？
A: 设置环境变量 `PYTHONUTF8=1`，项目中所有文件读写均已强制指定 `utf-8` 编码。

更多问题请参阅 [docs/troubleshooting.md](docs/troubleshooting.md)。

---

## 参考文献

1. [LLaMA-Factory: Unified Efficient Fine-Tuning of 100+ LLMs](https://github.com/hiyouga/LlamaFactory)
2. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
3. [XL-Sum: Large-Scale Multilingual Abstractive Summarization](https://aclanthology.org/2021.findings-acl.413/)

