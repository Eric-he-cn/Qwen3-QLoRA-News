# 基于SFT的新闻结构化摘要助手（QLoRA微调Qwen3-4B）

基于 **Qwen3 + LLaMA-Factory** 的参数高效微调（PEFT）流水线，面向消费级 GPU 的结构化新闻摘要生成。

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.8+](https://img.shields.io/badge/CUDA-12.8%2B-green.svg)](https://developer.nvidia.com/cuda-downloads)

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 项目结构](#2-项目结构)
- [3. 快速开始](#3-快速开始)
- [4. 端到端流程](#4-端到端流程)
- [5. 评测方法](#5-评测方法)
- [6. 实验结果](#6-实验结果)
- [7. 技术细节](#7-技术细节)
- [8. 常见问题](#8-常见问题)
- [9. 参考文献](#9-参考文献)

---

## 1. 项目概述

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

### 1.1 核心特性

- **固定字段结构化输出**：固定格式输出，零后处理直接适配 UI 渲染
- **QLoRA 参数高效微调**：4-bit 量化 + LoRA 低秩适配，消费级显卡 16GB 显存可训练
- **异步数据标注流水线**：DeepSeek API 异步并发打标（5 并发 ~50 条/分钟），成本 <￥15/1000 条
- **多维度评测体系**：ROUGE-L、格式合规率、推理延迟三维评测
- **基座/微调对比工具**：内置并排对比模式，量化微调收益

---

### 1.2 系统要求

| 组件 | 规格要求 |
|------|---------|
| **GPU** | NVIDIA RTX 3090 / 4090 / 5060 Ti 及以上（≥16GB 显存） |
| **CUDA** | 12.8+ |
| **Python** | 3.11 |
| **磁盘空间** | ~30GB（模型 + 数据集 + checkpoints） |

---

## 2. 项目结构

```
Qwen3-QLoRA-News/
├── README.md                              # 项目主文档（含实验结果与使用指南）
├── requirements.txt                       # Python 依赖清单（精确版本号）
├── .gitignore                             # 排除大文件：outputs/、data/raw/ 等
│
└── projects/edge_news_summarizer/
    │
    ├── configs/                           # LLaMA-Factory YAML 配置文件
    │   ├── train_qwen3_4b_qlora_news.yaml # Qwen3-4B QLoRA 训练配置（已验证）
    │   ├── train_qwen3_8b_qlora_news.yaml # Qwen3-8B QLoRA 训练配置（备用）
    │   ├── infer_news.yaml                # 微调模型批量推理配置（batch=4）
    │   └── infer_news_base.yaml           # 基座模型批量推理配置（max_new_tokens=2048）
    │
    ├── data/                              # 数据流水线（大文件已 .gitignore）
    │   ├── raw/                           # 原始 XL-Sum 采集数据（未处理）
    │   ├── labeled/                       # DeepSeek API 标注中间产物
    │   ├── cleaned/                       # 最终可用数据集
    │   │   ├── train.json                 # 训练集（3,843 条）
    │   │   ├── val.json                   # 验证集（480 条）
    │   │   └── test.json                  # 测试集（481 条）
    │   ├── prompts/
    │   │   └── label_prompt_news_structured.txt  # DeepSeek 标注 prompt 模板
    │   └── reports/                       # 数据质量检查报告（JSON）
    │
    ├── scripts/                           # 自动化流水线脚本
    │   ├── 01_collect_news.py             # 从 XL-Sum 采集原始新闻数据
    │   ├── 02_generate_labels_api.py      # 调用 DeepSeek API 批量生成结构化标签
    │   ├── 03_validate_and_clean.py       # 格式校验、去重、质量快照统计（整合临时质量脚本）
    │   ├── 04_split_dataset.py            # 切分数据集 + instruction 统一刷新 + token 长度统计
    │   ├── 05_register_dataset_info.py    # 向 LLaMA-Factory dataset_info.json 注册数据集
    │   ├── 06_eval_rouge_and_format.py    # 在线 benchmark + 离线评测（Base/SFT 独立加载）
    │   ├── 07_benchmark_latency.py        # 单条推理延迟基准测试
    │   └── 08_demo_cli.py                 # 交互/批量/对比演示（支持 thinking 开关）
    │
    ├── outputs/                           # 训练与评测产物（已 .gitignore）
    │   ├── checkpoints/
    │   │   └── qwen3-4b-qlora-news-v2/    # LoRA adapter 权重（63 MB，合并前）
    │   ├── merged/
    │   │   └── qwen3-4b-news-v2/          # 合并后完整模型权重（~8 GB）
    │   ├── eval/                          # Benchmark 评测结果
    │   │   ├── group_A/                   # Base 模型推理结果（481 条）
    │   │   │   ├── predictions_raw.jsonl
    │   │   │   ├── rouge_report.json      # R-1=0.6644, R-2=0.4211
    │   │   │   ├── format_report.json     # 50 bad cases（类别越界为主）
    │   │   │   └── bad_cases.jsonl
    │   │   ├── group_B/                   # Base + thinking 推理结果（481 条）
    │   │   ├── group_C/                   # SFT V2 推理结果（481 条）
    │   │   │   ├── predictions_raw.jsonl
    │   │   │   ├── rouge_report.json      # R-1=0.7653, R-2=0.5232
    │   │   │   ├── format_report.json     # 0 bad cases，全部指标 100%
    │   │   │   └── bad_cases.jsonl
    │   │   ├── benchmark_summary.json     # A/B/C 汇总指标
    │   │   └── analysis_report.json       # 多组对比分析报告
    │   └── logs/                          # 训练日志（trainer_log.jsonl）
    │
    └── docs/                              # 开发文档
        ├── dev_plan.md                    # 项目开发计划与阶段拆解
        ├── labeling_guideline.md          # 标注规范：6 字段格式定义与示例
        └── troubleshooting.md             # 常见问题排查记录
```

---

## 3. 快速开始

### 3.1 环境配置

```bash
# 创建 conda 环境
conda create -n my_sft python=3.11 -y
conda activate my_sft

# 安装 PyTorch（以CUDA 12.8为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 验证 GPU 可用性
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

### 3.2 安装 LLaMA-Factory

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e ".[torch,metrics]"
```

### 3.3 安装项目依赖

```bash
pip install -U datasets pandas tqdm pydantic python-dotenv openai \
               rouge-score jieba bitsandbytes pyyaml jsonlines
```

### 3.4 配置 API 凭据（以deepseek为例）

```bash
cp projects/edge_news_summarizer/.env.example projects/edge_news_summarizer/.env
# 编辑 .env 填入：
# OPENAI_API_KEY=sk-xxxx          # DeepSeek API Key
# OPENAI_API_BASE=https://api.deepseek.com  
# OPENAI_MODEL=deepseek-chat  
```

### 3.5 下载基座模型

```python
from huggingface_hub import snapshot_download

# Qwen3-4B
snapshot_download("Qwen/Qwen3-4B", local_dir=r"D:\LLM\models\Qwen3-4B")

# Qwen3-8B
snapshot_download("Qwen/Qwen3-8B", local_dir=r"D:\LLM\models\Qwen3-8B")
```

---

## 4. 端到端流程

### 4.1 数据构建流程

构建高质量训练数据分为 5 个步骤，每个步骤对应一个独立脚本。所有脚本可在项目任意目录执行，路径已自动处理。

#### 4.1.1 Step 1: 原始数据采集

从 XL-Sum 数据集（BBC 多语言新闻）采集中英文混合数据，过滤涉政内容。

```bash
python projects/edge_news_summarizer/scripts/01_collect_news.py \
  --source xlsum --lang mixed --max_samples 6000
```

**输出**：`data/raw/news_raw.jsonl` (过滤后约4,800 条记录)

#### 4.1.2 Step 2: 结构化标注

使用 DeepSeek API 异步生成六字段结构化标签。

```bash
python projects/edge_news_summarizer/scripts/02_generate_labels_api.py \
  --max_samples 0 --concurrency 5
```

**输出**：`data/labeled/news_labeled_v1.jsonl` 

#### 4.1.3 Step 3: 校验与清洗

校验字段完整性、去重、检查格式合规性。

```bash
python projects/edge_news_summarizer/scripts/03_validate_and_clean.py

# 可选：输出质量快照与随机样本预览
python projects/edge_news_summarizer/scripts/03_validate_and_clean.py \
  --sample_preview_count 5 \
  --quality_snapshot_path projects/edge_news_summarizer/data/reports/quality_snapshot.json
```

**输出**：`data/cleaned/cleaned_all.jsonl`, `data/reports/data_quality_report.md`, `data/reports/quality_snapshot.json`

#### 4.1.4 Step 4: 数据集划分

按 8:1:1 划分训练集/验证集/测试集。

```bash
python projects/edge_news_summarizer/scripts/04_split_dataset.py \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

# 可选：统一 instruction + 统计 token 长度分布
python projects/edge_news_summarizer/scripts/04_split_dataset.py \
  --refresh_instruction \
  --analyze_tokens \
  --tokenizer_path D:/LLM/models/Qwen3-4B
```

**输出**：`train.json` (3,843) / `val.json` (480) / `test.json` (481) / `test_manual_eval.json` (100) / `data/reports/token_length_report.json`（可选）

#### 4.1.5 Step 5: 注册到 LLaMA-Factory

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

### 4.2 模型训练

本项目采用 **QLoRA（Quantized Low-Rank Adaptation）** 微调方法，在 16GB 显存上高效训练 4B/8B 模型。技术细节见 [Technical Details](#技术细节) 章节。

#### 4.2.1 训练命令

在 **LlamaFactory 根目录**执行：

```bash
conda activate my_sft

# Qwen3-4B（推荐先验证，训练约 3~4 小时）
llamafactory-cli train projects/edge_news_summarizer/configs/train_qwen3_4b_qlora_news.yaml

# Qwen3-8B（生产级质量，需调整配置）
llamafactory-cli train projects/edge_news_summarizer/configs/train_qwen3_8b_qlora_news.yaml
```

**训练信息**（4B 模型，RTX 5060 Ti 16GB）：
- **总步数**：651 步（3 epochs × 217 steps/epoch）
- **训练时长**：~3.4 小时（包含评估与保存开销）
- **显存占用**：建议以训练时 `nvidia-smi` 实测为准（受 `cutoff_len`、`grad_accum` 等参数影响）
- **Checkpoints**：每 50 步保存一次，保留最近 5 个

#### 4.2.2 合并 LoRA 权重（可选）

训练完成后，可将 LoRA adapter 合并回基座模型用于独立部署：

```bash
llamafactory-cli export \
  --model_name_or_path D:/LLM/models/Qwen3-4B \
  --adapter_name_or_path projects/edge_news_summarizer/outputs/checkpoints/qwen3-4b-qlora-news-v2 \
  --template qwen3_nothink \
  --finetuning_type lora \
  --export_dir projects/edge_news_summarizer/outputs/merged/qwen3-4b-news-v2
```



### 4.3 推理与应用

#### 4.3.1 交互式 CLI

```bash
# 仅微调模型
python projects/edge_news_summarizer/scripts/08_demo_cli.py \
  --model_path D:/LLM/models/Qwen3-4B \
  --adapter_path projects/edge_news_summarizer/outputs/checkpoints/qwen3-4b-qlora-news-v2

# 对比模式（基座 vs 微调，推荐）
python projects/edge_news_summarizer/scripts/08_demo_cli.py \
  --model_path D:/LLM/models/Qwen3-4B \
  --adapter_path projects/edge_news_summarizer/outputs/checkpoints/qwen3-4b-qlora-news-v2 \
  --compare
```

#### 4.3.2 批量对比推理

从测试集提取样本，生成基座/微调并排对比结果：

```bash
# 交互式对比
python projects/edge_news_summarizer/scripts/08_demo_cli.py \
  --model_path D:/LLM/models/Qwen3-4B \
  --adapter_path projects/edge_news_summarizer/outputs/checkpoints/qwen3-4b-qlora-news-v2 \
  --compare

# 批量对比（从测试集取 50 条，保存对比结果）
python projects/edge_news_summarizer/scripts/08_demo_cli.py \
  --model_path D:/LLM/models/Qwen3-4B \
  --adapter_path projects/edge_news_summarizer/outputs/checkpoints/qwen3-4b-qlora-news-v2 \
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

## 5. 评测方法

本项目主要从**文本生成质量**和**推理性能**两个维度进行量化评估。

### 5.1 在线 Benchmark 对比评测（推荐）

`06_eval_rouge_and_format.py` 支持 `benchmark` 模式，在同一次运行中对比 Base 与 SFT 模型，断点续传、自动写入结果：

```bash
# 仅跑 Group A（Base）和 Group C（SFT V2），跳过 thinking 模式
python -u projects/edge_news_summarizer/scripts/06_eval_rouge_and_format.py \
  --mode benchmark \
  --n_samples 0 \
  --batch_size 4 \
  --skip_think

# 单独跑 Group B（Base + thinking），建议 batch_size=2
python -u projects/edge_news_summarizer/scripts/06_eval_rouge_and_format.py \
  --mode benchmark \
  --n_samples 0 \
  --batch_size 2 \
  --only_groups B

# 单独用合并后的 SFT 模型跑 Group C（推荐）
python -u projects/edge_news_summarizer/scripts/06_eval_rouge_and_format.py \
  --mode benchmark \
  --n_samples 0 \
  --batch_size 4 \
  --merged_model projects/edge_news_summarizer/outputs/merged/qwen3-4b-news-v2 \
  --only_groups C
```

当前 `benchmark` 已改为 **Base/SFT 独立加载**：Base 组直接使用基座权重，不再通过 PeftModel 包装后禁用 adapter。输出目录 `outputs/eval/group_{A,B,C}/` 中每组独立保存 JSONL + ROUGE + 格式报告，`benchmark_summary.json` 汇总关键指标。

### 5.2 离线评测模式（已有预测文件时）

```bash
# 计算 ROUGE 分数与格式合规率（已有 predictions JSONL 时使用）
python projects/edge_news_summarizer/scripts/06_eval_rouge_and_format.py \
  --test projects/edge_news_summarizer/data/cleaned/test.json \
  --predictions projects/edge_news_summarizer/outputs/eval/group_C/predictions_raw.jsonl
```

**评估指标说明**：

| 维度 | 指标 | SFT V2 实测 | 说明 |
|------|------|------------|------|
| **内容重叠** | ROUGE-1 | 0.7653 | 单词级重叠（jieba 分词） |
| | ROUGE-2 | 0.5232 | 双词组重叠（Bigram） |
| | ROUGE-L | 0.7347 | 最长公共子序列 |
| **格式合规** | 字段完整率 | 100% | 6 个预定义字段全部存在 |
| | 类别合规率 | 100% | 事件类别在白名单内（支持组合类别） |
| | 要点格式率 | 100% | 核心要点包含 ≥3 条编号列表 |

### 5.3 推理延迟基准测试

在目标部署硬件上评估单条推理速度（P50/P95 分位延迟）：

```bash
python projects/edge_news_summarizer/scripts/07_benchmark_latency.py \
  --model_path projects/edge_news_summarizer/outputs/merged/qwen3-4b-news-v2 \
  --num_samples 20
```


---

## 6. 实验结果

本节汇总 Qwen3-4B QLoRA SFT V2 的量化评测结果，覆盖文本质量、格式合规性与推理效率三类指标。

### 6.1 实验配置

| 项目 | 配置 |
|------|------|
| **基座模型** | Qwen3-4B |
| **微调方法** | QLoRA（4-bit NF4，LoRA rank=8，alpha=16）|
| **训练数据** | 3,843 条（DeepSeek API 标注，来源 XL-Sum）|
| **验证集** | 480 条 |
| **测试集** | 481 条 |
| **训练步数** | 651 steps（3 epochs）|
| **训练时长** | ~3.4 小时|
| **评测框架** | HuggingFace generate + rouge-score + jieba 分词（在线推理）|
| **推理配置** | batch=4，BF16，thinking=False，temperature=0.1 |
| **SFT 推理方式** | LoRA 合并后完整模型|

### 6.2 训练过程指标

| 指标 | 数值 |
|------|------|
| 最终 Train Loss | 0.9530 |
| 最终 Eval Loss | 0.9662 |
| 困惑度（Perplexity）| exp(0.9662) ≈ **2.63** |
| 最终 Checkpoint | Step 651（3 epochs 完整训练）|

> Step 651 为最终保存点。训练-验证 loss gap 约为 0.013，模型收敛状态良好，无明显过拟合信号。

### 6.3 评测结果（测试集 481 条）

#### 6.3.1 文本质量（ROUGE，jieba 分词）

> **实验条件**：A/B/C 三组均使用相同系统提示词。A=Base 不思考，B=Base 思考，C=SFT V2（不思考）。

| 指标 | Group A（Base 不思考） | Group B（Base 思考） | Group C（SFT V2 不思考） | 最优 |
|------|------------------------|----------------------|---------------------------|------|
| **ROUGE-1** | 0.6644 | 0.6895 | **0.7653** | C |
| **ROUGE-2** | 0.4211 | 0.4117 | **0.5232** | C |
| **ROUGE-L** | 0.6348 | 0.6446 | **0.7347** | C |

> B 相比 A：ROUGE-1 +0.0251，ROUGE-2 -0.0094，ROUGE-L +0.0098。该任务下，thinking 对重叠质量未表现出稳定增益。

#### 6.3.2 格式合规性

| 指标 | Group A（Base 不思考） | Group B（Base 思考） | Group C（SFT V2 不思考） | 最优 |
|------|------------------------|----------------------|---------------------------|------|
| **必需字段完整率** | 97.1% | **100.0%** | **100.0%** | B/C |
| **类别合规率** | 89.6% | 88.8% | **100.0%** | C |
| **要点格式率**（≥3 条编号） | 92.1% | 45.3% | **100.0%** | C |
| **时间信息完整率** | 97.1% | **100.0%** | **100.0%** | B/C |
| **平均要点条数** | 2.81 | 1.90 | **3.27** | C |
| **格式错误样本** | 50 / 481 | 54 / 481 | **0 / 481** | C |

> B 组在「要点格式率」和「类别合规率」上劣于 A 组，说明 Base 开启 thinking 后并未自动改善结构化输出稳定性。

#### 6.3.3 推理效率（RTX 5060 Ti 16GB）

| 组别 | 配置 | 平均速度 | 相对 A 组 | 说明 |
|------|------|----------|-----------|------|
| **Group A** | Base 不思考，batch=4 | 4.2 s/条 | 1.00x | 历史固定基准 |
| **Group B** | Base 思考，batch=2 | 17.7 s/条 | 4.21x 慢 | 本次完整运行 |
| **Group C** | SFT V2（merged）不思考，batch=4 | **2.9 s/条** | 0.69x（更快） | 历史固定基准 |

> 结论：thinking 模式带来显著时延开销（B 约为 A 的 4.21 倍），但并未在格式/ROUGE 上形成稳定收益；C 在质量与速度上保持最优平衡。

#### 6.3.4 结果口径说明

- A、C 速度来自既有完整基准（batch=4，no-think）。
- B 速度来自本次补跑完整任务（batch=2，thinking）。
- 三组样本规模一致，均为测试集 481 条。
- 详细字段见 `projects/edge_news_summarizer/outputs/eval/benchmark_summary.json`。

### 6.4 关键观察

1. **C 组质量优势稳定**：C 在 ROUGE-1/2/L 三项均为最优，较 A 分别提升 +0.1009/+0.1021/+0.0999。

2. **B 组未形成整体质量优势**：B 虽在 ROUGE-1/L 略高于 A，但 ROUGE-2 下降，且格式层面（类别合规、要点编号）明显劣化。

3. **thinking 延迟成本显著**：B 平均 17.7 s/条，较 A 慢 4.21 倍；在本任务场景下，时延成本高于收益。

4. **结构化输出最佳方案仍是 SFT V2 no-think**：C 同时实现 0 bad cases、100% 格式指标与更低时延（2.9 s/条），是当前部署优选。

---

## 7. 技术细节

### 7.1 微调参数配置

#### 7.1.1 LoRA 与量化设置
| 参数 | 4B 配置 | 8B 配置 | 设计意图 |
|------|---------|---------|----------|
| `quantization_bit` | 4 | 4 | 使用 NF4 格式量化，4B 权重仅约 2.5GB，大幅降低显存门槛 |
| `lora_target` | all | all | 覆盖所有线性层 (Attention + MLP)，最大化适应能力 |
| `lora_rank` | 8 | 16 | 8B 模型参数空间更大，提高 Rank 以保证表达能力 |
| `lora_alpha` | 16 | 32 | 保持 `alpha/rank = 2` 的缩放比例 |

#### 7.1.2 训练动态与显存优化
| 参数 | 4B 配置 | 8B 配置 | 设计意图 |
|------|---------|---------|----------|
| `batch_size` | 1 | 1 | 配合梯度累积使用，将单步显存占用降至最低 |
| `grad_accum` | 16 | 16 | 模拟有效 Batch Size = 16，稳定梯度下降方向 |
| `cutoff_len` | 1024 | 512 | 8B 模型显存紧张，限制序列长度以防 OOM |
| `learning_rate` | 2e-4 | 1e-4 | 8B 模型训练稳定性要求更高，降低 LR |

### 7.2 梯度累积原理

为在 16GB 显存上模拟大 Batch 训练，采用了 `batch_size=1` + `gradient_accumulation_steps=16` 策略：

1. **前向/反向传播**：每次只计算 1 条数据，累加梯度但不更新权重。
2. **权重更新**：每累积 16 次后，进行一次 Optimizer Step 并清空梯度。
3. **效果**：数学上等价于 Batch Size = 16，但显存峰值仅为 Batch Size = 1 的水平。

### 7.3 显存观测（推理阶段，Qwen3-4B）

下表为在 RTX 5060 Ti 16GB 上执行 A/B/C 基准时的 `nvidia-smi` 观察区间（非理论估算）：

| 组别 | 运行配置 | 观测显存占用（MiB） | 结论 |
|------|----------|---------------------|------|
| **Group A** | Base，不思考，batch=4 | 约 11,000 ~ 12,000 | 可稳定运行 |
| **Group B** | Base，thinking，batch=2 | 约 11,700 ~ 12,300 | 可稳定运行 |
| **Group C** | SFT merged，不思考，batch=4 | 约 10,800 ~ 11,800 | 可稳定运行 |

> 备注：当前文档仅保留实测推理显存结论，不再给出未经复核的训练阶段分项估算。训练显存请以训练日志与实时监控为准。

---

## 8. 常见问题

Q: 为什么生成的摘要有时候不包含【核心要点】？
A: 可能是训练步数不足或 `cutoff_len` 截断了输出。建议检查 ROUGE 报告中的 format rate。

Q: 如何解决 Windows 下的编码错误？
A: 设置环境变量 `PYTHONUTF8=1`，项目中所有文件读写均已强制指定 `utf-8` 编码。

更多问题请参阅 [docs/troubleshooting.md](docs/troubleshooting.md)。

---

## 9. 参考文献

1. [LLaMA-Factory: Unified Efficient Fine-Tuning of 100+ LLMs](https://github.com/hiyouga/LlamaFactory)
2. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
3. [XL-Sum: Large-Scale Multilingual Abstractive Summarization](https://aclanthology.org/2021.findings-acl.413/)

