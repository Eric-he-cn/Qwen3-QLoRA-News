# 项目调试记录

本文档记录 Qwen3-4B QLoRA 新闻摘要项目全流程调试过程中遇到的实际问题及解决方案，按问题类型分节整理，供后续维护和类似项目参考。

---

## 1. 推理性能问题

### 1.1 Thinking 模式推理速度极低（36~40 s/条）

**发现时机**：Group B（Base + thinking）启动后，前几条日志显示单条耗时 31~40 s，全集 481 条估算需约 5 小时。

**根因定位**：

在 `06_eval_rouge_and_format.py` 中存在如下硬编码逻辑：

```python
eff_batch = 1 if think else batch_size
```

该行强制 thinking 组退化为单条推理（batch=1），无论传入 `--batch_size` 为何值。

**解决方案**：

删除该条件分支，统一使用用户传入的 `batch_size`：

```python
eff_batch = batch_size  # 移除 thinking 强制 batch=1 限制
```

Group B 改用 `--batch_size 2` 重新启动，平均速度从 ~37 s/条 提升至 ~17.7 s/条，全集总耗时 ~2.3 小时。

**结论**：thinking 模式本身带来的 token 生成量增大是延迟上升的主因，但 batch 层面没有理由固定为 1。

---

### 1.2 Thinking 模式 batch 上限建议

在 RTX 5060 Ti 16GB 上实测：

| batch_size | 观测速度 | 显存（推理阶段） | 是否稳定 |
|------------|----------|------------------|----------|
| 1 | ~37 s/条 | ~10,500 MiB | 稳定 |
| 2 | ~17.7 s/条 | ~11,700 MiB | 稳定 |
| 4 | 未测试（thinking 模式） | 预计 >13,000 MiB | 未知 |

建议 thinking 模式从 `batch=2` 开始，观察显存后再调整。

---

## 2. 模型加载架构问题

### 2.1 Base 模型通过 PeftModel 路径加载（架构混淆）

**发现时机**：代码审查发现 Group A（Base）和 Group C（SFT）均通过 `PeftModel.from_pretrained` 加载，Base 组通过 `disable_adapter()` 关闭 adapter 来模拟"纯基座"行为。

**问题分析**：

- `disable_adapter()` 是 Peft 提供的上下文管理器，逻辑上禁用 adapter，但 Base 模型仍被 Peft 框架包裹，权重形态与直接加载略有差异。
- 在 `generate()` 调用路径、显存分配和 `past_key_values` 缓存等细节上，行为可能与真实基座模型不完全等价。
- 基座对比实验的前提是"仅有无 adapter 的差异"，使用 Peft 包裹会引入额外变量。

**解决方案**：

重构 `06_eval_rouge_and_format.py`，将 Base 与 SFT 分离为两个独立加载函数：

```python
def _load_base_model(model_path, device_map, torch_dtype):
    """直接加载基座权重，不经过 PEFT。"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device_map, torch_dtype=torch_dtype
    )
    return model

def _load_sft_model(model_path, adapter_path, merged_model_path, ...):
    """
    优先使用合并后完整模型（merged_model_path）。
    不可用时，加载 base + adapter 并在线合并（merge_and_unload）。
    """
    ...
```

Group A 始终走 `_load_base_model`，Group C 始终走 `_load_sft_model`，两者不再通过同一个 PeftModel 实例切换。

---

## 3. Benchmark 稳定性问题

### 3.1 Benchmark 进程意外终止，已跑结果丢失

**发现时机**：首次运行 Group B 时因脚本错误被迫终止，4 条已完成的推理结果全部丢失，需要从头重跑。

**解决方案**：为每个 Group 实现 JSONL 即时追加 + 启动时断点检测：

```python
# 启动时检查已有 checkpoint
checkpoint_path = group_dir / "infer_checkpoint.jsonl"
done_ids = set()
if checkpoint_path.exists():
    with open(checkpoint_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            done_ids.add(obj["id"])

# 推理后立刻写盘，不等批次结束
with open(checkpoint_path, "a", encoding="utf-8") as f:
    f.write(json.dumps({...}, ensure_ascii=False) + "\n")
```

效果：重启后自动跳过已完成样本，只继续未完成部分，无需重跑。

---

### 3.2 Group C Benchmark 中途中止（停在 184/481）

**发现时机**：Group C 首次运行时因脚本报错在 184 条时停止。

**处理过程**：

1. 利用上述断点续传机制尝试恢复。
2. 由于同期决定将 Group C 切换为使用"合并后的 merged 模型"（推理速度更快，架构更干净），最终选择清空 Group C checkpoint，使用 merged 模型重新完整运行，保证结果来源一致。

**最终配置**：

```bash
python -u projects/edge_news_summarizer/scripts/06_eval_rouge_and_format.py \
  --mode benchmark \
  --n_samples 0 \
  --batch_size 4 \
  --merged_model projects/edge_news_summarizer/outputs/merged/qwen3-4b-news-v2 \
  --only_groups C
```

---

## 4. 环境与工具链问题

### 4.1 llamafactory-cli 在 Windows 下路径解析失败

**现象**：执行 `conda run -n my_sft llamafactory-cli train ...` 偶发返回 exit code 1，报错信息指向找不到命令或路径错误，在不同终端会话中表现不一致。

**根因**：`conda run` 本质上是在子进程中执行，conda 的 PATH 注入有时未正确将 `Scripts/` 目录加入可见路径，导致 `llamafactory-cli` 可执行文件找不到。

**解决方案**：

方案 A（推荐）：在已激活目标环境的 shell 中直接调用：

```powershell
conda activate my_sft
llamafactory-cli train projects/.../train_qwen3_4b_qlora_news_v2.yaml
```

方案 B：使用完整绝对路径绕过 PATH 查找：

```powershell
C:\Users\<user>\miniconda3\envs\my_sft\Scripts\llamafactory-cli.exe train ...
```

**经验**：Windows 下脚本调试时，优先在已激活环境的 PowerShell 中执行命令，而非依赖 `conda run`。

---

### 4.2 XL-Sum / HuggingFace 数据集下载超时

**现象**：`01_collect_news.py` 第一次运行时 `datasets.load_dataset("GEM/xlsum", ...)` 连接超时。

**解决方案**：

```bash
# 设置镜像加速（国内网络）
$env:HF_ENDPOINT = "https://hf-mirror.com"
python projects/edge_news_summarizer/scripts/01_collect_news.py ...
```

或在代码开头添加：

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

---

### 4.3 Windows 控制台中文输出乱码

**现象**：训练脚本或评测脚本输出的中文内容在 PowerShell 中显示为方块或乱码。

**解决方案**：

```powershell
# 临时：设置 UTF-8 代码页
chcp 65001

# 永久：在 PowerShell profile 中添加
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Python 侧：设置环境变量
$env:PYTHONUTF8 = "1"
```

项目内所有文件读写均已强制指定 `encoding="utf-8"`，但终端显示需单独处理。

---

## 5. 评测结果一致性问题

### 5.1 三组速度来源不统一，对比口径混乱

**现象**：A 和 C 的速度数据来自历史固定基准运行，B 的速度来自本次补跑，直接放在同一对比表中缺乏说明。

**解决方案**：

在 `benchmark_summary.json` 中为每条记录增加 `speed_source` 字段：

```json
{
  "group": "A",
  "per_sample_s": 4.2,
  "speed_source": "历史固定基准(batch=4, no-think)"
},
{
  "group": "B",
  "per_sample_s": 17.7,
  "speed_source": "本次完整运行(batch=2, thinking)"
},
{
  "group": "C",
  "per_sample_s": 2.9,
  "speed_source": "历史固定基准(merged, batch=4, no-think)"
}
```

在 README 6.3.3 和 6.3.4 中补充口径说明段落，方便读者理解数字的来源语境。

**教训**：对比实验中，若不同组的数据来源于不同时间/条件，必须在记录时标注来源，而非只记录数值。

---

### 5.2 README 中理论 VRAM 估算与实测偏差

**现象**：README 第 7.3 节原先给出训练阶段分项估算总计 ~6.6 GB，同时断言"batch=2 会导致超出 16GB 物理显存"。但实际推理阶段（batch=4，no-think）观测显存约 11,000~12,000 MiB，且 thinking 模式 batch=2 也能稳定运行（约 11,700 MiB），两者均与原文结论矛盾。

**根因**：原文混淆了**训练阶段**（有梯度、优化器状态）和**推理阶段**（无梯度，只加载推理权重）的显存需求，且训练侧数字本身也未经实测验证。

**解决方案**：将 7.3 节改写为"推理阶段实测显存观测"表格，基于实际运行 `nvidia-smi` 观察区间，删除未经复核的理论分项；训练显存改为提示"以训练日志与实时监控为准"。

---

## 6. 代码工程问题

### 6.1 临时脚本与正式流水线脱节

**发现时机**：代码审查时发现 `scripts/` 目录中存在 9 个临时脚本，功能与官方 `01~08` 重叠但各自维护，互不协调：

| 临时脚本 | 功能 | 重叠的官方脚本 |
|----------|------|----------------|
| `infer_base_hf.py` | 基座模型 HF 原生推理 | `06_eval_rouge_and_format.py` |
| `infer_base_vllm.py` | 基座模型 vLLM 推理 | `06_eval_rouge_and_format.py` |
| `_check_quality.py` | 数据质量快照 | `03_validate_and_clean.py` |
| `_tmp_token_stats.py` | token 长度统计 | `04_split_dataset.py` |
| `update_instruction.py` | instruction 字段刷新 | `04_split_dataset.py` |
| `_tmp_base_test.py` / `_tmp_compare_sft_before_after.py` | 效果对比 | `08_demo_cli.py` |

**解决方案**：

- 将临时脚本中有价值的逻辑（质量快照、token 分析、instruction 刷新）合并进对应官方脚本并加参数控制开关。
- 删除其余临时脚本和备份文件（`.bak`）。
- 统一以 `01~08` 单入口维护，脚本数量从 15 个收敛至 8 个。

**经验**：在 SFT 实验项目中，快速验证类脚本应尽早合并或清理，避免形成中长期维护双轨问题。

---

### 6.2 Eval loss / checkpoint 指标记录不一致

**现象**：README 和配置文件中的训练指标（Train Loss、Eval Loss、训练时长、Best Checkpoint）在不同章节中有多处不一致的版本，导致阅读困惑。

**根因**：项目迭代了 v1 → v2 两个训练版本，文档更新滞后于实验进展，部分章节仍保留 v1 数据。

**解决方案**：以最终 v2 完整训练结果为准统一更新（Train Loss=0.9530，Eval Loss=0.9662，Step=651，时长~3.4h），同时在 README 中明确所有数字均来自 V2 训练。

---

## 7. 快速故障排查检查列表

遇到问题时按以下顺序排查：

```
□ 1. conda 环境是否已激活（conda activate my_sft）
□ 2. GPU 是否可见（python -c "import torch; print(torch.cuda.is_available())"）
□ 3. 中文输出是否乱码（chcp 65001 / PYTHONUTF8=1）
□ 4. 脚本路径是否基于项目根目录（D:\LLM\MySFT\LLM_SFT\）
□ 5. Benchmark checkpoint 是否被意外保留（检查 outputs/eval/group_*/infer_checkpoint.jsonl 条数）
□ 6. 显存是否足够（nvidia-smi 查看当前占用）
□ 7. LLaMA-Factory 数据集是否注册（dataset_info.json 是否有目标 key）
```

更多通用环境问题见 [troubleshooting.md](troubleshooting.md)。
