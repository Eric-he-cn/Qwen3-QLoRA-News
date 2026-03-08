#!/usr/bin/env python3
"""
06_eval_rouge_and_format.py

本脚本负责两类评测任务：
1) 离线评测（eval）：对已有预测文件计算 ROUGE 与格式合规率。
2) 在线基准（benchmark）：直接加载模型推理并完成 A/B/C 三组对比。

A/B/C 组定义（默认）：
- A：Base 不思考（enable_thinking=False）
- B：Base 思考（enable_thinking=True）
- C：SFT V2 不思考（优先使用合并模型；若未提供合并模型则自动 merge LoRA）

关键约束：
- Base 组始终使用“纯基座权重”直接推理，不通过 PeftModel 包装后再禁用适配器。
- 断点续跑：每条样本立即写入 checkpoint（JSONL），中断后可自动续跑。
- Thinking 组允许 batch>1（建议不超过 2），以便在显存允许时提升吞吐。
"""

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
EVAL_DIR = PROJECT_DIR / "outputs" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TEST = PROJECT_DIR / "data" / "cleaned" / "test.json"
DEFAULT_PREDICTIONS = Path(
    "D:/LLM/LlamaFactory/projects/edge_news_summarizer/outputs/eval/generated_predictions.jsonl"
)
DEFAULT_BASE_MODEL = "D:/LLM/models/Qwen3-4B"
DEFAULT_ADAPTER_PATH = str(PROJECT_DIR / "outputs" / "checkpoints" / "qwen3-4b-qlora-news-v2")

REQUIRED_SECTIONS = [
    "【一句话摘要】",
    "【核心要点】",
    "【事件类别】",
    "【主要主体】",
    "【时间信息】",
    "【潜在影响】",
]

VALID_CATEGORIES = {
    "政治", "经济", "科技", "文化", "社会", "军事", "体育", "健康", "环境", "国际", "历史", "旅游", "财经",
    "technology", "finance", "politics", "society", "sports", "culture",
    "international", "military", "environment", "health",
}

MEDIUM_SYSTEM_PROMPT = (
    "你是专业的新闻编辑助手。请对新闻内容进行结构化摘要，严格按以下6个标签顺序输出，禁止使用 Markdown：\n"
    "【一句话摘要】【核心要点】【事件类别】【主要主体】【时间信息】【潜在影响】\n\n"
    "其中【核心要点】用阿拉伯数字编号列出至少3条；\n"
    "【事件类别】只能从以下选择：政治、经济、科技、文化、社会、军事、体育、健康、环境、国际、历史、旅游、财经"
)

MAX_TOKENS = {"A": 800, "B": 3072, "C": 800}


def load_test_data(path: Path) -> list[dict]:
    """加载测试数据，统一返回 list[dict]。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def strip_think_block(text: str) -> str:
    """移除 <think>...</think> 思维链块并返回清洗后的正文。"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def load_predictions(path: Path, strip_think: bool = False) -> list[str]:
    """加载预测文件（JSONL 或纯文本），可选剥离 think 块。"""
    predictions: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pred = obj.get("predict", obj.get("generated_text", obj.get("output", line)))
            except json.JSONDecodeError:
                pred = line
            predictions.append(strip_think_block(pred) if strip_think else pred)
    return predictions


def compute_rouge(references: list[str], predictions: list[str], use_jieba: bool = True) -> dict:
    """计算 ROUGE-1/2/L（中文优先 jieba 分词，失败则回退字符级）。"""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("[ERROR] 请先安装 rouge-score: pip install rouge-score", file=sys.stderr)
        sys.exit(1)

    metrics = ["rouge1", "rouge2", "rougeL"]

    if use_jieba:
        try:
            import jieba

            scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=False, tokenizer=None)
            scores = {m: [] for m in metrics}
            for ref, pred in zip(references, predictions):
                ref_tok = " ".join(jieba.cut(ref))
                pred_tok = " ".join(jieba.cut(pred))
                result = scorer.score(ref_tok, pred_tok)
                for m in metrics:
                    scores[m].append(result[m].fmeasure)
            return {k: sum(v) / len(v) if v else 0.0 for k, v in scores.items()}
        except ImportError:
            print("[WARN] jieba 未安装，回退到字符级 ROUGE")

    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=False)
    scores = {m: [] for m in metrics}
    for ref, pred in zip(references, predictions):
        result = scorer.score(" ".join(list(ref)), " ".join(list(pred)))
        for m in metrics:
            scores[m].append(result[m].fmeasure)
    return {k: sum(v) / len(v) if v else 0.0 for k, v in scores.items()}


def check_format(text: str) -> dict:
    """检查单条输出的结构字段、类别、要点数量与时间字段。"""
    result: dict[str, object] = {}

    for section in REQUIRED_SECTIONS:
        result[f"has_{section}"] = section in text
    result["all_sections_present"] = all(result[f"has_{section}"] for section in REQUIRED_SECTIONS)

    cat_match = re.search(r"【事件类别】\s*\n?\s*([^\n【]+)", text)
    if cat_match:
        category = cat_match.group(1).strip().lower()
        result["valid_category"] = any(v.lower() in category or category in v.lower() for v in VALID_CATEGORIES)
        result["extracted_category"] = cat_match.group(1).strip()
    else:
        result["valid_category"] = False
        result["extracted_category"] = ""

    bullet_match = re.search(r"【核心要点】(.*?)(?:【|$)", text, re.DOTALL)
    if bullet_match:
        bullets = re.findall(r"^\s*\d+[\.、．]\s*.+", bullet_match.group(1), re.MULTILINE)
        result["bullet_count"] = len(bullets)
        result["valid_bullets"] = len(bullets) >= 3
    else:
        result["bullet_count"] = 0
        result["valid_bullets"] = False

    result["has_time_info"] = bool(re.search(r"【时间信息】\s*\n?\s*([^\n【]+)", text))
    return result


def evaluate_format(predictions: list[str]) -> tuple[dict, list[int]]:
    """批量评测格式合规率并返回 bad case 索引。

    bad case 判定条件（三选一触发即计入）：
    - 必需字段不完整（any section missing）
    - 事件类别不在白名单（invalid category）
    - 核心要点编号条数不足 3 条（valid_bullets=False）
    """
    rows = [check_format(pred) for pred in predictions]
    n = len(rows)

    report = {
        "total": n,
        "all_sections_pass_rate": 0.0,
        "valid_category_rate": 0.0,
        "valid_bullets_rate": 0.0,
        "has_time_info_rate": 0.0,
        "avg_bullet_count": 0.0,
        "missing_field_rate": 0.0,
        "invalid_category_rate": 0.0,
    }
    if n == 0:
        return report, []

    report["all_sections_pass_rate"] = sum(bool(r["all_sections_present"]) for r in rows) / n
    report["valid_category_rate"] = sum(bool(r["valid_category"]) for r in rows) / n
    report["valid_bullets_rate"] = sum(bool(r["valid_bullets"]) for r in rows) / n
    report["has_time_info_rate"] = sum(bool(r["has_time_info"]) for r in rows) / n
    report["avg_bullet_count"] = sum(int(r["bullet_count"]) for r in rows) / n
    # 以下两项可由上方字段推导，保留是为了兼容旧报告格式
    report["missing_field_rate"] = 1.0 - report["all_sections_pass_rate"]
    report["invalid_category_rate"] = 1.0 - report["valid_category_rate"]

    # bad case：字段缺失、类别越界、要点不足三条——三者任一触发均计入
    # 修复前仅检查前两项，导致要点不足的样本被漏统计
    bad_indices = [
        i for i, row in enumerate(rows)
        if (
            not bool(row["all_sections_present"])
            or not bool(row["valid_category"])
            or not bool(row["valid_bullets"])
        )
    ]
    return report, bad_indices


def _clear_torch_cache() -> None:
    """释放 Python 与 CUDA 缓存，降低切换模型时的显存峰值。"""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def _load_tokenizer(model_path: str):
    """加载 tokenizer，并统一设置左填充与 pad_token。

    注意事项：
    - padding_side="left"：batch 推理时右侧为有效 token，左填充对 generate 友好。
    - pad_token_id 缺失时以 eos_token_id 代替，防止生成时报 ValueError。
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 批量推理必须左填充：模型 attention 从右往左读有效 token
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        # Qwen 系列 tokenizer 通常没有独立 pad_token，以 eos 代替
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def _load_base_model(base_model_path: str):
    """加载纯基座模型（不注入 LoRA）。

    设计要点：
    - 直接使用 AutoModelForCausalLM，不经过 PeftModel 包装。
    - 这与 Group A 的对比语义一致：测量"纯基座 + 系统提示"的能力上限。
    - device_map="auto" 让 Accelerate 按显存自动分配层（单卡即全量 GPU）。
    """
    import torch
    from transformers import AutoModelForCausalLM

    print(f"[Benchmark] 加载 Base 模型: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,   # Qwen3 原生 BF16，不可改为 float16
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()  # 关闭 dropout 等训练专用层，确保推理确定性
    return model


def _load_sft_model(base_model_path: str, adapter_path: str, merged_model_path: str | None):
    """加载 SFT 模型。

    优先策略（按速度/架构干净程度排序）：
    1. merged_model_path 已提供  → 直接加载合并后完整权重，无额外开销。
    2. 仅提供 adapter_path       → 加载 base + adapter，调用 merge_and_unload
                                   使 adapter 权重永久融入 base，返回普通模型。

    说明：不使用 disable_adapter() 方案，避免 PeftModel 包裹对推理路径的干扰。
    """
    import torch
    from transformers import AutoModelForCausalLM

    # ── 路径1：直接加载已合并模型（推荐，速度最快）──────────────────────────
    if merged_model_path:
        print(f"[Benchmark] 加载已合并 SFT 模型: {merged_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            merged_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        return model

    # ── 路径2：在线 merge（训练产物未导出时的备选）──────────────────────────
    if not adapter_path:
        raise ValueError("未提供 adapter_path，无法加载未合并 SFT 模型。")

    print(f"[Benchmark] 加载 Base + LoRA 并执行 merge: {adapter_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    try:
        from peft import PeftModel
    except ImportError:
        print("[ERROR] 缺少 peft，请安装: pip install peft", file=sys.stderr)
        sys.exit(1)

    peft_model = PeftModel.from_pretrained(base, adapter_path)
    merged = peft_model.merge_and_unload()
    merged.eval()
    return merged


def _infer_group(
    model,
    tokenizer,
    samples: list[dict],
    enable_thinking: bool,
    max_new_tokens: int,
    group_label: str,
    checkpoint_path: Path,
    batch_size: int,
) -> tuple[list[str], float]:
    """执行单组推理，支持断点续跑与批量生成。"""
    import torch

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    n = len(samples)
    preds: list[str | None] = [None] * n
    start_idx = 0

    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            done = [json.loads(line) for line in f if line.strip()]
        if done:
            # 将已完成的预测按 index 填回 preds 列表
            for row in done:
                idx = int(row["index"])
                if 0 <= idx < n:
                    preds[idx] = row["raw"]
            print(
                f"  [{group_label}] 检测到 checkpoint，已恢复 {sum(1 for p in preds if p is not None)} 条，"
                f"跳过已完成样本继续推理...",
                flush=True,
            )

    # 使用 preds 中仍为 None 的索引作为待推理集合
    # 相比原来 range(start_idx, n) 的方式，此方案能正确处理 checkpoint 中存在空洞的情况
    # （例如批次中途崩溃导致部分 index 未写入）
    remaining = [i for i in range(n) if preds[i] is None]
    if not remaining:
        elapsed = 0.0
        print(f"  [{group_label}] checkpoint 已覆盖全部样本，跳过推理。", flush=True)
        return [p if p is not None else "" for p in preds], elapsed

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    # 以追加模式打开 checkpoint 文件，推理后每条立即写盘，防止进程中断导致丢失
    ckpt_f = open(checkpoint_path, "a", encoding="utf-8")

    t0 = time.time()
    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=n,
            initial=start_idx,
            desc=f"Group {group_label}",
            unit="条",
            dynamic_ncols=True,
            file=sys.stdout,
        )

    try:
        for pos in range(0, len(remaining), batch_size):
            batch_indices = remaining[pos: pos + batch_size]

            # ── 构建本批次的 chat 格式输入 ──────────────────────────────
            prompts = []
            for idx in batch_indices:
                messages = [
                    {"role": "system", "content": MEDIUM_SYSTEM_PROMPT},
                    {"role": "user", "content": samples[idx]["input"]},
                ]
                prompts.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                )

            # ── tokenize & 移到 GPU ─────────────────────────────────────
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            # 记录 padding 后的输入长度，解码时只保留新生成的部分
            padded_len = inputs["input_ids"].shape[1]

            # ── 无梯度推理，temperature=0.1 保证输出基本稳定 ─────────────
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    repetition_penalty=1.1,  # 轻度抑制重复生成
                    pad_token_id=tokenizer.pad_token_id,
                )

            # ── 解码并逐条写入 checkpoint ────────────────────────────────
            for b_idx, sample_idx in enumerate(batch_indices):
                # 仅保留新生成 token，去除 prompt 部分
                new_ids = output_ids[b_idx][padded_len:]
                pred = tokenizer.decode(new_ids, skip_special_tokens=True)
                preds[sample_idx] = pred
                # 每条立即 flush，确保中断后 checkpoint 完整可恢复
                ckpt_f.write(json.dumps({"index": sample_idx, "raw": pred}, ensure_ascii=False) + "\n")
                ckpt_f.flush()

            if pbar is not None:
                pbar.update(len(batch_indices))
            else:
                elapsed_now = time.time() - t0
                print(f"  [{group_label}] {batch_indices[-1] + 1}/{n} 完成  {elapsed_now:.0f}s elapsed", flush=True)

    finally:
        # 无论是否发生异常，都保证文件句柄正常关闭（防止数据未 flush）
        if pbar is not None:
            pbar.close()
        ckpt_f.close()

    elapsed = time.time() - t0
    done_count = max(len(remaining), 1)
    print(
        f"  [{group_label}] 全部 {n} 条完成，总耗时 {elapsed:.1f}s ({elapsed / done_count:.1f}s/条, batch={batch_size})",
        flush=True,
    )
    return [p if p is not None else "" for p in preds], elapsed


def _eval_group(
    group_label: str,
    preds_raw: list[str],
    refs: list[str],
    test_data: list[dict],
    output_dir: Path,
    elapsed: float,
) -> dict:
    """评测单组结果并持久化为 group 目录下的报告文件。"""
    clean_preds = [strip_think_block(p) for p in preds_raw]
    rouge = compute_rouge(refs, clean_preds)
    fmt_report, bad_indices = evaluate_format(clean_preds)

    group_dir = output_dir / f"group_{group_label}"
    group_dir.mkdir(parents=True, exist_ok=True)

    with open(group_dir / "predictions_raw.jsonl", "w", encoding="utf-8") as f:
        for sample, raw, clean in zip(test_data, preds_raw, clean_preds):
            f.write(
                json.dumps(
                    {
                        "input": sample.get("input", ""),
                        "reference": sample.get("output", ""),
                        "raw": raw,
                        "clean": clean,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    (group_dir / "rouge_report.json").write_text(json.dumps(rouge, ensure_ascii=False, indent=2), encoding="utf-8")
    (group_dir / "format_report.json").write_text(
        json.dumps(fmt_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with open(group_dir / "bad_cases.jsonl", "w", encoding="utf-8") as f:
        for idx in bad_indices:
            f.write(json.dumps({"index": idx, "clean": clean_preds[idx]}, ensure_ascii=False) + "\n")

    summary = {
        "group": group_label,
        "n_samples": len(preds_raw),
        "elapsed_s": round(elapsed, 1),
        "per_sample_s": round(elapsed / max(len(preds_raw), 1), 1),
        "rouge1": round(rouge["rouge1"], 4),
        "rouge2": round(rouge["rouge2"], 4),
        "rougeL": round(rouge["rougeL"], 4),
        "all_sections_pass": f"{fmt_report['all_sections_pass_rate']:.1%}",
        "valid_category": f"{fmt_report['valid_category_rate']:.1%}",
        "valid_bullets": f"{fmt_report['valid_bullets_rate']:.1%}",
        "bad_cases": len(bad_indices),
    }

    print(f"\n  示例输出（Group {group_label} 第1条）：")
    print("  " + "-" * 50)
    if clean_preds:
        print("  " + clean_preds[0].replace("\n", "\n  ")[:400])
    else:
        print("  (空)")
    print("  " + "-" * 50)
    return summary


def run_benchmark(
    base_model_path: str,
    adapter_path: str,
    test_data: list[dict],
    output_dir: Path,
    batch_size: int = 1,
    skip_think: bool = False,
    merged_model_path: str | None = None,
    only_groups: str | None = None,
) -> None:
    """运行 A/B/C 在线推理对比并输出汇总报告。

    组别定义：
      A = Base 不思考（基线，最快）
      B = Base 思考（测试 thinking 模式对质量的影响）
      C = SFT V2 不思考（目标模型）

    设计要点：
    - 按 model_kind 复用已加载模型，避免 A→B 切换时重复加载（同为 base）。
    - A→C 切换时主动 del + empty_cache，释放显存后再加载 SFT。
    - tokenizer 只加载一次；C 单独运行时优先用 merged 路径避免不必要的 base 加载。
    """
    refs = [row.get("output", "") for row in test_data]
    n = len(test_data)

    # 三组定义：(标签, 模型种类, 是否思考, max_new_tokens, 描述)
    groups = [
        ("A", "base", False, MAX_TOKENS["A"], "Base 不思考"),
        ("B", "base", True,  MAX_TOKENS["B"], "Base 思考"),
        ("C", "sft",  False, MAX_TOKENS["C"], "SFT V2 不思考"),
    ]

    # ── 按命令行参数过滤运行的组别 ─────────────────────────────────────────
    if skip_think:
        groups = [g for g in groups if g[0] != "B"]
        print("[Benchmark] 已跳过 Group B（Base 思考模式）", flush=True)

    if only_groups:
        allow = [x.strip() for x in only_groups.split(",") if x.strip()]
        groups = [g for g in groups if g[0] in allow]
        print(f"[Benchmark] 仅运行指定组别: {allow}", flush=True)

    if not groups:
        print("[ERROR] 无有效组别可运行。", file=sys.stderr)
        sys.exit(1)

    # 若 C 需要 SFT 但未提供任何 SFT 路径，提前报错
    need_sft = any(g[1] == "sft" for g in groups)
    if need_sft and not merged_model_path and not adapter_path:
        print("[ERROR] Group C 需要 --merged_model 或 --adapter_path", file=sys.stderr)
        sys.exit(1)

    # tokenizer 策略：仅跑 C 且有 merged 模型时，直接用 merged 路径避免加载 base
    tokenizer_model_path = (
        merged_model_path
        if (len(groups) == 1 and groups[0][0] == "C" and merged_model_path)
        else base_model_path
    )
    print(f"[Benchmark] 加载 tokenizer: {tokenizer_model_path}")
    tokenizer = _load_tokenizer(tokenizer_model_path)

    # current_kind 记录当前已加载模型的种类（base/sft），
    # 相同种类连续运行时跳过重新加载（A→B 均为 base，只加载一次）
    current_kind: str | None = None
    current_model = None
    all_summaries: list[dict] = []

    for label, model_kind, think, max_tok, desc in groups:
        eff_batch = batch_size
        if think and batch_size > 2:
            # thinking 模式每条输出 token 量远超 no-think，>2 容易 OOM
            print(f"[WARN] Group {label} 为 thinking 模式，建议 batch_size<=2，当前为 {batch_size}")

        # ── 按需切换模型：同 kind 连续跑时复用已加载权重 ──────────────────
        if current_kind != model_kind or current_model is None:
            if current_model is not None:
                # 主动释放前一个模型的显存，防止同时驻留两个模型导致 OOM
                del current_model
                _clear_torch_cache()
            if model_kind == "base":
                current_model = _load_base_model(base_model_path)
            else:
                current_model = _load_sft_model(base_model_path, adapter_path, merged_model_path)
            current_kind = model_kind

        group_dir = output_dir / f"group_{label}"
        group_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = group_dir / "infer_checkpoint.jsonl"

        print(f"\n{'=' * 60}", flush=True)
        print(f"  Group {label}：{desc}  (max_new_tokens={max_tok}, batch={eff_batch}, n={n})", flush=True)
        print(f"  model_kind={model_kind}", flush=True)
        print(f"  checkpoint => {ckpt_path}", flush=True)
        print(f"{'=' * 60}", flush=True)

        preds_raw, elapsed = _infer_group(
            model=current_model,
            tokenizer=tokenizer,
            samples=test_data,
            enable_thinking=think,
            max_new_tokens=max_tok,
            group_label=label,
            checkpoint_path=ckpt_path,
            batch_size=eff_batch,
        )

        summary = _eval_group(label, preds_raw, refs, test_data, output_dir, elapsed)
        all_summaries.append(summary)

    if current_model is not None:
        del current_model
    _clear_torch_cache()

    print("\n" + "=" * 72)
    print("  三组对比结果汇总")
    print("=" * 72)
    print(f"{'组别':<6} {'描述':<14} {'R-1':>6} {'R-2':>6} {'R-L':>6} {'全节':>7} {'类别':>7} {'要点':>7} {'坏例':>5} {'耗时/条':>8}")
    print("-" * 72)

    desc_map = {"A": "Base不思考", "B": "Base思考", "C": "SFT-V2不思考"}
    for row in all_summaries:
        print(
            f"  {row['group']:<4} {desc_map[row['group']]:<14} "
            f"{row['rouge1']:>6.4f} {row['rouge2']:>6.4f} {row['rougeL']:>6.4f} "
            f"{row['all_sections_pass']:>7} {row['valid_category']:>7} {row['valid_bullets']:>7} "
            f"{row['bad_cases']:>5} {row['per_sample_s']:>7.1f}s"
        )
    print("=" * 72)

    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[完成] 汇总结果已保存: {summary_path}")


def run_eval_mode(args) -> None:
    """执行离线评测分支（已有预测文件）。"""
    test_path = Path(args.test)
    pred_path = Path(args.predictions)

    if not test_path.exists():
        print(f"[ERROR] 测试集不存在: {test_path}", file=sys.stderr)
        sys.exit(1)
    if not pred_path.exists():
        print(f"[ERROR] 预测结果不存在: {pred_path}", file=sys.stderr)
        sys.exit(1)

    test_data = load_test_data(test_path)
    predictions = load_predictions(pred_path, strip_think=args.strip_think)
    references = [row.get("output", "") for row in test_data]

    n = min(len(references), len(predictions))
    if len(references) != len(predictions):
        print(f"[WARN] 测试集({len(references)})与预测({len(predictions)})数量不一致，取前 {n} 条")
    references = references[:n]
    predictions = predictions[:n]

    print(f"[INFO] 评测 {n} 条样本...")

    rouge_scores = compute_rouge(references, predictions, use_jieba=not args.no_jieba)
    format_report, bad_indices = evaluate_format(predictions)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rouge_path = output_dir / "rouge_report.json"
    rouge_path.write_text(json.dumps(rouge_scores, ensure_ascii=False, indent=2), encoding="utf-8")

    format_path = output_dir / "format_report.json"
    format_path.write_text(json.dumps(format_report, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(output_dir / "bad_cases.jsonl", "w", encoding="utf-8") as f:
        for i in bad_indices:
            f.write(
                json.dumps(
                    {
                        "index": i,
                        "reference": references[i],
                        "prediction": predictions[i],
                        "input": test_data[i].get("input", ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"[INFO] ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"[INFO] ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"[INFO] ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print(f"[INFO] 格式通过率: {format_report['all_sections_pass_rate']:.2%}")
    print(f"[INFO] 类别合规率: {format_report['valid_category_rate']:.2%}")
    print(f"[INFO] 要点格式率: {format_report['valid_bullets_rate']:.2%}")
    print(f"[INFO] Bad cases: {len(bad_indices)}")
    print(f"[INFO] 报告输出目录: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="ROUGE + 格式评测脚本")
    parser.add_argument("--mode", type=str, default="eval", choices=["eval", "benchmark"], help="运行模式")
    parser.add_argument("--test", type=str, default=str(DEFAULT_TEST), help="测试集路径")
    parser.add_argument("--predictions", type=str, default=str(DEFAULT_PREDICTIONS), help="离线预测文件路径")
    parser.add_argument("--output_dir", type=str, default=str(EVAL_DIR), help="评测输出目录")
    parser.add_argument("--no_jieba", action="store_true", help="禁用 jieba 分词")
    parser.add_argument("--strip_think", action="store_true", help="离线评测时先剥离 think 块")

    parser.add_argument("--n_samples", type=int, default=10, help="benchmark 样本数，0 表示全量")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL, help="基座模型路径")
    parser.add_argument("--adapter_path", type=str, default=DEFAULT_ADAPTER_PATH, help="LoRA adapter 路径")
    parser.add_argument("--merged_model", type=str, default="", help="合并后的 SFT 模型路径")
    parser.add_argument("--batch_size", type=int, default=1, help="推理 batch 大小")
    parser.add_argument("--skip_think", action="store_true", help="跳过 Group B")
    parser.add_argument("--only_groups", type=str, default="", help="仅运行指定组别，如 A,C")
    return parser


def main() -> None:
    """脚本入口。"""
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "eval":
        run_eval_mode(args)
        return

    test_data = load_test_data(Path(args.test))
    n = args.n_samples if args.n_samples > 0 else len(test_data)
    test_data = test_data[:n]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Benchmark] 使用 {len(test_data)} 条样本，开始在线推理评测...")
    run_benchmark(
        base_model_path=args.base_model,
        adapter_path=args.adapter_path,
        test_data=test_data,
        output_dir=output_dir,
        batch_size=args.batch_size,
        skip_think=args.skip_think,
        merged_model_path=args.merged_model if args.merged_model else None,
        only_groups=args.only_groups if args.only_groups else None,
    )


if __name__ == "__main__":
    main()
