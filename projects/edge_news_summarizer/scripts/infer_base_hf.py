#!/usr/bin/env python3
"""
infer_base_hf.py
使用 HuggingFace Transformers 对基座模型（Qwen3-4B）进行批量推理。

特性：
  - 每处理 SAVE_EVERY 条自动保存一次，防止长时间运行后丢失结果
  - 支持断点续推：已存在输出文件时自动跳过已完成的条目，从上次中断处继续
  - 输出格式与 LlamaFactory generated_predictions.jsonl 完全一致

运行环境：my_sft
用法：
  cd D:\LLM\MySFT\LLM_SFT\projects\edge_news_summarizer
  python scripts/infer_base_hf.py                     # 首次运行或断点续推
  python scripts/infer_base_hf.py --restart           # 强制从头重跑（忽略已有结果）

输出：outputs/eval_base_v2/generated_predictions.jsonl

评测（运行完成后）：
  python scripts/06_eval_rouge_and_format.py \
    --predictions outputs/eval_base_v2/generated_predictions.jsonl \
    --output_dir  outputs/eval_base_v2 \
    --strip_think
"""

import argparse
import json
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DEFAULT_MODEL      = "D:/LLM/models/Qwen3-4B"
DEFAULT_TEST       = PROJECT_DIR / "data" / "cleaned" / "test.json"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "eval_base_v2"
SAVE_EVERY         = 20          # 每处理 20 条保存一次
MAX_NEW_TOKENS     = 2048
TEMPERATURE        = 0.1
TOP_P              = 0.9
REPETITION_PENALTY = 1.1


# ─────────────────────────── 工具函数 ─────────────────────────────────────────

def load_test_data(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def load_done_indices(out_path: Path) -> set[int]:
    """读取已有输出文件，返回已完成的样本序号集合。"""
    done = set()
    if not out_path.exists():
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "_idx" in obj:
                    done.add(obj["_idx"])
            except json.JSONDecodeError:
                pass
    return done


def append_result(out_path: Path, idx: int, predict: str, label: str) -> None:
    """追加单条结果（含 _idx 字段用于断点定位，评测脚本会忽略多余字段）。"""
    line = json.dumps({"_idx": idx, "predict": predict, "label": label},
                      ensure_ascii=False)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def reorder_output(out_path: Path, total: int) -> None:
    """推理结束后将结果按原始顺序排列，去掉 _idx 辅助字段，覆写文件。"""
    records: dict[int, dict] = {}
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records[obj["_idx"]] = obj

    ordered = [records[i] for i in range(total) if i in records]
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in ordered:
            clean = {"predict": obj["predict"], "label": obj["label"]}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
    print(f"[INFO] 结果已排序并写入: {out_path}")


def build_prompt(tokenizer, sample: dict) -> str:
    """构建 Qwen3 chat prompt（开启思考模式）。"""
    messages = [
        {"role": "system", "content": sample["instruction"]},
        {"role": "user",   "content": sample["input"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,   # 允许 <think> 块，评测时用 --strip_think 剥离
    )


# ─────────────────────────── 主流程 ───────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="基座模型 HF 批量推理（支持断点续推）")
    parser.add_argument("--model",       type=str, default=DEFAULT_MODEL)
    parser.add_argument("--test",        type=str, default=str(DEFAULT_TEST))
    parser.add_argument("--output_dir",  type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top_p",       type=float, default=TOP_P)
    parser.add_argument("--repetition_penalty", type=float, default=REPETITION_PENALTY)
    parser.add_argument("--save_every",  type=int, default=SAVE_EVERY,
                        help="每处理 N 条打印一次进度（每条推理后立即保存）")
    parser.add_argument("--restart",     action="store_true",
                        help="忽略已有结果，从头重新推理")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "generated_predictions.jsonl"

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    test_data = load_test_data(Path(args.test))
    total = len(test_data)
    print(f"[INFO] 测试集: {total} 条")

    # ── 断点续推：跳过已完成的样本 ────────────────────────────────────────────
    if args.restart and out_path.exists():
        out_path.unlink()
        print("[INFO] --restart: 已删除旧结果，从头开始")

    done_indices = load_done_indices(out_path)
    remaining = [i for i in range(total) if i not in done_indices]

    if done_indices:
        print(f"[INFO] 断点续推: 已完成 {len(done_indices)} 条，"
              f"剩余 {len(remaining)} 条")
    else:
        print(f"[INFO] 首次运行，共需处理 {total} 条")

    if not remaining:
        print("[INFO] 所有样本已完成，直接进行排序整理")
        reorder_output(out_path, total)
        return

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    print(f"[INFO] 加载模型: {args.model} (BF16)")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[INFO] 模型加载完成，设备: {next(model.parameters()).device}")

    # ── 逐条推理 ──────────────────────────────────────────────────────────────
    t_start = time.time()
    t_batch  = time.time()

    print(f"[INFO] 开始推理（每完成 {args.save_every} 条打印进度）...")
    print(f"       max_new_tokens={args.max_new_tokens}, "
          f"temperature={args.temperature}, "
          f"repetition_penalty={args.repetition_penalty}")
    print()

    for pos, idx in enumerate(remaining, 1):
        sample = test_data[idx]
        prompt = build_prompt(tokenizer, sample)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                do_sample=(args.temperature > 0),
                pad_token_id=tokenizer.eos_token_id,
            )

        # 只取新生成部分
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        predict = tokenizer.decode(new_ids, skip_special_tokens=True)

        # 立即保存（追加写入，不丢失已有结果）
        append_result(out_path, idx, predict, sample["output"])

        # 进度打印
        if pos % args.save_every == 0 or pos == len(remaining):
            elapsed   = time.time() - t_start
            batch_t   = time.time() - t_batch
            done_total = len(done_indices) + pos
            eta_s     = (elapsed / pos) * (len(remaining) - pos)
            print(f"  [{done_total:4d}/{total}] "
                  f"最近{min(pos, args.save_every)}条耗时 {batch_t:.1f}s | "
                  f"总耗时 {elapsed/60:.1f}min | "
                  f"预计剩余 {eta_s/60:.1f}min")
            t_batch = time.time()

    # ── 排序并整理输出文件 ─────────────────────────────────────────────────────
    total_elapsed = time.time() - t_start
    print(f"\n[INFO] 推理完成！共耗时 {total_elapsed/60:.1f} 分钟")
    reorder_output(out_path, total)

    print()
    print("=" * 60)
    print("评测命令（在 my_sft 环境中运行）：")
    print(f"  cd D:\\LLM\\MySFT\\LLM_SFT\\projects\\edge_news_summarizer")
    print(f"  conda activate my_sft")
    print(f"  python scripts/06_eval_rouge_and_format.py \\")
    print(f"    --predictions {out_path} \\")
    print(f"    --output_dir  {output_dir} \\")
    print(f"    --strip_think")


if __name__ == "__main__":
    main()
