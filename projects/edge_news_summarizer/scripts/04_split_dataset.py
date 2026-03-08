#!/usr/bin/env python3
"""
04_split_dataset.py

将清洗后的数据按比例划分为 train / val / test，并提供两项扩展能力：
1) 可选统一刷新 instruction 字段（替代临时脚本 update_instruction.py）。
2) 可选统计训练样本 token 长度分布（替代临时脚本 _tmp_token_stats.py）。

输入：
- data/cleaned/cleaned_all.jsonl

输出：
- data/cleaned/train.json
- data/cleaned/val.json
- data/cleaned/test.json
- data/cleaned/test_manual_eval.json
- data/reports/token_length_report.json（仅在 --analyze_tokens 时输出）
"""

import argparse
import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_CLEANED_DIR = PROJECT_DIR / "data" / "cleaned"
DATA_REPORTS_DIR = PROJECT_DIR / "data" / "reports"
DATA_CLEANED_DIR.mkdir(parents=True, exist_ok=True)
DATA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INPUT = DATA_CLEANED_DIR / "cleaned_all.jsonl"
DEFAULT_TOKEN_REPORT = DATA_REPORTS_DIR / "token_length_report.json"
DEFAULT_TOKENIZER_PATH = "D:/LLM/models/Qwen3-4B"

MEDIUM_SYSTEM_PROMPT = (
    "你是专业的新闻编辑助手。请对新闻内容进行结构化摘要，严格按以下6个标签顺序输出，禁止使用 Markdown：\n"
    "【一句话摘要】【核心要点】【事件类别】【主要主体】【时间信息】【潜在影响】\n\n"
    "其中【核心要点】用阿拉伯数字编号列出至少3条；\n"
    "【事件类别】只能从以下选择：政治、经济、科技、文化、社会、军事、体育、健康、环境、国际、历史、旅游、财经"
)


def load_jsonl(path: Path) -> list[dict]:
    """读取 JSONL 文件并返回记录列表。"""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(rows: list[dict], path: Path) -> None:
    """保存 JSON（list 格式）文件。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 已保存 {len(rows)} 条到 {path}")


def split_dataset(
    rows: list[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """按比例随机切分数据集。"""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test 比例之和必须为 1.0")

    rng = random.Random(seed)
    shuffled = rows.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def refresh_instruction(rows: list[dict], instruction_text: str) -> None:
    """原地刷新 instruction 字段。"""
    for row in rows:
        row["instruction"] = instruction_text


def load_instruction_text(instruction_file: str | None) -> str:
    """加载统一 instruction 文本，优先读取文件，否则使用内置中等约束版本。"""
    if instruction_file:
        path = Path(instruction_file)
        if not path.exists():
            raise FileNotFoundError(f"instruction 文件不存在: {path}")
        return path.read_text(encoding="utf-8").strip()
    return MEDIUM_SYSTEM_PROMPT


def analyze_token_lengths(
    train_rows: list[dict],
    tokenizer_path: str,
    cutoff_len: int,
) -> dict:
    """统计训练数据在 chat template 下的 token 长度分布。

    用途：
    - 评估 cutoff_len 是否合理（超长样本会被截断，可能丢失输出字段）。
    - 指导选择合适的 cutoff_len 权衡覆盖率与显存占用。

    输出字段说明：
    - over_cutoff / over_cutoff_rate：超出 cutoff 的样本数及比例。
    - cumulative[N]：token<=N 的样本数及比例，便于快速判断分位点。
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("[WARN] 未安装 transformers，跳过 token 长度统计。")
        return {}

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    lengths: list[int] = []

    for row in train_rows:
        instruction = row.get("instruction", "")
        user_input = row.get("input", "")
        output = row.get("output", "")

        # 将 instruction + input 合并为 user 消息（与 LLaMA-Factory 训练格式一致）
        user_content = instruction + ("\n" + user_input if user_input else "")
        messages = [
            {"role": "system",    "content": MEDIUM_SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": output},
        ]

        # apply_chat_template 生成与实际训练完全一致的文本（含特殊 token）
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        lengths.append(len(tokenizer.encode(text)))

    if not lengths:
        return {}

    total = len(lengths)
    over_cutoff = sum(1 for l in lengths if l > cutoff_len)

    # 累计分位分布，便于判断各 cutoff 阈值下的覆盖率
    cumulative = {}
    for threshold in [512, 768, 900, 1024, 1280, 1536, 2048]:
        count = sum(1 for l in lengths if l <= threshold)
        cumulative[str(threshold)] = {"count": count, "rate": count / total}

    return {
        "tokenizer_path": tokenizer_path,
        "num_samples": total,
        "cutoff_len": cutoff_len,
        "over_cutoff": over_cutoff,
        "over_cutoff_rate": over_cutoff / total,
        "avg_len": round(sum(lengths) / total, 1),
        "min_len": min(lengths),
        "max_len": max(lengths),
        "cumulative": cumulative,
    }


def main() -> None:
    """脚本入口。"""
    parser = argparse.ArgumentParser(description="数据集切分脚本")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="输入 JSONL 路径")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manual_eval_count", type=int, default=100)

    parser.add_argument("--refresh_instruction", action="store_true", help="切分后统一刷新 instruction")
    parser.add_argument("--instruction_file", type=str, default="", help="统一 instruction 文本文件")

    parser.add_argument("--analyze_tokens", action="store_true", help="输出训练集 token 长度统计")
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--cutoff_len", type=int, default=1024)
    parser.add_argument("--token_report", type=str, default=str(DEFAULT_TOKEN_REPORT))
    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print("[ERROR] train_ratio + val_ratio + test_ratio 必须等于 1.0", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在: {input_path}", file=sys.stderr)
        print("请先运行 03_validate_and_clean.py", file=sys.stderr)
        sys.exit(1)

    rows = load_jsonl(input_path)
    print(f"[INFO] 读取 {len(rows)} 条记录")
    if len(rows) < 10:
        print(f"[WARN] 数据量较少（{len(rows)} 条），建议至少 100 条以上。")

    train, val, test = split_dataset(rows, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

    if args.refresh_instruction:
        instruction_text = load_instruction_text(args.instruction_file if args.instruction_file else None)
        refresh_instruction(train, instruction_text)
        refresh_instruction(val, instruction_text)
        refresh_instruction(test, instruction_text)
        print("[INFO] 已统一刷新 train/val/test 的 instruction 字段")

    print("\n[INFO] 划分结果：")
    print(f"  训练集: {len(train)} 条 ({len(train) / len(rows) * 100:.1f}%)")
    print(f"  验证集: {len(val)} 条 ({len(val) / len(rows) * 100:.1f}%)")
    print(f"  测试集: {len(test)} 条 ({len(test) / len(rows) * 100:.1f}%)")

    save_json(train, DATA_CLEANED_DIR / "train.json")
    save_json(val, DATA_CLEANED_DIR / "val.json")
    save_json(test, DATA_CLEANED_DIR / "test.json")

    rng = random.Random(args.seed + 1)
    manual_eval = rng.sample(test, min(args.manual_eval_count, len(test)))
    save_json(manual_eval, DATA_CLEANED_DIR / "test_manual_eval.json")

    if args.analyze_tokens:
        token_report = analyze_token_lengths(
            train_rows=train,
            tokenizer_path=args.tokenizer_path,
            cutoff_len=args.cutoff_len,
        )
        token_report_path = Path(args.token_report)
        token_report_path.parent.mkdir(parents=True, exist_ok=True)
        token_report_path.write_text(json.dumps(token_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] token 长度报告已保存: {token_report_path}")

    print("\n[INFO] 数据集划分完成！")
    print("[INFO] 下一步：运行 05_register_dataset_info.py 注册数据集")


if __name__ == "__main__":
    main()
