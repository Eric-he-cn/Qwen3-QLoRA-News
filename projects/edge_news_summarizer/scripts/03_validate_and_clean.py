#!/usr/bin/env python3
"""
03_validate_and_clean.py

对 API 标注后的新闻数据进行校验、清洗与质量统计，输出可直接用于训练的数据。

主要职责：
1) 结构校验：6 个固定字段、类别白名单、核心要点数量。
2) 长度过滤：输入/输出长度范围约束。
3) 数据去重：按 instruction + input 去重。
4) 质量快照：输出长度分布、语言来源分布、时间信息覆盖率、随机样本预览。

输入：
- data/labeled/news_labeled_v1.jsonl

输出：
- data/cleaned/cleaned_all.jsonl
- data/reports/labeling_stats.json
- data/reports/data_quality_report.md
- data/reports/quality_snapshot.json
"""

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_CLEANED_DIR = PROJECT_DIR / "data" / "cleaned"
DATA_REPORTS_DIR = PROJECT_DIR / "data" / "reports"
DATA_CLEANED_DIR.mkdir(parents=True, exist_ok=True)
DATA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INPUT = PROJECT_DIR / "data" / "labeled" / "news_labeled_v1.jsonl"
DEFAULT_OUTPUT = DATA_CLEANED_DIR / "cleaned_all.jsonl"
DEFAULT_STATS = DATA_REPORTS_DIR / "labeling_stats.json"
DEFAULT_REPORT = DATA_REPORTS_DIR / "data_quality_report.md"
DEFAULT_SNAPSHOT = DATA_REPORTS_DIR / "quality_snapshot.json"

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

TIME_NONE_MARKERS = ["无时间", "未提及", "无", "不明确", "未知", "N/A", "None", "没有提到", "暂无"]


def check_sections(output_text: str) -> tuple[bool, list[str]]:
    """检查输出是否包含全部必需字段。"""
    missing = [section for section in REQUIRED_SECTIONS if section not in output_text]
    return len(missing) == 0, missing


def extract_category(output_text: str) -> str:
    """提取【事件类别】字段文本。"""
    match = re.search(r"【事件类别】\s*\n?\s*([^\n【]+)", output_text)
    return match.group(1).strip() if match else ""


def check_category(category: str) -> bool:
    """判断类别是否命中白名单（支持宽松匹配）。"""
    cat_lower = category.lower().strip()
    return any(v.lower() in cat_lower or cat_lower in v.lower() for v in VALID_CATEGORIES)


def check_bullet_points(output_text: str, min_points: int = 3) -> tuple[bool, int]:
    """检查【核心要点】中编号条目数是否达到下限。"""
    match = re.search(r"【核心要点】(.*?)(?:【|$)", output_text, re.DOTALL)
    if not match:
        return False, 0
    bullets = re.findall(r"^\s*\d+[\.、．]\s*.+", match.group(1), re.MULTILINE)
    return len(bullets) >= min_points, len(bullets)


def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    """按 instruction + input 去重，返回 (唯一记录, 去重数)。

    注意：原先使用 Python hash() 存在哈希碰撞风险（概率极低但不可忽视）。
    已改为直接存储字符串键，彻底消除碰撞隐患。
    """
    # 使用字符串 set 而非 hash，避免 hash 碰撞导致正常记录被错误删除
    seen: set[str] = set()
    unique_records: list[dict] = []
    dup_count = 0

    for row in records:
        # 以 instruction + input 内容为透明 key，不依赖哈希
        key = row.get("instruction", "") + "\x00" + row.get("input", "")
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        unique_records.append(row)

    return unique_records, dup_count


def validate_record(
    record: dict,
    strict: bool,
    min_input_len: int,
    max_input_len: int,
    min_output_len: int,
    max_output_len: int,
) -> tuple[bool, list[str]]:
    """校验单条记录并返回 (是否通过, 错误列表)。

    校验维度：
    1. 长度约束：input/output 字符数在合理区间内。
    2. 字段完整性：6 个必需标签全部出现。
    3. 类别合规性：strict=True 时类别必须命中白名单。
    4. 要点数量：编号列表至少 3 条。
    """
    errors: list[str] = []
    input_text = record.get("input", "")
    output_text = record.get("output", "")

    # ── 长度过滤：过短说明内容无效，过长超出训练上下文窗口 ──────────────────
    if len(input_text) < min_input_len:
        errors.append(f"input_too_short({len(input_text)})")
    if len(input_text) > max_input_len:
        errors.append(f"input_too_long({len(input_text)})")
    if len(output_text) < min_output_len:
        errors.append(f"output_too_short({len(output_text)})")
    if len(output_text) > max_output_len:
        errors.append(f"output_too_long({len(output_text)})")

    # ── 字段完整性：6 个标签缺一不可 ────────────────────────────────────────
    sections_ok, missing_sections = check_sections(output_text)
    if not sections_ok:
        errors.append(f"missing_sections:{','.join(missing_sections)}")

    # ── 类别合规性 ───────────────────────────────────────────────────────────
    category = extract_category(output_text)
    if not category:
        errors.append("no_category")
    elif strict and not check_category(category):
        # 非严格模式下允许白名单外类别（如组合类别"科技/财经"），降低过滤率
        errors.append(f"invalid_category:'{category}'")

    # ── 要点数量：少于 3 条会导致训练样本格式不一致 ──────────────────────────
    bullets_ok, bullet_count = check_bullet_points(output_text)
    if not bullets_ok:
        errors.append(f"insufficient_bullets({bullet_count})")

    return len(errors) == 0, errors


def summarize_quality(records: list[dict], sample_preview_count: int, seed: int) -> dict:
    """生成质量快照：字段完整率、长度分布、语言分布、时间信息覆盖率与抽样。"""
    total = len(records)
    if total == 0:
        return {
            "total": 0,
            "all_sections_pass_rate": 0.0,
            "output_length": {"avg": 0, "min": 0, "max": 0, "short_lt_200": 0, "short_lt_200_rate": 0.0},
            "language_dist": {},
            "time_info_specific_rate": 0.0,
            "samples": [],
        }

    outputs = [row.get("output", "") for row in records]
    full_ok = sum(1 for text in outputs if all(section in text for section in REQUIRED_SECTIONS))

    lengths = [len(text) for text in outputs]
    short_count = sum(1 for v in lengths if v < 200)

    lang_counter = Counter(row.get("source_lang", row.get("lang", "?")) for row in records)

    time_with_info = 0
    for text in outputs:
        idx = text.find("【时间信息】")
        if idx >= 0:
            snippet = text[idx + 7: idx + 60].strip()
            if snippet and not any(marker in snippet for marker in TIME_NONE_MARKERS):
                time_with_info += 1

    samples: list[dict] = []
    if sample_preview_count > 0:
        rng = random.Random(seed)
        picked = rng.sample(records, min(sample_preview_count, total))
        for row in picked:
            samples.append(
                {
                    "id": row.get("id", ""),
                    "input_preview": row.get("input", "")[:120],
                    "output_preview": row.get("output", "")[:200],
                }
            )

    return {
        "total": total,
        "all_sections_pass_rate": full_ok / total,
        "output_length": {
            "avg": round(sum(lengths) / total, 1),
            "min": min(lengths),
            "max": max(lengths),
            "short_lt_200": short_count,
            "short_lt_200_rate": short_count / total,
        },
        "language_dist": dict(lang_counter),
        "time_info_specific_rate": time_with_info / total,
        "samples": samples,
    }


def generate_report(stats: dict, output_path: Path) -> None:
    """生成 Markdown 数据质量报告。"""
    total = stats["total"]
    passed = stats["passed"]

    report = f"""# 数据质量报告

## 概览

| 指标 | 数量 | 比例 |
|------|------|------|
| 原始记录 | {total} | 100% |
| 通过校验 | {passed} | {(passed / total * 100 if total > 0 else 0):.1f}% |
| 去重丢弃 | {stats['duplicates']} | {(stats['duplicates'] / total * 100 if total > 0 else 0):.1f}% |

## 错误分布

| 错误类型 | 数量 |
|----------|------|
"""

    for err_type, count in sorted(stats["error_counts"].items(), key=lambda x: -x[1]):
        report += f"| {err_type} | {count} |\n"

    report += """
## 类别分布

| 类别 | 数量 |
|------|------|
"""

    for cat, count in sorted(stats["category_dist"].items(), key=lambda x: -x[1]):
        report += f"| {cat} | {count} |\n"

    output_path.write_text(report, encoding="utf-8")
    print(f"[INFO] 质量报告已保存: {output_path}")


def main() -> None:
    """脚本入口。"""
    parser = argparse.ArgumentParser(description="数据校验与清洗脚本")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="输入 JSONL 路径")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="清洗后 JSONL 输出路径")
    parser.add_argument("--strict", action="store_true", help="严格类别校验（必须命中白名单）")

    parser.add_argument("--min_input_len", type=int, default=50)
    parser.add_argument("--max_input_len", type=int, default=4000)
    parser.add_argument("--min_output_len", type=int, default=100)
    parser.add_argument("--max_output_len", type=int, default=2000)

    parser.add_argument("--stats_path", type=str, default=str(DEFAULT_STATS), help="统计 JSON 输出路径")
    parser.add_argument("--report_path", type=str, default=str(DEFAULT_REPORT), help="Markdown 报告路径")
    parser.add_argument("--quality_snapshot_path", type=str, default=str(DEFAULT_SNAPSHOT), help="质量快照 JSON 输出路径")
    parser.add_argument("--sample_preview_count", type=int, default=3, help="质量快照抽样条数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    records: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[INFO] 读取 {len(records)} 条记录")

    records, dup_count = deduplicate(records)
    print(f"[INFO] 去重后 {len(records)} 条（去重 {dup_count} 条）")

    passed: list[dict] = []
    failed: list[dict] = []
    error_counts: defaultdict[str, int] = defaultdict(int)
    category_dist: defaultdict[str, int] = defaultdict(int)

    for row in records:
        ok, errors = validate_record(
            row,
            strict=args.strict,
            min_input_len=args.min_input_len,
            max_input_len=args.max_input_len,
            min_output_len=args.min_output_len,
            max_output_len=args.max_output_len,
        )
        if ok:
            passed.append(row)
            category_dist[extract_category(row.get("output", ""))] += 1
        else:
            for err in errors:
                key = err.split("(")[0].split(":")[0]
                error_counts[key] += 1
            failed.append({**row, "_errors": errors})

    print(f"[INFO] 通过校验: {len(passed)} 条 / 失败: {len(failed)} 条")

    cleaned_rows = [
        {
            "instruction": row.get("instruction", "你是一位专业的新闻编辑助手，请对新闻进行结构化摘要分析。"),
            "input": row.get("input", ""),
            "output": row.get("output", ""),
        }
        for row in passed
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in cleaned_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[INFO] 清洗数据已保存: {output_path}")

    stats = {
        "total": len(records) + dup_count,
        "passed": len(passed),
        "duplicates": dup_count,
        "error_counts": dict(error_counts),
        "category_dist": dict(category_dist),
    }

    stats_path = Path(args.stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] 统计信息已保存: {stats_path}")

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generate_report(stats, report_path)

    quality_snapshot = summarize_quality(
        records=passed,
        sample_preview_count=args.sample_preview_count,
        seed=args.seed,
    )
    quality_snapshot_path = Path(args.quality_snapshot_path)
    quality_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    quality_snapshot_path.write_text(json.dumps(quality_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] 质量快照已保存: {quality_snapshot_path}")

    total = stats["total"]
    print("\n===== 数据清洗摘要 =====")
    print(f"原始记录: {total}")
    print(f"通过率: {len(passed) / max(total, 1) * 100:.1f}%")
    print(f"错误类型分布: {dict(error_counts)}")


if __name__ == "__main__":
    main()
