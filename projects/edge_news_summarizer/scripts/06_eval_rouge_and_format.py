#!/usr/bin/env python3
"""
06_eval_rouge_and_format.py
对模型推理结果进行 ROUGE 评测和格式正确率评测。

输入：
  - 测试集：data/cleaned/test.json（含 reference output）
  - 模型推理结果：outputs/eval/generated_predictions.jsonl（LLaMA-Factory 推理输出）

输出：
  - outputs/eval/rouge_report.json
  - outputs/eval/format_report.json
  - outputs/eval/bad_cases.jsonl（格式失败的样本）

用法：
  python scripts/06_eval_rouge_and_format.py
  python scripts/06_eval_rouge_and_format.py --predictions outputs/eval/generated_predictions.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
EVAL_DIR = PROJECT_DIR / "outputs" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TEST = PROJECT_DIR / "data" / "cleaned" / "test.json"
# LlamaFactory 实际输出路径（相对于 D:\LLM\LlamaFactory 运行）
DEFAULT_PREDICTIONS = Path("D:/LLM/LlamaFactory/projects/edge_news_summarizer/outputs/eval/generated_predictions.jsonl")

REQUIRED_SECTIONS = [
    "【一句话摘要】",
    "【核心要点】",
    "【事件类别】",
    "【主要主体】",
    "【时间信息】",
    "【潜在影响】",
]

VALID_CATEGORIES = {
    "科技", "财经", "政治", "社会", "体育", "文化", "国际", "军事", "环境", "健康",
    "technology", "finance", "politics", "society", "sports", "culture",
    "international", "military", "environment", "health",
}


def load_test_data(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def strip_think_block(text: str) -> str:
    """剥离 Qwen3 思维链块 <think>...</think>，返回实际答案部分。
    若无 think 块则原样返回。剥离后去掉首尾空白。
    """
    # 贪婪匹配整个 think 块（含嵌套情况用非贪婪）
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return stripped.strip()


def load_predictions(path: Path, strip_think: bool = False) -> list[str]:
    """加载模型预测结果（每行一个 JSON 或纯文本）。
    strip_think=True 时自动剥离 <think>...</think> 块（用于基座模型评测）。
    """
    predictions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # LLaMA-Factory 输出格式：{"predict": "...", "label": "..."}
                pred = obj.get("predict", obj.get("generated_text", obj.get("output", line)))
                if strip_think:
                    pred = strip_think_block(pred)
                predictions.append(pred)
            except json.JSONDecodeError:
                pred = strip_think_block(line) if strip_think else line
                predictions.append(pred)
    return predictions


def compute_rouge(references: list[str], predictions: list[str],
                  use_jieba: bool = True) -> dict:
    """计算 ROUGE 分数。"""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("[ERROR] 请先安装 rouge-score: pip install rouge-score", file=sys.stderr)
        sys.exit(1)

    if use_jieba:
        try:
            import jieba

            def tokenize_zh(text):
                return " ".join(jieba.cut(text))

            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                              use_stemmer=False,
                                              tokenizer=None)
            scores = {"rouge1": [], "rouge2": [], "rougeL": []}
            for ref, pred in zip(references, predictions):
                ref_tok = tokenize_zh(ref)
                pred_tok = tokenize_zh(pred)
                result = scorer.score(ref_tok, pred_tok)
                for k in scores:
                    scores[k].append(result[k].fmeasure)
        except ImportError:
            print("[WARN] jieba 未安装，使用字符级 ROUGE（适合中文）")
            use_jieba = False

    if not use_jieba:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for ref, pred in zip(references, predictions):
            # 字符分割（中文友好）
            ref_chars = " ".join(list(ref))
            pred_chars = " ".join(list(pred))
            result = scorer.score(ref_chars, pred_chars)
            for k in scores:
                scores[k].append(result[k].fmeasure)

    return {k: sum(v) / len(v) if v else 0.0 for k, v in scores.items()}


def check_format(text: str) -> dict:
    """检查单条预测的格式合规性。"""
    results = {}

    # 1. 必需字段
    for section in REQUIRED_SECTIONS:
        results[f"has_{section}"] = section in text
    results["all_sections_present"] = all(results[f"has_{s}"] for s in REQUIRED_SECTIONS)

    # 2. 类别合规
    cat_match = re.search(r"【事件类别】\s*\n?\s*([^\n【]+)", text)
    if cat_match:
        cat = cat_match.group(1).strip().lower()
        results["valid_category"] = any(v.lower() in cat or cat in v.lower()
                                        for v in VALID_CATEGORIES)
        results["extracted_category"] = cat_match.group(1).strip()
    else:
        results["valid_category"] = False
        results["extracted_category"] = ""

    # 3. 要点格式
    bullet_section = re.search(r"【核心要点】(.*?)(?:【|$)", text, re.DOTALL)
    if bullet_section:
        bullets = re.findall(r"^\s*\d+[\.、．]\s*.+", bullet_section.group(1), re.MULTILINE)
        results["bullet_count"] = len(bullets)
        results["valid_bullets"] = len(bullets) >= 3
    else:
        results["bullet_count"] = 0
        results["valid_bullets"] = False

    # 4. 时间信息存在
    time_match = re.search(r"【时间信息】\s*\n?\s*([^\n【]+)", text)
    results["has_time_info"] = bool(time_match)

    return results


def evaluate_format(predictions: list[str]) -> tuple[dict, list[int]]:
    """批量格式评测。"""
    all_results = [check_format(p) for p in predictions]
    bad_case_indices = []

    aggregated = {
        "total": len(predictions),
        "all_sections_pass_rate": 0.0,
        "valid_category_rate": 0.0,
        "valid_bullets_rate": 0.0,
        "has_time_info_rate": 0.0,
        "avg_bullet_count": 0.0,
        "missing_field_rate": 0.0,
        "invalid_category_rate": 0.0,
    }

    if not all_results:
        return aggregated, bad_case_indices

    n = len(all_results)
    aggregated["all_sections_pass_rate"] = sum(r["all_sections_present"] for r in all_results) / n
    aggregated["valid_category_rate"] = sum(r["valid_category"] for r in all_results) / n
    aggregated["valid_bullets_rate"] = sum(r["valid_bullets"] for r in all_results) / n
    aggregated["has_time_info_rate"] = sum(r["has_time_info"] for r in all_results) / n
    aggregated["avg_bullet_count"] = sum(r["bullet_count"] for r in all_results) / n
    aggregated["missing_field_rate"] = 1.0 - aggregated["all_sections_pass_rate"]
    aggregated["invalid_category_rate"] = 1.0 - aggregated["valid_category_rate"]

    bad_case_indices = [i for i, r in enumerate(all_results)
                        if not r["all_sections_present"] or not r["valid_category"]]

    return aggregated, bad_case_indices


def main():
    parser = argparse.ArgumentParser(description="ROUGE + 格式评测脚本")
    parser.add_argument("--test", type=str, default=str(DEFAULT_TEST),
                        help="测试集路径（含 reference output）")
    parser.add_argument("--predictions", type=str, default=str(DEFAULT_PREDICTIONS),
                        help="模型推理结果路径")
    parser.add_argument("--output_dir", type=str, default=str(EVAL_DIR))
    parser.add_argument("--no_jieba", action="store_true", help="禁用 jieba 分词")
    parser.add_argument("--strip_think", action="store_true",
                        help="评测前剥离 <think>...</think> 块（基座模型推理结果使用）")
    args = parser.parse_args()

    test_path = Path(args.test)
    pred_path = Path(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not test_path.exists():
        print(f"[ERROR] 测试集不存在: {test_path}", file=sys.stderr)
        sys.exit(1)
    if not pred_path.exists():
        print(f"[ERROR] 预测结果不存在: {pred_path}", file=sys.stderr)
        print("请先运行模型推理，或使用 --predictions 指定结果文件。", file=sys.stderr)
        sys.exit(1)

    test_data = load_test_data(test_path)
    predictions = load_predictions(pred_path, strip_think=args.strip_think)
    references = [r.get("output", "") for r in test_data]

    n = min(len(references), len(predictions))
    if len(references) != len(predictions):
        print(f"[WARN] 测试集({len(references)})与预测({len(predictions)})数量不一致，取前 {n} 条")
    references = references[:n]
    predictions = predictions[:n]

    print(f"[INFO] 评测 {n} 条样本...")

    # ROUGE 评测
    print("[INFO] 计算 ROUGE 分数...")
    rouge_scores = compute_rouge(references, predictions, use_jieba=not args.no_jieba)
    print(f"[INFO] ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"[INFO] ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"[INFO] ROUGE-L: {rouge_scores['rougeL']:.4f}")

    rouge_path = output_dir / "rouge_report.json"
    rouge_path.write_text(json.dumps(rouge_scores, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] ROUGE 报告已保存: {rouge_path}")

    # 格式评测
    print("[INFO] 评测格式合规性...")
    format_report, bad_indices = evaluate_format(predictions)
    print(f"[INFO] 格式通过率: {format_report['all_sections_pass_rate']:.2%}")
    print(f"[INFO] 类别合规率: {format_report['valid_category_rate']:.2%}")
    print(f"[INFO] 要点格式率: {format_report['valid_bullets_rate']:.2%}")
    print(f"[INFO] Bad cases 数量: {len(bad_indices)}")

    format_path = output_dir / "format_report.json"
    format_path.write_text(json.dumps(format_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] 格式报告已保存: {format_path}")

    # Bad cases
    bad_cases_path = output_dir / "bad_cases.jsonl"
    with open(bad_cases_path, "w", encoding="utf-8") as f:
        for i in bad_indices:
            bad = {
                "index": i,
                "reference": references[i],
                "prediction": predictions[i],
                "input": test_data[i].get("input", ""),
            }
            f.write(json.dumps(bad, ensure_ascii=False) + "\n")
    print(f"[INFO] Bad cases 已保存: {bad_cases_path} ({len(bad_indices)} 条)")

    print("\n===== 评测完成 =====")
    print(json.dumps({**rouge_scores, **format_report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
