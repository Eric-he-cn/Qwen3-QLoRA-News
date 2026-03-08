#!/usr/bin/env python3
"""
07_benchmark_latency.py
对本地模型进行推理性能评测：平均延迟、P50/P95 延迟、显存占用。

用法：
  python scripts/07_benchmark_latency.py --model_path ./models/Qwen2.5-3B-Instruct
  python scripts/07_benchmark_latency.py --model_path ./outputs/merged/qwen25-3b-news --num_samples 50
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
EVAL_DIR = PROJECT_DIR / "outputs" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TEST = PROJECT_DIR / "data" / "cleaned" / "test.json"
LATENCY_REPORT = EVAL_DIR / "latency_report.json"

SYSTEM_PROMPT = "你是一位专业的新闻编辑助手，请对新闻进行结构化摘要分析。"


def load_test_samples(path: Path, num_samples: int) -> list[dict]:
    """加载测试样本，返回指定数量的前缀切片。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data[:num_samples]
    return [data]


def build_prompt(record: dict) -> str:
    """从样本中提取输入字段作为用户 prompt。"""
    return record.get("input", "")


def run_benchmark(model_path: str, adapter_path: str | None, test_samples: list[dict],
                  max_new_tokens: int, device: str) -> dict:
    """加载模型并运行推理延迟评测。"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[ERROR] 请先安装 transformers 和 torch", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"[INFO] 加载模型: {model_path}")
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": device if device != "cpu" else None,
    }
    if device == "cuda":
        load_kwargs["torch_dtype"] = torch.float16
    elif device == "cpu":
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    # 加载 LoRA adapter（如果提供）
    if adapter_path:
        try:
            from peft import PeftModel
            print(f"[INFO] 加载 LoRA adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
            print("[INFO] LoRA 权重已合并")
        except ImportError:
            print("[ERROR] 请先安装 peft: pip install peft", file=sys.stderr)
            sys.exit(1)

    model.eval()

    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
        torch.cuda.reset_peak_memory_stats()

    latencies = []
    outputs = []

    print(f"\n[INFO] 开始评测 {len(test_samples)} 条样本 (max_new_tokens={max_new_tokens})...")
    print("-" * 60)

    for i, sample in enumerate(test_samples, 1):
        prompt = build_prompt(sample)
        if not prompt:
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # 使用 chat template
        try:
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt"
            )
        except Exception:
            text = f"{SYSTEM_PROMPT}\n\n{prompt}"
            input_ids = tokenizer(text, return_tensors="pt").input_ids

        if device == "cuda" and torch.cuda.is_available():
            input_ids = input_ids.to("cuda")

        start = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        end = time.perf_counter()
        elapsed = end - start
        latencies.append(elapsed)

        output_ids = generated[0][input_ids.shape[-1]:]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        outputs.append(output_text)

        print(f"[{i:3d}/{len(test_samples)}] 耗时: {elapsed:.3f}s | 输入 tokens: {input_ids.shape[-1]}")

    # 统计
    if not latencies:
        print("[ERROR] 没有有效的推理结果", file=sys.stderr)
        sys.exit(1)

    latencies_sorted = sorted(latencies)

    def percentile(sorted_data, p):
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]

    report = {
        "model_path": model_path,
        "adapter_path": adapter_path,
        "num_samples": len(latencies),
        "max_new_tokens": max_new_tokens,
        "device": device,
        "avg_latency_s": statistics.mean(latencies),
        "median_latency_s": statistics.median(latencies),
        "p50_latency_s": percentile(latencies_sorted, 50),
        "p95_latency_s": percentile(latencies_sorted, 95),
        "min_latency_s": min(latencies),
        "max_latency_s": max(latencies),
        "stddev_latency_s": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
    }

    # 显存统计
    if device == "cuda":
        try:
            import torch
            peak_mem_bytes = torch.cuda.max_memory_allocated()
            report["peak_gpu_memory_mb"] = peak_mem_bytes / 1024 / 1024
            print(f"\n[INFO] 峰值显存: {report['peak_gpu_memory_mb']:.1f} MB")
        except Exception:
            pass

    return report


def main():
    """脚本入口。"""
    parser = argparse.ArgumentParser(description="推理延迟评测脚本")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径（基座模型或已合并权重）")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="LoRA adapter 路径（可选，不合并时使用）")
    parser.add_argument("--test", type=str, default=str(DEFAULT_TEST),
                        help="测试集路径")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="评测样本数（默认: 20）")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="最大生成 token 数（默认: 512）")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="推理设备（默认: cuda）")
    parser.add_argument("--output", type=str, default=str(LATENCY_REPORT))
    args = parser.parse_args()

    test_path = Path(args.test)
    if not test_path.exists():
        print(f"[ERROR] 测试集不存在: {test_path}", file=sys.stderr)
        sys.exit(1)

    test_samples = load_test_samples(test_path, args.num_samples)
    print(f"[INFO] 加载 {len(test_samples)} 条测试样本")

    report = run_benchmark(
        args.model_path, args.adapter_path, test_samples,
        args.max_new_tokens, args.device
    )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n===== 延迟评测报告 =====")
    print(f"平均延迟:   {report['avg_latency_s']:.3f}s")
    print(f"P50 延迟:   {report['p50_latency_s']:.3f}s")
    print(f"P95 延迟:   {report['p95_latency_s']:.3f}s")
    print(f"最大延迟:   {report['max_latency_s']:.3f}s")
    if "peak_gpu_memory_mb" in report:
        print(f"峰值显存:   {report['peak_gpu_memory_mb']:.1f} MB")
    print(f"\n报告已保存: {output_path}")


if __name__ == "__main__":
    main()
