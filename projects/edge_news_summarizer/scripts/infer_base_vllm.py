#!/usr/bin/env python3
"""
infer_base_vllm.py
使用 vLLM 对基座模型（Qwen3-4B）进行批量推理，输出格式与 LlamaFactory 生成的
generated_predictions.jsonl 完全一致，可直接送入 06_eval_rouge_and_format.py 评测。

运行环境：my_vllm（需安装 vllm）
用法：
  python scripts/infer_base_vllm.py
  python scripts/infer_base_vllm.py --model D:/LLM/models/Qwen3-4B --batch_size 16

输出：
  outputs/eval_base_vllm/generated_predictions.jsonl
"""

import argparse
import json
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DEFAULT_MODEL = "D:/LLM/models/Qwen3-4B"
DEFAULT_TEST = PROJECT_DIR / "data" / "cleaned" / "test.json"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "eval_base_vllm"


def load_test_data(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def build_prompt(tokenizer, sample: dict) -> str:
    """使用 Qwen3 tokenizer 的 chat template 构建 prompt。
    开启思考模式（enable_thinking=True），让基座模型自然输出 <think> 块后再给出答案。
    """
    messages = [
        {"role": "system", "content": sample["instruction"]},
        {"role": "user",   "content": sample["input"]},
    ]
    # add_generation_prompt=True：在末尾追加 <|im_start|>assistant\n<think>\n
    # enable_thinking=True：Qwen3 官方 chat_template 支持此参数
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return text


def main():
    parser = argparse.ArgumentParser(description="vLLM 基座模型批量推理")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="模型路径")
    parser.add_argument("--test", type=str, default=str(DEFAULT_TEST),
                        help="测试集路径")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="输出目录")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="每批请求数（vLLM continuous batching 下可设更大值）")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="最大生成 token 数（含 think 块）")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                        help="vLLM 占用显存比例，RTX 16GB 建议 0.85-0.92")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载数据 ──────────────────────────────────────────────
    test_data = load_test_data(Path(args.test))
    print(f"[INFO] 测试集: {len(test_data)} 条")

    # ── 初始化 vLLM ──────────────────────────────────────────
    print(f"[INFO] 加载模型: {args.model}")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,          # prompt + 生成总长度上限
        trust_remote_code=True,
        # 关闭 vLLM 自带的 thinking 处理，让模型自然生成后我们手动剥离
        # （vLLM 0.8+ 对 Qwen3 有原生支持，也可用 ReasoningConfig，但手动剥离更稳定）
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    # ── 构建所有 prompt ───────────────────────────────────────
    print("[INFO] 构建 prompts...")
    prompts = [build_prompt(tokenizer, s) for s in test_data]
    labels  = [s["output"] for s in test_data]

    # ── 批量推理 ─────────────────────────────────────────────
    print(f"[INFO] 开始推理，batch_size={args.batch_size}（vLLM 内部 continuous batching）")
    t0 = time.time()

    # vLLM generate() 接受全量 prompt 列表，内部自动 continuous batching
    outputs = llm.generate(prompts, sampling_params)

    elapsed = time.time() - t0
    print(f"[INFO] 推理完成，耗时 {elapsed/60:.1f} 分钟，"
          f"平均 {elapsed/len(test_data):.2f}s/条")

    # ── 保存结果 ─────────────────────────────────────────────
    out_path = output_dir / "generated_predictions.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for output, label in zip(outputs, labels):
            pred = output.outputs[0].text
            line = json.dumps({"predict": pred, "label": label}, ensure_ascii=False)
            f.write(line + "\n")

    print(f"[INFO] 结果已保存: {out_path}（{len(outputs)} 条）")
    print()
    print("评测命令（在 my_sft 环境中运行）：")
    print(f"  python scripts/06_eval_rouge_and_format.py \\")
    print(f"    --predictions {out_path} \\")
    print(f"    --output_dir {output_dir} \\")
    print(f"    --strip_think")


if __name__ == "__main__":
    main()
