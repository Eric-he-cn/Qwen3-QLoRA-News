#!/usr/bin/env python3
"""
08_demo_cli.py
新闻结构化摘要 CLI Demo。

功能：
  - 交互式或批量模式
  - 支持基座模型 / 微调模型（自动加载 LoRA adapter）
  - 支持 --compare 模式：同一输入同时跑基座模型（含系统提示词）和微调模型，逐条对比输出
  - 输入：新闻标题 + 正文（命令行交互 / txt / json 文件）
  - 输出：结构化摘要（终端显示 + 可选保存）
  - 显示推理耗时

用法（交互模式，仅基座模型）：
  python scripts/08_demo_cli.py --model_path D:/LLM/models/Qwen3-4B

用法（微调模型）：
  python scripts/08_demo_cli.py \
    --model_path D:/LLM/models/Qwen3-4B \
        --adapter_path outputs/checkpoints/qwen3-4b-qlora-news-v2

用法（对比模式：基座 vs 微调，交互式）：
  python scripts/08_demo_cli.py \
    --model_path D:/LLM/models/Qwen3-4B \
        --adapter_path outputs/checkpoints/qwen3-4b-qlora-news-v2 \
    --compare

用法（对比模式：批量处理并保存对比结果）：
  python scripts/08_demo_cli.py \
    --model_path D:/LLM/models/Qwen3-4B \
        --adapter_path outputs/checkpoints/qwen3-4b-qlora-news-v2 \
    --compare \
    --input_file data/cleaned/test.json \
    --output_file outputs/eval/compare_outputs.jsonl \
    --num_samples 50
"""

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
EVAL_DIR = PROJECT_DIR / "outputs" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_TEMPLATE_FILE = PROJECT_DIR / "data" / "prompts" / "label_prompt_news_structured.txt"

SYSTEM_PROMPT = "你是一位专业的新闻编辑助手，请对新闻进行结构化摘要分析。"


def load_prompt_template() -> str:
    """加载用户提示模板，不存在时使用内置兜底模板。"""
    if PROMPT_TEMPLATE_FILE.exists():
        return PROMPT_TEMPLATE_FILE.read_text(encoding="utf-8")
    return (
        "新闻标题：{title}\n\n新闻正文：\n{content}\n\n"
        "请按以下格式输出结构化摘要：\n"
        "【一句话摘要】\n【核心要点】\n【事件类别】\n【主要主体】\n【时间信息】\n【潜在影响】"
    )


def load_model(model_path: str, adapter_path: str | None = None,
               device: str = "cuda", quantize: bool = False):
    """加载模型和 tokenizer。"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[ERROR] 请先安装 transformers: pip install transformers torch", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"[INFO] 加载模型: {model_path}")
    load_kwargs = {"trust_remote_code": True}

    if device == "cuda":
        import torch
        if quantize:
            try:
                load_kwargs["quantization_config"] = __import__(
                    "transformers").BitsAndBytesConfig(load_in_4bit=True)
                load_kwargs["device_map"] = "auto"
            except Exception as e:
                print(f"[WARN] 4-bit 量化失败: {e}，回退到 bfloat16")
                # Qwen3 原生 BF16 训练，使用 bfloat16 而不是 float16
                # float16 在长序列/大 embedding 下容易引发 NaN
                load_kwargs["torch_dtype"] = torch.bfloat16
                load_kwargs["device_map"] = "auto"
        else:
            # Qwen3 必须使用 bfloat16，不得改为 float16
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = __import__("torch").float32

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    if adapter_path:
        try:
            from peft import PeftModel
            print(f"[INFO] 加载 LoRA adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            print("[INFO] LoRA adapter 加载成功")
        except ImportError:
            print("[ERROR] 请先安装 peft: pip install peft", file=sys.stderr)
            sys.exit(1)

    model.eval()
    print("[INFO] 模型加载完成！\n")
    return model, tokenizer


def generate_summary(model, tokenizer, title: str, content: str,
                     prompt_template: str, max_new_tokens: int = 512,
                     temperature: float = 0.1,
                     enable_thinking: bool = False) -> tuple[str, float]:
    """生成结构化摘要，返回（摘要文本, 耗时秒）。"""
    import torch

    # 构建输入
    content_truncated = content[:3000] if len(content) > 3000 else content
    user_content = prompt_template.format(title=title, content=content_truncated)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
        )
    except Exception:
        text = f"{SYSTEM_PROMPT}\n\n{user_content}"
        input_ids = tokenizer(text, return_tensors="pt").input_ids

    # 检测设备
    try:
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
    except Exception:
        pass

    start = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.01,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    elapsed = time.perf_counter() - start

    output_ids = generated[0][input_ids.shape[-1]:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output_text.strip(), elapsed


def print_result(title: str, output: str, elapsed: float) -> None:
    """打印单模型推理结果。"""
    print("\n" + "=" * 60)
    print(f"新闻标题：{title[:60]}")
    print("=" * 60)
    print(output)
    print("-" * 60)
    print(f"⏱ 推理耗时：{elapsed:.3f}s")
    print("=" * 60 + "\n")


def interactive_mode(model, tokenizer, prompt_template: str,
                     max_new_tokens: int, temperature: float,
                     enable_thinking: bool, save_path: Path | None) -> None:
    """交互式 CLI 模式。"""
    print("===== 新闻结构化摘要 Demo =====")
    print("输入 'quit' 或 'exit' 退出，输入 'clear' 清屏\n")

    results = []

    while True:
        title = input("请输入新闻标题（或 'quit' 退出）：").strip()
        if title.lower() in ("quit", "exit", "q"):
            break
        if title.lower() == "clear":
            print("\033[2J\033[H")
            continue

        print("请输入新闻正文（输入空行结束）：")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        content = "\n".join(lines).strip()

        if not content:
            print("[WARN] 正文为空，跳过。\n")
            continue

        print("\n[INFO] 正在生成摘要...")
        output, elapsed = generate_summary(
            model, tokenizer, title, content, prompt_template,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
        print_result(title, output, elapsed)

        result = {"title": title, "content": content[:500], "output": output, "elapsed_s": elapsed}
        results.append(result)

        if save_path:
            with open(save_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\n共生成 {len(results)} 条摘要。")
    if save_path:
        print(f"结果已保存: {save_path}")


def batch_mode(model, tokenizer, prompt_template: str, input_file: Path,
               output_file: Path, num_samples: int, max_new_tokens: int,
               temperature: float, enable_thinking: bool) -> None:
    """批量处理模式。"""
    with open(input_file, encoding="utf-8") as f:
        if input_file.suffix == ".json":
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
        else:
            data = [json.loads(line) for line in f if line.strip()]

    if num_samples > 0:
        data = data[:num_samples]

    print(f"[INFO] 批量处理 {len(data)} 条记录...")

    with open(output_file, "w", encoding="utf-8") as out_f:
        for i, record in enumerate(data, 1):
            title = record.get("title", "")
            content = record.get("content", record.get("input", ""))

            # 如果是 Alpaca 格式，从 input 字段提取
            if not title and "input" in record:
                input_text = record["input"]
                # 尝试提取标题
                lines = input_text.split("\n")
                for line in lines:
                    if line.startswith("新闻标题："):
                        title = line.replace("新闻标题：", "").strip()
                    elif line.startswith("新闻正文："):
                        content_start = input_text.find("新闻正文：")
                        content = input_text[content_start + 5:].strip() if content_start >= 0 else input_text

            if not content:
                continue

            print(f"[{i:3d}/{len(data)}] 处理: {title[:50] or '(无标题)'}")
            output, elapsed = generate_summary(
                model, tokenizer, title, content, prompt_template,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                enable_thinking=enable_thinking,
            )

            result = {
                "id": record.get("id", f"batch_{i}"),
                "title": title,
                "reference": record.get("output", ""),
                "prediction": output,
                "elapsed_s": elapsed,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

            if i % 10 == 0:
                out_f.flush()

    print(f"\n[INFO] 批量处理完成！结果已保存: {output_file}")


def print_compare(title: str, base_out: str, ft_out: str,
                  base_elapsed: float, ft_elapsed: float) -> None:
    """并排打印基座模型与微调模型的对比输出。"""
    sep = "=" * 70
    half = "-" * 70
    print(f"\n{sep}")
    print(f"新闻标题：{title[:80]}")
    print(sep)
    print(f"【基座模型（Base + 系统提示词）】  耗时: {base_elapsed:.3f}s")
    print(half)
    print(base_out if base_out else "(无输出)")
    print(sep)
    print(f"【微调模型（Fine-tuned + 系统提示词）】  耗时: {ft_elapsed:.3f}s")
    print(half)
    print(ft_out if ft_out else "(无输出)")
    print(sep + "\n")


def compare_batch_mode(base_model, base_tokenizer,
                       ft_model, ft_tokenizer,
                       prompt_template: str, input_file: Path,
                       output_file: Path, num_samples: int,
                       max_new_tokens: int, temperature: float,
                       enable_thinking: bool) -> None:
    """对比批量模式：同一输入分别跑基座和微调模型，输出对比结果。"""
    with open(input_file, encoding="utf-8") as f:
        if input_file.suffix == ".json":
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
        else:
            data = [json.loads(line) for line in f if line.strip()]

    if num_samples > 0:
        data = data[:num_samples]

    print(f"[INFO] 对比模式：处理 {len(data)} 条记录（基座 vs 微调）...")

    with open(output_file, "w", encoding="utf-8") as out_f:
        for i, record in enumerate(data, 1):
            title = record.get("title", "")
            content = record.get("content", record.get("input", ""))
            if not title and "input" in record:
                input_text = record["input"]
                for line in input_text.split("\n"):
                    if line.startswith("新闻标题："):
                        title = line.replace("新闻标题：", "").strip()
                content_start = input_text.find("新闻正文：")
                if content_start >= 0:
                    content = input_text[content_start + 5:].strip()
            if not content:
                continue

            print(f"[{i:3d}/{len(data)}] 处理: {title[:50] or '(无标题)'}")

            base_out, base_t = generate_summary(
                base_model, base_tokenizer, title, content,
                prompt_template,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                enable_thinking=enable_thinking,
            )
            ft_out, ft_t = generate_summary(
                ft_model, ft_tokenizer, title, content,
                prompt_template,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                enable_thinking=enable_thinking,
            )

            print_compare(title, base_out, ft_out, base_t, ft_t)

            result = {
                "id": record.get("id", f"compare_{i}"),
                "title": title,
                "reference": record.get("output", ""),
                "base_prediction": base_out,
                "ft_prediction": ft_out,
                "base_elapsed_s": base_t,
                "ft_elapsed_s": ft_t,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            if i % 10 == 0:
                out_f.flush()

    print(f"\n[INFO] 对比完成！结果已保存: {output_file}")


def compare_interactive_mode(base_model, base_tokenizer,
                              ft_model, ft_tokenizer,
                              prompt_template: str,
                              max_new_tokens: int,
                              temperature: float,
                              enable_thinking: bool,
                              save_path: Path | None) -> None:
    """对比交互式模式。"""
    print("===== 新闻摘要对比 Demo（Base vs Fine-tuned）=====")
    print("输入 'quit' 退出\n")
    results = []
    while True:
        title = input("请输入新闻标题（或 'quit' 退出）：").strip()
        if title.lower() in ("quit", "exit", "q"):
            break
        print("请输入新闻正文（空行结束）：")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        content = "\n".join(lines).strip()
        if not content:
            print("[WARN] 正文为空，跳过。\n")
            continue

        print("\n[INFO] 生成基座模型输出...")
        base_out, base_t = generate_summary(
            base_model, base_tokenizer, title, content, prompt_template,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
        print("[INFO] 生成微调模型输出...")
        ft_out, ft_t = generate_summary(
            ft_model, ft_tokenizer, title, content, prompt_template,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )

        print_compare(title, base_out, ft_out, base_t, ft_t)

        result = {"title": title, "content": content[:500],
                  "base_prediction": base_out, "ft_prediction": ft_out,
                  "base_elapsed_s": base_t, "ft_elapsed_s": ft_t}
        results.append(result)
        if save_path:
            with open(save_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\n共生成 {len(results)} 条对比结果。")
    if save_path:
        print(f"结果已保存: {save_path}")


def main():
    """脚本入口。"""
    parser = argparse.ArgumentParser(description="新闻结构化摘要 CLI Demo")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径（对比模式下同时作为基座模型和微调模型的基础）")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="LoRA adapter 路径（微调模型，对比模式必须提供）")
    parser.add_argument("--compare", action="store_true",
                        help="对比模式：同时运行基座模型（无adapter）和微调模型（有adapter），逐条对比输出")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--quantize", action="store_true", help="使用 4-bit 量化（需要 bitsandbytes）")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--enable_thinking", action="store_true",
                        help="启用 thinking 模式（会显著增加时延）")

    # 批量模式参数
    parser.add_argument("--input_file", type=str, default=None,
                        help="批量输入文件路径（JSON 或 JSONL）")
    parser.add_argument("--output_file", type=str,
                        default=str(EVAL_DIR / "demo_outputs.jsonl"),
                        help="批量输出文件路径")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="批量模式处理样本数（0 表示全部）")

    args = parser.parse_args()

    prompt_template = load_prompt_template()

    # ── 对比模式 ──────────────────────────────────────────────
    if args.compare:
        if not args.adapter_path:
            print("[ERROR] --compare 模式需要同时提供 --adapter_path", file=sys.stderr)
            sys.exit(1)

        print("[INFO] 对比模式：加载基座模型（无 adapter）...")
        base_model, base_tokenizer = load_model(
            args.model_path, adapter_path=None,
            device=args.device, quantize=args.quantize)

        print("[INFO] 对比模式：加载微调模型（含 adapter）...")
        ft_model, ft_tokenizer = load_model(
            args.model_path, adapter_path=args.adapter_path,
            device=args.device, quantize=args.quantize)

        if args.input_file:
            input_path = Path(args.input_file)
            if not input_path.exists():
                print(f"[ERROR] 输入文件不存在: {input_path}", file=sys.stderr)
                sys.exit(1)
            compare_batch_mode(
                base_model, base_tokenizer,
                ft_model, ft_tokenizer,
                prompt_template, input_path,
                Path(args.output_file), args.num_samples, args.max_new_tokens,
                args.temperature, args.enable_thinking,
            )
        else:
            compare_interactive_mode(
                base_model, base_tokenizer,
                ft_model, ft_tokenizer,
                prompt_template, args.max_new_tokens,
                args.temperature, args.enable_thinking,
                Path(args.output_file) if args.output_file else None)
        return

    # ── 普通模式 ──────────────────────────────────────────────
    model, tokenizer = load_model(args.model_path, args.adapter_path,
                                  args.device, args.quantize)

    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"[ERROR] 输入文件不存在: {input_path}", file=sys.stderr)
            sys.exit(1)
        batch_mode(model, tokenizer, prompt_template, input_path,
                   Path(args.output_file), args.num_samples, args.max_new_tokens,
                   args.temperature, args.enable_thinking)
    else:
        interactive_mode(model, tokenizer, prompt_template,
                         args.max_new_tokens,
                         args.temperature,
                         args.enable_thinking,
                         Path(args.output_file) if args.output_file else None)


if __name__ == "__main__":
    main()
