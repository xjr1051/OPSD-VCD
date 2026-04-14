import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MMHal-Bench response generation with Qwen2.5-VL.")
    parser.add_argument("--model-path", type=str, required=True, help="Model path or adapter path.")
    parser.add_argument(
        "--base-model-path",
        type=str,
        default=None,
        help="Optional base model path when --model-path points to a PEFT adapter.",
    )
    parser.add_argument(
        "--processor-path",
        type=str,
        default=None,
        help="Optional processor/tokenizer path. Defaults to base-model-path or model-path.",
    )
    parser.add_argument("--template-file", type=str, required=True, help="MMHal response_template.json path.")
    parser.add_argument("--image-folder", type=str, required=True, help="MMHal images folder.")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSON with model answers.")
    parser.add_argument("--summary-file", type=str, required=True, help="Output summary JSON.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated tokens.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="float16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for model loading.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        help="Attention implementation passed to from_pretrained.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for generation.")
    parser.add_argument("--num-chunks", type=int, default=1, help="Split all samples into N chunks.")
    parser.add_argument("--chunk-idx", type=int, default=0, help="Current chunk index.")
    return parser.parse_args()


def resolve_dtype(torch_dtype: str):
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[torch_dtype]


def load_model(model_path: str, base_model_path: Optional[str], torch_dtype, attn_implementation: str):
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    if base_model_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("peft is required when --base-model-path is provided.") from exc

        base_model = AutoModelForImageTextToText.from_pretrained(base_model_path, **model_kwargs)
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)

    model.eval()
    return model


def get_chunk(items: List[dict], num_chunks: int, chunk_idx: int) -> List[dict]:
    if num_chunks <= 0:
        raise ValueError("num-chunks must be > 0")
    if chunk_idx < 0 or chunk_idx >= num_chunks:
        raise ValueError(f"chunk-idx must be in [0, {num_chunks - 1}]")
    if not items:
        return []
    chunk_size = (len(items) + num_chunks - 1) // num_chunks
    start = chunk_idx * chunk_size
    end = min(len(items), start + chunk_size)
    return items[start:end]


def move_to_device(batch: dict):
    if torch.cuda.is_available():
        return {k: v.to("cuda") if hasattr(v, "to") else v for k, v in batch.items()}
    return {k: v.to("cpu") if hasattr(v, "to") else v for k, v in batch.items()}


def build_prompt_text(processor, image: Image.Image, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def resolve_image_path(image_folder: str, item: dict) -> str:
    image_path = item.get("image_path")
    if image_path:
        p = Path(image_path)
        if p.exists():
            return str(p)
        p2 = Path(image_folder) / image_path
        if p2.exists():
            return str(p2)

    image_src = item.get("image_src", "")
    filename = os.path.basename(image_src)
    p3 = Path(image_folder) / filename
    if p3.exists():
        return str(p3)

    raise FileNotFoundError(f"Cannot resolve image for item with image_src={image_src}")


def summarize_outputs(rows: List[dict]) -> Dict:
    by_type: Dict[str, int] = {}
    for row in rows:
        t = row.get("question_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    avg_len = 0.0
    if rows:
        avg_len = sum(len((r.get("model_answer") or "").split()) for r in rows) / len(rows)

    return {
        "num_samples": len(rows),
        "avg_response_words": avg_len,
        "question_type_counts": dict(sorted(by_type.items())),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    template = json.loads(Path(args.template_file).read_text(encoding="utf-8"))
    samples = get_chunk(template, args.num_chunks, args.chunk_idx)
    if not samples:
        raise RuntimeError("No MMHal samples found for this chunk.")

    processor_path = args.processor_path or args.base_model_path or args.model_path
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "padding_side"):
        processor.tokenizer.padding_side = "left"
    elif hasattr(processor, "padding_side"):
        processor.padding_side = "left"

    model = load_model(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        torch_dtype=resolve_dtype(args.torch_dtype),
        attn_implementation=args.attn_implementation,
    )

    output_path = Path(args.output_file)
    summary_path = Path(args.summary_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    do_sample = args.temperature is not None and args.temperature > 0
    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")

    out_rows: List[dict] = []
    for start in tqdm(range(0, len(samples), args.batch_size), desc="Generating MMHal answers"):
        batch = samples[start : start + args.batch_size]

        prompts = []
        texts = []
        images = []
        image_paths = []

        for item in batch:
            question = (item.get("question") or "").strip()
            prompt = question
            image_path = resolve_image_path(args.image_folder, item)
            image = Image.open(image_path).convert("RGB")

            prompts.append(prompt)
            image_paths.append(image_path)
            images.append(image)
            texts.append(build_prompt_text(processor, image, prompt))

        model_inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        for image in images:
            image.close()

        model_inputs = move_to_device(model_inputs)
        input_token_len = model_inputs["input_ids"].shape[1]

        generate_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
            if args.top_k is not None:
                generate_kwargs["top_k"] = args.top_k

        with torch.inference_mode():
            output_ids = model.generate(**model_inputs, **generate_kwargs)

        outputs = processor.batch_decode(
            output_ids[:, input_token_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for item, image_path, answer in zip(batch, image_paths, outputs):
            out_item = dict(item)
            out_item["image_path"] = image_path
            out_item["model_answer"] = answer.strip()
            out_rows.append(out_item)

    output_path.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = summarize_outputs(out_rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[done] outputs: {output_path}")
    print(f"[done] summary: {summary_path}")
    print(f"[info] num_samples={summary['num_samples']}, avg_response_words={summary['avg_response_words']:.2f}")


if __name__ == "__main__":
    main()
