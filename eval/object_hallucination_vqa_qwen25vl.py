import argparse
import json
import math
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="POPE answer generation with Qwen2.5-VL (official-style output format).")
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
    parser.add_argument("--image-folder", type=str, required=True, help="Folder containing images referenced by POPE questions.")
    parser.add_argument("--question-file", type=str, required=True, help="POPE question file (.json or .jsonl).")
    parser.add_argument("--answers-file", type=str, required=True, help="Output answer file in JSONL.")
    parser.add_argument("--num-chunks", type=int, default=1, help="Split questions into N chunks.")
    parser.add_argument("--chunk-idx", type=int, default=0, help="Current chunk index.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum generated tokens.")
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for generation. Increase to improve GPU utilization.",
    )
    return parser.parse_args()


def load_questions(question_file: str):
    content = Path(question_file).read_text(encoding="utf-8").strip()
    if not content:
        return []
    if content[0] == "[":
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {question_file}")
        return data
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def get_chunk(items, num_chunks: int, chunk_idx: int):
    if num_chunks <= 0:
        raise ValueError("num_chunks must be > 0")
    if chunk_idx < 0 or chunk_idx >= num_chunks:
        raise ValueError(f"chunk_idx must be in [0, {num_chunks - 1}]")
    if not items:
        return []
    chunk_size = int(math.ceil(len(items) / float(num_chunks)))
    start = chunk_idx * chunk_size
    end = min(len(items), start + chunk_size)
    return items[start:end]


def resolve_dtype(torch_dtype: str):
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[torch_dtype]


def load_model(model_path: str, base_model_path: str, torch_dtype, attn_implementation: str):
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


def build_prompt(processor, image: Image.Image, question: str) -> dict:
    prompt = question.strip() + " Please answer this question with one word."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    return prompt, model_inputs


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


def move_to_device(batch: dict):
    if torch.cuda.is_available():
        return {k: v.to("cuda") if hasattr(v, "to") else v for k, v in batch.items()}
    return {k: v.to("cpu") if hasattr(v, "to") else v for k, v in batch.items()}


def main():
    args = parse_args()
    set_seed(args.seed)

    questions = load_questions(args.question_file)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

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

    answers_path = Path(args.answers_file)
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    model_id = os.path.basename(os.path.normpath(args.model_path))

    do_sample = args.temperature is not None and args.temperature > 0
    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")

    with answers_path.open("w", encoding="utf-8") as writer:
        for start in tqdm(range(0, len(questions), args.batch_size), desc="Generating POPE answers"):
            batch_samples = questions[start : start + args.batch_size]

            qids = []
            image_files = []
            prompts = []
            images = []
            texts = []

            for sample in batch_samples:
                qid = sample.get("question_id")
                image_file = sample.get("image")
                question = sample.get("text") or sample.get("question") or sample.get("prompt")
                if qid is None or image_file is None or question is None:
                    raise ValueError(f"Malformed question sample: {sample}")

                prompt = question.strip() + " Please answer this question with one word."
                image_path = os.path.join(args.image_folder, image_file)
                image = Image.open(image_path).convert("RGB")

                qids.append(qid)
                image_files.append(image_file)
                prompts.append(prompt)
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
                generate_kwargs.update(
                    {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                    }
                )
                if args.top_k is not None:
                    generate_kwargs["top_k"] = args.top_k

            with torch.inference_mode():
                output_ids = model.generate(**model_inputs, **generate_kwargs)

            outputs = processor.batch_decode(
                output_ids[:, input_token_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for qid, image_file, prompt, text in zip(qids, image_files, prompts, outputs):
                writer.write(
                    json.dumps(
                        {
                            "question_id": qid,
                            "prompt": prompt,
                            "text": text.strip(),
                            "model_id": model_id,
                            "image": image_file,
                            "metadata": {},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
