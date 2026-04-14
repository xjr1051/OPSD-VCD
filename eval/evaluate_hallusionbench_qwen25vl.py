import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HallusionBench generation with Qwen2.5-VL.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--base-model-path", type=str, default=None)
    parser.add_argument("--processor-path", type=str, default=None)
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--image-root", type=str, required=True, help="Root directory that contains VD/ and VS/ folders.")
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-dtype", type=str, default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
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


def build_messages(question: str, image: Optional[Image.Image]) -> List[Dict]:
    prompt = (
        question.strip()
        + "\n\nAnswer only with one word: Yes, No, or Uncertain."
        + " Do not add explanations."
    )
    if image is None:
        content = [{"type": "text", "text": prompt}]
    else:
        content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
    return [{"role": "user", "content": content}]


def find_image_candidate(path: Path) -> Optional[Path]:
    if path.exists():
        return path
    parent = path.parent
    if not parent.exists():
        return None

    stem = path.stem.lower()
    ext = path.suffix.lower()
    for cand in parent.iterdir():
        if cand.is_file() and cand.stem.lower() == stem and cand.suffix.lower() == ext:
            return cand

    for cand in parent.iterdir():
        if cand.is_file() and cand.stem.lower() == stem and cand.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            return cand
    return None


def resolve_image_path(image_root: str, item: dict) -> Optional[str]:
    visual_input = str(item.get("visual_input", "0"))
    if visual_input == "0":
        return None

    filename = item.get("filename")
    if filename:
        fn = str(filename).lstrip("./")
        p = find_image_candidate(Path(image_root) / fn)
        if p is not None:
            return str(p)

    category = str(item.get("category", "")).strip()
    subcategory = str(item.get("subcategory", "")).strip()
    set_id = str(item.get("set_id", "")).strip()
    figure_id = str(item.get("figure_id", "")).strip()
    if category and subcategory and set_id and figure_id:
        p2 = find_image_candidate(Path(image_root) / category / subcategory / f"{set_id}_{figure_id}.png")
        if p2 is not None:
            return str(p2)

    raise FileNotFoundError(f"Cannot resolve image for row set_id={set_id}, figure_id={figure_id}")


def parse_prediction_to_label(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)

    if not t:
        return "2"

    if re.search(r"\buncertain\b|\bunknown\b|\bnot sure\b|\bcannot\b|\bcan't\b|\bunable\b", t):
        return "2"

    if re.match(r"^(yes|yeah|yep)\b", t):
        return "1"
    if re.match(r"^(no|nope)\b", t):
        return "0"

    m = re.search(r"\b(yes|no)\b", t)
    if m:
        return "1" if m.group(1) == "yes" else "0"

    return "2"


def run_group(
    model,
    processor,
    rows: List[dict],
    texts: List[str],
    images: Optional[List[Image.Image]],
    do_sample: bool,
    args: argparse.Namespace,
) -> List[str]:
    if not rows:
        return []

    if images is None:
        model_inputs = processor(text=texts, return_tensors="pt", padding=True)
    else:
        model_inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

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
    return [o.strip() for o in outputs]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    rows = json.loads(Path(args.data_file).read_text(encoding="utf-8"))
    for i, row in enumerate(rows):
        row["_orig_index"] = i

    rows = get_chunk(rows, args.num_chunks, args.chunk_idx)
    if not rows:
        raise RuntimeError("No HallusionBench rows found for this chunk.")

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

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    do_sample = args.temperature is not None and args.temperature > 0
    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")

    outputs: List[dict] = []
    missing_image_rows = 0
    for start in tqdm(range(0, len(rows), args.batch_size), desc="HallusionBench generation"):
        batch_rows = rows[start : start + args.batch_size]

        img_group_rows, img_group_texts, img_group_images = [], [], []
        txt_group_rows, txt_group_texts = [], []

        for row in batch_rows:
            q = str(row.get("question", ""))
            try:
                img_path = resolve_image_path(args.image_root, row)
            except FileNotFoundError:
                out = dict(row)
                out["model_prediction"] = "Uncertain"
                out["pred_answer"] = "2"
                out["missing_image"] = True
                outputs.append(out)
                missing_image_rows += 1
                continue
            if img_path is None:
                msgs = build_messages(q, None)
                txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                txt_group_rows.append((row, None))
                txt_group_texts.append(txt)
            else:
                img = Image.open(img_path).convert("RGB")
                msgs = build_messages(q, img)
                txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                img_group_rows.append((row, img_path))
                img_group_texts.append(txt)
                img_group_images.append(img)

        img_outputs = run_group(model, processor, [r for r, _ in img_group_rows], img_group_texts, img_group_images, do_sample, args)
        for img in img_group_images:
            img.close()

        txt_outputs = run_group(model, processor, [r for r, _ in txt_group_rows], txt_group_texts, None, do_sample, args)

        for (row, img_path), pred in zip(img_group_rows, img_outputs):
            out = dict(row)
            out["filename"] = f"./{Path(img_path).relative_to(Path(args.image_root)).as_posix()}"
            out["model_prediction"] = pred
            out["pred_answer"] = parse_prediction_to_label(pred)
            outputs.append(out)

        for (row, _), pred in zip(txt_group_rows, txt_outputs):
            out = dict(row)
            out["model_prediction"] = pred
            out["pred_answer"] = parse_prediction_to_label(pred)
            outputs.append(out)

    outputs = sorted(outputs, key=lambda x: x.get("_orig_index", 10**9))
    out_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] outputs: {out_path}")
    print(f"[info] num_samples={len(outputs)}")
    if missing_image_rows:
        print(f"[warn] missing_image_rows={missing_image_rows}")


if __name__ == "__main__":
    main()
