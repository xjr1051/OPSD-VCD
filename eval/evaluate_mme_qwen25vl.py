import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"]

PERCEPTION_CATEGORIES = {
    "existence",
    "count",
    "position",
    "color",
    "poster",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "ocr",
}

COGNITION_CATEGORIES = {
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MME evaluation with Qwen2.5-VL.")
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
    parser.add_argument(
        "--mme-root",
        type=str,
        default="data/MME/MME_Benchmark_release_version",
        help="MME root directory (or its parent containing MME_Benchmark_release_version).",
    )
    parser.add_argument("--answers-file", type=str, required=True, help="Output predictions JSONL file.")
    parser.add_argument("--metrics-file", type=str, required=True, help="Output metrics JSON file.")
    parser.add_argument("--summary-file", type=str, required=True, help="Output markdown summary file.")
    parser.add_argument(
        "--categories",
        type=str,
        default="",
        help="Comma-separated categories to run. Empty means all categories.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling.")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Maximum generated tokens.")
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


def canonical_category_name(name: str) -> str:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key.startswith("poster"):
        return "poster"
    if key in {"commonsense", "commonsense_reason"}:
        return "commonsense_reasoning"
    if key in {"numerical", "numerical_reasoning"}:
        return "numerical_calculation"
    if key in {"translation", "text_translate"}:
        return "text_translation"
    if key in {"code", "coding_reasoning"}:
        return "code_reasoning"
    return key


def resolve_mme_root(mme_root: str) -> Path:
    root = Path(mme_root)
    if not root.exists():
        raise FileNotFoundError(f"MME root does not exist: {mme_root}")
    if (root / "MME_Benchmark_release_version").is_dir():
        root = root / "MME_Benchmark_release_version"
    if (root / "MME_Benchmark").is_dir():
        root = root / "MME_Benchmark"
    return root


def list_category_dirs(mme_root: Path, selected_categories: Optional[set]) -> List[Path]:
    dirs = []
    for child in sorted(mme_root.iterdir()):
        if not child.is_dir():
            continue
        has_questions = (child / "questions_answers_YN").is_dir() or any(child.glob("*.txt"))
        if not has_questions:
            continue

        canonical = canonical_category_name(child.name)
        if selected_categories and canonical not in selected_categories:
            continue
        dirs.append(child)
    return dirs


def normalize_yes_no(text: str) -> str:
    t = text.strip().lower()
    if not t:
        return "unknown"

    m = re.search(r"\b(yes|no)\b", t)
    if m:
        return m.group(1)

    if t.startswith("yes") or t.startswith("y"):
        return "yes"
    if t.startswith("no") or t.startswith("n"):
        return "no"
    return "unknown"


def parse_qa_line(line: str) -> Tuple[Optional[str], Optional[str]]:
    raw = line.strip()
    if not raw:
        return None, None

    parts = [p.strip() for p in raw.split("\t") if p.strip()]
    if len(parts) >= 2:
        question, gt = parts[0], normalize_yes_no(parts[1])
        if gt in {"yes", "no"}:
            return question, gt

    m = re.match(r"^(.*?)(?:\s+)(yes|no)\s*$", raw, flags=re.IGNORECASE)
    if m:
        question = m.group(1).strip()
        gt = normalize_yes_no(m.group(2))
        if gt in {"yes", "no"}:
            return question, gt

    return None, None


def find_image_path(category_dir: Path, image_hint: Optional[str], stem_hint: Optional[str]) -> Path:
    candidates = []
    if image_hint:
        candidates.append(category_dir / image_hint)
        candidates.append(category_dir / "images" / image_hint)

    if stem_hint:
        for ext in IMAGE_EXTS:
            candidates.append(category_dir / f"{stem_hint}{ext}")
            candidates.append(category_dir / "images" / f"{stem_hint}{ext}")

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Unable to resolve image in category={category_dir.name}, image_hint={image_hint}, stem_hint={stem_hint}"
    )


def load_mme_samples(mme_root: Path, selected_categories: Optional[set]) -> List[dict]:
    samples: List[dict] = []
    qid = 0

    for category_dir in list_category_dirs(mme_root, selected_categories):
        category_name = canonical_category_name(category_dir.name)
        qa_dir = category_dir / "questions_answers_YN"

        if qa_dir.is_dir():
            txt_files = sorted(qa_dir.glob("*.txt"))
            for txt_file in txt_files:
                pair_key = txt_file.stem
                image_path = find_image_path(category_dir, image_hint=None, stem_hint=pair_key)

                lines = [line.strip() for line in txt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                for idx, line in enumerate(lines):
                    question, gt = parse_qa_line(line)
                    if not question or not gt:
                        continue
                    samples.append(
                        {
                            "question_id": qid,
                            "category": category_name,
                            "pair_key": f"{category_name}:{pair_key}",
                            "pair_idx": idx,
                            "question": question,
                            "gt": gt,
                            "image_path": str(image_path),
                        }
                    )
                    qid += 1
            continue

        txt_files = sorted(category_dir.glob("*.txt"))

        # MME cognition categories often store one txt per image under the category root.
        paired_txt = []
        for txt_file in txt_files:
            try:
                image_path = find_image_path(category_dir, image_hint=None, stem_hint=txt_file.stem)
                paired_txt.append((txt_file, image_path))
            except FileNotFoundError:
                pass

        if paired_txt:
            for txt_file, image_path in paired_txt:
                pair_key = txt_file.stem
                lines = [line.strip() for line in txt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                for idx, line in enumerate(lines):
                    question, gt = parse_qa_line(line)
                    if not question or not gt:
                        continue
                    samples.append(
                        {
                            "question_id": qid,
                            "category": category_name,
                            "pair_key": f"{category_name}:{pair_key}",
                            "pair_idx": idx,
                            "question": question,
                            "gt": gt,
                            "image_path": str(image_path),
                        }
                    )
                    qid += 1
            continue

        for txt_file in txt_files:
            lines = [line.strip() for line in txt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            for idx, line in enumerate(lines):
                parts = [p.strip() for p in line.split("\t")]
                if len(parts) < 3:
                    continue
                image_name, question, gt = parts[0], parts[1], parts[2]
                gt = normalize_yes_no(gt)
                if gt not in {"yes", "no"}:
                    continue
                image_path = find_image_path(category_dir, image_hint=image_name, stem_hint=Path(image_name).stem)
                pair_key = f"{category_name}:{Path(image_name).stem}"
                samples.append(
                    {
                        "question_id": qid,
                        "category": category_name,
                        "pair_key": pair_key,
                        "pair_idx": idx,
                        "question": question,
                        "gt": gt,
                        "image_path": str(image_path),
                    }
                )
                qid += 1

    return samples


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


def evaluate_samples(args: argparse.Namespace, samples: List[dict]) -> List[dict]:
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

    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")

    do_sample = args.temperature is not None and args.temperature > 0
    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
        if args.top_k is not None:
            generate_kwargs["top_k"] = args.top_k

    outputs: List[dict] = []
    for start in tqdm(range(0, len(samples), args.batch_size), desc="Generating MME answers"):
        batch_samples = samples[start : start + args.batch_size]

        images = []
        texts = []
        prompts = []
        for sample in batch_samples:
            prompt = sample["question"].strip() + " Please answer yes or no."
            image = Image.open(sample["image_path"]).convert("RGB")
            prompts.append(prompt)
            images.append(image)
            texts.append(build_prompt_text(processor, image, prompt))

        model_inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        for image in images:
            image.close()

        model_inputs = move_to_device(model_inputs)
        input_token_len = model_inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = model.generate(**model_inputs, **generate_kwargs)

        decoded = processor.batch_decode(
            output_ids[:, input_token_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for sample, prompt, pred_text in zip(batch_samples, prompts, decoded):
            pred_norm = normalize_yes_no(pred_text)
            outputs.append(
                {
                    "question_id": sample["question_id"],
                    "category": sample["category"],
                    "pair_key": sample["pair_key"],
                    "pair_idx": sample["pair_idx"],
                    "image": os.path.basename(sample["image_path"]),
                    "prompt": prompt,
                    "gt": sample["gt"],
                    "pred": pred_norm,
                    "raw_text": pred_text.strip(),
                    "correct": int(pred_norm == sample["gt"]),
                }
            )

    return outputs


def compute_mme_metrics(records: List[dict]) -> Dict:
    cat_records: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        cat_records[r["category"]].append(r)

    per_category = {}
    for category, rows in sorted(cat_records.items()):
        total = len(rows)
        correct = sum(r["correct"] for r in rows)
        acc = (correct / total * 100.0) if total > 0 else 0.0

        pair_groups: Dict[str, List[int]] = defaultdict(list)
        for r in rows:
            pair_groups[r["pair_key"]].append(int(r["correct"]))

        total_pairs = len(pair_groups)
        # Official MME style: acc+ counts the percentage of image-pairs where all paired questions are correct.
        pair_correct = sum(1 for flags in pair_groups.values() if len(flags) >= 2 and all(flags))
        acc_plus = (pair_correct / total_pairs * 100.0) if total_pairs > 0 else 0.0
        score = acc + acc_plus

        per_category[category] = {
            "num_questions": total,
            "num_pairs": total_pairs,
            "num_correct": correct,
            "num_pair_correct": pair_correct,
            "acc": acc,
            "acc_plus": acc_plus,
            "score": score,
        }

    perception_score = 0.0
    cognition_score = 0.0
    for category, m in per_category.items():
        if category in PERCEPTION_CATEGORIES:
            perception_score += m["score"]
        if category in COGNITION_CATEGORIES:
            cognition_score += m["score"]

    total_score = perception_score + cognition_score
    return {
        "per_category": per_category,
        "totals": {
            "perception": perception_score,
            "cognition": cognition_score,
            "total": total_score,
        },
        "num_predictions": len(records),
    }


def write_summary_markdown(summary_file: Path, metrics: Dict):
    lines = []
    totals = metrics["totals"]
    lines.append("# MME Evaluation Summary")
    lines.append("")
    lines.append(f"- Perception: {totals['perception']:.2f}")
    lines.append(f"- Cognition: {totals['cognition']:.2f}")
    lines.append(f"- Total: {totals['total']:.2f}")
    lines.append("")
    lines.append("| Category | Questions | Pairs | Acc | Acc+ | Score |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for category, m in sorted(metrics["per_category"].items()):
        lines.append(
            f"| {category} | {m['num_questions']} | {m['num_pairs']} | {m['acc']:.2f} | {m['acc_plus']:.2f} | {m['score']:.2f} |"
        )
    summary_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    set_seed(args.seed)

    mme_root = resolve_mme_root(args.mme_root)
    selected_categories = None
    if args.categories.strip():
        selected_categories = {
            canonical_category_name(c.strip()) for c in args.categories.split(",") if c.strip()
        }

    samples = load_mme_samples(mme_root, selected_categories)
    if not samples:
        raise RuntimeError(f"No MME samples found under {mme_root}")
    samples = get_chunk(samples, args.num_chunks, args.chunk_idx)
    if not samples:
        raise RuntimeError(
            f"No samples in chunk {args.chunk_idx}/{args.num_chunks} under {mme_root}."
        )

    answers_path = Path(args.answers_file)
    metrics_path = Path(args.metrics_file)
    summary_path = Path(args.summary_file)
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    records = evaluate_samples(args, samples)
    with answers_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics = compute_mme_metrics(records)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_markdown(summary_path, metrics)

    print(f"[done] answers: {answers_path}")
    print(f"[done] metrics: {metrics_path}")
    print(f"[done] summary: {summary_path}")
    print(f"[score] total={metrics['totals']['total']:.2f}, perception={metrics['totals']['perception']:.2f}, cognition={metrics['totals']['cognition']:.2f}")


if __name__ == "__main__":
    main()
