import argparse
import csv
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List

import torch
from PIL import Image
from transformers import pipeline


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "made",
    "of",
    "on",
    "or",
    "the",
    "their",
    "there",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whose",
    "why",
    "with",
}

BOUNDARY_WORDS = {
    "and",
    "are",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "made",
    "of",
    "on",
    "or",
    "to",
    "was",
    "were",
    "with",
}

GENERIC_OBJECTS = {
    "animal",
    "baby",
    "bear",
    "bird",
    "boy",
    "cat",
    "child",
    "dog",
    "face",
    "girl",
    "guy",
    "horse",
    "lady",
    "lizard",
    "man",
    "men",
    "monster",
    "person",
    "people",
    "puppy",
    "robot",
    "sheep",
    "woman",
    "women",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build pseudo target-object boxes for HaloQuest with open-vocab detection.")
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--image-column", type=str, default="image")
    parser.add_argument("--question-column", type=str, default="question")
    parser.add_argument("--detector-model", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--label-keep-threshold", type=float, default=0.05)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--max-ngram", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    return list(OrderedDict.fromkeys([x.strip() for x in items if x and x.strip()]))


def tokenize_question(question: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", question.lower())


def extract_article_chunks(question: str) -> List[str]:
    tokens = tokenize_question(question)
    chunks = []
    for i, tok in enumerate(tokens):
        if tok not in {"a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "their"}:
            continue
        chunk = []
        for nxt in tokens[i + 1 :]:
            if nxt in BOUNDARY_WORDS:
                break
            chunk.append(nxt)
            if len(chunk) >= 3:
                break
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks


def clean_phrase(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower())
    text = re.sub(r"[?.,!]+$", "", text)
    return text


def head_noun_candidate(phrase: str) -> str:
    tokens = [t for t in tokenize_question(phrase) if t not in STOPWORDS]
    if not tokens:
        return ""
    return tokens[-1]


def extract_focus_phrases(question: str) -> List[str]:
    q = clean_phrase(question)
    focuses = []

    prefixes = [
        "what color is the ",
        "what color are the ",
        "what outfit color does the ",
        "what outfit colour does the ",
        "is the ",
        "are the ",
        "does the ",
        "do the ",
    ]
    for prefix in prefixes:
        if q.startswith(prefix):
            focuses.append(q[len(prefix) :])
            break

    m = re.search(r"how many (.+?)(?: is| are| does| do| can| should| would| could| the )", q)
    if m:
        focuses.append(m.group(1))

    for phrase in list(focuses):
        if "'s " in phrase:
            owner, tail = phrase.split("'s ", 1)
            focuses.append(tail)
            if " on the " in tail:
                focuses.append(tail.split(" on the ", 1)[0])
            if " of the " in tail:
                focuses.append(tail.split(" of the ", 1)[0])
            if " to the " in tail:
                focuses.append(tail.split(" to the ", 1)[0])
            if " made out of " in tail:
                focuses.append(tail.split(" made out of ", 1)[0])
        else:
            for sep in (" on the ", " of the ", " to the ", " with the "):
                if sep in phrase:
                    focuses.append(phrase.split(sep, 1)[0])

    heads = [head_noun_candidate(p) for p in focuses]
    focuses.extend([h for h in heads if h])
    return dedupe_keep_order(focuses)


def extract_ngram_candidates(question: str, max_ngram: int) -> List[str]:
    tokens = [t for t in tokenize_question(question) if t not in STOPWORDS]
    cands = []
    for n in range(min(max_ngram, len(tokens)), 0, -1):
        for i in range(0, len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            cands.append(phrase)
    return cands


def extract_candidate_labels(question: str, max_candidates: int, max_ngram: int) -> List[str]:
    seeds = []
    seeds.extend(extract_focus_phrases(question))
    seeds.extend(extract_article_chunks(question))
    seeds.extend(extract_ngram_candidates(question, max_ngram))
    seeds.append(question.strip().rstrip("?"))

    filtered = []
    for cand in dedupe_keep_order(seeds):
        words = cand.split()
        if not words:
            continue
        if all(w in STOPWORDS for w in words):
            continue
        filtered.append(cand)
        if len(filtered) >= max_candidates:
            break
    return filtered


def label_is_generic(label: str) -> bool:
    tokens = tokenize_question(label)
    if not tokens:
        return True
    return tokens[-1] in GENERIC_OBJECTS


def has_specific_focus(candidates: List[str]) -> bool:
    for cand in candidates:
        if cand and not label_is_generic(cand):
            return True
    return False


def resolve_detector_text_limit(detector) -> int:
    tokenizer = getattr(detector, "tokenizer", None)
    max_length = getattr(tokenizer, "model_max_length", 16) if tokenizer is not None else 16
    if not isinstance(max_length, int) or max_length <= 0 or max_length > 256:
        return 16
    return max_length


def clip_candidate_to_fit(label: str, detector, max_length: int) -> str:
    tokenizer = getattr(detector, "tokenizer", None)
    if tokenizer is None:
        return label.strip()

    words = [word for word in label.strip().split() if word]
    while words:
        candidate = " ".join(words)
        tokenized = tokenizer(candidate, add_special_tokens=True, truncation=False, return_attention_mask=False)
        input_ids = tokenized.get("input_ids", [])
        if len(input_ids) <= max_length:
            return candidate
        words = words[:-1]
    return ""


def sanitize_candidate_labels(candidates: List[str], detector) -> List[str]:
    max_length = resolve_detector_text_limit(detector)
    clipped = []
    for cand in candidates:
        fitted = clip_candidate_to_fit(cand, detector, max_length)
        if fitted:
            clipped.append(fitted)
    return dedupe_keep_order(clipped)


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else -1
    detector = pipeline(
        task="zero-shot-object-detection",
        model=args.detector_model,
        device=device,
    )

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    for extra in ["target_object", "pseudo_bboxes", "pseudo_bbox_scores", "pseudo_box_source", "pseudo_candidates"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    total = len(rows) if args.limit <= 0 else min(args.limit, len(rows))
    found = 0

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(rows[:total]):
            image_path = row.get(args.image_column, "")
            question = row.get(args.question_column, "")
            raw_candidates = extract_candidate_labels(question, args.max_candidates, args.max_ngram)
            candidates = sanitize_candidate_labels(raw_candidates, detector)

            row["pseudo_candidates"] = json.dumps(candidates, ensure_ascii=False)
            row["target_object"] = ""
            row["pseudo_bboxes"] = "[]"
            row["pseudo_bbox_scores"] = "[]"
            row["pseudo_box_source"] = "none"
            specific_focus = has_specific_focus(candidates)

            if image_path and candidates and Path(image_path).exists():
                try:
                    with Image.open(image_path) as pil_image:
                        image = pil_image.convert("RGB")
                        detections = detector(image, candidate_labels=candidates, threshold=args.score_threshold)
                        image.close()

                    if detections:
                        ranked = sorted(detections, key=lambda x: float(x["score"]), reverse=True)
                        best = ranked[0]
                        non_generic = [det for det in ranked if not label_is_generic(str(det["label"]))]
                        if non_generic and float(non_generic[0]["score"]) >= float(best["score"]) - 0.12:
                            best = non_generic[0]

                        keep_label = str(best["label"])
                        if not (specific_focus and label_is_generic(keep_label)):
                            kept = [
                                det
                                for det in detections
                                if str(det["label"]) == keep_label and float(det["score"]) >= args.label_keep_threshold
                            ]
                            if kept:
                                boxes = []
                                scores = []
                                for det in kept:
                                    box = det["box"]
                                    boxes.append([float(box["xmin"]), float(box["ymin"]), float(box["xmax"]), float(box["ymax"])])
                                    scores.append(float(det["score"]))

                                row["target_object"] = keep_label
                                row["pseudo_bboxes"] = json.dumps(boxes, ensure_ascii=False)
                                row["pseudo_bbox_scores"] = json.dumps(scores, ensure_ascii=False)
                                row["pseudo_box_source"] = "owlvit"
                                found += 1
                except Exception as exc:
                    row["pseudo_box_source"] = f"error:{type(exc).__name__}"
                    print(f"[warn] row={idx} image={image_path} failed: {exc}")

            writer.writerow(row)
            if (idx + 1) % 100 == 0 or idx + 1 == total:
                print(f"[progress] {idx + 1}/{total} rows, found={found}")

        for row in rows[total:]:
            writer.writerow(row)

    print(f"[done] input={input_csv}")
    print(f"[done] output={output_csv}")
    print(f"[done] found_boxes={found}/{total}")


if __name__ == "__main__":
    main()
