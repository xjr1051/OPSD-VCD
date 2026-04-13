import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official-style POPE metric computation.")
    parser.add_argument("--gt_files", type=str, required=True, help="Ground-truth POPE label file (json/jsonl lines).")
    parser.add_argument("--gen_files", type=str, required=True, help="Generated answer file (jsonl).")
    parser.add_argument(
        "--strict_order",
        action="store_true",
        help="Match official script behavior strictly by checking line-by-line question_id order.",
    )
    parser.add_argument("--out_file", type=str, default=None, help="Optional path to dump metrics as JSON.")
    return parser.parse_args()


def load_json_lines(path: str):
    file_path = Path(path)
    content = file_path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    if content[0] == "[":
        loaded = json.loads(content)
        if not isinstance(loaded, list):
            raise ValueError(f"Expected list in {path}")
        return loaded
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(gt_files, gen_files, strict_order: bool):
    if strict_order and len(gt_files) != len(gen_files):
        raise ValueError(f"Length mismatch: gt={len(gt_files)} vs gen={len(gen_files)}")

    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    unknown = 0
    yes_answers = 0

    if strict_order:
        iterator = zip(gt_files, gen_files)
    else:
        gen_map = {item["question_id"]: item for item in gen_files}
        iterator = ((gt, gen_map.get(gt["question_id"])) for gt in gt_files)

    for gt_line, gen_line in iterator:
        idx = gt_line.get("question_id")
        gt_answer = gt_line.get("label")

        if gen_line is None:
            unknown += 1
            continue

        if strict_order:
            gen_idx = gen_line.get("question_id")
            if idx != gen_idx:
                raise AssertionError(f"question_id mismatch: gt={idx}, gen={gen_idx}")

        gen_answer = gen_line.get("text", "")

        gt_answer = str(gt_answer).lower().strip()
        gen_answer = str(gen_answer).lower().strip()

        # Official POPE rule: presence match on 'yes'/'no'.
        if gt_answer == "yes":
            if "yes" in gen_answer:
                true_pos += 1
                yes_answers += 1
            else:
                false_neg += 1
        elif gt_answer == "no":
            if "no" in gen_answer:
                true_neg += 1
            else:
                false_pos += 1
                yes_answers += 1
        else:
            unknown += 1

    total_questions = len(gt_files)
    precision = safe_div(true_pos, true_pos + false_pos)
    recall = safe_div(true_pos, true_pos + false_neg)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(true_pos + true_neg, total_questions)
    yes_proportion = safe_div(yes_answers, total_questions)
    unknown_prop = safe_div(unknown, total_questions)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "yes": yes_proportion,
        "unknow": unknown_prop,
        "true_pos": true_pos,
        "true_neg": true_neg,
        "false_pos": false_pos,
        "false_neg": false_neg,
        "unknown": unknown,
        "total_questions": total_questions,
    }


def main():
    args = parse_args()

    gt_files = load_json_lines(os.path.expanduser(args.gt_files))
    gen_files = load_json_lines(os.path.expanduser(args.gen_files))
    metrics = compute_metrics(gt_files, gen_files, strict_order=args.strict_order)

    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1: {metrics['f1']}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"yes: {metrics['yes']}")
    print(f"unknow: {metrics['unknow']}")

    if args.out_file:
        out_path = Path(args.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
