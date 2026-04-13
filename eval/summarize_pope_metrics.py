import argparse
import json
import re
from pathlib import Path


PATTERN = re.compile(r"(?P<model>.+)_coco_pope_(?P<split>random|popular|adversarial)\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize POPE metrics into markdown table.")
    parser.add_argument("--metrics-dir", type=str, required=True, help="Directory containing per-split metric JSON files.")
    parser.add_argument("--output-file", type=str, default=None, help="Optional markdown output path.")
    return parser.parse_args()


def read_rows(metrics_dir: Path):
    rows = []
    for file_path in sorted(metrics_dir.glob("*_coco_pope_*.json")):
        match = PATTERN.match(file_path.name)
        if not match:
            continue
        data = json.loads(file_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": match.group("model"),
                "split": match.group("split"),
                "accuracy": data.get("accuracy", 0.0),
                "f1": data.get("f1", 0.0),
                "precision": data.get("precision", 0.0),
                "recall": data.get("recall", 0.0),
                "yes": data.get("yes", 0.0),
            }
        )
    return rows


def format_md(rows):
    lines = [
        "| model | split | accuracy | f1 | precision | recall | yes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {split} | {accuracy:.4f} | {f1:.4f} | {precision:.4f} | {recall:.4f} | {yes:.4f} |".format(
                **row
            )
        )

    model_names = sorted({row["model"] for row in rows})
    if model_names:
        lines.append("")
        lines.append("| model | avg_accuracy | avg_f1 | avg_precision | avg_recall | avg_yes |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for model in model_names:
            sub = [r for r in rows if r["model"] == model]
            count = len(sub)
            lines.append(
                "| {model} | {avg_accuracy:.4f} | {avg_f1:.4f} | {avg_precision:.4f} | {avg_recall:.4f} | {avg_yes:.4f} |".format(
                    model=model,
                    avg_accuracy=sum(r["accuracy"] for r in sub) / count,
                    avg_f1=sum(r["f1"] for r in sub) / count,
                    avg_precision=sum(r["precision"] for r in sub) / count,
                    avg_recall=sum(r["recall"] for r in sub) / count,
                    avg_yes=sum(r["yes"] for r in sub) / count,
                )
            )

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    rows = read_rows(metrics_dir)
    if not rows:
        raise ValueError(f"No metric files found in {metrics_dir}")

    markdown = format_md(rows)
    print(markdown)

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
