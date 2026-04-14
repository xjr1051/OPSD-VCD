import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score HallusionBench with official-style metrics.")
    parser.add_argument("--input-file", type=str, required=True, help="Model outputs JSON file.")
    parser.add_argument("--output-json", type=str, required=True, help="Summary JSON output.")
    parser.add_argument("--output-md", type=str, required=True, help="Summary markdown output.")
    parser.add_argument("--correctness-entry", type=str, default="pred_correctness")
    return parser.parse_args()


def assign_correctness(rows: List[dict], correctness_entry: str) -> List[dict]:
    for r in rows:
        c = int(r[correctness_entry])
        if r["category"] == "VS" and int(r["figure_id"]) == 0:
            r["correct"] = 1 if c in (1, 2) else 0
        else:
            r["correct"] = 1 if c == 1 else 0
    return rows


def get_eval_fig(rows: List[dict]) -> Dict:
    eval_fig_dict: Dict[str, Tuple[int, int]] = {}
    for r in rows:
        if r["category"] == "VS" and str(r["figure_id"]) == "0":
            continue
        name = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["figure_id"])])
        if name in eval_fig_dict:
            c, t = eval_fig_dict[name]
            eval_fig_dict[name] = (c + r["correct"], t + 1)
        else:
            eval_fig_dict[name] = (r["correct"], 1)

    stat = {
        "note": "all accuracy per image (consistency test)",
        "total": len(eval_fig_dict),
        "correct": 0,
        "wrong": 0,
        "inconsistent": 0,
        "score": 0.0,
    }

    for c, t in eval_fig_dict.values():
        if c == t:
            stat["correct"] += 1
        elif c == 0:
            stat["wrong"] += 1
        else:
            stat["inconsistent"] += 1
        stat["score"] += (c / t)

    if stat["total"] > 0:
        stat["score"] /= stat["total"]
    return stat


def get_eval_all(rows: List[dict], model_correctness_entry: str) -> Dict:
    eval_all_dict = {}
    stat = {"LH": 0, "VI": 0, "Mix": 0}

    for r in rows:
        name = "_".join([
            r["category"],
            r["subcategory"],
            str(r["set_id"]),
            str(r["figure_id"]),
            str(r["question_id"]),
        ])
        eval_all_dict[name] = r["correct"]

        if str(r["category"]) == "VD":
            if str(r["figure_id"]) == "0":
                if str(r[model_correctness_entry]) in ("0", "2"):
                    stat["VI"] += 1
            else:
                if str(r[model_correctness_entry]) == "0":
                    stat["Mix"] += 1
                elif str(r[model_correctness_entry]) == "2":
                    stat["VI"] += 1
        else:
            if str(r["visual_input"]) == "0":
                if str(r[model_correctness_entry]) == "0":
                    stat["LH"] += 1
            else:
                if str(r[model_correctness_entry]) == "0":
                    stat["Mix"] += 1
                elif str(r[model_correctness_entry]) == "2":
                    stat["VI"] += 1

    stat["note"] = "all accuracy per question"
    stat["total"] = len(eval_all_dict)
    stat["correct"] = sum(eval_all_dict.values())
    stat["wrong"] = stat["total"] - stat["correct"]
    return stat


def get_eval_pair_all(rows: List[dict], model_correctness_entry: str) -> Dict:
    orig_correctness = {}
    for r in rows:
        if str(r["figure_id"]) == "0":
            key = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["question_id"])])
            orig_correctness[key] = r[model_correctness_entry]

    eval_pair = {}
    analysis_pair = {}
    counter = 0
    lh_counter = 0
    vi_counter = 0
    both_counter = 0

    for r in rows:
        name = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["question_id"])])
        if name in eval_pair:
            c, t = eval_pair[name]
            eval_pair[name] = (c + r["correct"], t + 1)
        else:
            eval_pair[name] = (r["correct"], 1)
        counter += 1

        analysis = (0, 0)
        if str(r["figure_id"]) == "0":
            if str(r["category"]) == "VD":
                if str(r[model_correctness_entry]) in ("0", "2"):
                    analysis = (0, 1)
            else:
                if str(r[model_correctness_entry]) == "0":
                    analysis = (1, 0)
        else:
            key = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["question_id"])])
            orig_c = orig_correctness[key]
            if str(r["category"]) == "VD":
                if str(orig_c) == "1" and str(r[model_correctness_entry]) == "0":
                    if str(r.get("same", "0")) == "1":
                        analysis = (1, 1)
                    else:
                        analysis = (0, 1)
                elif str(orig_c) == "1" and str(r[model_correctness_entry]) == "2":
                    analysis = (0, 1)
                elif str(r[model_correctness_entry]) in ("0", "2"):
                    analysis = (0, 1)
            else:
                if str(orig_c) == "0":
                    if str(r[model_correctness_entry]) == "0" and str(r.get("same", "0")) == "1":
                        analysis = (1, 0)
                    elif str(r[model_correctness_entry]) == "0":
                        analysis = (1, 1)
                    elif str(r[model_correctness_entry]) == "2":
                        analysis = (1, 1)
                elif str(orig_c) == "2":
                    if str(r[model_correctness_entry]) in ("0", "2"):
                        analysis = (0, 1)
                else:
                    if str(r[model_correctness_entry]) == "2":
                        analysis = (0, 1)
                    elif str(r[model_correctness_entry]) == "0":
                        if str(r["visual_input"]) == "1":
                            analysis = (0, 1)
                        elif str(r["visual_input"]) == "2":
                            if str(r.get("same", "0")) == "1":
                                analysis = (1, 0)
                            else:
                                analysis = (0, 1)

        if analysis[0] > 0 and analysis[1] > 0:
            both_counter += 1
        elif analysis[0] > 0:
            lh_counter += 1
        elif analysis[1] > 0:
            vi_counter += 1

        if name in analysis_pair:
            lh, vi = analysis_pair[name]
            analysis_pair[name] = (lh + analysis[0], vi + analysis[1])
        else:
            analysis_pair[name] = analysis

    stat = {
        "note": "all accuracy per question pair",
        "total": len(eval_pair),
        "total_q": counter,
        "correct": 0,
        "wrong": 0,
        "LH": 0,
        "VI": 0,
        "Mix": 0,
        "LH_cg": lh_counter,
        "VI_cg": vi_counter,
        "Mix_cg": both_counter,
    }

    for k in eval_pair.keys():
        c, t = eval_pair[k]
        lh, vi = analysis_pair[k]
        if c == t:
            stat["correct"] += 1
        else:
            stat["wrong"] += 1
        if lh > 0 and vi > 0:
            stat["Mix"] += 1
        elif lh > 0:
            stat["LH"] += 1
        elif vi > 0:
            stat["VI"] += 1

    return stat


def get_eval_pair_easy(rows: List[dict]) -> Dict:
    d = {}
    counter = 0
    for r in rows:
        if str(r["visual_input"]) == "2":
            continue
        name = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["question_id"])])
        if name in d:
            c, t = d[name]
            d[name] = (c + r["correct"], t + 1)
        else:
            d[name] = (r["correct"], 1)
        counter += 1

    stat = {"note": "all accuracy per question pair", "total": len(d), "total_q": counter, "correct": 0, "wrong": 0}
    for c, t in d.values():
        if c == t:
            stat["correct"] += 1
        else:
            stat["wrong"] += 1
    return stat


def get_eval_pair_hard(rows: List[dict]) -> Dict:
    d = {}
    counter = 0
    for r in rows:
        if str(r["visual_input"]) != "2":
            continue
        name = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["question_id"])])
        if name in d:
            c, t = d[name]
            d[name] = (c + r["correct"], t + 1)
        else:
            d[name] = (r["correct"], 1)
        counter += 1

    stat = {"note": "all accuracy per question pair", "total": len(d), "total_q": counter, "correct": 0, "wrong": 0}
    for c, t in d.values():
        if c == t:
            stat["correct"] += 1
        else:
            stat["wrong"] += 1
    return stat


def yes_ratio_stats(rows: List[dict]) -> Dict:
    yes_gt = [int(i["gt_answer"]) for i in rows]
    yes_pred = [int(int(i["correct"]) == int(i["gt_answer"])) for i in rows]
    fp_sample = [i for i in rows if int(i["correct"]) == 0]
    fp = [int(i["gt_answer"]) for i in fp_sample] if fp_sample else [0]
    return {
        "diff": (sum(yes_pred) / len(yes_pred)) - (sum(yes_gt) / len(yes_gt)),
        "fp": (len(fp) - sum(fp)) / len(fp),
    }


def build_same_field(rows: List[dict]) -> None:
    orig_pred = {}
    for r in rows:
        if str(r["figure_id"]) == "0":
            key = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["question_id"])])
            orig_pred[key] = str(r["pred_answer"])

    for r in rows:
        if str(r["figure_id"]) == "0":
            r["same"] = "1"
            continue
        key = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["question_id"])])
        if key in orig_pred and str(r["pred_answer"]) == orig_pred[key]:
            r["same"] = "1"
        else:
            r["same"] = "0"


def to_percent(num: int, den: int) -> float:
    return round(100.0 * num / den, 4) if den else 0.0


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    rows = json.loads(input_path.read_text(encoding="utf-8"))
    for r in rows:
        gt = str(r.get("gt_answer", ""))
        pred = str(r.get("pred_answer", "2"))
        if pred == "2":
            r[args.correctness_entry] = "2"
        elif pred == gt:
            r[args.correctness_entry] = "1"
        else:
            r[args.correctness_entry] = "0"

    build_same_field(rows)
    rows = assign_correctness(rows, correctness_entry=args.correctness_entry)

    vd = [r for r in rows if r["category"] == "VD"]
    vs = [r for r in rows if r["category"] == "VS"]

    all_q = get_eval_all(rows, args.correctness_entry)
    vd_q = get_eval_all(vd, args.correctness_entry)
    vs_q = get_eval_all(vs, args.correctness_entry)

    all_pair = get_eval_pair_all(rows, args.correctness_entry)
    easy = get_eval_pair_easy(rows)
    hard = get_eval_pair_hard(rows)
    vd_pair = get_eval_pair_all(vd, args.correctness_entry)
    vs_pair = get_eval_pair_all(vs, args.correctness_entry)
    easy_vd = get_eval_pair_easy(vd)
    hard_vd = get_eval_pair_hard(vd)
    easy_vs = get_eval_pair_easy(vs)
    hard_vs = get_eval_pair_hard(vs)

    fig_all = get_eval_fig(rows)
    fig_vd = get_eval_fig(vd)
    fig_vs = get_eval_fig(vs)

    q_acc = to_percent(all_q["correct"], all_q["total"])
    pair_acc = to_percent(all_pair["correct"], all_pair["total"])
    fig_acc = to_percent(fig_all["correct"], fig_all["total"])
    easy_acc = to_percent(easy["correct"], easy["total"])
    hard_acc = to_percent(hard["correct"], hard["total"])

    stats = yes_ratio_stats(rows)

    summary = {
        "num_questions": len(rows),
        "num_vd_questions": len(vd),
        "num_vs_questions": len(vs),
        "leaderboard_metrics": {
            "acc_per_question_pair": pair_acc,
            "acc_per_figure": fig_acc,
            "acc_per_easy_question": easy_acc,
            "acc_per_hard_question": hard_acc,
            "acc_per_question": q_acc,
        },
        "question_accuracy": {
            "VD": to_percent(vd_q["correct"], vd_q["total"]),
            "VS": to_percent(vs_q["correct"], vs_q["total"]),
            "Overall": q_acc,
        },
        "question_pair_accuracy": {
            "VD": {
                "Easy": to_percent(easy_vd["correct"], easy_vd["total"]),
                "Hard": to_percent(hard_vd["correct"], hard_vd["total"]),
                "Total": to_percent(vd_pair["correct"], vd_pair["total"]),
            },
            "VS": {
                "Easy": to_percent(easy_vs["correct"], easy_vs["total"]),
                "Hard": to_percent(hard_vs["correct"], hard_vs["total"]),
                "Total": to_percent(vs_pair["correct"], vs_pair["total"]),
            },
            "Overall": {
                "Easy": easy_acc,
                "Hard": hard_acc,
                "Total": pair_acc,
            },
        },
        "figure_accuracy": {
            "VD": {
                "Correct": to_percent(fig_vd["correct"], fig_vd["total"]),
                "Wrong": round(to_percent(fig_vd["inconsistent"], fig_vd["total"]) + to_percent(fig_vd["wrong"], fig_vd["total"]), 4),
                "Score": round(fig_vd["score"], 4),
            },
            "VS": {
                "Correct": to_percent(fig_vs["correct"], fig_vs["total"]),
                "Wrong": round(to_percent(fig_vs["inconsistent"], fig_vs["total"]) + to_percent(fig_vs["wrong"], fig_vs["total"]), 4),
                "Score": round(fig_vs["score"], 4),
            },
            "Overall": {
                "Correct": to_percent(fig_all["correct"], fig_all["total"]),
                "Wrong": round(to_percent(fig_all["inconsistent"], fig_all["total"]) + to_percent(fig_all["wrong"], fig_all["total"]), 4),
                "Score": round(fig_all["score"], 4),
            },
        },
        "analysis": {
            "yes_no_bias_pct_diff": round(stats["diff"], 6),
            "yes_no_bias_fp_ratio": round(stats["fp"], 6),
            "consistency_correct": to_percent(fig_all["correct"], fig_all["total"]),
            "consistency_inconsistent": to_percent(fig_all["inconsistent"], fig_all["total"]),
            "consistency_wrong": to_percent(fig_all["wrong"], fig_all["total"]),
            "lh_pct": round(100 * all_pair["LH_cg"] / max(1, (all_pair["LH_cg"] + all_pair["VI_cg"] + all_pair["Mix_cg"])), 4),
            "vi_pct": round(100 * all_pair["VI_cg"] / max(1, (all_pair["LH_cg"] + all_pair["VI_cg"] + all_pair["Mix_cg"])), 4),
            "mix_pct": round(100 * all_pair["Mix_cg"] / max(1, (all_pair["LH_cg"] + all_pair["VI_cg"] + all_pair["Mix_cg"])), 4),
        },
        "note": "This run uses deterministic Yes/No/Uncertain parsing from model outputs (no GPT correctness judge).",
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = []
    lines.append("# HallusionBench Summary")
    lines.append("")
    lines.append(summary["note"])
    lines.append("")
    lines.append("## Leaderboard Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Acc per question pair (qAcc) | {pair_acc:.4f} |")
    lines.append(f"| Acc per figure (fAcc) | {fig_acc:.4f} |")
    lines.append(f"| Acc per easy question | {easy_acc:.4f} |")
    lines.append(f"| Acc per hard question | {hard_acc:.4f} |")
    lines.append(f"| Acc per question (aAcc) | {q_acc:.4f} |")
    lines.append("")
    lines.append("## Question Accuracy")
    lines.append("")
    lines.append("| Split | Accuracy |")
    lines.append("|---|---:|")
    lines.append(f"| VD | {summary['question_accuracy']['VD']:.4f} |")
    lines.append(f"| VS | {summary['question_accuracy']['VS']:.4f} |")
    lines.append(f"| Overall | {summary['question_accuracy']['Overall']:.4f} |")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] summary json: {out_json}")
    print(f"[done] summary md: {out_md}")
    print(f"[score] qAcc={pair_acc:.4f}, fAcc={fig_acc:.4f}, aAcc={q_acc:.4f}")


if __name__ == "__main__":
    main()
