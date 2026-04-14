import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List

from openai import OpenAI


PROMPT_TEMPLATE = '''Please act as an impartial and objective judge and evaluate the quality of the response provided by a Large Multimodal Model (LMM) to the user question. Your evaluation should be mainly based on whether the response is informative, and whether the response contains any hallucination. Hallucination, in this context, refers to a situation where the LMM generates a response that includes information not present or implied in the image or previous conversation. A hallucination could be a false claim about an object, action, emotion, or any other detail that is not grounded in the image.

For clarity, consider these examples:

### Example 1:
- Image Contents: Lynx, Cat
- Question: How many cats can you see in this image?
- Standard Human-Generated Answer: There are two cats visible in the image.
- LMM Response to Evaluate: There is one cat in the image.
- Analysis: The LMM's response is false because it claims there is only one cat in the image, when in fact there are two.
- Hallucination: Yes.

### Example 2:
- Image Contents: Tree, Plant, Panda, Animal
- Question: What can you see from this photo?
- Standard Human-Generated Answer: In this photo we can see branches. We can see a panda on a branch. The background is blurry.
- LMM Response to Evaluate: In the photo, I can see a large black and white panda sitting on a tree branch. The image captures the panda's unique appearance and its natural habitat, showcasing the beauty and rarity of these animals.
- Analysis: The LMM's response is not false because it does not make any false claims about the image contents. For example, it does not claim that the panda is eating bamboo.
- Hallucination: No.

### Example 3:
- Image Contents: Fashion accessory, Mammal, Dog, Umbrella
- Question: Describe this image in details.
- Standard Human-Generated Answer: In this picture I can see a dog laying under a yellow umbrella on a beach, and there is a rope on the right side. In the background I can see few persons talking or lying. At the top of the image, I can see the blue and clear sky.
- LMM Response to Evaluate: The image features a brown dog laying on the sand at the beach, underneath a yellow umbrella. The dog is positioned near the water, and there are people in the background. The beach is sandy, and the sun is shining, creating a pleasant atmosphere. The dog is enjoying the shade provided by the umbrella, which is a common way to protect pets from the sun's harmful rays. The presence of people in the background suggests that the beach is a popular spot for leisure activities, such as sunbathing, swimming, and relaxing.
- Analysis: The LMM's response is false because it claims that the dog is positioned near the water, but there is no water in the image.
- Hallucination: Yes.

### Example 4:
- Image Contents: Clothing, Luggage and bags, Man, Building, Woman, Person
- Question: Where is this photo taken?
- Standard Human-Generated Answer: From the slogan "Welcome to Houston, Texas", we can infer that this photo is taken at an airport in Houston, Texas.
- LMM Response to Evaluate: This photo is taken at an airport, specifically at the entrance of a terminal. However, the specific location of this airport cannot be determined from the image.
- Analysis: Although the LMM's response does not explicitly mention the location "Houston, Texas" and does not perfectly answer the question, it makes no false claims about the image contents. Therefore, this response is not considered hallucination. It is OK if the LMM fails to directly answer the question, expresses uncertainty, or cannot include all the objects or details in the human-generated answer.
- Hallucination: No.

### Example 5:
- Image Contents: Tree, Tower
- Question: What environmental concerns are involved in this photo?
- Standard Human-Generated Answer: This photo shows a smokestack emitting smoke. It may cause severe air pollution which threatens human health.
- LMM Response to Evaluate: In the photo, there are several environmental concerns related to the smokestack emitting smoke. The smoke from the smokestack is a byproduct of industrial processes, which can contribute to air pollution, climate change, and negative impacts on human health. The smoke contains harmful pollutants, such as particulate matter, sulfur dioxide, nitrogen oxides, and carbon monoxide, which can lead to respiratory problems, heart disease, and other health issues. Additionally, the smoke contributes to the greenhouse effect, which can lead to global warming and climate change. The smokestack's emissions also affect the environment, as they can harm wildlife, vegetation, and ecosystems. Therefore, it is essential to address these environmental concerns by implementing measures to reduce emissions and promote sustainable practices in industrial processes.
- Analysis: Although the LMM's response is significantly longer than the standard human-generated answer, it does not contain any false claims about the image contents. Instead, it provides additional general information about the environmental concerns, which can be inferred from the smoke emission. Such detailed analysis or reasoning should be considered as a positive aspect, as long as it contains no false claims.
- Hallucination: No.

With these examples in mind, please help me evaluate whether the response by the LMM is informative, and whether hallucination exists in it, based on the comparison between the LMM's response and the factual information provided in the image contents, question, and the standard human-generated answer below.

Please note that the standard human-generated answer may only contain factual information but may not give a detailed analysis. Also, the standard human-generated answer may not be completely comprehensive in describing all the objects and their attributes, so please be a bit more cautious during evalutation. LMM's detailed analysis or reasoning should be encouraged.

To evaluate the LMM responses, first, begin your evaluation by providing a short explanation. Second, after providing your explanation, you must rate the response by choosing from the following options:
- Rating: 6, very informative with good analysis or reasoning, no hallucination
- Rating: 5, very informative, no hallucination
- Rating: 4, somewhat informative, no hallucination
- Rating: 3, not informative, no hallucination
- Rating: 2, very informative, with hallucination
- Rating: 1, somewhat informative, with hallucination
- Rating: 0, not informative, with hallucination

### Image Contents
{}

### Question
{}

### Standard Human-Generated Answer
{}

### LMM Response to Evaluate
{}
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MMHal official-style GPT judge scoring.")
    parser.add_argument("--response-file", type=str, required=True, help="MMHal responses JSON file.")
    parser.add_argument("--output-file", type=str, required=True, help="Raw judge responses JSON file.")
    parser.add_argument("--scored-file", type=str, required=True, help="Parsed per-sample scores JSON file.")
    parser.add_argument("--summary-file", type=str, required=True, help="Judge metrics summary JSON file.")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Judge API key. If omitted, reads JUDGE_API_KEY / OPENAI_API_KEY from environment.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL (e.g., https://yunwu.ai or https://api.openai.com/v1).",
    )
    parser.add_argument("--model", type=str, default="gpt-4-0314", help="Judge model.")
    parser.add_argument("--sleep-sec", type=float, default=1.0, help="Delay between API requests.")
    parser.add_argument("--max-retries", type=int, default=20, help="Max retries per sample.")
    return parser.parse_args()


def normalize_api_base(api_base: str) -> str:
    base = (api_base or "").strip()
    if not base:
        return ""
    base = base.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    return base


def load_records(path: Path) -> List[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("response-file must contain a JSON list")
    return records


def build_prompt(record: dict) -> str:
    image_content = record.get("image_content", [])
    if isinstance(image_content, list):
        image_content_str = ", ".join(str(x) for x in image_content)
    else:
        image_content_str = str(image_content)

    return PROMPT_TEMPLATE.format(
        image_content_str,
        record.get("question", ""),
        record.get("gt_answer", ""),
        record.get("model_answer", ""),
    )


def parse_rating(text: str) -> int:
    matches = re.findall(r"rating\s*:\s*([0-6])", text.lower())
    if len(matches) == 1:
        return int(matches[0])
    return 0


def summarize(scored_rows: List[dict]) -> Dict:
    n = len(scored_rows)
    scores = [int(r["rating"]) for r in scored_rows]
    halluc = [1 if s < 3 else 0 for s in scores]

    by_type: Dict[str, List[int]] = {}
    for r in scored_rows:
        t = r.get("question_type", "unknown")
        by_type.setdefault(t, []).append(int(r["rating"]))

    avg_by_type = {k: sum(v) / len(v) for k, v in sorted(by_type.items())}

    # Keep official script behavior for reference: index mod 8 buckets.
    mod8: List[List[int]] = [[] for _ in range(8)]
    for i, s in enumerate(scores):
        mod8[i % 8].append(s)
    avg_mod8 = [sum(v) / len(v) if v else 0.0 for v in mod8]

    return {
        "num_samples": n,
        "avg_score": sum(scores) / max(1, n),
        "hallucination_rate": sum(halluc) / max(1, n),
        "avg_score_by_question_type": avg_by_type,
        "avg_score_by_index_mod8": avg_mod8,
        "rating_scale": {
            "6": "very informative with good analysis or reasoning, no hallucination",
            "5": "very informative, no hallucination",
            "4": "somewhat informative, no hallucination",
            "3": "not informative, no hallucination",
            "2": "very informative, with hallucination",
            "1": "somewhat informative, with hallucination",
            "0": "not informative, with hallucination",
        },
    }


def main() -> None:
    args = parse_args()

    api_key = args.api_key or os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key: provide --api-key or set JUDGE_API_KEY / OPENAI_API_KEY")

    api_base = normalize_api_base(args.api_base or os.getenv("JUDGE_API_BASE") or os.getenv("OPENAI_BASE_URL") or "")

    response_path = Path(args.response_file)
    output_path = Path(args.output_file)
    scored_path = Path(args.scored_file)
    summary_path = Path(args.summary_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(response_path)
    if api_base:
        print(f"[config] api_base={api_base}", flush=True)
        client = OpenAI(api_key=api_key, base_url=api_base)
    else:
        print("[config] api_base=<openai default>", flush=True)
        client = OpenAI(api_key=api_key)

    raw_rows: List[dict] = []
    scored_rows: List[dict] = []

    for i, record in enumerate(records):
        prompt = build_prompt(record)
        text = None
        err = None
        for attempt in range(1, args.max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                text = (resp.choices[0].message.content or "").strip()
                err = None
                break
            except Exception as exc:  # noqa: BLE001
                err = str(exc)
                print(f"[retry] idx={i} attempt={attempt}/{args.max_retries} err={exc}", flush=True)
                time.sleep(min(2 * attempt, 20))

        if text is None:
            raise RuntimeError(f"judge request failed at idx={i}: {err}")

        rating = parse_rating(text)
        raw_rows.append(
            {
                "index": i,
                "id": record.get("id"),
                "question_type": record.get("question_type", "unknown"),
                "judge_text": text,
            }
        )
        scored_rows.append(
            {
                "index": i,
                "id": record.get("id"),
                "question_type": record.get("question_type", "unknown"),
                "rating": rating,
            }
        )
        print(f"[judge] idx={i} rating={rating}", flush=True)
        time.sleep(max(args.sleep_sec, 0.0))

    summary = summarize(scored_rows)
    summary.update(
        {
            "judge_model": args.model,
            "judge_api_base": api_base or "<openai default>",
            "response_file": str(response_path),
            "output_file": str(output_path),
            "scored_file": str(scored_path),
        }
    )

    output_path.write_text(json.dumps(raw_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    scored_path.write_text(json.dumps(scored_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[done] judge raw: {output_path}")
    print(f"[done] judge scored: {scored_path}")
    print(f"[done] judge summary: {summary_path}")
    print(f"[score] avg_score={summary['avg_score']:.4f}")
    print(f"[score] hallucination_rate={summary['hallucination_rate']:.4f}")


if __name__ == "__main__":
    main()
