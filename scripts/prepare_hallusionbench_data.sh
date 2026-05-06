#!/usr/bin/env bash
set -euo pipefail

REPO_ID=${REPO_ID:-rayguan/HallusionBench}
HALLUSION_ROOT=${HALLUSION_ROOT:-data/HallusionBench}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}

mkdir -p "${HALLUSION_ROOT}"

python - <<'PY'
import os
import json
import shutil
import time
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import hf_hub_download, list_repo_files

repo_id = os.environ.get("REPO_ID", "rayguan/HallusionBench")
local_dir = os.environ.get("HALLUSION_ROOT", "data/HallusionBench")

Path(local_dir).mkdir(parents=True, exist_ok=True)


def local_exists(root: Path, rel: str) -> bool:
  target = root / "data" / rel
  if target.exists():
    return True
  parent = target.parent
  if not parent.exists():
    return False
  stem = target.stem.lower()
  ext = target.suffix.lower()
  for cand in parent.glob("*"):
    if cand.is_file() and cand.stem.lower() == stem and cand.suffix.lower() == ext:
      return True
  return False


def ensure_alias(root: Path, expected_rel: str, actual_rel: str) -> None:
  expected = root / "data" / expected_rel
  actual = root / "data" / actual_rel
  if expected == actual or expected.exists() or not actual.exists():
    return
  expected.parent.mkdir(parents=True, exist_ok=True)
  rel_target = os.path.relpath(actual, expected.parent)
  try:
    expected.symlink_to(rel_target)
  except OSError:
    shutil.copy2(actual, expected)

for meta_file in ["HallusionBench.json", "HallusionBench_result_sample.json", "README.md", "evaluation.py", "utils.py"]:
  hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename=meta_file,
    local_dir=local_dir,
  )

json_path = Path(local_dir) / "HallusionBench.json"
data = json.loads(json_path.read_text(encoding="utf-8"))

required = sorted(
  {
    str(r.get("filename", "")).lstrip("./")
    for r in data
    if str(r.get("visual_input", "0")) != "0" and r.get("filename")
  }
)

hub_files = []
for attempt in range(1, 6):
  try:
    hub_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    break
  except Exception as exc:  # noqa: BLE001
    if attempt == 5:
      raise RuntimeError(f"failed to list dataset files for {repo_id}: {exc}")
    time.sleep(min(2 * attempt, 10))

hub_data_files = [f for f in hub_files if f.startswith("data/")]
hub_rel_to_actual = {}
for f in hub_data_files:
  rel = f[len("data/"):]
  key = rel.lower()
  if key not in hub_rel_to_actual:
    hub_rel_to_actual[key] = rel

hub_stem_to_actual = {}
for rel in hub_rel_to_actual.values():
  p = Path(rel)
  stem_key = str(p.with_suffix("")).lower()
  if stem_key not in hub_stem_to_actual:
    hub_stem_to_actual[stem_key] = rel


def resolve_hub_rel(rel: str):
  k = rel.lower()
  if k in hub_rel_to_actual:
    return hub_rel_to_actual[k]
  stem_key = str(Path(rel).with_suffix("")).lower()
  return hub_stem_to_actual.get(stem_key)

missing = [
  rel
  for rel in required
  if not local_exists(Path(local_dir), rel)
]

print(f"[info] required_images={len(required)}, missing_images={len(missing)}")

not_found_on_hub = []

for idx, rel in enumerate(missing, start=1):
  actual_rel = resolve_hub_rel(rel)
  if actual_rel is None:
    not_found_on_hub.append(rel)
    continue

  filename = f"data/{actual_rel}"
  ok = False
  for attempt in range(1, 6):
    try:
      hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        local_dir=local_dir,
      )
      ok = True
      break
    except Exception as exc:  # noqa: BLE001
      if attempt == 5:
        raise RuntimeError(f"download failed for {filename}: {exc}")
      time.sleep(min(2 * attempt, 10))

  if ok:
    ensure_alias(Path(local_dir), rel, actual_rel)

  if idx % 20 == 0 or idx == len(missing):
    print(f"[progress] downloaded_missing={idx}/{len(missing)}")

if not_found_on_hub:
  print(f"[warn] missing_on_hub={len(not_found_on_hub)}")
  for rel in not_found_on_hub[:20]:
    print(f"[warn] not found on hub: data/{rel}")

print(f"[done] HallusionBench targeted download ready: {local_dir}")
PY

if [[ ! -f "${HALLUSION_ROOT}/HallusionBench.json" ]]; then
  echo "[error] missing HallusionBench.json under ${HALLUSION_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${HALLUSION_ROOT}/data" ]]; then
  echo "[error] missing data dir under ${HALLUSION_ROOT}" >&2
  exit 1
fi

required_count=$(HALLUSION_ROOT="${HALLUSION_ROOT}" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["HALLUSION_ROOT"])
data = json.loads((root / "HallusionBench.json").read_text(encoding="utf-8"))
required = {
  str(r.get("filename", "")).lstrip("./")
  for r in data
  if str(r.get("visual_input", "0")) != "0" and r.get("filename")
}
print(len(required))
PY
)

have_required=$(HALLUSION_ROOT="${HALLUSION_ROOT}" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["HALLUSION_ROOT"])
data = json.loads((root / "HallusionBench.json").read_text(encoding="utf-8"))
required = {
  str(r.get("filename", "")).lstrip("./")
  for r in data
  if str(r.get("visual_input", "0")) != "0" and r.get("filename")
}


def local_exists(rel: str) -> bool:
  target = root / "data" / rel
  if target.exists():
    return True
  parent = target.parent
  if not parent.exists():
    return False
  stem = target.stem.lower()
  ext = target.suffix.lower()
  for cand in parent.glob("*"):
    if cand.is_file() and cand.stem.lower() == stem and cand.suffix.lower() == ext:
      return True
  return False


missing = [rel for rel in required if not local_exists(rel)]
print(len(required) - len(missing))
PY
)

echo "[info] required_images=${required_count}, available_required_images=${have_required}"

echo "[done] HallusionBench data ready at ${HALLUSION_ROOT}"
