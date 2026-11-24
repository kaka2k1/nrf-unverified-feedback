#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
py=../.venv/bin/python
out=results/q1_e1
items=data/metatool_items.jsonl
free -g
mkdir -p "$out"
"$py" - "$items" "$out/annotations" <<'PY'
import json
import math
from pathlib import Path
import sys

from bfcl_breakeven_cpu import SOURCES

items = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
counts = {item["id"]: len(item["candidates"]) for item in items}
if len(items) != 995 or len(counts) != len(items):
    raise ValueError("Expected 995 distinct MetaTool inputs")
for model in ["Qwen3-0.6B", *SOURCES]:
    path = Path(sys.argv[2]) / f"annot_{model}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != len(counts) or {row["id"] for row in rows} != counts.keys():
        raise ValueError(f"Incomplete or duplicate annotations: {path}")
    for row in rows:
        n = counts[row["id"]]
        scores = row["probs"]
        if (
            len(scores) != n
            or any(not math.isfinite(score) or score < 0 for score in scores)
            or not math.isclose(sum(scores), 1.0, abs_tol=1e-6)
            or row["pred"] != max(range(n), key=scores.__getitem__)
        ):
            raise ValueError(f"Invalid prediction for {row['id']} in {path}")
PY
/usr/bin/time -f 'Peak host RSS %M KiB' nice -n 10 "$py" q1_e4_cpu.py \
  --setup bfcl --items "$items" --annot-dir "$out/annotations" --seeds 20 --workers 1 \
  --cache "$out/cache" --out "$out"
/usr/bin/time -f 'Peak host RSS %M KiB' nice -n 10 "$py" scripts/analyze_bfcl_feedback.py \
  --items "$items" --annot-dir "$out/annotations" --out "$out/source_context.md"
sed -i '1s/on BFCL/on MetaTool/' "$out/source_context.md"
/usr/bin/time -f 'Peak host RSS %M KiB' nice -n 10 "$py" -m scripts.q1_followup_results metatool
