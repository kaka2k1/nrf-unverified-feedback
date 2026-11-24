"""Convert MetaTool Task2-Subtask1 (similar tools) to the BFCL item schema."""
import argparse
import json
from pathlib import Path
import random
import re

from prep_bfcl import assign_contexts, DESC_CHARS

TOOL = re.compile(r"^\d+\. tool name: (.*?), tool description: (.*)$", re.MULTILINE)
START = "[List of Tools with Names and Descriptions Start]\n"
END = "[List of Tools with Names and Descriptions End]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Pinned upstream Task2-Subtask1.json")
    parser.add_argument("--out", type=Path, default=Path("data/metatool_items.jsonl"))
    args = parser.parse_args()
    items = []
    for row in json.loads(args.source.read_text()):
        block = row["action_prompt"].split(START, 1)[1].split(END, 1)[0]
        pairs = TOOL.findall(block)
        names = [name for name, _ in pairs]
        if len(names) != 10 or len(set(names)) != 10 or row["tool"] not in names:
            raise ValueError(f"Invalid candidate set for MetaTool index {row['index']}")
        # Shuffle candidate order independently of the gold label.
        random.Random(int(row["index"])).shuffle(pairs)
        candidates = [
            dict(name=name, description=desc[:DESC_CHARS]) for name, desc in pairs
        ]
        items.append(dict(id=f"metatool-similar-{row['index']}",
                          group="|".join(sorted(names)), query=row["query"].strip(),
                          candidates=candidates,
                          gold=[c["name"] for c in candidates].index(row["tool"])))
    if len({it["id"] for it in items}) != len(items):
        raise ValueError("Duplicate MetaTool IDs")
    assign_contexts(items)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(it) + "\n" for it in items))
    print(f"Wrote {len(items)} items to {args.out}")


if __name__ == "__main__":
    main()
