"""Collect corrective feedback from several open LLMs ("feedback sources") on BFCL function selection.

For each model and item: P(candidate | query, candidates) via constrained scoring of '{"name": "<fn>"}'.
Output: <out>/annot_<model-slug>.jsonl with {id, probs, pred, sec}.

python annotate_bfcl.py --items data/bfcl_items.jsonl --models Qwen/Qwen2.5-0.5B-Instruct --out out
"""
import argparse
from contextlib import nullcontext
import json
import os
import time

import numpy as np
import torch

from kaggle_qlora import seq_logprob

SYSTEM = ("You are a function-calling assistant. Given the user's request and the available functions, choose the "
          'single function that should be called. Reply only with JSON: {"name": "<function name>"}.')


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def build_prompt(tok, it):
    funcs = "\n".join(f"- {c['name']}: {c['description']}" for c in it["candidates"])
    user = f"Available functions:\n{funcs}\n\nUser request: {it['query']}"
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:  # noqa: BLE001  (templates without system role)
        msgs = [{"role": "user", "content": SYSTEM + "\n\n" + user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def score_items(model, tok, items, bs, dtype, device="cuda"):
    seqs, idx = [], []
    for i, it in enumerate(items):
        p = tok(build_prompt(tok, it), add_special_tokens=False).input_ids
        for j, c in enumerate(it["candidates"]):
            comp = tok(json.dumps({"name": c["name"]}), add_special_tokens=False).input_ids
            seqs.append((p, comp))
            idx.append((i, j))
    order = np.argsort([len(p) + len(c) for p, c in seqs])
    scores = [np.full(len(it["candidates"]), -np.inf) for it in items]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for s in range(0, len(order), bs):
        b = order[s:s + bs]
        L = max(len(seqs[k][0]) + len(seqs[k][1]) for k in b)
        ids = torch.full((len(b), L), pad, dtype=torch.long)
        att = torch.zeros((len(b), L), dtype=torch.long)
        lab = torch.full((len(b), L), -100, dtype=torch.long)
        for r, k in enumerate(b):
            p, c = seqs[k]
            ids[r, :len(p) + len(c)] = torch.tensor(p + c)
            att[r, :len(p) + len(c)] = 1
            lab[r, len(p):len(p) + len(c)] = torch.tensor(c)
        ids, att, lab = (tensor.to(device) for tensor in (ids, att, lab))
        with torch.autocast("cuda", dtype=dtype) if device == "cuda" else nullcontext():
            per, _ = seq_logprob(model, ids, att, lab)
        for k, v in zip(b, per.tolist()):
            i, j = idx[k]
            scores[i][j] = v
    out = []
    for sc in scores:
        z = np.exp(sc - sc.max())
        out.append((z / z.sum()).tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/bfcl_items.jsonl")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--quantize", nargs="*", default=[], metavar="MODEL_ID",
                    help="Load these sources with 4-bit NF4 on CUDA")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    os.makedirs(a.out, exist_ok=True)
    items = [json.loads(line) for line in open(a.items)]
    if a.limit:
        items = items[:a.limit]
    cuda = a.device == "cuda"
    if not cuda and a.quantize:
        ap.error("--quantize requires CUDA")
    cap = torch.cuda.get_device_capability(0)[0] if cuda else 0
    dtype = (torch.bfloat16 if cap >= 8 else torch.float16) if cuda else torch.float32
    env = dict(gpu=torch.cuda.get_device_name(0) if cuda else None, torch=torch.__version__,
               transformers=__import__("transformers").__version__, dtype=str(dtype), quantize=a.quantize)
    json.dump(env, open(f"{a.out}/env_annot.json", "w"))
    log(env)
    for name in a.models:
        slug = name.split("/")[-1]
        f = f"{a.out}/annot_{slug}.jsonl"
        if os.path.exists(f):
            log("skip", name)
            continue
        model = None
        try:
            t0 = time.time()
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            tok = AutoTokenizer.from_pretrained(name)
            kwargs = {}
            if name in a.quantize:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
                )
            model = AutoModelForCausalLM.from_pretrained(
                name, dtype=dtype, device_map={"": a.device}, **kwargs
            )
            model.config.use_cache = False
            model.eval()
            probs = score_items(model, tok, items, a.bs, dtype, a.device)
            sec = time.time() - t0
            acc = np.mean([int(np.argmax(p) == it["gold"]) for p, it in zip(probs, items)])
            with open(f + ".tmp", "w") as fo:
                for p, it in zip(probs, items):
                    fo.write(json.dumps(dict(id=it["id"], probs=p, pred=int(np.argmax(p)))) + "\n")
            os.replace(f + ".tmp", f)
            peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
            log(f"{name}: acc {acc:.3f} sec {sec:.0f} peak {peak:.2f}GB")
        except Exception as e:  # noqa: BLE001
            import traceback
            log(f"!!! {name} failed: {e}\n{traceback.format_exc()[-1500:]}")
        finally:
            del model
            if cuda:
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
