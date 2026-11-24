"""Learning from REAL LLM-generated corrective feedback on BFCL function selection (GPU, Qwen3 QLoRA).

Sources = open LLMs whose function choices were collected by annotate_bfcl.py. Per run (seed):
  - item-level split: locked test (n_test), clean seed set (n_seed), adapt pool (rest)
  - every adapt item receives the choice of ONE randomly assigned source as feedback; the verifier reveals the gold
    function for a random verify_rate fraction
  - context = data-driven cluster of the item's function set (scripts/prep_bfcl.py)
Methods reuse nrf.reliability (weighting methods only; label spaces differ per item).

python real_feedback.py --annot-dir annot --sources a b c --seeds 0 1 2 --verify-rate 0.25 --out out
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kaggle_qlora as kq  # noqa: E402
from nrf.reliability import method_weights, weight_diagnostics  # noqa: E402

SYSTEM = ("You are a function-calling assistant. Given the user's request and the available functions, choose the "
          'single function that should be called. Reply only with JSON: {"name": "<function name>"}.')


class Enc:
    def __init__(self, tok, view="full"):
        self.tok = tok
        self.view = view  # "full": names + descriptions; "names": under-documented tools (names only)
        self.cache = {}

    def prompt_ids(self, it):
        if it["id"] not in self.cache:
            if self.view == "names":
                funcs = "\n".join(f"- {c['name']}" for c in it["candidates"])
            else:
                funcs = "\n".join(f"- {c['name']}: {c['description']}" for c in it["candidates"])
            user = f"Available functions:\n{funcs}\n\nUser request: {it['query']}"
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
            p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            self.cache[it["id"]] = self.tok(p, add_special_tokens=False).input_ids
        return self.cache[it["id"]]

    def completion_ids(self, it, j):
        return self.tok(json.dumps({"name": it["candidates"][j]["name"]}) + "<|im_end|>",
                        add_special_tokens=False).input_ids


@torch.no_grad()
def score(model, enc, items, bs=16):
    model.eval()
    seqs, idx = [], []
    for i, it in enumerate(items):
        p = enc.prompt_ids(it)
        for j in range(len(it["candidates"])):
            seqs.append((p, enc.completion_ids(it, j)))
            idx.append((i, j))
    order = np.argsort([len(p) + len(c) for p, c in seqs])
    sc = [np.full(len(it["candidates"]), -np.inf) for it in items]
    for s in range(0, len(order), bs):
        b = order[s:s + bs]
        ids, att, lab = kq.collate([seqs[k] for k in b], enc.tok.pad_token_id, "cuda")
        with torch.autocast("cuda", dtype=kq.AMP_DTYPE):
            lp, _ = kq.seq_logprob(model, ids, att, lab)
        for k, v in zip(b, lp.tolist()):
            i, j = idx[k]
            sc[i][j] = v
    out = []
    for x in sc:
        z = np.exp(x - x.max())
        out.append(z / z.sum())
    return out


def evaluate(model, enc, test):
    probs = score(model, enc, test)
    pred = np.array([int(np.argmax(p)) for p in probs])
    gold = np.array([it["gold"] for it in test])
    ctx = np.array([it["context"] for it in test])
    r = dict(acc=float((pred == gold).mean()))
    r["acc_by_ctx"] = {int(c): float((pred == gold)[ctx == c].mean()) for c in np.unique(ctx)}
    return r


def load_annotations(annot_dir, sources, items):
    ids = [it["id"] for it in items]
    preds = {}
    for s in sources:
        f = os.path.join(annot_dir, f"annot_{s}.jsonl")
        if not os.path.exists(f):
            cand = glob.glob(os.path.join(annot_dir, "**", f"annot_{s}.jsonl"), recursive=True)
            f = cand[0]
        d = {json.loads(line)["id"]: json.loads(line)["pred"] for line in open(f)}
        preds[s] = np.array([d[i] for i in ids])
    return preds


def make_split(items, preds, sources, seed, n_test, n_seed, verify_rate):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(items))
    te, sd, ad = perm[:n_test], perm[n_test:n_test + n_seed], perm[n_test + n_seed:]
    src = rng.integers(len(sources), size=len(ad))
    y_fb = np.array([preds[sources[s]][i] for s, i in zip(src, ad)])
    gold = np.array([items[i]["gold"] for i in ad])
    ver = rng.random(len(ad)) < verify_rate
    adapt = pd.DataFrame(dict(source=src, domain=[items[i]["context"] for i in ad], y_fb=y_fb, y_true=gold,
                              verified=ver, text=[items[i]["query"] for i in ad]))
    return te, sd, ad, adapt


def observable(adapt):
    obs = adapt.drop(columns=["y_true"]).copy()
    obs["y_ver"] = np.where(adapt.verified, adapt.y_true, -1)
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/bfcl_items.jsonl")
    ap.add_argument("--annot-dir", default="annot")
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--methods", nargs="+", default=["naive", "verifier_corrected", "conf_filter", "source_prior",
                                                     "instance_rel", "ctx_prior", "ctx_eb", "oracle_clean"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--verify-rate", type=float, default=0.25)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--n-seed", type=int, default=64)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--seed-steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--targets", nargs="+", default=["q_proj", "v_proj"])
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--learner-view", choices=["full", "names"], default="names")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    kq.AMP_DTYPE = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    info = kq.env_info(a)
    info["amp_dtype"] = str(kq.AMP_DTYPE)
    json.dump(info, open(f"{a.out}/env.json", "w"), indent=1)
    kq.log(json.dumps(info))
    items = [json.loads(line) for line in open(a.items)]
    preds = load_annotations(a.annot_dir, a.sources, items)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"
    enc = Enc(tok, a.learner_view)
    resf = open(f"{a.out}/results.jsonl", "a")

    def record(r):
        resf.write(json.dumps(r) + "\n")
        resf.flush()
        kq.log("RESULT", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                                     if k not in ("losses", "acc_by_ctx")}))

    for seed in a.seeds:
        te, sd, ad, adapt = make_split(items, preds, a.sources, 1000 + seed, a.n_test, a.n_seed, a.verify_rate)
        test = [items[i] for i in te]
        obs = observable(adapt)
        base = dict(seed=seed, verify_rate=a.verify_rate, noise_type="bfcl_llm", noise=float((adapt.y_fb != adapt.y_true).mean()),
                    realized_noise=float((adapt.y_fb != adapt.y_true).mean()), model=a.model, steps=a.max_steps,
                    sources=a.sources, learner_view=a.learner_view)
        kq.log(f"=== seed {seed}: feedback error {base['noise']:.3f}, verified {adapt.verified.mean():.3f}")
        seed_items = [(enc.prompt_ids(items[i]), enc.completion_ids(items[i], items[i]["gold"]), 1.0) for i in sd]
        args_like = argparse.Namespace(**{**vars(a), "batch": a.batch, "lr": a.lr})
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model, gpm = kq.load_model(args_like, "cuda")
        r = evaluate(model, enc, test)
        record(dict(base, method="frozen", sec=time.time() - t0, peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30, **r))
        model = kq.add_lora(model, gpm, args_like)
        t0 = time.time()
        kq.train_lora(model, enc, seed_items, args_like, "cuda", a.seed_steps, run_seed=seed)
        p_ad = score(model, enc, [items[i] for i in ad])
        p_seed_fb = np.array([p[y] for p, y in zip(p_ad, adapt.y_fb)])
        r = evaluate(model, enc, test)
        record(dict(base, method="seed_only", steps=a.seed_steps, sec=time.time() - t0,
                    peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30, **r))
        del model
        kq.free()
        for m in a.methods:
            try:
                labels, w, q = method_weights(m, obs, p_seed_fb, tau=a.tau, gamma=a.gamma,
                                              y_true_for_oracle=adapt.y_true.to_numpy() if m == "oracle_clean" else None)
                diag = weight_diagnostics(obs, adapt.y_true.to_numpy(), labels, w, q)
                tr = seed_items + [(enc.prompt_ids(items[i]), enc.completion_ids(items[i], int(y)), float(wi))
                                   for i, y, wi in zip(ad, labels, w) if wi > 0]
                kq.log(f"--- {m}: {len(tr)} items, diag {json.dumps({k: round(v, 3) for k, v in diag.items()})}")
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                model, gpm = kq.load_model(args_like, "cuda")
                model = kq.add_lora(model, gpm, args_like)
                losses = kq.train_lora(model, enc, tr, args_like, "cuda", a.max_steps, run_seed=seed)
                r = evaluate(model, enc, test)
                record(dict(base, method=m, sec=time.time() - t0, peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30,
                            final_loss=float(np.mean(losses[-10:])), **diag, **r))
                del model
                kq.free()
            except Exception:  # noqa: BLE001
                kq.log(f"!!! {m} failed\n{traceback.format_exc()}")
                record(dict(base, method=m, error=traceback.format_exc()[-2000:]))
                torch.cuda.empty_cache()
    kq.log("done")


if __name__ == "__main__":
    main()
