"""Qwen3 QLoRA weighted-SFT experiment on the noisy-feedback tool-calling benchmark (GPU).

Protocol per (benchmark seed, noise):
  frozen      base model, zero-shot (no training)
  seed_only   LoRA trained on the clean seed set only; it also provides p_seed(y_fb | x) for the adapt pool
  <method>    fresh LoRA from the SAME base checkpoint, trained on seed (w=1) + adapt feedback weighted by the
              method, identical step budget for every method
Evaluation: constrained scoring (sum log-prob of each of the 4 candidate tool calls -> argmax) on the locked
test set (seen + held-out API families), plus greedy-generation exact match on the first --n-gen test items.

Smoke:  python kaggle_qlora.py --smoke --out /kaggle/working/smoke
Main:   python kaggle_qlora.py --seeds 0 1 2 --noise 0.4 --methods naive verifier_corrected source_prior vcfl ca_vcfl
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import re
import sys
import time
import traceback
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nrf import VERBS, make_benchmark, method_weights, observable_view, tool_names, weight_diagnostics  # noqa: E402

SYSTEM = ("You are a tool-calling agent. Choose exactly one tool that fulfils the user's request and reply "
          'only with a JSON object of the form {"name": "<tool>"}.')


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- data -> token ids
class Encoder:
    def __init__(self, tok, seed=0):
        self.tok = tok
        self.rng = np.random.default_rng(seed)
        self.cache = {}

    def prompt_ids(self, domain, text):
        key = (domain, text)
        if key not in self.cache:
            tools = tool_names(domain)
            order = np.random.default_rng(zlib.crc32(f"{domain}|{text}".encode())).permutation(4)
            user = "Available tools: " + ", ".join(tools[i] for i in order) + "\nRequest: " + text
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
            p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            self.cache[key] = self.tok(p, add_special_tokens=False).input_ids
        return self.cache[key]

    def completion_ids(self, domain, verb_idx):
        tool = tool_names(domain)[verb_idx]
        return self.tok(json.dumps({"name": tool}) + "<|im_end|>", add_special_tokens=False).input_ids


def collate(seqs, pad_id, device):
    """seqs: list of (prompt_ids, completion_ids). Right padding; labels only on completion."""
    L = max(len(p) + len(c) for p, c in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    lab = torch.full((len(seqs), L), -100, dtype=torch.long)
    for i, (p, c) in enumerate(seqs):
        n = len(p) + len(c)
        ids[i, :n] = torch.tensor(p + c)
        att[i, :n] = 1
        lab[i, len(p):n] = torch.tensor(c)
    return ids.to(device), att.to(device), lab.to(device)


def seq_logprob(model, ids, att, lab):
    """Per-sequence (sum log-prob over completion tokens, n completion tokens)."""
    # apply the LM head only at completion positions (full-vocab logits for every position would cost GBs)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    h = base.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
    tgt = lab[:, 1:]
    mask = tgt != -100
    sel = base.lm_head(h[mask]).float()
    lp = torch.log_softmax(sel, -1).gather(-1, tgt[mask].unsqueeze(-1)).squeeze(-1)
    per = torch.zeros(ids.shape[0], device=ids.device, dtype=torch.float32)
    rows = mask.nonzero()[:, 0]
    per.index_add_(0, rows, lp)
    return per, mask.sum(1)


@torch.no_grad()
def score_candidates(model, enc, df, device, bs=32):
    """Returns probs [N,4] over the 4 tools (softmax of total log-prob)."""
    model.eval()
    seqs, idx = [], []
    for i, (d, t) in enumerate(zip(df.domain, df.text)):
        p = enc.prompt_ids(d, t)
        for v in range(4):
            seqs.append((p, enc.completion_ids(d, v)))
            idx.append((i, v))
    order = np.argsort([len(p) + len(c) for p, c in seqs])
    scores = np.zeros((len(df), 4))
    for s in range(0, len(order), bs):
        b = order[s:s + bs]
        ids, att, lab = collate([seqs[j] for j in b], enc.tok.pad_token_id, device)
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            lp, _ = seq_logprob(model, ids, att, lab)
        for j, v in zip(b, lp.tolist()):
            scores[idx[j]] = v
    z = scores - scores.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


@torch.no_grad()
def generate_check(model, enc, df, device, n, bs=16):
    if n <= 0:
        return None
    model.eval()
    tok = enc.tok
    tok.padding_side = "left"
    hits = 0
    sub = df.iloc[:n]
    for s in range(0, len(sub), bs):
        part = sub.iloc[s:s + bs]
        prompts = [enc.prompt_ids(d, t) for d, t in zip(part.domain, part.text)]
        L = max(map(len, prompts))
        ids = torch.full((len(prompts), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        for i, p in enumerate(prompts):
            ids[i, L - len(p):] = torch.tensor(p)
            att[i, L - len(p):] = 1
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            out = model.generate(input_ids=ids.to(device), attention_mask=att.to(device), max_new_tokens=24,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        for i, (d, y) in enumerate(zip(part.domain, part.y_true)):
            txt = tok.decode(out[i, L:], skip_special_tokens=True)
            m = re.search(r'"name"\s*:\s*"([^"]+)"', txt)
            hits += int(bool(m) and m.group(1) == tool_names(d)[y])
    tok.padding_side = "right"
    return hits / len(sub)


# ----------------------------------------------------------------------------- model
def load_model(args, device):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kw = dict(torch_dtype=AMP_DTYPE, device_map={"": 0})
    if not args.no_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_use_double_quant=True,
                                                       bnb_4bit_compute_dtype=AMP_DTYPE)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.grad_ckpt)
    return model, get_peft_model


def add_lora(model, get_peft_model, args):
    from peft import LoraConfig
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules=args.targets,
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    for p in model.parameters():  # LoRA weights in fp32 (needed for fp16 GradScaler)
        if p.requires_grad:
            p.data = p.data.float()
    return model


def train_lora(model, enc, items, args, device, steps, run_seed):
    """items: list of (prompt_ids, completion_ids, weight>0). Weighted mean-token NLL, normalised by the
    global mean weight so the effective learning rate is comparable across methods."""
    torch.manual_seed(run_seed)
    rng = np.random.default_rng(run_seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP_DTYPE == torch.float16))
    wbar = float(np.mean([w for _, _, w in items]))
    model.train()
    perm, pos, losses = rng.permutation(len(items)), 0, []
    for step in range(steps):
        if pos + args.batch > len(perm):
            perm, pos = rng.permutation(len(items)), 0
        batch = [items[j] for j in perm[pos:pos + args.batch]]
        pos += args.batch
        ids, att, lab = collate([(p, c) for p, c, _ in batch], enc.tok.pad_token_id, device)
        w = torch.tensor([x[2] for x in batch], device=device, dtype=torch.float32)
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            lp, n = seq_logprob(model, ids, att, lab)
        nll = -lp / n.clamp(min=1)
        loss = (w * nll).sum() / (len(batch) * wbar)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        losses.append(float(loss))
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            log(f"    step {step + 1}/{steps} loss {np.mean(losses[-10:]):.4f}")
    return losses


def evaluate(model, enc, test, device, args):
    p = score_candidates(model, enc, test, device, bs=args.eval_bs)
    pred = p.argmax(1)
    y = test.y_true.to_numpy()
    from sklearn.metrics import f1_score
    seen = (test.split == "seen").to_numpy()
    return dict(acc=float((pred == y).mean()), acc_seen=float((pred == y)[seen].mean()),
                acc_held=float((pred == y)[~seen].mean()) if (~seen).any() else float("nan"),
                acc_policy=float((pred == y)[test.policy_item.to_numpy()].mean())
                if test.policy_item.any() else float("nan"),
                macro_f1=float(f1_score(y, pred, average="macro")),
                gen_acc=generate_check(model, enc, test, device, args.n_gen))


def free():
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- main
def env_info(args):
    import importlib
    info = dict(python=sys.version.split()[0], platform=platform.platform(), torch=torch.__version__,
                cuda=torch.version.cuda, args=vars(args))
    for m in ["transformers", "peft", "accelerate", "bitsandbytes", "sklearn"]:
        try:
            info[m] = importlib.import_module(m).__version__
        except Exception as e:  # noqa: BLE001
            info[m] = f"ERR {e}"
    if torch.cuda.is_available():
        info["gpus"] = [dict(name=torch.cuda.get_device_name(i),
                             mem_gb=round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2))
                        for i in range(torch.cuda.device_count())]
        info["bf16"] = torch.cuda.is_bf16_supported()
    return info


AMP_DTYPE = torch.float16


def main():
    global AMP_DTYPE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--methods", nargs="+", default=["naive", "verified_only", "verifier_corrected", "conf_filter",
                                                     "source_prior", "vcfl", "ctx_prior", "ca_vcfl",
                                                     "oracle_clean"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--noise", type=float, nargs="+", default=[0.4])
    ap.add_argument("--noise-type", default="context")
    ap.add_argument("--verify-rate", type=float, default=0.25)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--seed-steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--targets", nargs="+", default=["q_proj", "v_proj"])
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--n-test-seen", type=int, default=300)
    ap.add_argument("--n-test-held", type=int, default=200)
    ap.add_argument("--n-gen", type=int, default=64)
    ap.add_argument("--eval-bs", type=int, default=16)
    ap.add_argument("--skip-frozen", action="store_true")
    ap.add_argument("--time-budget-min", type=float, default=600)
    ap.add_argument("--amp", choices=["auto", "fp16", "bf16"], default="auto")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    if args.smoke:
        args.max_steps, args.seed_steps, args.n_test_seen, args.n_test_held, args.n_gen = 5, 5, 8, 8, 4
        args.methods = ["naive", "source_prior", "vcfl", "ca_vcfl"]
    os.makedirs(args.out, exist_ok=True)
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda:0")
    # T4 (sm_75) reports bf16 "supported" via slow emulation -> only use bf16 natively (sm_80+)
    if args.amp == "bf16" or (args.amp == "auto" and torch.cuda.get_device_capability(0)[0] >= 8):
        AMP_DTYPE = torch.bfloat16
    info = env_info(args)
    info["amp_dtype"] = str(AMP_DTYPE)
    log(json.dumps(info, indent=1))
    json.dump(info, open(f"{args.out}/env.json", "w"), indent=1)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    enc = Encoder(tok)
    res_f = open(f"{args.out}/results.jsonl", "a")
    t_start = time.time()

    def record(r):
        res_f.write(json.dumps(r) + "\n")
        res_f.flush()
        log("RESULT", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                                  if k not in ("losses",)}))

    for s in args.seeds:
        for nz in args.noise:
            b = make_benchmark(seed=1000 + s, noise=nz, noise_type=args.noise_type, verify_rate=args.verify_rate,
                               n_test_seen=args.n_test_seen, n_test_held=args.n_test_held)
            seed_df, adapt, test = b["seed"], b["adapt"], b["test"]
            obs = observable_view(adapt)
            base = dict(seed=s, noise=nz, noise_type=args.noise_type, verify_rate=args.verify_rate,
                        realized_noise=b["meta"]["realized_noise"], model=args.model, steps=args.max_steps)
            log(f"=== seed {s} noise {nz} realized {b['meta']['realized_noise']:.3f} held={b['meta']['heldout_domains']}")
            seed_items = [(enc.prompt_ids(d, t), enc.completion_ids(d, y), 1.0)
                          for d, t, y in zip(seed_df.domain, seed_df.text, seed_df.y_true)]

            # frozen + seed model (shared by all methods of this (seed, noise))
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            model, gpm = load_model(args, device)
            if not args.skip_frozen and nz == args.noise[0]:
                r = evaluate(model, enc, test, device, args)
                record(dict(base, method="frozen", sec=time.time() - t0,
                            peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30, **r))
            model = add_lora(model, gpm, args)
            t0 = time.time()
            train_lora(model, enc, seed_items, args, device, args.seed_steps, run_seed=s)
            p_adapt = score_candidates(model, enc, adapt, device, bs=args.eval_bs)
            p_seed_fb = p_adapt[np.arange(len(adapt)), obs.y_fb.to_numpy()]
            r = evaluate(model, enc, test, device, args)
            record(dict(base, method="seed_only", steps=args.seed_steps, sec=time.time() - t0,
                        peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30,
                        seed_model_adapt_acc=float((p_adapt.argmax(1) == adapt.y_true.to_numpy()).mean()), **r))
            del model
            free()

            for m in args.methods:
                if (time.time() - t_start) / 60 > args.time_budget_min:
                    log("time budget exhausted; stopping")
                    return
                try:
                    labels, w, q = method_weights(m, obs, p_adapt, tau=args.tau, gamma=args.gamma,
                                                  y_true_for_oracle=adapt.y_true.to_numpy()
                                                  if m == "oracle_clean" else None)
                    diag = weight_diagnostics(obs, adapt.y_true.to_numpy(), labels, w, q)
                    items = seed_items + [(enc.prompt_ids(d, t), enc.completion_ids(d, int(y)), float(wi))
                                          for d, t, y, wi in zip(adapt.domain, adapt.text, labels, w) if wi > 0]
                    log(f"--- {m}: {len(items)} items, diag {json.dumps({k: round(v, 3) for k, v in diag.items()})}")
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.time()
                    model, gpm = load_model(args, device)
                    model = add_lora(model, gpm, args)
                    losses = train_lora(model, enc, items, args, device, args.max_steps, run_seed=s)
                    t_train = time.time() - t0
                    r = evaluate(model, enc, test, device, args)
                    record(dict(base, method=m, sec=time.time() - t0, train_sec=t_train,
                                peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30,
                                final_loss=float(np.mean(losses[-10:])), losses=losses, **diag, **r))
                    del model
                    free()
                except Exception:  # noqa: BLE001
                    log(f"!!! method {m} failed\n{traceback.format_exc()}")
                    record(dict(base, method=m, error=traceback.format_exc()[-2000:]))
                    torch.cuda.empty_cache()
    log(f"done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
