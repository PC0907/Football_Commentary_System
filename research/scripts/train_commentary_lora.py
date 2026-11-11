"""QLoRA fine-tuning of Qwen3.5 for state -> commentary (variant G2).

  python scripts/train_commentary_lora.py --model Qwen/Qwen3.5-4B --data-dir cache/commentary \
      --out outputs/commentary/lora_g2 --epochs 1 --bs 8 --accum 2 [--limit 64 --max-steps 10]

Language-model linear layers only (full-attention q/k/v/o, linear-attention in_proj_qkv /
in_proj_z / out_proj, MLP); the vision tower is untouched. Loss on the assistant (caption)
tokens only. Writes the adapter + log.jsonl to --out.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from commentary_llm import SYSTEM, build_prompt  # noqa: E402
from fcs.progress import Progress  # noqa: E402

TARGETS = (r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|"
           r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)")


def encode(tok, e, max_len):
    prompt = build_prompt(tok, e)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(e["target"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p_ids + t_ids)[:max_len]
    labels = ([-100] * len(p_ids) + t_ids)[:max_len]
    return ids, labels


def target_only_loss(model, ids, att, lab):
    """Causal-LM loss computing logits ONLY at supervised positions (caption tokens).
    Avoids materialising (batch x seq x 248k-vocab) logits, the dominant memory cost."""
    inner = model.get_base_model()
    h = inner.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state   # [B, T, d]
    tgt = lab[:, 1:]
    mask = tgt != -100
    hs = h[:, :-1][mask]                                  # hidden states that predict a target token
    logits = inner.lm_head(hs).float()
    return torch.nn.functional.cross_entropy(logits, tgt[mask])


def collate(batch, pad_id):
    L = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), L), pad_id); lab = torch.full((len(batch), L), -100); att = torch.zeros((len(batch), L), dtype=torch.long)
    for k, (i, l) in enumerate(batch):
        ids[k, :len(i)] = torch.tensor(i); lab[k, :len(l)] = torch.tensor(l); att[k, :len(i)] = 1
    return ids, lab, att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--data-dir", default=str(ROOT / "cache" / "commentary"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "commentary" / "lora_g2"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-steps", type=int)
    a = ap.parse_args()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                           bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(a.model, quantization_config=q, dtype=torch.float16, device_map="cuda")
    # NB: peft's prepare_model_for_kbit_training up-casts embeddings / lm_head to fp32
    # (~1.9 GB for Qwen's 248k vocabulary) — too much for a 4 GB GPU; do the essentials only.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ex = [json.loads(l) for l in open(Path(a.data_dir) / "train.jsonl")]
    random.Random(0).shuffle(ex)
    if a.limit:
        ex = ex[:a.limit]
    data = [encode(tok, e, a.max_len) for e in ex]
    steps = math.ceil(len(data) * a.epochs / (a.bs * a.accum))
    if a.max_steps:
        steps = min(steps, a.max_steps)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    prog = Progress(steps, f"QLoRA {Path(a.model).name} r={a.rank} ({n_tr / 1e6:.1f}M trainable)", "step")
    model.train()
    it, step, t0, run = 0, 0, time.time(), 0.0
    order = list(range(len(data)))
    while step < steps:
        random.shuffle(order)
        for i in range(0, len(order) - a.bs + 1, a.bs):
            ids, lab, att = collate([data[j] for j in order[i:i + a.bs]], tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = target_only_loss(model, ids.cuda(), att.cuda(), lab.cuda()) / a.accum
            loss.backward(); run += loss.item(); it += 1
            if it % a.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                prog.update(1, desc=f"QLoRA {Path(a.model).name} loss {run:.3f}")
                if step % 25 == 0 or step == steps:
                    with open(out / "log.jsonl", "a") as f:
                        f.write(json.dumps(dict(step=step, loss=run, lr=sched.get_last_lr()[0], t=time.time() - t0)) + "\n")
                run = 0.0
                if step >= steps:
                    break
    model.save_pretrained(out / "adapter")
    prog.close()
    print(f"done: {steps} steps in {(time.time() - t0) / 60:.1f} min -> {out / 'adapter'}")


if __name__ == "__main__":
    main()
