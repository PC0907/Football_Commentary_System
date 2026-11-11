"""Commentary generation with a small local LLM (Qwen3.5) from the structured game state.

  python scripts/commentary_llm.py --split valid --n 300 --model Qwen/Qwen3.5-2B [--adapter DIR] [--tag g1]

G1 = zero-shot (no adapter); G2/G3 = with a LoRA adapter trained by train_commentary_lora.py.
Writes outputs/commentary/<tag>_<split>.jsonl and prints MatchTime-protocol metrics.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fcs.commentary.dataset import state_to_text  # noqa: E402
from fcs.commentary.metrics import caption_scores_matchtime  # noqa: E402
from fcs.progress import Progress  # noqa: E402

SYSTEM = ("You are a live football (soccer) text commentator. Given the match state and the labelled "
          "actions around the current moment, write ONE short commentary sentence in the style of a live "
          "text feed, present tense. Never use real names: write [PLAYER] for any player, [TEAM] for any "
          "team, [REFEREE] for the referee, [COACH] for a coach. Output only the sentence.")


def build_prompt(tok, e):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": state_to_text(e["state"])}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load_model(name, adapter=None, four_bit=True):
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                           bnb_4bit_use_double_quant=True) if four_bit else None
    model = AutoModelForImageTextToText.from_pretrained(name, quantization_config=q, dtype=torch.float16,
                                                        device_map="cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def clean(text):
    t = text.strip().split("\n")[0].strip()
    return t.strip('"').strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--adapter")
    ap.add_argument("--tag", default="g1")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=48)
    a = ap.parse_args()
    ex = [json.loads(l) for l in open(ROOT / "cache" / "commentary" / f"{a.split}.jsonl")]
    if a.n and a.n < len(ex):
        import random
        random.Random(0).shuffle(ex)
        ex = ex[:a.n]
    tok, model = load_model(a.model, a.adapter)
    preds = []
    prog = Progress(len(ex), f"commentary {a.tag} {a.split} ({Path(a.model).name})", "caption")
    t0 = time.time()
    for i in range(0, len(ex), a.bs):
        batch = ex[i:i + a.bs]
        enc = tok([build_prompt(tok, e) for e in batch], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for e, o in zip(batch, out[:, enc["input_ids"].shape[1]:]):
            preds.append(clean(tok.decode(o, skip_special_tokens=True)))
        prog.update(len(batch))
    prog.close()
    d = ROOT / "outputs" / "commentary"; d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{a.tag}_{a.split}.jsonl", "w") as f:
        for e, p in zip(ex, preds):
            f.write(json.dumps(dict(game=e["game"], half=e["half"], t=e["t"], pred=p, ref=e["target"],
                                    state=state_to_text(e["state"]))) + "\n")
    m = caption_scores_matchtime(preds, [e["target"] for e in ex])
    m["sec_per_caption"] = (time.time() - t0) / len(ex)
    (d / f"{a.tag}_{a.split}_metrics.json").write_text(json.dumps(m, indent=1))
    print(a.tag, a.split, " ".join(f"{k} {v:.2f}" for k, v in m.items()))
    for e, p in list(zip(ex, preds))[:3]:
        print("  IN :", state_to_text(e["state"])[:150]); print("  OUT:", p); print("  REF:", e["target"])


if __name__ == "__main__":
    main()
