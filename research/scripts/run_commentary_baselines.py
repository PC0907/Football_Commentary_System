"""Non-LLM commentary baselines on SN-Caption-test-align (3,267 captions).

G0  template: a fixed anonymised phrase for the labelled action nearest to the caption time
R0  consensus retrieval: among TRAIN captions whose nearest action has the same label,
    the one with the highest mean TF-IDF cosine similarity to the others ("most typical");
    captions with no nearby action use the consensus of all no-action captions.
Outputs outputs/commentary/<name>_test.jsonl and a summary JSON with metrics.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fcs.commentary.metrics import caption_scores, caption_scores_matchtime  # noqa: E402
from fcs.progress import Progress  # noqa: E402

TEMPLATES = {
    "Goal": "Goal! [PLAYER] ([TEAM]) finds the back of the net.",
    "Shots on target": "[PLAYER] ([TEAM]) fires a shot on target, but the goalkeeper makes the save.",
    "Shots off target": "[PLAYER] ([TEAM]) tries his luck from distance, but the shot goes wide.",
    "Corner": "[TEAM] win a corner kick.",
    "Foul": "[PLAYER] ([TEAM]) commits a foul.",
    "Yellow card": "[PLAYER] ([TEAM]) is shown a yellow card by [REFEREE].",
    "Red card": "[PLAYER] ([TEAM]) is sent off!",
    "Substitution": "[TEAM] make a substitution, [PLAYER] replaces [PLAYER].",
    "Offside": "[PLAYER] ([TEAM]) is caught offside.",
    "Clearance": "[PLAYER] ([TEAM]) clears the ball away.",
    "Ball out of play": "The ball goes out of play.",
    "Throw-in": "[TEAM] have a throw-in.",
    "Direct free-kick": "[TEAM] have a free kick.",
    "Indirect free-kick": "[TEAM] have an indirect free kick.",
    "Kick-off": "The match is underway.",
    "Penalty": "Penalty to [TEAM]!",
    None: "The game continues.",
}


def nearest_label(e):
    ev = e["state"]["events"]
    return min(ev, key=lambda x: abs(x["dt"]))["label"] if ev else None


def load(split):
    return [json.loads(l) for l in open(ROOT / "cache" / "commentary" / f"{split}.jsonl")]


def consensus(captions):
    from sklearn.feature_extraction.text import TfidfVectorizer
    caps = captions[:3000]
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(caps)
    sim = (X @ X.T).toarray()
    return caps[int(np.argmax(sim.mean(1)))]


def main():
    tr, te = load("train"), load("test")
    refs = [e["target"] for e in te]
    out = {}
    prog = Progress(3, "commentary baselines", "step")
    preds = [TEMPLATES.get(nearest_label(e), TEMPLATES[None]) for e in te]
    out["G0-template"] = caption_scores(preds, refs)
    out["G0-template[matchtime-protocol]"] = caption_scores_matchtime(preds, refs); prog.update(1)
    groups = defaultdict(list)
    for e in tr:
        groups[nearest_label(e)].append(e["target"])
    cons = {k: consensus(v) for k, v in groups.items()}
    preds_r = [cons.get(nearest_label(e), cons[None]) for e in te]
    out["R0-retrieval"] = caption_scores(preds_r, refs)
    out["R0-retrieval[matchtime-protocol]"] = caption_scores_matchtime(preds_r, refs); prog.update(1)
    d = ROOT / "outputs" / "commentary"; d.mkdir(parents=True, exist_ok=True)
    for name, P in (("G0-template", preds), ("R0-retrieval", preds_r)):
        with open(d / f"{name}_test.jsonl", "w") as f:
            for e, p in zip(te, P):
                f.write(json.dumps(dict(game=e["game"], half=e["half"], t=e["t"], pred=p, ref=e["target"])) + "\n")
    (d / "baselines_test.json").write_text(json.dumps(out, indent=1)); prog.update(1); prog.close()
    for k, v in out.items():
        print(k, " ".join(f"{m} {x:.2f}" for m, x in v.items()))


if __name__ == "__main__":
    main()
