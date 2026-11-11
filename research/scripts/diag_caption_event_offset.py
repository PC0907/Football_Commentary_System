"""Offset between aligned caption times and Labels-v2 action times, per event type.
Captions are keyword-matched to an event type; for each, the nearest action of that type in
the same half gives (action_time - caption_time). Writes outputs/commentary/offsets.json."""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fcs.commentary.dataset import _sn_game_dir, load_actions
KW = {"Goal": r"\bgoal!|scores|the score is now|finds the net|back of the net",
      "Yellow card": r"yellow card|booked|bookable", "Corner": r"\bcorner\b",
      "Substitution": r"substitution|replaces|comes on", "Offside": r"offside"}
off = defaultdict(list)
for split in ("valid", "test"):
    for l in open(ROOT / f"cache/commentary/{split}.jsonl"):
        e = json.loads(l)
        acts = None
        for lab, rx in KW.items():
            if re.search(rx, e["target"], re.I):
                if acts is None:
                    acts = load_actions(_sn_game_dir(e["game"]))
                c = [a["t"] - e["t"] for a in acts if a["half"] == e["half"] and a["label"] == lab]
                if c:
                    off[lab].append(min(c, key=abs))
res = {}
for lab, v in off.items():
    v = np.array(v)
    res[lab] = dict(n=int(len(v)), median_s=float(np.median(v)), p10=float(np.percentile(v, 10)),
                    p90=float(np.percentile(v, 90)), within_m25_p5=float(np.mean((v >= -25) & (v <= 5))),
                    within_pm60=float(np.mean(np.abs(v) <= 60)))
(ROOT / "outputs/commentary/offsets.json").write_text(json.dumps(res, indent=1))
for k, r in res.items():
    print(f"{k:13s} n={r['n']:5d} median {r['median_s']:+7.1f}s  p10 {r['p10']:+7.1f}  p90 {r['p90']:+7.1f}  in[-25,+5] {100*r['within_m25_p5']:4.0f}%  |dt|<=60 {100*r['within_pm60']:4.0f}%")
