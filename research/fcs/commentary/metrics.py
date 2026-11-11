"""Caption metrics as in MatchTime / SN-Caption: BLEU-1..4, METEOR, ROUGE-L, CIDEr via
pycocoevalcap with PTB tokenisation (single reference per sample). Reported ×100.
"""
from __future__ import annotations

from typing import Dict, List


def caption_scores_matchtime(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """Exactly MatchTime's evaluation/score_single.py protocol: raw (untokenised) strings,
    BLEU via BleuScorer(option='closest'), METEOR, ROUGE-L (mean), CIDEr. ×100."""
    from pycocoevalcap.bleu.bleu_scorer import BleuScorer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    gts = {i: [r] for i, r in enumerate(refs)}
    res = {i: [p] for i, p in enumerate(preds)}
    bs = BleuScorer(n=4)
    for i in gts:
        bs += (res[i][0], gts[i])
    b, _ = bs.compute_score(option="closest", verbose=0)
    out = {f"BLEU-{k + 1}": 100 * b[k] for k in range(4)}
    out["METEOR"] = 100 * Meteor().compute_score(gts, res)[0]
    _, rl = Rouge().compute_score(gts, res)
    out["ROUGE-L"] = 100 * float(rl.mean())
    out["CIDEr"] = 100 * Cider().compute_score(gts, res)[0]
    return out


def caption_scores(preds: List[str], refs: List[str]) -> Dict[str, float]:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    gts = {i: [{"caption": r}] for i, r in enumerate(refs)}
    res = {i: [{"caption": p if p.strip() else "."}] for i, p in enumerate(preds)}
    tok = PTBTokenizer()
    gts, res = tok.tokenize(gts), tok.tokenize(res)
    out = {}
    b, _ = Bleu(4).compute_score(gts, res, verbose=0)
    for k in range(4):
        out[f"BLEU-{k + 1}"] = 100 * b[k]
    out["METEOR"] = 100 * Meteor().compute_score(gts, res)[0]
    out["ROUGE-L"] = 100 * Rouge().compute_score(gts, res)[0]
    out["CIDEr"] = 100 * Cider().compute_score(gts, res)[0]
    return out
