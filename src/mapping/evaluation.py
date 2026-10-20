"""
src/mapping/evaluation.py
---------------------------
Evaluate the semantic mapper (and its thresholds) against LABELLED examples,
when you have them. This repo ships NO ground-truth labels, so the default
thresholds (candidate 0.35 / weak 0.25) are uncalibrated starting points.

Labelled example file (JSON):
    {"examples": [
        {"id": "ex1", "text": "<behaviour narrative>", "expected_technique_ids": ["T1059.001"]},
        {"id": "ex2", "text": "<benign narrative>",    "expected_technique_ids": []}      # expected: no match
    ]}

Metrics:
    top-k hit rate / MRR  : is an expected technique in the ranked list? (exact id)
    family hit rate       : parent technique counts (T1059 matches T1059.001)
    per-threshold table   : for each candidate threshold t, examples whose top-1
                            score >= t are "predicted"; precision, recall (over
                            positive examples), and the share of negative
                            examples correctly left unmatched.
Small label sets overfit: treat the suggested threshold as a hint, not a result.
"""

import json
from typing import Dict, List, Optional, Sequence


def load_examples(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    ex = payload.get("examples") if isinstance(payload, dict) else payload
    if not isinstance(ex, list) or not ex:
        raise ValueError(f"{path}: expected a non-empty 'examples' list")
    for i, e in enumerate(ex):
        if not isinstance(e, dict) or not e.get("text") or "expected_technique_ids" not in e:
            raise ValueError(f"{path}: example #{i} needs 'text' and 'expected_technique_ids'")
    return ex


def _family(tid: str) -> str:
    return tid.split(".")[0]


def evaluate_mapping(examples: List[Dict], mapper, thresholds: Optional[Sequence[float]] = None,
                     ks: Sequence[int] = (1, 3, 5)) -> Dict:
    thresholds = list(thresholds) if thresholds is not None else [round(0.05 * i, 2) for i in range(2, 15)]
    saved_k = mapper.top_k
    mapper.top_k = max(max(ks), saved_k)
    try:
        vecs = mapper.embed_texts([e["text"] for e in examples])
        ranked = [mapper.rank_vector(v) for v in vecs]
    finally:
        mapper.top_k = saved_k

    pos = [i for i, e in enumerate(examples) if e["expected_technique_ids"]]
    neg = [i for i, e in enumerate(examples) if not e["expected_technique_ids"]]

    def first_hit(i, family=False):
        want = set(examples[i]["expected_technique_ids"])
        if family:
            want = {_family(w) for w in want}
        for r, c in enumerate(ranked[i], start=1):
            got = _family(c.technique_id) if family else c.technique_id
            if got in want:
                return r
        return None

    out: Dict = {"n_examples": len(examples), "n_positive": len(pos), "n_negative": len(neg)}
    if pos:
        hits = {i: first_hit(i) for i in pos}
        fam = {i: first_hit(i, True) for i in pos}
        out["top_k_hit_rate"] = {f"top{k}": round(sum(1 for h in hits.values() if h and h <= k) / len(pos), 4) for k in ks}
        out["family_top_k_hit_rate"] = {f"top{k}": round(sum(1 for h in fam.values() if h and h <= k) / len(pos), 4) for k in ks}
        out["mrr"] = round(sum((1.0 / h) for h in hits.values() if h) / len(pos), 4)

    table = []
    for t in thresholds:
        predicted = [i for i in range(len(examples)) if ranked[i] and ranked[i][0].similarity_score >= t]
        correct = [i for i in predicted if i in pos and first_hit(i) == 1]
        precision = len(correct) / len(predicted) if predicted else None
        recall = len(correct) / len(pos) if pos else None
        neg_ok = sum(1 for i in neg if i not in predicted) / len(neg) if neg else None
        f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0
        table.append({"threshold": t, "predicted": len(predicted),
                      "precision_top1": None if precision is None else round(precision, 4),
                      "recall_top1": None if recall is None else round(recall, 4),
                      "negatives_left_unmatched": None if neg_ok is None else round(neg_ok, 4),
                      "f1": round(f1, 4)})
    out["threshold_table"] = table
    if pos:
        best = max(table, key=lambda r: (r["f1"], r["threshold"]))
        out["suggested_candidate_threshold"] = best["threshold"] if best["f1"] > 0 else None
        out["note"] = ("Suggested threshold maximises top-1 F1 on THESE examples only; "
                       f"{len(examples)} examples is a small sample - validate before relying on it.")
    return out
