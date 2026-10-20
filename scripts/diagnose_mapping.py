import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Inspect per-example semantic ATT&CK mapping rankings."
    )
    ap.add_argument("--examples", required=True)
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "cache", "mitre"))
    ap.add_argument("--corpus-file")
    ap.add_argument("--embedder", choices=["sbert", "lexical"], default="sbert")
    ap.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--platform", action="append", dest="platforms")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if a.top_k < 1:
        ap.error("--top-k must be at least 1")

    from src.mapping.embedders import (
        EmbedderUnavailable, LexicalCorpusEmbedder, SBertEmbedder
    )
    from src.mapping.evaluation import load_examples
    from src.mapping.semantic_mapper import SemanticMitreMapper, TechniqueCorpus
    from src.mapping.text_prep import technique_embedding_text
    from src.mitre.corpus import CorpusError
    from src.mitre.updater import load_corpus_for_run

    try:
        examples = load_examples(a.examples)
        corpus = load_corpus_for_run(a.cache_dir, a.corpus_file)
        emb = (
            LexicalCorpusEmbedder(
                technique_embedding_text(t) for t in corpus.active()
            )
            if a.embedder == "lexical"
            else SBertEmbedder(a.embedding_model)
        )
        plats = [
            p for p in (a.platforms or ["Windows"])
            if p.lower() != "all"
        ] or None
        tc = TechniqueCorpus.from_mitre(
            corpus, emb, a.cache_dir, platforms=plats
        )
        mapper = SemanticMitreMapper(tc, emb, top_k=a.top_k)
        vectors = mapper.embed_texts([e["text"] for e in examples])
        rankings = [mapper.rank_vector(v) for v in vectors]
    except (ValueError, OSError, CorpusError, EmbedderUnavailable) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    rows = []
    for e, ranked in zip(examples, rankings):
        expected = set(e["expected_technique_ids"])
        expected_rank = next(
            (c.rank for c in ranked if c.technique_id in expected), None
        )
        top = ranked[0] if ranked else None
        rows.append({
            "id": e.get("id", ""),
            "category": e.get("category", (
                "positive" if expected else "benign"
            )),
            "expected_technique_ids": sorted(expected),
            "expected_rank_in_top_k": expected_rank,
            "top1_matches_expected": bool(
                expected and top and top.technique_id in expected
            ),
            "top1_technique_id": top.technique_id if top else None,
            "top1_technique_name": top.technique_name if top else None,
            "top1_similarity_score": (
                top.similarity_score if top else None
            ),
            "top1_status": top.mapping_status if top else None,
            "top1_at_or_above_threshold": bool(
                top and top.similarity_score >= a.threshold
            ),
            "ranked_predictions": [
                {
                    "rank": c.rank,
                    "technique_id": c.technique_id,
                    "technique_name": c.technique_name,
                    "similarity_score": c.similarity_score,
                    "mapping_status": c.mapping_status,
                }
                for c in ranked
            ],
        })

    positives = [r for r in rows if r["expected_technique_ids"]]
    negatives = [r for r in rows if not r["expected_technique_ids"]]
    summary = {
        "n_examples": len(rows),
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "positive_top1_hits": sum(
            r["top1_matches_expected"] for r in positives
        ),
        "positive_expected_found_in_top_k": sum(
            r["expected_rank_in_top_k"] is not None for r in positives
        ),
        "negative_top1_scores_at_or_above_threshold": sum(
            r["top1_at_or_above_threshold"] for r in negatives
        ),
        "threshold_used": a.threshold,
        "threshold_note": (
            "Similarity scores are not probabilities. This threshold is "
            "for diagnosis only; it does not change mapper configuration."
        ),
    }
    result = {
        "corpus": corpus.version_label,
        "model": emb.model_id,
        "top_k": a.top_k,
        "summary": summary,
        "examples": rows,
    }

    if a.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f'Corpus: {result["corpus"]} | Model: {result["model"]}')
        print(json.dumps(summary, indent=2))
        for r in rows:
            print(
                f'\n{r["id"]} [{r["category"]}] '
                f'expected_rank={r["expected_rank_in_top_k"]} '
                f'top1={r["top1_technique_id"]} '
                f'score={r["top1_similarity_score"]}'
            )
            if r["expected_technique_ids"]:
                print("  expected:", ", ".join(r["expected_technique_ids"]))
            for c in r["ranked_predictions"]:
                print(
                    f'  {c["rank"]}. {c["technique_id"]} '
                    f'{c["technique_name"]} '
                    f'score={c["similarity_score"]} '
                    f'status={c["mapping_status"]}'
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
