"""
scripts/evaluate_mapping.py
------------------------------
Evaluate the semantic MITRE mapper and its thresholds against LABELLED
examples you provide (format: docs/examples/labeled_examples.example.json).
The repository ships no ground truth, so thresholds stay uncalibrated until
you run this on real labels.

    python scripts\\evaluate_mapping.py --examples my_labels.json
    python scripts\\evaluate_mapping.py --examples my_labels.json --embedder lexical --json
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", required=True)
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "cache", "mitre"))
    ap.add_argument("--corpus-file")
    ap.add_argument("--embedder", choices=["sbert", "lexical"], default="sbert")
    ap.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--platform", action="append", dest="platforms")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    from src.mapping.embedders import EmbedderUnavailable, LexicalCorpusEmbedder, SBertEmbedder
    from src.mapping.evaluation import evaluate_mapping, load_examples
    from src.mapping.semantic_mapper import SemanticMitreMapper, TechniqueCorpus
    from src.mapping.text_prep import technique_embedding_text
    from src.mitre.corpus import CorpusError
    from src.mitre.updater import load_corpus_for_run

    try:
        examples = load_examples(a.examples)
        corpus = load_corpus_for_run(a.cache_dir, a.corpus_file)
        emb = (LexicalCorpusEmbedder(technique_embedding_text(t) for t in corpus.active())
               if a.embedder == "lexical" else SBertEmbedder(a.embedding_model))
    except (ValueError, OSError, CorpusError, EmbedderUnavailable) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    plats = [p for p in (a.platforms or ["Windows"]) if p.lower() != "all"] or None
    tc = TechniqueCorpus.from_mitre(corpus, emb, a.cache_dir, platforms=plats)
    mapper = SemanticMitreMapper(tc, emb, top_k=a.top_k)
    result = evaluate_mapping(examples, mapper)
    if a.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"examples: {len(examples)}   corpus: {corpus.version_label}   model: {emb.model_id}")
        if getattr(emb, "is_standin", False):
            print("NOTE: lexical stand-in embedder - numbers are not representative of SBERT.")
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
