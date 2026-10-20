"""
scripts/rebuild_technique_embeddings.py
-----------------------------------------
Re-embed the CACHED ATT&CK corpus (no download) and write the versioned
embedding cache. Normal runs build/reuse this cache automatically; use this
script to pre-warm it or to force a rebuild after changing the model.

    python scripts\\rebuild_technique_embeddings.py
    python scripts\\rebuild_technique_embeddings.py --embedder sbert --embedding-model all-MiniLM-L6-v2
    python scripts\\rebuild_technique_embeddings.py --embedder lexical      (offline stand-in)
    python scripts\\rebuild_technique_embeddings.py --corpus-file cache\\mitre\\enterprise-attack-v19.2-fa7bcbf8.json
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "cache", "mitre"))
    ap.add_argument("--corpus-file", help="explicit corpus file (default: the cache's CURRENT version)")
    ap.add_argument("--embedder", choices=["sbert", "lexical"], default="sbert")
    ap.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="HF model name or local folder")
    ap.add_argument("--no-force", action="store_true", help="reuse a valid cache instead of rebuilding")
    a = ap.parse_args(argv)

    from src.mapping.embedders import EmbedderUnavailable, LexicalCorpusEmbedder, SBertEmbedder
    from src.mapping.text_prep import technique_embedding_text
    from src.mitre.corpus import CorpusError
    from src.mitre.embedding_cache import load_or_build
    from src.mitre.updater import load_corpus_for_run

    try:
        corpus = load_corpus_for_run(a.cache_dir, a.corpus_file)
    except CorpusError as e:
        print(f"error: {e}\nRun: python scripts\\update_mitre_corpus.py", file=sys.stderr)
        return 1
    try:
        emb = (LexicalCorpusEmbedder(technique_embedding_text(t) for t in corpus.active())
               if a.embedder == "lexical" else SBertEmbedder(a.embedding_model))
    except EmbedderUnavailable as e:
        print(f"error: embedding model unavailable: {e}", file=sys.stderr)
        return 1
    if getattr(emb, "is_standin", False):
        print("NOTE: lexical stand-in embedder - development/testing only.")
    last = [0]

    def progress(done, total):
        if done - last[0] >= 200 or done == total:
            print(f"  embedded {done}/{total}")
            last[0] = done

    try:
        techs, vecs, status = load_or_build(corpus, emb, a.cache_dir, rebuild=not a.no_force, progress=progress)
    except Exception as e:
        print(f"error: embedding failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"corpus   : {corpus.version_label}  ({len(techs)} active techniques)")
    print(f"model    : {status['model_id']}   dim {status['dim']}")
    print(f"cache    : {status['cache']} -> {status['path']}")
    if status.get("pruned_stale"):
        print(f"pruned   : {', '.join(status['pruned_stale'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
