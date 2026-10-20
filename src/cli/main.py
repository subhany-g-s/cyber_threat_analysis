"""
src/cli/main.py
-----------------
Command-line front end over src.service.PipelineService. Presentation only:
it builds a PipelineConfig from flags, prints each stage as it completes and a
summary at the end. Exit code 0 = all stages ran, 1 = a stage failed,
2 = bad usage / unusable input path.
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
for _sub in ("graph", "gae", "clustering", "novelty"):          # legacy flat-import modules
    _p = os.path.join(PROJECT_ROOT, "src", _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

EPILOG = r"""
Windows examples (cmd or PowerShell, from the project folder):
  python run_pipeline.py --help
  python run_pipeline.py --input data\sample_2000_events.json
  python run_pipeline.py --input data\events.csv --input-format csv --top-k 5
  python run_pipeline.py --input data\sample_2000_events.json --export-dir outputs\run1
  python run_pipeline.py --input data\sample_2000_events.json --llm ollama --ollama-model llama3.1
  python run_pipeline.py --input data\sample_2000_events.json --neo4j     (needs NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD)
  python scripts\update_mitre_corpus.py                  (explicit ATT&CK download; analysis itself never downloads)
  python scripts\rebuild_technique_embeddings.py         (re-embed the cached ATT&CK corpus)
  python scripts\evaluate_mapping.py --examples labeled.json

Nothing is written to disk unless --export-dir is given.
"""


def _defaults():
    from src.service.config import PipelineConfig
    return PipelineConfig()


DEFAULTS = _defaults()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="run_pipeline.py", description="SUBHANY cyber threat analysis pipeline (local CLI)",
                                 epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("input")
    g.add_argument("--input", default="data/sample_2000_events.json", help="log file (default JSONL; see --input-format)")
    g.add_argument("--input-format", choices=["auto", "jsonl", "json-array", "csv"], default="auto")
    g = ap.add_argument_group("pretrained GAE (never retrained)")
    g.add_argument("--pretrained-model", default=DEFAULTS.checkpoint_path)
    g.add_argument("--pretrained-schema", default=DEFAULTS.feature_schema_path)
    g.add_argument("--embeddings-npy", help="use precomputed embeddings instead of running the GAE")
    g.add_argument("--embeddings-mapping", help="json with row_to_event_uid for --embeddings-npy")
    g = ap.add_argument_group("clustering / novelty")
    g.add_argument("--min-cluster-size", type=int, default=20)
    g.add_argument("--min-samples", type=int, default=10)
    g = ap.add_argument_group("MITRE ATT&CK semantic mapping")
    g.add_argument("--mitre-cache-dir", default="cache/mitre")
    g.add_argument("--corpus-file", help="explicit corpus file (legacy 15-entry JSON allowed, flagged in output)")
    g.add_argument("--embedder", choices=["sbert", "lexical"], default="sbert",
                   help="'lexical' = offline TF-IDF stand-in for dev/tests; scores are NOT comparable to SBERT")
    g.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="HF model name or local folder")
    g.add_argument("--top-k", type=int, default=5)
    g.add_argument("--candidate-threshold", type=float, default=0.35)
    g.add_argument("--weak-threshold", type=float, default=0.25)
    g.add_argument("--platform", action="append", dest="platforms",
                   help="restrict techniques to a platform (repeatable; default Windows; 'all' = no filter)")
    g.add_argument("--rebuild-embeddings", action="store_true", help="ignore the embedding cache for this run")
    g = ap.add_argument_group("noise review")
    g.add_argument("--noise-review-top-n", type=int, default=10, help="noise units given full semantic analysis")
    g.add_argument("--noise-group-min-size", type=int, default=2)
    g = ap.add_argument_group("GraphRAG")
    g.add_argument("--retrieval-depth", type=int, default=2)
    g.add_argument("--neo4j", action="store_true", help="also sync to Neo4j (env NEO4J_URI/USER/PASSWORD); falls back to local")
    g.add_argument("--ontology", help="ontology-module JSON in the subhany.ontology-entities contract")
    g = ap.add_argument_group("report")
    g.add_argument(
    "--llm",
    choices=["auto", "none", "ollama", "gemini"],
    default="auto",
    help="LLM backend: auto, none, ollama, or gemini",
)
    g.add_argument("--ollama-model", default=DEFAULTS.ollama_model, help="env: SUBHANY_OLLAMA_MODEL")
    g.add_argument("--ollama-host", default=DEFAULTS.ollama_host, help="env: OLLAMA_HOST")
    g.add_argument("--ollama-timeout", type=int, default=DEFAULTS.ollama_timeout, help="seconds per LLM request")
    g.add_argument("--ollama-num-ctx", type=int, default=DEFAULTS.ollama_num_ctx, help="context window requested from Ollama")
    g.add_argument("--report-top-n", type=int, default=DEFAULTS.report_top_n, help="findings shown in the primary report (and analysed by the LLM)")
    g = ap.add_argument_group("output")
    g.add_argument("--export-dir", help="write result files here (default: nothing is written)")
    g.add_argument("--json", action="store_true", help="print the full RunResult as JSON on stdout (no stage text)")
    g.add_argument("--no-report", action="store_true", help="omit the final report text from the terminal")
    g.add_argument("--quiet", action="store_true", help="print only the summary")
    return ap


def config_from_args(a):
    from src.service.config import PipelineConfig
    plats = tuple(a.platforms or ["Windows"])
    if any(p.lower() == "all" for p in plats):
        plats = ()
    return PipelineConfig(
        input_path=a.input, input_format=a.input_format, project_root=PROJECT_ROOT,
        checkpoint_path=a.pretrained_model, feature_schema_path=a.pretrained_schema,
        precomputed_embeddings=a.embeddings_npy, precomputed_mapping=a.embeddings_mapping,
        min_cluster_size=a.min_cluster_size, min_samples=a.min_samples, mitre_cache_dir=a.mitre_cache_dir,
        corpus_file=a.corpus_file, embedder=a.embedder, embedding_model=a.embedding_model, top_k=a.top_k,
        candidate_threshold=a.candidate_threshold, weak_threshold=a.weak_threshold, platforms=plats,
        rebuild_embeddings=a.rebuild_embeddings, noise_review_top_n=a.noise_review_top_n,
        noise_group_min_size=a.noise_group_min_size, retrieval_depth=a.retrieval_depth, neo4j_enabled=a.neo4j,
        ontology_path=a.ontology, llm=a.llm, ollama_model=a.ollama_model, ollama_host=a.ollama_host,
        ollama_timeout=a.ollama_timeout, ollama_num_ctx=a.ollama_num_ctx, report_top_n=a.report_top_n, export_dir=a.export_dir)


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    for flag, val in (("--top-k", a.top_k), ("--min-cluster-size", a.min_cluster_size)):
        if val < 1:
            print(f"error: {flag} must be >= 1", file=sys.stderr)
            return 2
    if not (0.0 <= a.weak_threshold <= a.candidate_threshold <= 1.0):
        print("error: need 0 <= --weak-threshold <= --candidate-threshold <= 1", file=sys.stderr)
        return 2

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    from src.cli.render import render_header, render_stage, render_summary
    from src.service.pipeline import PipelineService

    cfg = config_from_args(a)
    if not a.json:
        _safe_print(render_header(cfg.to_dict()))
    cb = None if (a.json or a.quiet) else (lambda st: _safe_print(render_stage(st)))
    res = PipelineService(cfg, on_stage=cb).run()
    if a.json:
        sys.stdout.write(json.dumps(res.to_dict(), indent=2, default=str, ensure_ascii=True) + "\n")
    else:
        _safe_print(render_summary(res, show_report=not a.no_report, detail_units=a.report_top_n))
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
