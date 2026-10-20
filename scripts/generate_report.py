"""
scripts/generate_report.py
------------------------------
CLI: "Generate a threat report for cluster 3"

Usage:
    python scripts/generate_report.py --cluster-id 3
    python scripts/generate_report.py --cluster-id 3 --llm template   # no model server required
    python scripts/generate_report.py --all                           # every retrievable cluster
    python scripts/generate_report.py --list                          # show available cluster_ids
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.rag.llm_clients import OllamaLLMClient, TemplateLLMClient
from src.rag.rag_pipeline import generate_cluster_report, generate_reports_for_clusters
from src.rag.report_schema import ReportParseError
from src.rag.retriever import RagIndex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster-id", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--llm", choices=["ollama", "template"], default="ollama",
                     help="'template' is a non-LLM stand-in for testing without a model server. See README.")
    ap.add_argument("--ollama-model", default="llama3.1")
    ap.add_argument("--reports-dir", default="output/reports")
    args = ap.parse_args()

    index = RagIndex(output_dir=args.output_dir)

    if args.list:
        print("Available cluster_ids:", index.available_cluster_ids())
        return

    if args.llm == "template":
        print("WARNING: --llm template is a non-LLM stand-in for testing, NOT real report generation. See README.")
        llm = TemplateLLMClient()
    else:
        llm = OllamaLLMClient(model=args.ollama_model)

    os.makedirs(args.reports_dir, exist_ok=True)

    if args.all:
        cluster_ids = index.available_cluster_ids()
        reports = generate_reports_for_clusters(cluster_ids, index, llm, on_error="skip")
        for r in reports:
            path = os.path.join(args.reports_dir, f"cluster_{r.cluster_id}_report.json")
            with open(path, "w") as f:
                json.dump(r.as_dict(), f, indent=2)
        print(f"\nGenerated {len(reports)}/{len(cluster_ids)} reports in {args.reports_dir}/")
        return

    if args.cluster_id is None:
        ap.error("Provide --cluster-id N, --all, or --list.")

    try:
        report = generate_cluster_report(args.cluster_id, index, llm)
    except KeyError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except ReportParseError as e:
        print(f"ERROR: report generation failed validation: {e}")
        sys.exit(1)

    path = os.path.join(args.reports_dir, f"cluster_{report.cluster_id}_report.json")
    with open(path, "w") as f:
        json.dump(report.as_dict(), f, indent=2)

    print(f"\n{'='*60}")
    print(report.title)
    print('='*60)
    print(f"\n{report.summary}\n")
    print("Observed Behavior:")
    print(f"  {report.observed_behavior}\n")
    print("Likely Techniques:")
    if not report.likely_techniques:
        print("  (none -- no confident technique match in the retrieved context)")
    for t in report.likely_techniques:
        print(f"  - {t.technique_id} {t.technique_name} [{t.confidence_label}, sim={t.similarity_score}]")
        print(f"      {t.reasoning}")
    print(f"\nRisk Assessment:\n  {report.risk_assessment}")
    print(f"\n(generated_by={report.generated_by}, cluster_size={report.cluster_size}, "
          f"max_novelty_score={report.max_novelty_score})")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
