# Cleanup and change audit

Nothing was deleted that is original data, evidence, a user-created report or a model asset. Detection outputs were
re-checked against the original code (md5 of `embedding_clusters.csv`, `novelty_scores.csv`,
`semantic_mitre_mappings.json`, `cluster_profiles_v2.json` identical on the sample run, lexical embedder).

## New files
`src/rag/findings.py` (deterministic findings, priority, merge, plan), `src/rag/finding_llm.py` (Ollama analysis + validation),
`src/rag/analyst_render.py` (primary report renderer), `tests/test_analyst_report.py`, `.gitignore`, `outputs/` (+ `.gitkeep`,
`archive/README.md`), `models/`, this file.

## Moved
| From | To | Why |
|---|---|---|
| `output/gae_model.pt`, `output/gae_feature_schema.json` | `models/` | model assets were mixed with generated output; code defaults/tests/docs updated |
| `run_out/` | `outputs/archive/run_out_sbert_2000/` | earlier generated export; kept, not deleted |
| `run_out_noise100/` | `outputs/archive/run_out_noise100/` | same |

## Removed (regenerable only)
`.pytest_cache/`, all `__pycache__/` directories.

## Merged / consolidated in code
Per-unit LLM prose path in `evidence_report.py` (`llm_prompt`, `validate_llm_prose`, `apply_llm`, mock labelling) was replaced by the
finding-level analysis in `finding_llm.py`; Ollama defaults now live only in `src/service/config.py`.

## Retained although they look redundant (not provably unused)
* `output/` (legacy cluster results, `reports/`): defaults of the legacy scripts/`RagIndex` point at it; they are earlier-phase results.
* `scripts/legacy_run_pipeline_v1.py`, `scripts/generate_report.py`, `src/rag/retriever.py`, `rag_pipeline.py`, `rag_export.py`: legacy file-based RAG, still covered by tests.
* `config/mitre_technique_corpus.json` (15-entry legacy corpus): selectable with `--corpus-file`.
* `docs/examples/windows_mapping_eval_v2.json.bak`: no proof it is a pure duplicate.
* `docs/README_prior_phase.md`: history.
* `.venv` of the original archive: untouched in the original; not part of the cleaned tree.

## Dependencies
None added. Ollama is called with the standard library only.

## Pre-existing lint
`pyflakes` reports unused imports in untouched legacy files (e.g. `src/graph/graph_builder.py`); left alone deliberately.
