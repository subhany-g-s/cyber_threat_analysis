# SUBHANY cyber threat analysis pipeline

Parse logs -> event graph -> **pretrained** GAE embeddings (never retrained) -> HDBSCAN clusters -> novelty ->
profiles -> semantic ATT&CK mapping -> GraphRAG retrieval -> **prioritized analyst report** (optionally written up
by a local LLM through Ollama, with a labelled deterministic fallback).
Processing lives in a UI-independent service; the terminal CLI is one front end (see `docs/dashboard_readiness.md`).
Everything stays in memory; files are written only with `--export-dir`.

## Quick start (Windows, cmd or PowerShell)

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts\update_mitre_corpus.py                 :: one explicit ATT&CK download (internet needed once)
python run_pipeline.py --input data\sample_2000_events.json --export-dir outputs\run1
```

`torch` / `torch-geometric` may need the platform-specific command from pytorch.org / pyg.org. The first SBERT run
downloads `all-MiniLM-L6-v2` from Hugging Face. Windows was not available when this was built (verified on Linux, Python 3.13).

| Task | Command |
|---|---|
| Full analysis, report printed in the terminal | `python run_pipeline.py --input data\sample_2000_events.json` |
| Write result files | add `--export-dir outputs\run1` |
| CSV / JSON-array input | `--input data\events.csv --input-format csv` |
| Machine-readable result | add `--json` |
| Update ATT&CK corpus | `python scripts\update_mitre_corpus.py` (`--list`, `--info`, `--use <version>`, `--from-file bundle.json`) |
| Rebuild technique embeddings | `python scripts\rebuild_technique_embeddings.py` |
| Evaluate thresholds on your labels | `python scripts\evaluate_mapping.py --examples labels.json` |
| Offline stand-in embedder (dev/tests only) | `--embedder lexical` |
| Neo4j (optional) | set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, then `--neo4j` |
| Tests | `python -m pytest tests -q` |

Analysis never downloads anything. Without a cached ATT&CK corpus stage 8 fails with an instruction to run the updater.

## The report

The default report (`analyst_report.md`, also printed after a run) has five parts:

* **A Executive summary** - scope, findings by priority, main reasons, evidence limitations, analysis mode.
* **B Prioritized findings** - each with priority (documented criteria), status, what was observed, why it matters,
  evidence references (`EV:<id>` for events), a limited ATT&CK assessment, missing evidence, benign explanations,
  next steps, what can / cannot be concluded.
* **C Overall assessment** - patterns, whether one explanation fits, what can and cannot be concluded.
* **D Investigation plan** - ordered actions, each with why, which finding, and what would strengthen or weaken it.
* **E Technical appendix** - pointers to the full detail, which is kept separately (`evidence_report.*`, `run_result.json`, `noise_review.json`, CSVs).

How findings are built (no detection algorithm is changed; this layer only reads the results):

* Priority is a transparent ladder (Critical/High/Medium/Low/Informational) from novelty tier, **observed indicators**
  (hidden bidirectional characters, script/screensaver extensions, encoded commands, autostart registry keys,
  staging paths, public destinations) and noise/corroboration. ATT&CK similarity scores are **not** part of it.
* Units are merged only when they share a process id or an identical indicator artefact. Same host, process name,
  cluster or technique never merges anything; same-host/time neighbours are listed as "related, not merged".
* Noise units outside `--noise-review-top-n` are still scanned for observable indicators and surfaced if they carry one.
* A technique is attached to a finding with its **own** evidence: events whose individual text similarity reaches the
  candidate threshold. Otherwise it is marked unit-level context only and its fit is capped at "weak".
  Similarity is a ranking signal, not a probability and not proof.

## LLM analysis (Ollama) and fallback

```bat
ollama serve
ollama pull llama3.1
python run_pipeline.py --input data\sample_2000_events.json --llm ollama --ollama-model llama3.1
```

* `--llm auto` (default) uses Ollama if it answers, otherwise the deterministic fallback; `--llm none` never calls it.
  Defaults live in `src/service/config.py`; override with `--ollama-model`, `--ollama-host`, `--ollama-timeout`,
  `--ollama-num-ctx`, or the env vars `SUBHANY_OLLAMA_MODEL` / `OLLAMA_HOST`.
* The model gets one compact JSON evidence package per finding (top `--report-top-n`, default 5) and must answer in a
  fixed JSON schema. Output is validated in code: unknown evidence/technique/event ids, paths, IPs, hosts or files that
  are not in the package, over-claims ("confirmed attack", attribution) and contradictory ATT&CK fits cause the whole
  analysis for that finding to be rejected; a "plausible" fit without event-level support is capped to "weak".
  The model cannot change scores, priorities, merging or evidence.
* Failure never stops the run: no retries, the first connection failure (or two consecutive failures) ends LLM use, and
  the report says so. The report header always states the mode: DETERMINISTIC, LLM-ASSISTED or LLM-PARTIAL; LLM-written
  items are tagged `[LLM]`.

## Outputs (`--export-dir outputs/<name>`)

| File | Contents |
|---|---|
| `analyst_report.md` / `analyst_report.json` | the prioritized report (primary) and its structured form with the prioritization rules and evidence index |
| `evidence_report.md` / `.txt` / `.json` | technical appendix: every unit, all candidates with scores, passages, graph paths, validation |
| `run_result.json`, `noise_review.json` | run metrics; every noise event with its measured signals |
| `embedding_clusters.csv`, `novelty_scores.csv`, `cluster_profiles_v2.json`, `semantic_mitre_mappings.json`, `rag_ready_clusters.json`, `ontology_findings.json` | stage outputs |

`outputs/` is git-ignored; earlier exports are kept in `outputs/archive/`. The pretrained GAE lives in `models/`.

## The ten stages

1 parse -> 2 normalize -> 3 event graph -> 4 GAE (checkpoint/schema validation, inference only) -> 5 HDBSCAN ->
6 novelty -> 7 cluster profiles + **noise review** -> 8 semantic ATT&CK mapping -> 9 GraphRAG -> 10 report.
Each stage prints status (`OK/WARN/FAIL/SKIP`), real metrics, warnings, elapsed time and a plain explanation.
A failure in stages 1-7 skips the rest (nothing downstream is meaningful); stage 8/9/10 problems (missing corpus,
Neo4j down, LLM unavailable) degrade with a warning.

## Layout

```
run_pipeline.py            launcher
src/service/   config, result, adapters, gae_stage, pipeline (orchestrator), export
src/cli/       render (terminal), main (flags)
src/rag/       findings (deterministic analyst layer), finding_llm (Ollama analysis + validation), analyst_render,
               evidence_report + report_render (technical appendix), llm_clients, legacy file-based RAG modules
src/mitre|mapping|noise|graphrag|ontology|graph|gae|clustering|novelty|profiling   pipeline stages
models/        pretrained GAE checkpoint + feature schema      cache/mitre/   versioned ATT&CK corpus + embeddings
data/ config/ docs/ scripts/ tests/    sample data, legacy corpus, contracts/methodology, utilities, tests
output/        legacy-pipeline results kept for the legacy scripts (see docs/CLEANUP_AUDIT.md)
outputs/       new run outputs (git-ignored); outputs/archive/ holds earlier exports
```

## Semantics you should hold onto

* **Similarity is not probability.** Scores are cosine similarity between text embeddings. Defaults
  (candidate 0.35, weak 0.25) are **uncalibrated** - no labelled ground truth ships with this repo. Candidates are
  leads; "NO_MATCH" and "WEAK_ONLY" are valid outcomes and no technique is forced.
* **Noise is preserved.** HDBSCAN noise (cluster -1) is never dropped or assigned to a normal cluster. Every noise
  event is listed with its attributes, graph neighbours, novelty signals and a rank. Events sharing a process id
  form small review groups (`N#`); the rest are single events (`E#`). The top `--noise-review-top-n` units
  (default 10, ordered by novelty) get semantic mapping, retrieval and a report section. Listed "signals" are
  measured properties, not causes; noise is not assumed malicious.
* **GraphRAG claims are measured.** Each unit reports whether graph traversal added evidence beyond its own events.
  Local mode (NetworkX) is the default; Neo4j is optional and the run says which mode was used.
* **Report sections are separated:** observed facts / computed findings / semantic candidates / retrieved ATT&CK
  references / graph evidence / ontology facts / hedged inferences / uncertainty / recommended steps. Facts are
  built deterministically with evidence ids. An LLM may only add prose, which is rejected if it cites unknown
  techniques/events/evidence or over-claims. Mock/template output is labelled MOCK.
* **Ontology:** the ontology module is a separate piece of work. `docs/ontology_contract.md` defines a versioned
  JSON contract in both directions; nothing about its real schema is assumed.

## How to read the results

Novelty means "unusual in this dataset", not "malicious". A finding is a lead for an analyst: "Suspicious" means an
observable indicator was seen in the events themselves; "Needs investigation" means unusual with limited direct
indicators. Nothing in the report confirms an incident, names an actor or proves an ATT&CK technique.

## Known limitations

* Not exercised in the build environment: SBERT (model download blocked), a real Ollama server, a live Neo4j server,
  Windows. Ollama is covered by a local mock HTTP server and fakes; sample runs here used `--embedder lexical`.
* Lexical-embedder scores measure word overlap only; at default thresholds it yields no ATT&CK candidates. Lowering
  thresholds (`--candidate-threshold 0.08 --weak-threshold 0.04`) exercises those code paths but is not a real result.
* Similarity thresholds are uncalibrated (no labelled ground truth ships with the repo).
* Indicator classes are a small, fixed list of observable markers, not a detection ruleset.
* Neo4j queries are validated only against a fake driver.
* The older file-based RAG path (`scripts/generate_report.py`, `src/rag/retriever.py` ...) and
  `scripts/legacy_run_pipeline_v1.py` are retained for compatibility; they cover clusters only.
* Prior-phase README: `docs/README_prior_phase.md`.
