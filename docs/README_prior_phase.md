> **Superseded by the root README.md.** (The pretrained GAE files `gae_model.pt` / `gae_feature_schema.json` now live in `models/`, not `output/`.) This is the previous phase README, kept for history.

# cag-system — Cluster-Level Semantic MITRE Mapping

Continues from an existing, trained GAE + clustering pipeline (not
retrained here) and adds: cluster profiling → text narrative → SBERT
embedding similarity → top-K MITRE technique candidates → RAG-ready
export. No rule-based or per-event mapping anywhere in this stage.

## Project structure

```
cag-system/
├── config/
│   └── mitre_technique_corpus.json   # 15 technique id/name/tactic/description entries
├── cache/                            # technique_embeddings.npy + technique_ids.json (written by embed_technique_corpus.py)
├── data/
│   └── sample_2000_events.json
├── output/                           # pretrained GAE artifacts (gae_model.pt, gae_feature_schema.json)
│                                      # + this stage's outputs (see below)
├── src/
│   ├── graph/           # parser.py, normalizer.py, graph_builder.py -- UNCHANGED from the prior phase
│   ├── gae/             # gae_features.py, gae_model.py, train_gae.py -- UNCHANGED, loaded not retrained
│   ├── clustering/      # cluster_embeddings.py -- UNCHANGED (HDBSCAN on raw 32D embeddings)
│   ├── profiling/
│   │   └── cluster_profiler.py    # NEW: cluster -> structured profile + text narrative
│   ├── mapping/
│   │   ├── embedders.py           # NEW: TextEmbedder interface + SBertEmbedder (production)
│   │   ├── semantic_mapper.py     # NEW: cosine-similarity cluster -> top-K technique mapping
│   │   └── embed_technique_corpus.py  # NEW: one-time offline corpus embedding + cache
│   └── rag/
│       └── rag_export.py          # NEW: assembles the RAG-ready per-cluster bundle
├── scripts/
│   └── run_pipeline.py            # end-to-end driver (parse -> ... -> RAG export)
├── tests/
│   ├── test_profiling_and_mapping.py  # 7 unit tests, synthetic data, no network required
│   └── tfidf_embedder.py              # offline TextEmbedder stand-in for tests/dev (see below)
└── requirements.txt
```

## Run it

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --input data/sample_2000_events.json
```

This uses `SBertEmbedder` (`all-MiniLM-L6-v2`) by default — the
production path. It needs network access to Hugging Face's model hub on
first run (cached locally afterwards). If you're offline or want a fast
dev loop, pass `--embedder tfidf` — see the "What was actually tested
here" section below for why this matters.

Outputs land in `output/`:
- `cluster_profiles_v2.json` — structured profile + text narrative per cluster
- `semantic_mitre_mappings.json` — top-K technique candidates per cluster
- `rag_ready_clusters.json` — the combined, RAG-ready bundle

## What was actually tested in this sandbox, and what wasn't

Being direct about this because it affects how to read the sample output:

**Could not test:** `SBertEmbedder` end-to-end. This sandbox's network
egress doesn't reach `huggingface.co` (model weights live there), so
`SentenceTransformer("all-MiniLM-L6-v2")` cannot download here. The class
is implemented correctly against the `sentence-transformers` API
(installed, v6.1.0, confirmed importable) and follows the same interface
the test suite validates against the `TfidfEmbedder` stand-in — but I
have not personally observed it produce a real embedding in this
environment. **Run it on your own machine (with normal internet access)
before trusting its output**, and ideally re-run
`tests/test_profiling_and_mapping.py` with `SBertEmbedder` swapped in for
`TfidfEmbedder` once you can.

**Was tested, real output:** everything else. `scripts/run_pipeline.py
--embedder tfidf` ran the real 3,000-event sample through the real
trained GAE (loaded, not retrained), real HDBSCAN clustering (65
clusters, 729 noise points), real cluster profiling, and the real
mapping/RAG-export code — only the embedding backend was swapped for a
TF-IDF stand-in so it could run without network access. All 7 unit tests
pass against that same stand-in.

**What this means for the similarity scores you'll see in the sample
output:** they're low (0.04–0.09, all `NO_CONFIDENT_MATCH` against the
default 0.35 floor). TF-IDF measures literal word overlap, not semantic
meaning — a cluster narrative built from Windows Security-audit
boilerplate text ("An attempt was made to access an object...") shares
almost no vocabulary with a technique description like "Adversary or
script invokes PowerShell...", even when the underlying behavior is
related. **Do not read these specific numbers as the system's real
performance** — they're a mapping-logic smoke test, not a measurement of
the semantic mapper's actual discriminative power, which depends on
SBERT actually running.

## A real, important limitation found during this testing (not hypothetical)

The dataset's one confirmed-interesting event — the RTLO-masquerading
`.scr` file (`event_uid 43dc5a8d-...`) launched by `explorer.exe`,
previously identified by the old per-event rule engine as
`T1036.002`/`T1204.002` evidence — **is HDBSCAN noise (`cluster_id =
-1`) under this clustering configuration**, confirmed directly:

```
cluster_id for target event: -1  is_noise: True
```

Per this project's explicit design rule (and your stated constraint:
mapping must be cluster-level only), noise is excluded from Stage 7
entirely — `cluster_profiler.py` raises if you try to profile `cluster_id
== -1`, and `semantic_mapper.py` raises if a noise profile somehow
reaches it anyway. **The practical consequence: in the current
configuration, this cluster-only pipeline produces zero MITRE candidates
for the one event in this sample dataset that's actually worth
flagging.** This is a direct, structural tension between two of your
own requirements — "cluster-level only" and "don't lose real signal" —
not a bug I can quietly code around. Options, if this matters for your
evaluation:
1. Tune `min_cluster_size`/`min_samples` lower so rarer-but-coherent
   behavior forms small clusters instead of falling to noise (cheapest
   fix, may increase noise-cluster fragmentation elsewhere).
2. Accept the gap for this stage and rely on your existing novelty
   scoring (Phase 2) to surface noise events for separate human review —
   i.e., semantic MITRE mapping covers *clustered* behavior, novelty
   scoring covers *everything else*, and both feed the eventual RAG
   report rather than one subsuming the other.
3. Add a deliberately separate, clearly-labeled "singleton mapping" path
   for individual noise events — this would be event-level, so it must
   not be presented as part of the cluster-level mapping stage; flagging
   this instead of silently doing it, since it conflicts with your stated
   constraint.

I didn't pick one of these for you — it's a scope decision, not an
implementation bug.

## Stage 4 (new): RAG report generation

```
INPUT (cluster_id) -> RETRIEVER -> CONTEXT BUILDER -> LLM -> FINAL REPORT
```

```bash
python scripts/generate_report.py --list                        # show available cluster_ids
python scripts/generate_report.py --cluster-id 3                # requires a running local Ollama server (see below)
python scripts/generate_report.py --cluster-id 3 --llm template # no model server required, for testing/demo
python scripts/generate_report.py --all --llm template          # batch, writes output/reports/cluster_N_report.json
```

**New files:** `src/rag/retriever.py` (structured lookup + optional
semantic search over cluster narratives), `src/rag/context_builder.py`
(formats retrieved context into the prompt block), `src/rag/prompt_templates.py`,
`src/rag/llm_clients.py` (`OllamaLLMClient` production / `TemplateLLMClient`
test stand-in), `src/rag/report_schema.py` (validated output contract +
anti-hallucination check), `src/rag/rag_pipeline.py` (orchestrator),
`scripts/generate_report.py` (CLI). Also added `src/novelty/novelty_detection.py`
(the existing, unmodified Phase-2 module) and extended `run_pipeline.py`
to compute and persist `novelty_scores.csv` + `embedding_clusters.csv` —
these didn't exist in this project's output before and the RAG retriever
needs them to join full cluster membership against novelty scores.

**Why Ollama, not a hosted API:** matches this project's original stated
architecture ("Local LLM using Ollama") and keeps cluster narratives
(which are built from real event data) from leaving the analyst's own
environment — relevant for exercise/operational log data like this
dataset's.

**Anti-hallucination, enforced in code, not just prompted:**
`report_schema.py::parse_llm_json_response` checks every `technique_id`
the LLM's response cites against the exact set of `technique_id`s present
in the context it was given. A cited technique not in that set raises
`ReportParseError` — the report is rejected, not silently kept. Verified
by `tests/test_rag_pipeline.py::test_schema_rejects_hallucinated_technique_id`.

### What was actually tested here (same honesty as the SBERT note above)

**Could not test:** `OllamaLLMClient` end-to-end — no local Ollama server
running in this sandbox. Confirmed it fails with a clean, expected
`URLError: Connection refused` (not a crash in this project's code) when
no server is present. Install Ollama and pull a model
(`ollama pull llama3.1`) on your own machine before trusting real report
output.

**Was tested, real output:** the full retriever → context builder →
schema validation chain, using `TemplateLLMClient` (a non-LLM stand-in
that assembles a schema-valid JSON report directly from structured
context, clearly labeled `generated_by: "template"` everywhere). Ran
`--all` against the real 65-cluster sample output: **65/65 reports
generated successfully**, each traceable back to its cluster's
`representative_event_uids`. All 9 unit tests pass, including the
hallucination-rejection test. **The prose quality of `TemplateLLMClient`'s
output is not representative of real LLM output** — it's a structural
test fixture (see its docstring), not a writing sample.

## Removed from the prior (incorrect) system

- `config/mitre_rules.json`, `mitre_mapper.py` (rule engine, per-event
  mapping, bitmask/suffix conditions, co-occurrence boosts) — fully
  retired, not imported anywhere in this stage.

## requirements.txt additions

```
sentence-transformers>=2.2
scikit-learn>=1.3   # TF-IDF stand-in for tests/offline dev only
```
