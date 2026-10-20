# Semantic MITRE Mapping — Methodology (paper-ready)

## Method summary

Each non-noise HDBSCAN cluster is converted into a deterministic text
narrative aggregating its member events' dominant EventID distribution,
implicated hosts and processes, network destinations, temporal span, and
representative evidence text (command lines / audit messages,
deduplicated). This narrative is embedded with a sentence-level
transformer (Sentence-BERT) and compared via cosine similarity against
pre-embedded MITRE ATT&CK technique descriptions, using the same encoder
for both sides of the comparison. The top-K most similar techniques are
returned per cluster with their similarity score, with a minimum-score
floor below which no technique is reported as a confident match. Mapping
therefore operates on aggregated cluster-level behavior, not individual
log events, and is derived entirely from learned semantic similarity, not
hand-written detection conditions.

## Why cluster-level, not event-level

A single log event is frequently ambiguous in isolation — the same
EventID (e.g., Sysmon Process Access) is logged for both benign
inter-service queries and credential-access attempts, differing only in
specific field combinations an event-level view may not surface
coherently. Aggregating to the cluster level, where cluster membership is
already informed by the graph-structural and feature-based GAE embedding
(i.e., clusters already reflect shared process/host/temporal context,
not just superficial field equality), gives the mapping stage a richer,
pre-correlated behavioral signal to work from — analogous to how prior
work on GNN-based event-sequence reconstruction found sequence-level
representations preserve more tactical coherence than independently
scored events (cf. cohesion gains reported for GNN-based sequencing over
PID-window and session-based heuristics in related literature on
graph-based log correlation).

## Why embedding similarity, not rule-based matching

A rule-based mapper requires an explicit, hand-maintained condition for
every technique it can recognize, is brittle to paraphrased or
differently-structured evidence, and cannot generalize beyond the
conditions its author anticipated. Replacing this with embedding
similarity against official technique descriptions lets the mapper
recognize techniques whose *behavioral description* is semantically close
to the cluster's narrative, without requiring a bespoke condition per
technique — trading hand-authored precision for broader recall and lower
maintenance cost as the technique corpus grows. This mirrors the general
shift from lexical (e.g., TF-IDF) to dense semantic representations
observed elsewhere in security-log analysis, where literal token overlap
is a poor proxy for behavioral similarity between differently-phrased
descriptions of the same underlying technique.

## Scalability and generalization

Because technique embeddings are computed once and cached, inference cost
per run scales with the number of clusters, not the number of raw events
— consistent with the broader design goal of this pipeline (compress
events → clusters → a small number of mapping decisions). Because the
mapper has no technique-specific code path, extending coverage to a new
MITRE technique requires only adding one entry (id/name/tactic/
description) to the technique corpus and re-embedding it — no new
conditional logic, no redeployment of mapping code.

## Honest scope note for publication

This design deliberately omits two things common in more elaborate
mapping systems: (1) a symbolic/ontology layer fused with the semantic
score (explicitly out of scope here, owned separately), and (2) any
supervised fine-tuning of the embedding model on labeled
security-technique text. Both would likely raise precision at the cost
of added complexity and labeled-data dependency; reporting results from
this system should describe it as a **zero-shot semantic retrieval
baseline** over cluster narratives, not compare its raw similarity scores
against metrics from a fine-tuned or hybrid classifier evaluated on a
labeled technique-mapping test set, since those are not measuring the
same thing.
