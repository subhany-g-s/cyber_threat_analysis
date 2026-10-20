# Ontology integration contract

The ontology creation/population module is separate work. **It is not in this repository and its schema is not
assumed.** This pipeline therefore defines two small, versioned JSON contracts. `entity_type`, `relationship_type`
and `ontology_uri` are opaque strings that the ontology module owns; they are stored verbatim.

Machine-readable schemas: `src/ontology/schemas/ontology-entities.v1.schema.json`,
`src/ontology/schemas/pipeline-findings.v1.schema.json`. Validators: `src/ontology/contract.py`.
Examples: `docs/examples/ontology_entities.example.json` (placeholders only).

## 1. `subhany.ontology-entities` v1.x  (ontology module -> pipeline)

Top level: `contract`, `contract_version` (semver; major must be 1), `producer{name,version?}`, `entities[]`,
`relationships[]`.

Entity: `entity_id` (unique), `entity_type`, `canonical_name`, `ontology_uri?`, `aliases?`, `identifiers?[{scheme,value}]`,
`source_id`, `status` (`observed` | `inferred`), `confidence?{value 0..1, method, calibrated?}`, `evidence?[{text|reference|event_uid}]`.

Relationship: `source_entity_id`, `target_entity_id` (must exist in the document), `relationship_type`, `source_id`, `status`,
`confidence?`, `evidence?`.

* **Confidence** is the producer's own number; `method` says what it means; it counts as calibrated only if `calibrated: true`.
  It is never combined with this pipeline's cosine similarity.
* **`observed`** = stated by a source; **`inferred`** = derived, and must carry evidence or a confidence method.
* **Entity resolution (evidence only).** Links to this pipeline's events are created only from
  (a) an explicit `evidence.event_uid`, or (b) an exact match on a strong identifier scheme:
  `event_uid`, `ip_address`, `file_sha256`, `file_sha1`, `file_md5`, `process_guid`.
  A matching name/alias alone is reported under `possible_name_matches` with `identity_asserted: false` - no edge is created.
  Same type+name with different ids is warned about and never merged. Unlinked entities remain separate and are counted.
  Event citations that do not exist in the run are counted as `dangling_event_references`.
* In the graph: `OntologyEntity` nodes, `ONTOLOGY_RELATION` edges (the ontology's relationship type is a property),
  `EVIDENCED_BY` (entity -> event) and `RESOLVED_TO` (entity -> observed IP entity). Facts reach the report only through
  these edges and are labelled as coming from the ontology module.

Enable with `--ontology path\to\entities.json`. An invalid file is rejected with the first error and the run continues without it.

## 2. `subhany.pipeline-findings` v1.x  (pipeline -> ontology module)

Written as `ontology_findings.json` by `--export-dir`, or built in code with `src.ontology.export.export_findings(run_result)`.
`mapping{corpus_version, corpus_hash, attack_version, embedding_model, text_preprocessing, thresholds, similarity_semantics, is_standin_embedder}`;
one record per event: `event_uid`, `source_id`, `source_ref` (file:line), `cluster_id` (-1 = noise), `unit_id`, `unit_scope`
(`cluster | noise_group | noise_event | unreviewed_noise`), `novelty{score,label,signals,meaning}`, `behavior_summary`,
`technique_candidates[{technique_id,technique_name,similarity_score,status,rank,scope,mapping_method,reference_url}]`,
`supporting_event_uids`.
`similarity_score` is cosine similarity, **not** a probability; candidates are leads, not confirmed techniques.
Events in noise units that were not analysed carry `unit_scope: unreviewed_noise` and no candidates.

## 3. Optional Neo4j ingestion

`--neo4j` syncs the run's graph (Event, Entity, Cluster, TechniqueCandidate, ReferenceDocument, OntologyEntity) with
`MERGE` on unique-constrained stable keys (idempotent). Technique links are `CANDIDATE_TECHNIQUE` edges flagged
`inferred`. If the server or driver is unavailable the run says so and continues in local mode.
Credentials come only from environment variables. **Only tested against a fake driver.**
