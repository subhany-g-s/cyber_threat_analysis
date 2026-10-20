"""
src/graphrag/schema.py
------------------------
The PIPELINE-INTERNAL graph schema used by the GraphRAG layer (local
NetworkX store and the optional Neo4j store share it).

IMPORTANT - these are storage labels for this project's own outputs. They
are NOT the team's ontology: the separate ontology module owns its classes
and relationship vocabulary, and its records enter this graph only as
generic `OntologyEntity` nodes / `ONTOLOGY_RELATION` edges that carry the
module's own `entity_type` / `relationship_type` strings verbatim.

Node kinds (stable key in brackets):
    Event              [event_uid]            one log event (deterministic uuid5 of the raw record)
    Entity             [<type>:<value>]       an OBSERVED attribute value: host | process_image | user | ip_address
    Cluster            [<run_id>:<unit_id>]   a behaviour cluster (C7) or a noise unit (N2 group / E5 single event)
    TechniqueCandidate [technique_id]         an ATT&CK technique proposed as a CANDIDATE (never "confirmed")
    ReferenceDocument  [mitre:<tid>:<hash8>:<passage>]  trusted reference passage from the cached ATT&CK corpus
    OntologyEntity     [entity_id]            record supplied by the ontology module (contract v1)

Relationship types (a fixed whitelist - Cypher cannot parameterise types, so
only these strings are ever interpolated):
"""

EVENT = "Event"
ENTITY = "Entity"
CLUSTER = "Cluster"
TECHNIQUE = "TechniqueCandidate"
REFERENCE = "ReferenceDocument"
ONTOLOGY = "OntologyEntity"

NODE_KINDS = (EVENT, ENTITY, CLUSTER, TECHNIQUE, REFERENCE, ONTOLOGY)

# event <-> event (from src/graph/graph_builder.py; status = observed correlation)
SAME_PID = "SAME_PID"
SAME_HOST_TEMPORAL = "SAME_HOST_TEMPORAL"
EVENT_EVENT_RELS = (SAME_PID, SAME_HOST_TEMPORAL)

# event -> entity (observed attributes)
ON_HOST = "ON_HOST"
SOURCE_PROCESS = "SOURCE_PROCESS"
TARGET_PROCESS = "TARGET_PROCESS"
BY_USER = "BY_USER"
CONNECTED_TO = "CONNECTED_TO"
EVENT_ENTITY_RELS = (ON_HOST, SOURCE_PROCESS, TARGET_PROCESS, BY_USER, CONNECTED_TO)

# membership (computed by GAE + HDBSCAN). Noise units are NOT clusters: separate relationship.
IN_CLUSTER = "IN_CLUSTER"
IN_NOISE_UNIT = "IN_NOISE_UNIT"
HAS_REPRESENTATIVE = "HAS_REPRESENTATIVE"

# inferred semantic links (never confirmed)
CANDIDATE_TECHNIQUE = "CANDIDATE_TECHNIQUE"
DESCRIBED_BY = "DESCRIBED_BY"

# ontology module hand-off
ONTOLOGY_RELATION = "ONTOLOGY_RELATION"        # property relationship_type = the ontology's own string
EVIDENCED_BY = "EVIDENCED_BY"                  # OntologyEntity -> Event (explicit event_uid evidence only)
RESOLVED_TO = "RESOLVED_TO"                    # OntologyEntity -> Entity (strong-identifier match only)

ALL_RELS = (EVENT_EVENT_RELS + EVENT_ENTITY_RELS +
            (IN_CLUSTER, IN_NOISE_UNIT, HAS_REPRESENTATIVE, CANDIDATE_TECHNIQUE, DESCRIBED_BY,
             ONTOLOGY_RELATION, EVIDENCED_BY, RESOLVED_TO))

# provenance / status vocabulary on every edge
STATUS_OBSERVED = "observed"        # directly derived from log attributes / adjacency rules
STATUS_COMPUTED = "computed"        # produced by the GAE/HDBSCAN/novelty computation
STATUS_INFERRED = "inferred"        # semantic similarity or ontology inference - a lead, not a fact

# key (node kind) used for each unique constraint in Neo4j
UNIQUE_KEYS = {EVENT: "event_uid", ENTITY: "entity_key", CLUSTER: "cluster_key", TECHNIQUE: "technique_id",
               REFERENCE: "doc_id", ONTOLOGY: "entity_id"}


def entity_key(entity_type: str, value) -> str:
    v = str(value).strip()
    if entity_type in ("host", "process_image", "ip_address"):
        v = v.lower()
    return f"{entity_type}:{v}"
