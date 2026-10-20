"""
src/profiling/cluster_profiler.py
------------------------------------
Converts a cluster's member events into:
  (a) a structured profile dict (programmatic use: filtering, RAG metadata)
  (b) a deterministic text narrative (the ONLY thing the semantic mapper
      in src/mapping/ consumes)

Design rules:
  - cluster_id == -1 (HDBSCAN noise) is REJECTED here, not silently
    profiled. Noise has no coherent behavior to summarize; forcing a
    narrative onto it would hand the mapper something to falsely match
    against. Callers must filter noise out before calling this module
    (see build_cluster_profiles below, which does this for you).
  - The narrative is built from a FIXED TEMPLATE, not an LLM call. This
    keeps profiling deterministic, reproducible, and auditable -- the
    semantic mapper's input text is fully traceable back to the
    aggregated fields.
  - CommandLine/Message/RuleName text is pulled in ONLY for the
    human-readable narrative and the stored evidence list -- never fed
    into any numeric feature anywhere in this project.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.mapping.text_prep import display_safe


@dataclass
class ClusterProfile:
    cluster_id: int
    cluster_size: int
    dominant_event_ids: List[List[Any]]      # [[event_id, count], ...], most common first
    hostname_distribution: Dict[str, int]
    common_source_images: List[List[Any]]     # [[image_path, count], ...]
    common_target_images: List[List[Any]]
    common_destination_ips: List[List[Any]]
    common_protocols: List[List[Any]]
    severity_distribution: Dict[str, int]
    temporal_span: Dict[str, Optional[str]]   # {"earliest": ..., "latest": ...}
    representative_event_uids: List[str]
    evidence_snippets: List[str]              # CommandLine/Message fragments, for the narrative + RAG only
    text: str = ""                            # filled by cluster_to_text()
    # ---- additive fields (all defaulted; older callers/tests are unaffected) ----
    kind: str = "cluster"                     # "cluster" | "noise_group" | "noise_event"
    unit_id: str = ""                         # "C7" / "N2" / "E5"
    mapping_text: str = ""                    # behaviour-only text embedded by the semantic mapper
    event_descriptions: List[List[Any]] = field(default_factory=list)   # [[first message line, count], ...]
    registry_targets: List[List[Any]] = field(default_factory=list)
    loaded_modules: List[List[Any]] = field(default_factory=list)
    network_ports: List[List[Any]] = field(default_factory=list)
    log_channels: List[List[Any]] = field(default_factory=list)
    member_event_uids: List[str] = field(default_factory=list)
    command_texts: List[str] = field(default_factory=list)   # CommandLine / PowerShell script text (display_safe applied at render time)

    def as_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        return d


def _top_n(counter: Counter, n: int = 5) -> List[List[Any]]:
    return [[k, v] for k, v in counter.most_common(n)]


def build_cluster_profile(cluster_id: int, member_events: List[Dict],
                           representative_event_uids: List[str],
                           max_evidence_snippets: int = 3) -> ClusterProfile:
    """
    member_events: list of NormalizedEvent dicts (from normalizer.py's
    as_dict()) belonging to this cluster_id. Caller is responsible for
    excluding cluster_id == -1 (noise) -- see build_cluster_profiles().
    Noise is analysed separately through build_unit_profile().
    """
    if cluster_id == -1:
        raise ValueError(
            "cluster_id == -1 is HDBSCAN noise and must not be profiled as a "
            "behavioral cluster -- filter it out before calling build_cluster_profile(). "
            "Noise events/groups use build_unit_profile() and stay labelled as noise."
        )
    if not member_events:
        raise ValueError(f"cluster_id={cluster_id} has no member events.")
    profile = _build_profile(cluster_id, member_events, representative_event_uids, max_evidence_snippets,
                             kind="cluster", unit_id=f"C{cluster_id}")
    return profile


def build_unit_profile(unit_id: str, kind: str, member_events: List[Dict],
                       representative_event_uids: List[str], max_evidence_snippets: int = 3) -> ClusterProfile:
    """Profile of a NOISE unit (a single unclustered event or a small group of
    related unclustered events). cluster_id stays -1: the unit is never
    presented as a normal behavioural cluster, and its narrative says so."""
    if kind not in ("noise_group", "noise_event"):
        raise ValueError(f"kind must be 'noise_group' or 'noise_event', got {kind!r}")
    if not member_events:
        raise ValueError(f"noise unit {unit_id} has no member events.")
    return _build_profile(-1, member_events, representative_event_uids, max_evidence_snippets,
                          kind=kind, unit_id=unit_id)


def _first_line_key(raw: Dict[str, Any]) -> Optional[str]:
    msg = raw.get("Message")
    if not msg:
        return None
    first = str(msg).replace("\r", "\n").split("\n")[0].strip()
    first = first.split(":")[0].strip() if ":" in first else first
    return first[:90] or None


def _build_profile(cluster_id: int, member_events: List[Dict], representative_event_uids: List[str],
                   max_evidence_snippets: int, kind: str, unit_id: str) -> ClusterProfile:

    event_ids = Counter(e.get("event_id") for e in member_events if e.get("event_id") is not None)
    hostnames = Counter(e.get("hostname") for e in member_events if e.get("hostname"))
    src_images = Counter(e.get("source_image") for e in member_events if e.get("source_image"))
    tgt_images = Counter(e.get("target_image") for e in member_events if e.get("target_image"))
    dest_ips = Counter(e.get("destination_ip") for e in member_events if e.get("destination_ip"))
    protocols = Counter(e.get("protocol") for e in member_events if e.get("protocol"))
    severities = Counter(e.get("severity") for e in member_events if e.get("severity"))

    timestamps = sorted(e["timestamp"] for e in member_events if e.get("timestamp"))

    # ---- additional behavioural aggregates (observed values only, nothing inferred) ----
    raws = [(e.get("raw_event") or {}) for e in member_events]
    descriptions = Counter(k for k in (_first_line_key(r) for r in raws) if k)
    registry = Counter(str(r["TargetObject"])[:90] for r in raws if r.get("TargetObject"))
    modules = Counter(_basename(str(r["ImageLoaded"])) for r in raws if r.get("ImageLoaded"))
    ports = Counter(str(e.get("destination_port")) for e in member_events if e.get("destination_port") not in (None, ""))
    channels = Counter(str(r["Channel"]) for r in raws if r.get("Channel"))
    command_texts: List[str] = []
    seen_cmd = set()
    for r in raws:
        cmd = r.get("CommandLine")
        if not cmd and "powershell" in str(r.get("Channel", "")).lower() and r.get("Message"):
            cmd = r.get("Message")
        if not cmd:
            continue
        cmd = " ".join(str(cmd).split())[:200]
        if cmd[:60] in seen_cmd:
            continue
        seen_cmd.add(cmd[:60])
        command_texts.append(cmd)
        if len(command_texts) >= 3:
            break

    # Representative events are picked independently (medoid + top
    # membership-probability), but on this dataset they very often share
    # near-identical boilerplate Message text (e.g. three Security-auditing
    # events with the same templated "An attempt was made to access an
    # object..." preamble). Earlier versions of this function appended one
    # snippet per representative event with no dedup, producing a narrative
    # with the same paragraph repeated 2-3x -- noisy for a human reader and
    # wasted tokens for the embedder. We prefer CommandLine (highest
    # information density for technique-relevant behavior, e.g. actual
    # PowerShell arguments) over Message/RuleName, and dedupe on a
    # normalized prefix so repeated boilerplate collapses to one instance.
    rep_events = [e for e in member_events if e.get("event_uid") in representative_event_uids]
    evidence_snippets: List[str] = []
    seen_normalized = set()
    for e in rep_events:
        raw = e.get("raw_event") or {}
        snippet = raw.get("CommandLine") or raw.get("Message") or raw.get("RuleName")
        if not snippet:
            continue
        snippet = str(snippet)[:200]
        normalized = " ".join(snippet.split())[:80]  # collapse whitespace, compare on a short prefix
        if normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)
        evidence_snippets.append(snippet)
        if len(evidence_snippets) >= max_evidence_snippets:
            break

    profile = ClusterProfile(
        cluster_id=cluster_id,
        cluster_size=len(member_events),
        dominant_event_ids=_top_n(event_ids),
        hostname_distribution=dict(hostnames),
        common_source_images=_top_n(src_images),
        common_target_images=_top_n(tgt_images),
        common_destination_ips=_top_n(dest_ips),
        common_protocols=_top_n(protocols),
        severity_distribution=dict(severities),
        temporal_span={"earliest": timestamps[0] if timestamps else None,
                        "latest": timestamps[-1] if timestamps else None},
        representative_event_uids=representative_event_uids,
        evidence_snippets=evidence_snippets,
        kind=kind, unit_id=unit_id,
        event_descriptions=_top_n(descriptions, 4),
        registry_targets=_top_n(registry, 3),
        loaded_modules=_top_n(modules, 4),
        network_ports=_top_n(ports, 4),
        log_channels=_top_n(channels, 2),
        member_event_uids=[e["event_uid"] for e in member_events if e.get("event_uid")],
        command_texts=command_texts,
    )
    profile.text = cluster_to_text(profile)
    profile.mapping_text = cluster_to_mapping_text(profile)
    return profile


def _basename(path: Optional[str]) -> Optional[str]:
    """Image paths are full Windows paths; the narrative reads better
    with just the executable name (e.g. 'powershell.exe' not
    'C:\\Windows\\...\\powershell.exe'). Structured fields keep the full path."""
    if not path:
        return None
    return path.replace("/", "\\").split("\\")[-1]


def cluster_to_text(profile: ClusterProfile) -> str:
    """
    Fixed-template narrative. Every clause maps to one aggregated field --
    no free-text generation, so the output is fully reproducible and each
    sentence is traceable back to profile.as_dict().
    """
    parts = []

    if profile.kind in ("noise_group", "noise_event"):
        what = "Unclustered (HDBSCAN noise) event" if profile.kind == "noise_event" else \
            f"Unclustered (HDBSCAN noise) group {profile.unit_id} of {profile.cluster_size} related events"
        if profile.dominant_event_ids:
            dom_id, dom_count = profile.dominant_event_ids[0]
            parts.append(f"{what}; EventID {dom_id} x{dom_count}. Not assigned to any behavioural cluster.")
        else:
            parts.append(f"{what}. Not assigned to any behavioural cluster.")
    elif profile.dominant_event_ids:
        dom_id, dom_count = profile.dominant_event_ids[0]
        pct = round(100 * dom_count / profile.cluster_size, 1)
        parts.append(f"Cluster {profile.cluster_id} contains {profile.cluster_size} events, "
                      f"{pct}% of which are EventID {dom_id}.")
    else:
        parts.append(f"Cluster {profile.cluster_id} contains {profile.cluster_size} events.")

    src_names = [n for n in (_basename(p) for p, _ in profile.common_source_images[:3]) if n]
    tgt_names = [n for n in (_basename(p) for p, _ in profile.common_target_images[:3]) if n]
    if src_names and tgt_names:
        parts.append(f"Repeated execution of {', '.join(src_names)} observed, "
                      f"interacting with {', '.join(tgt_names)}.")
    elif src_names:
        parts.append(f"Repeated execution of {', '.join(src_names)} observed.")

    hosts = list(profile.hostname_distribution.keys())[:5]
    if len(hosts) == 1:
        parts.append(f"Activity confined to host {hosts[0]}.")
    elif hosts:
        parts.append(f"Activity observed across {len(profile.hostname_distribution)} hosts "
                      f"including {', '.join(hosts)}.")

    if profile.common_destination_ips:
        dst_ips = [ip for ip, _ in profile.common_destination_ips[:3]]
        proto_names = [p for p, _ in profile.common_protocols[:2]]
        proto_str = f" over protocol(s) {', '.join(str(p) for p in proto_names)}" if proto_names else ""
        parts.append(f"Network connections observed to {', '.join(dst_ips)}{proto_str}.")

    if profile.temporal_span.get("earliest") and profile.temporal_span.get("latest"):
        parts.append(f"Events span from {profile.temporal_span['earliest']} to {profile.temporal_span['latest']}.")

    if profile.evidence_snippets:
        quoted = "; ".join(f'"{s}"' for s in profile.evidence_snippets)
        parts.append(f"Representative command/evidence text: {quoted}.")

    return " ".join(parts)


def cluster_to_mapping_text(profile: ClusterProfile) -> str:
    """Behaviour-only text for the semantic mapper. Unlike `text` (a display
    narrative) it omits cluster ids, counts, host names and timestamps -
    tokens that carry no technique semantics and only add embedding noise.
    Every clause is an observed aggregate from the member events."""
    parts = []
    if profile.event_descriptions:
        total = max(profile.cluster_size, 1)
        parts.append("Windows event log activity: " + "; ".join(
            f"{d} ({round(100 * c / total)}%)" for d, c in profile.event_descriptions[:3]) + ".")
    src = [n for n in (_basename(p) for p, _ in profile.common_source_images[:3]) if n]
    tgt = [n for n in (_basename(p) for p, _ in profile.common_target_images[:3]) if n]
    if src and tgt:
        parts.append(f"Process {', '.join(src)} accessed or interacted with process {', '.join(tgt)}.")
    elif src:
        parts.append(f"Process {', '.join(src)} activity.")
    if profile.loaded_modules:
        parts.append("Loaded modules: " + ", ".join(m for m, _ in profile.loaded_modules[:4]) + ".")
    if profile.registry_targets:
        parts.append("Registry keys: " + "; ".join(k for k, _ in profile.registry_targets[:3]) + ".")
    if profile.common_destination_ips:
        ports = ", ".join(p for p, _ in profile.network_ports[:3])
        proto = ", ".join(str(p) for p, _ in profile.common_protocols[:2])
        parts.append("Network connections" + (f" to port {ports}" if ports else "")
                     + (f" protocol {proto}" if proto else "") + ".")
    if profile.command_texts:
        parts.append("Commands: " + " | ".join(f'"{c}"' for c in profile.command_texts[:2]) + ".")
    if not parts:
        parts.append("Windows event log activity with no further distinguishing attributes.")
    return " ".join(parts)


def build_cluster_profiles(cluster_assignments: List[Dict], events_by_uid: Dict[str, Dict],
                            medoid_event_uid_by_cluster: Optional[Dict[int, str]] = None,
                            top_prob_n: int = 3) -> List[ClusterProfile]:
    """
    High-level entrypoint: given cluster assignment rows (as produced by
    src/clustering/cluster_embeddings.py -- {event_uid, cluster_id,
    membership_probability}) and the normalized events dict, build one
    ClusterProfile per NON-NOISE cluster.

    Noise (cluster_id == -1) is filtered out here, once, so every other
    module in the pipeline (semantic mapper, RAG export) never has to
    remember to do it.
    """
    from collections import defaultdict

    by_cluster: Dict[int, List[Dict]] = defaultdict(list)
    for row in cluster_assignments:
        if row["cluster_id"] == -1:
            continue  # noise excluded -- see module docstring
        by_cluster[row["cluster_id"]].append(row)

    profiles = []
    for cluster_id, rows in sorted(by_cluster.items()):
        member_events = [events_by_uid[r["event_uid"]] for r in rows if r["event_uid"] in events_by_uid]

        rep_uids = []
        if medoid_event_uid_by_cluster and cluster_id in medoid_event_uid_by_cluster:
            rep_uids.append(medoid_event_uid_by_cluster[cluster_id])
        by_prob = sorted(rows, key=lambda r: -r.get("membership_probability", 0))
        for r in by_prob[:top_prob_n]:
            if r["event_uid"] not in rep_uids:
                rep_uids.append(r["event_uid"])

        profiles.append(build_cluster_profile(cluster_id, member_events, rep_uids))

    return profiles
