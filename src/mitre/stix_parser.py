"""
src/mitre/stix_parser.py
-------------------------
Turns an official MITRE ATT&CK Enterprise STIX 2.1 bundle into a flat list
of technique records (techniques AND sub-techniques), keeping revoked and
deprecated entries with an explicit `status` so they can be excluded from
active matching without losing their metadata.

The bundle is UNTRUSTED input: it is only ever read with json (never eval'd)
and every field access is defensive.

Output record (one dict per ATT&CK technique / sub-technique):
    technique_id        "T1059" / "T1059.001"
    technique_name      leaf name, e.g. "PowerShell"
    full_name           "Command and Scripting Interpreter: PowerShell"
    tactic              comma-joined tactic display names (back-compat key)
    tactics / tactic_shortnames / platforms
    description         cleaned text (citations / markdown / html stripped)
    is_subtechnique / parent_id
    status              "active" | "revoked" | "deprecated"
    revoked_by          technique_id that replaced a revoked entry (if known)
    detection           {"legacy_text": str, "strategies": [...]}  (see below)
    stix_id / url / version / created / modified / domains

Detection information: ATT&CK >= v18 expresses detection through
`x-mitre-detection-strategy` objects (linked by `detects` relationships) that
reference `x-mitre-analytic` objects. Older bundles used a free-text
`x_mitre_detection` field on the technique. Both are captured when present.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

TECHNIQUE_ID_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")

_CITATION_RE = re.compile(r"\(Citation:[^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:https?://|/)[^)]*\)")
_HTML_RE = re.compile(r"</?(?:code|br|i|b|em|strong|p)\s*/?>", re.IGNORECASE)
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\n{3,}")

MAX_ANALYTIC_CHARS = 600
MAX_ANALYTICS_PER_STRATEGY = 8


def clean_description(text: Optional[str]) -> str:
    """Strip STIX citation markers, markdown links and simple html tags."""
    if not text or not isinstance(text, str):
        return ""
    t = _CITATION_RE.sub("", text)
    t = _MD_LINK_RE.sub(r"\1", t)
    t = _HTML_RE.sub("", t)
    t = _WS_RE.sub(" ", t)
    t = _NL_RE.sub("\n\n", t)
    return t.strip()


@dataclass
class ParsedBundle:
    collection: Dict[str, Any] = field(default_factory=dict)
    techniques: List[Dict[str, Any]] = field(default_factory=list)
    tactics: Dict[str, str] = field(default_factory=dict)
    problems: List[Dict[str, str]] = field(default_factory=list)   # {kind, stix_id, detail}


def _attack_ref(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for ref in obj.get("external_references") or []:
        if isinstance(ref, dict) and ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
            return ref
    return None


def _title(shortname: str) -> str:
    return " ".join(w.capitalize() for w in shortname.replace("_", "-").split("-"))


def parse_bundle(bundle: Any) -> ParsedBundle:
    """Parse a STIX bundle (already json-decoded). Raises ValueError when the
    structure is not a STIX bundle at all; per-object problems are recorded in
    `ParsedBundle.problems` instead of aborting."""
    if not isinstance(bundle, dict) or bundle.get("type") != "bundle" or not isinstance(bundle.get("objects"), list):
        raise ValueError("Not a STIX bundle: expected an object with type='bundle' and an 'objects' list.")

    out = ParsedBundle()
    objects: List[Dict[str, Any]] = [o for o in bundle["objects"] if isinstance(o, dict)]
    by_id: Dict[str, Dict[str, Any]] = {o["id"]: o for o in objects if isinstance(o.get("id"), str)}

    # ---- collection metadata (ATT&CK release version, spec version) ----
    collections = [o for o in objects if o.get("type") == "x-mitre-collection"]
    if collections:
        c = collections[0]
        out.collection = {
            "name": c.get("name"),
            "attack_version": c.get("x_mitre_version"),
            "attack_spec_version": c.get("x_mitre_attack_spec_version"),
            "modified": c.get("modified"),
            "stix_id": c.get("id"),
        }
    else:
        out.problems.append({"kind": "no_collection_metadata", "stix_id": "",
                             "detail": "Bundle has no x-mitre-collection object; ATT&CK release version unknown."})

    # ---- tactics ----
    for o in objects:
        if o.get("type") == "x-mitre-tactic" and o.get("x_mitre_shortname"):
            out.tactics[o["x_mitre_shortname"]] = o.get("name") or _title(o["x_mitre_shortname"])

    # ---- relationships ----
    sub_of: Dict[str, str] = {}
    revoked_by: Dict[str, str] = {}
    detects: Dict[str, List[str]] = {}
    for o in objects:
        if o.get("type") != "relationship" or o.get("revoked") or o.get("x_mitre_deprecated"):
            continue
        rt, s, t = o.get("relationship_type"), o.get("source_ref"), o.get("target_ref")
        if not (isinstance(s, str) and isinstance(t, str)):
            continue
        if rt == "subtechnique-of":
            sub_of[s] = t
        elif rt == "revoked-by":
            revoked_by[s] = t
        elif rt == "detects" and s.startswith("x-mitre-detection-strategy--") and t.startswith("attack-pattern--"):
            detects.setdefault(t, []).append(s)

    def detection_for(stix_id: str) -> List[Dict[str, Any]]:
        strategies = []
        for sid in sorted(detects.get(stix_id, [])):
            ds = by_id.get(sid)
            if not ds or ds.get("revoked") or ds.get("x_mitre_deprecated"):
                continue
            analytics = []
            for aid in ds.get("x_mitre_analytic_refs") or []:
                an = by_id.get(aid)
                if not an or an.get("revoked") or an.get("x_mitre_deprecated"):
                    continue
                analytics.append({
                    "stix_id": aid,
                    "name": an.get("name"),
                    "platforms": an.get("x_mitre_platforms") or [],
                    "description": clean_description(an.get("description"))[:MAX_ANALYTIC_CHARS],
                    "log_sources": sorted({
                        ls.get("name") for ls in (an.get("x_mitre_log_source_references") or [])
                        if isinstance(ls, dict) and ls.get("name")
                    }),
                })
                if len(analytics) >= MAX_ANALYTICS_PER_STRATEGY:
                    break
            strategies.append({"stix_id": sid, "name": ds.get("name"), "analytics": analytics})
        return strategies

    # ---- techniques ----
    attack_patterns = [o for o in objects if o.get("type") == "attack-pattern"]
    stix_to_tid: Dict[str, str] = {}
    for o in attack_patterns:
        ref = _attack_ref(o)
        if ref is None:
            out.problems.append({"kind": "missing_id", "stix_id": str(o.get("id")),
                                 "detail": f"attack-pattern '{o.get('name')}' has no mitre-attack external_id; skipped."})
            continue
        tid = str(ref["external_id"]).strip()
        if not TECHNIQUE_ID_RE.match(tid):
            out.problems.append({"kind": "malformed_id", "stix_id": str(o.get("id")),
                                 "detail": f"attack-pattern '{o.get('name')}' has malformed technique id '{tid}'; skipped."})
            continue
        stix_to_tid[o["id"]] = tid

    for o in attack_patterns:
        tid = stix_to_tid.get(o.get("id"))
        if tid is None:
            continue
        ref = _attack_ref(o) or {}
        if o.get("revoked"):
            status = "revoked"
        elif o.get("x_mitre_deprecated"):
            status = "deprecated"
        else:
            status = "active"

        is_sub = bool(o.get("x_mitre_is_subtechnique")) or ("." in tid)
        parent_stix = sub_of.get(o["id"])
        parent_id = stix_to_tid.get(parent_stix) if parent_stix else None
        parent_obj = by_id.get(parent_stix) if parent_stix else None
        name = str(o.get("name") or "").strip()
        full_name = f"{parent_obj.get('name')}: {name}" if (is_sub and parent_obj and parent_obj.get("name")) else name

        phases = [p for p in (o.get("kill_chain_phases") or [])
                  if isinstance(p, dict) and p.get("kill_chain_name") == "mitre-attack" and p.get("phase_name")]
        shortnames = [p["phase_name"] for p in phases]
        tactic_names = [out.tactics.get(s, _title(s)) for s in shortnames]

        legacy = clean_description(o.get("x_mitre_detection"))
        out.techniques.append({
            "technique_id": tid,
            "technique_name": name,
            "full_name": full_name,
            "tactic": ", ".join(tactic_names),
            "tactics": tactic_names,
            "tactic_shortnames": shortnames,
            "platforms": list(o.get("x_mitre_platforms") or []),
            "description": clean_description(o.get("description")),
            "is_subtechnique": is_sub,
            "parent_id": parent_id,
            "status": status,
            "revoked_by": stix_to_tid.get(revoked_by.get(o["id"], "")) if status == "revoked" else None,
            "detection": {"legacy_text": legacy, "strategies": detection_for(o["id"])},
            "data_sources": list(o.get("x_mitre_data_sources") or []),
            "stix_id": o["id"],
            "url": ref.get("url"),
            "version": o.get("x_mitre_version"),
            "created": o.get("created"),
            "modified": o.get("modified"),
            "domains": list(o.get("x_mitre_domains") or []),
        })

    out.techniques.sort(key=lambda t: (t["technique_id"], t["status"] != "active", t["stix_id"]))
    return out
