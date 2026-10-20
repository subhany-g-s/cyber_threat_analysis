"""
src/mitre/corpus.py
--------------------
The versioned, locally cached MITRE ATT&CK technique corpus.

    MitreCorpus.techniques    every parsed technique / sub-technique
                              (active + revoked + deprecated, with status)
    MitreCorpus.active()      only entries eligible for semantic matching
    MitreCorpus.meta          source URL, ATT&CK release version, retrieval
                              time, content hash, statistics, parse problems

Normal analysis runs read this cache only -- they never touch the network.
The cache is written exclusively by scripts/update_mitre_corpus.py
(src/mitre/updater.py).

`content_hash` is a SHA-256 over the canonical technique records. It is the
key that invalidates the embedding cache, and it is independent of the
retrieval timestamp (re-downloading identical data does not change it).
"""

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

CORPUS_SCHEMA_VERSION = 1
MIN_DESCRIPTION_CHARS = 20


class CorpusError(Exception):
    """The corpus is missing, malformed, empty or fails validation."""


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def compute_content_hash(techniques: List[Dict[str, Any]]) -> str:
    canonical = json.dumps(sorted(techniques, key=lambda t: (t["technique_id"], t.get("stix_id", ""))),
                           sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_techniques(techniques: List[Dict[str, Any]]) -> ValidationReport:
    """Structural validation. Errors make a corpus unusable (refused at
    update time and at load time); warnings are surfaced but non-fatal."""
    rep = ValidationReport()
    if not isinstance(techniques, list) or not techniques:
        rep.errors.append("Corpus is empty: no technique records.")
        return rep

    from src.mitre.stix_parser import TECHNIQUE_ID_RE
    active_ids = Counter()
    all_ids = Counter()
    for i, t in enumerate(techniques):
        if not isinstance(t, dict):
            rep.errors.append(f"Record #{i} is not an object.")
            continue
        tid = t.get("technique_id")
        if not tid or not isinstance(tid, str):
            rep.errors.append(f"Record #{i} is missing technique_id.")
            continue
        if not TECHNIQUE_ID_RE.match(tid):
            rep.errors.append(f"Record #{i} has malformed technique_id '{tid}'.")
            continue
        if not (t.get("technique_name") or "").strip():
            rep.errors.append(f"{tid}: missing technique_name.")
        if t.get("status") not in ("active", "revoked", "deprecated"):
            rep.errors.append(f"{tid}: invalid status '{t.get('status')}'.")
        all_ids[(tid, t.get("status"))] += 1
        if t.get("status") == "active":
            active_ids[tid] += 1
            if len((t.get("description") or "").strip()) < MIN_DESCRIPTION_CHARS:
                rep.warnings.append(f"{tid}: active technique has an empty/very short description; "
                                    f"excluded from semantic matching.")
            if t.get("is_subtechnique") and not t.get("parent_id"):
                rep.warnings.append(f"{tid}: sub-technique has no resolvable parent.")

    dup_active = sorted(tid for tid, n in active_ids.items() if n > 1)
    if dup_active:
        rep.errors.append(f"Duplicate ACTIVE technique ids: {dup_active[:10]}"
                          + (" ..." if len(dup_active) > 10 else ""))
    dup_other = sorted({tid for (tid, st), n in all_ids.items() if n > 1 and st != "active"})
    if dup_other:
        rep.warnings.append(f"Duplicate revoked/deprecated technique ids: {dup_other[:10]}")
    if not active_ids:
        rep.errors.append("Corpus has no ACTIVE techniques; nothing could be matched against.")
    return rep


def compute_stats(techniques: List[Dict[str, Any]]) -> Dict[str, Any]:
    status = Counter(t["status"] for t in techniques)
    active = [t for t in techniques if t["status"] == "active"]
    matchable = [t for t in active if len((t.get("description") or "").strip()) >= MIN_DESCRIPTION_CHARS]
    tactic_counts: Counter = Counter()
    platform_counts: Counter = Counter()
    for t in active:
        for tac in t.get("tactics") or []:
            tactic_counts[tac] += 1
        for p in t.get("platforms") or []:
            platform_counts[p] += 1
    return {
        "total_records": len(techniques),
        "active": status.get("active", 0),
        "revoked": status.get("revoked", 0),
        "deprecated": status.get("deprecated", 0),
        "active_techniques": sum(1 for t in active if not t.get("is_subtechnique")),
        "active_subtechniques": sum(1 for t in active if t.get("is_subtechnique")),
        "matchable_active": len(matchable),
        "active_with_detection_info": sum(
            1 for t in active
            if (t.get("detection") or {}).get("strategies") or (t.get("detection") or {}).get("legacy_text")),
        "active_per_tactic": dict(sorted(tactic_counts.items())),
        "active_per_platform": dict(sorted(platform_counts.items())),
        "mean_description_chars": round(sum(len(t.get("description") or "") for t in active) / len(active), 1) if active else 0.0,
    }


@dataclass
class MitreCorpus:
    techniques: List[Dict[str, Any]]
    meta: Dict[str, Any] = field(default_factory=dict)
    path: Optional[str] = None

    # ------------------------------------------------------------ identity
    @property
    def content_hash(self) -> str:
        return self.meta.get("content_hash") or compute_content_hash(self.techniques)

    @property
    def attack_version(self) -> str:
        return str(self.meta.get("attack_version") or "unknown")

    @property
    def version_label(self) -> str:
        """e.g. 'enterprise-attack-v19.2-3fa91c2b'"""
        return f"{self.meta.get('domain', 'enterprise-attack')}-v{self.attack_version}-{self.content_hash[:8]}"

    # ------------------------------------------------------------ queries
    def active(self, platforms: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        """Entries eligible for semantic matching: status == active and a
        usable description. `platforms` (case-insensitive) restricts to
        techniques whose x_mitre_platforms intersect the given set."""
        wanted = {p.lower() for p in platforms} if platforms else None
        out = []
        for t in self.techniques:
            if t.get("status") != "active":
                continue
            if len((t.get("description") or "").strip()) < MIN_DESCRIPTION_CHARS:
                continue
            if wanted is not None and not ({p.lower() for p in t.get("platforms") or []} & wanted):
                continue
            out.append(t)
        return out

    def by_id(self, technique_id: str) -> Optional[Dict[str, Any]]:
        for t in self.techniques:
            if t["technique_id"] == technique_id and t["status"] == "active":
                return t
        for t in self.techniques:
            if t["technique_id"] == technique_id:
                return t
        return None

    def stats(self) -> Dict[str, Any]:
        return self.meta.get("stats") or compute_stats(self.techniques)

    # ----------------------------------------------------------------- I/O
    def to_dict(self) -> Dict[str, Any]:
        meta = dict(self.meta)
        meta["content_hash"] = self.content_hash
        meta["corpus_schema_version"] = CORPUS_SCHEMA_VERSION
        return {"meta": meta, "techniques": self.techniques}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)
        os.replace(tmp, path)          # atomic: a failed write never corrupts an existing cache
        self.path = path

    @classmethod
    def load(cls, path: str) -> "MitreCorpus":
        if not os.path.exists(path):
            raise CorpusError(f"MITRE corpus file not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise CorpusError(f"MITRE corpus file {path} is unreadable or corrupt: {e}")
        return cls.from_payload(payload, path)

    @classmethod
    def from_payload(cls, payload: Any, path: Optional[str] = None) -> "MitreCorpus":
        if not isinstance(payload, dict) or not isinstance(payload.get("techniques"), list):
            raise CorpusError("Corpus payload must be an object with a 'techniques' list.")
        techniques = payload["techniques"]
        meta = payload.get("meta") or {}
        rep = validate_techniques(techniques)
        if not rep.ok:
            raise CorpusError("Corpus failed validation: " + "; ".join(rep.errors))
        recomputed = compute_content_hash(techniques)
        if meta.get("content_hash") and meta["content_hash"] != recomputed:
            raise CorpusError(
                f"Corpus content hash mismatch (recorded {meta['content_hash'][:12]}..., "
                f"recomputed {recomputed[:12]}...): the cache file was modified after it was written. "
                f"Re-run scripts/update_mitre_corpus.py.")
        meta = dict(meta)
        meta["content_hash"] = recomputed
        return cls(techniques=techniques, meta=meta, path=path)

    @classmethod
    def from_legacy_curated_file(cls, path: str) -> "MitreCorpus":
        """Loads the OLD hand-written config/mitre_technique_corpus.json shape
        ({"techniques": [{technique_id, technique_name, tactic, description}]}).
        Only reachable through an explicit --corpus-file flag; the result is
        labelled `legacy_curated` everywhere it is displayed."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise CorpusError(f"Cannot read corpus file {path}: {e}")
        raw = payload.get("techniques") if isinstance(payload, dict) else None
        if not isinstance(raw, list) or not raw:
            raise CorpusError(f"{path}: no 'techniques' list.")
        techs = []
        for t in raw:
            tactic = t.get("tactic") or ""
            techs.append({
                "technique_id": t.get("technique_id"), "technique_name": t.get("technique_name"),
                "full_name": t.get("technique_name"), "tactic": tactic,
                "tactics": [x.strip() for x in tactic.split(",") if x.strip()], "tactic_shortnames": [],
                "platforms": ["Windows"], "description": t.get("description") or "",
                "is_subtechnique": "." in str(t.get("technique_id")), "parent_id": None,
                "status": "active", "revoked_by": None, "detection": {"legacy_text": "", "strategies": []},
                "data_sources": [], "stix_id": f"legacy::{t.get('technique_id')}", "url": None,
                "version": None, "created": None, "modified": None, "domains": [],
            })
        rep = validate_techniques(techs)
        if not rep.ok:
            raise CorpusError("Corpus failed validation: " + "; ".join(rep.errors))
        meta = {"domain": "legacy-curated", "attack_version": "n/a", "source_url": None,
                "legacy_curated": True, "stats": compute_stats(techs)}
        return cls(techniques=techs, meta=meta, path=path)
