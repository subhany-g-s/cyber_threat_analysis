"""
src/rag/findings.py
---------------------
The ANALYST-FACING layer of the report: turns the per-unit evidence reports
(see evidence_report.py - untouched, still the full technical record) into a
short list of prioritized, de-duplicated suspicious-behaviour FINDINGS.

This module is presentation / triage only. It reads the results of the
detection stages (clusters, novelty, mappings, retrieval) and never changes
them. Everything here is deterministic and works without any LLM; the LLM
step (finding_llm.py) only adds validated narrative on top.

What a finding is
    One unit (a cluster, a noise group or a noise event) or - when units are
    directly linked by a SHARED PROCESS ID - a small set of such units. Units
    that merely share a host, a time window, a process name or an ATT&CK
    candidate are NOT merged (they are cross-referenced instead).

How findings are prioritized (documented rules, NOT a calibrated severity model)
    Inputs, all taken from existing results:
      tier      novelty label present in the unit: HIGH_NOVELTY=3, SUSPICIOUS=2, LOW_NOVELTY=1, NORMAL=0
                (labels come from the unchanged novelty stage - statistical unusualness, not maliciousness)
      markers   observable indicators found directly in the unit's own events (see MARKERS below):
                strong / medium / weak
      noise     the unit is HDBSCAN noise (not part of any behaviour cluster)
      linked    the unit shares a process id with another unit that also carries a marker
    Ladder (first rule that matches wins):
      Critical  tier 3 AND a strong marker AND >=2 marker classes AND linked   (reserved; needs corroboration)
      High      strong marker AND (tier>=2 OR >=2 marker classes);  or  >=2 medium markers AND tier>=2
      Medium    strong marker;  or  medium marker AND tier>=1;  or  noise unit AND tier>=2
      Low       tier>=2;  or  tier>=1 AND (weak marker OR noise);  or  medium marker
      Informational  everything else (kept in the technical appendix, not listed)
    Similarity scores to ATT&CK techniques do NOT enter the ladder: they are uncalibrated and
    are not evidence that a technique occurred.
"""

import ipaddress
import re
from collections import Counter, defaultdict
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.mapping.text_prep import display_safe, has_unsafe_chars
from src.profiling.cluster_profiler import build_unit_profile

ANALYST_SCHEMA_VERSION = "1.0"
PRIORITIES = ["Critical", "High", "Medium", "Low", "Informational"]
_PRIO_RANK = {p: i for i, p in enumerate(PRIORITIES)}

PRIORITY_CRITERIA = [
    "Critical: HIGH_NOVELTY event + a strong indicator + at least two indicator classes + a shared-process-id link to another unit that also carries an indicator (reserved for corroborated cases).",
    "High: a strong indicator together with SUSPICIOUS-or-higher novelty (or a second indicator class); or two medium indicators with SUSPICIOUS-or-higher novelty.",
    "Medium: a strong indicator on its own; a medium indicator with any novelty above NORMAL; or an unclustered (noise) unit with SUSPICIOUS-or-higher novelty.",
    "Low: SUSPICIOUS-or-higher novelty without an indicator; LOW_NOVELTY with a weak indicator or in a noise unit; or a medium indicator in normal-novelty activity.",
    "Informational: everything else. Kept in the technical appendix, not listed as a finding.",
    "Priorities are transparent triage rules, not statistical confidence. Novelty = unusualness in this dataset; ATT&CK similarity = uncalibrated text similarity. Neither is a probability of compromise.",
]

# --------------------------------------------------------------------------- observable indicators
# Each is a directly observable property of one event's own fields. They are triage hints with
# stated limits (benign explanations below), not detections and not technique mappings.
MARKERS: Dict[str, Dict[str, str]] = {
    "control_chars": {
        "strength": "strong",
        "text": "a path or name contains hidden / bidirectional-override control characters such as U+202E, including their mis-decoded text form (can disguise a file extension)",
        "benign": "right-to-left or localised file names can contain bidirectional characters legitimately (rare in system paths)"},
    "script_like_extension": {
        "strength": "medium",
        "text": "a file or process uses a script- or screensaver-type extension (.scr, .pif, .hta, .vbs, .vbe, .jse, .wsf, .cpl, .com)",
        "benign": ".scr / .cpl / .com files also ship with legitimate software (screensavers, Control Panel applets)"},
    "encoded_command": {
        "strength": "medium",
        "text": "recorded command or script text contains an encoded-command, download-cradle or hidden-window token",
        "benign": "deployment and management tools often use encoded commands or download helpers"},
    "persistence_registry": {
        "strength": "medium",
        "text": "a registry value in a location commonly used for autostart or execution hijack (Run keys, IFEO, Winlogon, service ImagePath, AppInit_DLLs)",
        "benign": "installers, updaters and administrators routinely write Run and service keys"},
    "staging_path": {
        "strength": "weak",
        "text": "a file or process sits in a user-writable staging location (ProgramData, Temp, AppData, Users\\Public)",
        "benign": "installers and software updaters routinely write to ProgramData, Temp and AppData"},
    "external_destination": {
        "strength": "weak",
        "text": "a network connection goes to a non-private IP address",
        "benign": "connections to public addresses are normal for DNS, updates and cloud services"},
}
_STRENGTH_RANK = {"strong": 3, "medium": 2, "weak": 1}
_SCRIPT_EXT = (".scr", ".pif", ".hta", ".vbs", ".vbe", ".jse", ".wsf", ".cpl", ".com")
_STAGING_RE = re.compile(r"\\(programdata|temp|appdata|users\\public)\\", re.IGNORECASE)
_ENCODED_RE = re.compile(r"-e(nc|ncodedcommand)?\s+[A-Za-z0-9+/=]{20,}|frombase64string|downloadstring|downloadfile|"
                         r"net\.webclient|invoke-webrequest|\biex\b|invoke-expression|-w(indowstyle)?\s+hidden",
                         re.IGNORECASE)
_PERSIST_RE = re.compile(r"\\currentversion\\(run|runonce|runservices)\b|\\image file execution options\\|"
                         r"\\winlogon\\(shell|userinit|notify)|\\services\\[^\\]+\\(imagepath|servicedll)|appinit_dlls",
                         re.IGNORECASE)

RELATIONS = {"pid": "shares a process id with this finding", "host_time": "same host within 300 s, no shared process id"}


# --------------------------------------------------------------------------- small helpers
def _basename(path: Optional[str]) -> str:
    return str(path or "").replace("/", "\\").split("\\")[-1]


# Some log pipelines mis-decode UTF-8 bidirectional overrides (U+202A..U+202E = E2 80 AA..AE) as the three characters
# "a-circumflex, euro, <AA..AE>" (cp1252). The dataset's RTLO file name is stored that way, so both forms are detected.
_BIDI_MOJIBAKE_RE = re.compile("\u00e2\u20ac[\u00aa-\u00ae]")


def _has_hidden_chars(text: Any) -> bool:
    return has_unsafe_chars(text) or bool(text and _BIDI_MOJIBAKE_RE.search(str(text)))


def _show(s: Any) -> str:
    """display_safe, plus mis-decoded bidi overrides shown as the escape they stand for (\\u202e)."""
    s = _BIDI_MOJIBAKE_RE.sub(lambda m: "\\u%04x" % (0x202A + ord(m.group(0)[2]) - 0xAA), str(s if s is not None else ""))
    return display_safe(s)


def _clip(s: Any, n: int) -> str:
    s = _show(s)
    return s if len(s) <= n else s[: n - 3] + "..."


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def dedupe_texts(items: Iterable[Dict[str, Any]], key: str = "text", sim: float = 0.8) -> List[Dict[str, Any]]:
    """Order-preserving removal of repeated / near-identical items (token Jaccard >= sim)."""
    out: List[Dict[str, Any]] = []
    toks: List[Set[str]] = []
    for it in items:
        t = set(_norm_key(str(it.get(key, ""))).split())
        if not t:
            continue
        if any(len(t & o) / max(len(t | o), 1) >= sim for o in toks):
            continue
        out.append(it)
        toks.append(t)
    return out


class EvidenceIndex:
    """Short, stable, resolvable references for events: EV:<first 8 hex of the event_uid>
    (extended only if two events of the run would collide)."""

    def __init__(self, event_uids: Iterable[str]):
        self._full = sorted(set(event_uids))
        self._short: Dict[str, str] = {}
        for n in (8, 10, 12, 36):
            counts = Counter(u[:n] for u in self._full)
            if all(c == 1 for c in counts.values()) or n == 36:
                self._short = {u: u[:n] for u in self._full}
                break
        self._rev = {v: k for k, v in self._short.items()}

    def ref(self, uid: str) -> str:
        return "EV:" + self._short.get(uid, uid[:8])

    def resolve(self, ref: str) -> Optional[str]:
        return self._rev.get(ref[3:]) if ref.startswith("EV:") else None


# --------------------------------------------------------------------------- event views & markers
def _first_line(raw: Dict[str, Any]) -> Optional[str]:
    msg = raw.get("Message")
    if not msg:
        return None
    first = str(msg).replace("\r", "\n").split("\n")[0].strip()
    first = first.split(":")[0].strip() if ":" in first else first
    return first[:90] or None


def _val(raw: Dict[str, Any], *keys: str) -> Optional[str]:
    for k in keys:
        v = raw.get(k)
        if v not in (None, "", "-", "--"):
            return str(v)
    return None


def _first_token(cmd: str) -> str:
    cmd = cmd.strip()
    if cmd.startswith('"') and '"' in cmd[1:]:
        return cmd[1:cmd.index('"', 1)]
    return cmd.split()[0] if cmd.split() else cmd


def _artifact_key(detail: str) -> str:
    """Normalised identity of an indicator artefact: lower-case, device-path prefix mapped to a drive letter."""
    d = _first_token(detail).lower().replace("/", "\\")
    return re.sub(r"^\\device\\harddiskvolume\d+\\", lambda m: "c:\\", d)


def event_view(e: Dict[str, Any]) -> Dict[str, Any]:
    """The fields of one normalized event that the report talks about (observed values only)."""
    raw = e.get("raw_event") or {}
    image = _val(raw, "Image", "NewProcessName", "SourceImage", "ProcessName", "Application") or e.get("source_image")
    cmd = _val(raw, "CommandLine")
    if not cmd and "powershell" in str(raw.get("Channel", "")).lower():
        cmd = _val(raw, "Message")
    parent = _val(raw, "ParentImage", "ParentProcessName")
    registry = _val(raw, "TargetObject")
    target_file = _val(raw, "TargetFilename")
    obj = target_file or e.get("target_image") or registry or _val(raw, "ImageLoaded", "ObjectName")
    dest = None
    if e.get("destination_ip"):
        dest = f"{e['destination_ip']}:{e['destination_port']}" if e.get("destination_port") not in (None, "") else str(e["destination_ip"])
    elif raw.get("DestinationIp"):
        dest = f"{raw['DestinationIp']}:{raw.get('DestinationPort')}" if raw.get("DestinationPort") else str(raw["DestinationIp"])
    hashes = _val(raw, "Hashes")
    sha256 = re.search(r"SHA256=([0-9A-Fa-f]{64})", hashes).group(1).upper() if hashes and re.search(r"SHA256=([0-9A-Fa-f]{64})", hashes) else None
    return {"event_id": e.get("event_id"), "timestamp": e.get("timestamp"), "host": e.get("hostname"),
            "user": _val(raw, "User", "SubjectUserName") or _val(raw, "AccountName", "UserID") or e.get("user"),
            "description": _first_line(raw) or (f"EventID {e.get('event_id')}"),
            "image": image, "parent": parent, "parent_command": _val(raw, "ParentCommandLine"),
            "object": obj, "target_filename": target_file, "target_image": e.get("target_image"),
            "registry": registry, "module": _val(raw, "ImageLoaded"), "object_name": _val(raw, "ObjectName"),
            "command": cmd, "dest": dest, "dest_ip": e.get("destination_ip") or raw.get("DestinationIp"), "sha256": sha256,
            "pid": e.get("process_id") or _val(raw, "ProcessId", "NewProcessId")}


def event_markers(e: Dict[str, Any]) -> Dict[str, str]:
    """marker class -> short detail (the artefact itself), for the indicator classes present in this event."""
    v = event_view(e)
    out: Dict[str, str] = {}
    cands = [v["image"], e.get("source_image"), v["parent"], v["target_image"], v["target_filename"], v["registry"],
             v["module"], v["object_name"]]
    for p in cands:
        if p and _has_hidden_chars(p):
            out["control_chars"] = _clip(p, 110)
            break
    else:
        for c in (v["command"], v["parent_command"]):
            if c and _has_hidden_chars(c):
                out["control_chars"] = _clip(_first_token(c), 110)
                break
    for p in (v["target_filename"], v["target_image"], v["image"], e.get("source_image"), v["parent"]):
        if p and str(p).strip().lower().endswith(_SCRIPT_EXT):
            out["script_like_extension"] = _clip(p, 110)
            break
    if v["command"] and _ENCODED_RE.search(str(v["command"])):
        out["encoded_command"] = _clip(v["command"], 110)
    if v["registry"] and _PERSIST_RE.search(str(v["registry"])):
        out["persistence_registry"] = _clip(v["registry"], 110)
    for p in (v["target_filename"], v["target_image"], v["image"], e.get("source_image"), v["parent"]):
        if p and _STAGING_RE.search(str(p)):
            out["staging_path"] = _clip(p, 110)
            break
    ip = v["dest_ip"]
    if ip:
        try:
            a = ipaddress.ip_address(str(ip))
            if not (a.is_private or a.is_loopback or a.is_link_local or a.is_multicast or a.is_unspecified or a.is_reserved):
                out["external_destination"] = _clip(v["dest"], 60)
        except ValueError:
            pass
    return out


def describe_event(e: Dict[str, Any]) -> str:
    """One-line plain description of an event (observed fields only)."""
    v = event_view(e)
    s = f"EventID {v['event_id']} \"{_clip(v['description'], 60)}\""
    actor = _basename(v["image"])
    obj = v["dest"] or (_clip(v["object"], 90) if v["object"] else "")
    if v["parent"] and actor and not obj:
        s += f": {_clip(_basename(v['parent']), 40)} started {_clip(actor, 40)}"
        if v["command"]:
            s += f" with command line {_clip(v['command'], 90)}"
    elif actor and obj:
        s += f": {_clip(actor, 40)} -> {obj}"
    elif actor:
        s += f": {_clip(actor, 40)}"
    elif obj:
        s += f": {obj}"
    return s


def _marker_strength(classes: Iterable[str]) -> Dict[str, int]:
    c = Counter(MARKERS[k]["strength"] for k in classes)
    return {"strong": c.get("strong", 0), "medium": c.get("medium", 0), "weak": c.get("weak", 0)}


# --------------------------------------------------------------------------- priority ladder (pure)
def novelty_tier(nov: Dict[str, Any]) -> int:
    lc = (nov or {}).get("label_counts") or {}
    if lc.get("HIGH_NOVELTY"):
        return 3
    if lc.get("SUSPICIOUS"):
        return 2
    return 1 if lc.get("LOW_NOVELTY") else 0


def rate_priority(tier: int, classes: Set[str], is_noise: bool, linked_with_marker: bool) -> Tuple[str, str]:
    """(priority, plain-language reason). See the module docstring for the ladder."""
    st = _marker_strength(classes)
    strong, medium, weak, distinct = st["strong"], st["medium"], st["weak"], len(classes)
    if tier >= 3 and strong >= 1 and distinct >= 2 and linked_with_marker:
        return "Critical", "HIGH_NOVELTY, a strong indicator, several indicator classes and a shared-process-id link to another indicator-bearing unit"
    if strong >= 1 and (tier >= 2 or distinct >= 2):
        return "High", "a strong observable indicator " + ("with SUSPICIOUS-or-higher novelty" if tier >= 2 else "plus a second indicator class")
    if medium >= 2 and tier >= 2:
        return "High", "two medium indicator classes with SUSPICIOUS-or-higher novelty"
    if strong >= 1:
        return "Medium", "a strong observable indicator in otherwise normal-novelty activity"
    if medium >= 1 and tier >= 1:
        return "Medium", "a medium observable indicator with above-normal novelty"
    if is_noise and tier >= 2:
        return "Medium", "unclustered (noise) activity with SUSPICIOUS-or-higher novelty and no cluster to compare it with"
    if tier >= 2:
        return "Low", "SUSPICIOUS-or-higher novelty without any observable indicator"
    if tier >= 1 and (weak >= 1 or is_noise):
        return "Low", "slightly unusual " + ("noise activity" if is_noise else "activity with a weak indicator")
    if medium >= 1:
        return "Low", "a medium indicator in normal-novelty activity"
    return "Informational", "no above-normal novelty and no observable indicator"


def status_for(priority: str, classes: Set[str]) -> Tuple[str, str]:
    st = _marker_strength(classes)
    if st["strong"] or priority in ("Critical", "High"):
        return "Suspicious", "observable indicator(s) in the events themselves, not only a statistical score"
    if classes or priority == "Medium":
        return "Needs investigation", "unusual, with limited direct indicators"
    return "Insufficient evidence", "statistically unusual only; no observed indicator distinguishes it from rare routine activity"


# --------------------------------------------------------------------------- unit assessment
def _unit_links(units: Sequence[Any]) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, int]]]:
    """pid_links: unit -> units reached by a DIRECT shared-process-id edge from this unit's seeds.
    host_links: unit -> {other unit: events linked by same-host/time edges}."""
    pid: Dict[str, Set[str]] = defaultdict(set)
    host: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    ids = {u.unit_id for u in units}
    for u in units:
        for ev in (u.retrieval or {}).get("traversal_events", []):
            other = ev.get("unit_id")
            if ev.get("in_unit") or not other or other == u.unit_id or other not in ids:
                continue
            rel = str(ev.get("via_rel") or "")
            if ev.get("hop") == 1 and "SAME_PID" in rel:
                pid[u.unit_id].add(other)
                pid[other].add(u.unit_id)
            else:
                host[u.unit_id][other] += 1
                host[other][u.unit_id] += 1
    return pid, host


def _artifact_links(assess: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], Set[str]], Dict[Tuple[str, str], str]]:
    """Units whose events carry the SAME strong/medium indicator artefact (identical path / registry value / command text).
    Sharing the exact artefact (e.g. one specific file path) is a defensible link; sharing a host or process name is not."""
    owners: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    names: Dict[Tuple[str, str], str] = {}
    for uid, a in assess.items():
        for m in a["event_markers"].values():
            for cls, detail in m.items():
                if MARKERS[cls]["strength"] != "weak":
                    key = (cls, _artifact_key(detail))
                    owners[key].add(uid)
                    names.setdefault(key, _first_token(detail))
    links: Dict[str, Set[str]] = defaultdict(set)
    shared = {k: v for k, v in owners.items() if len(v) > 1}
    for units in shared.values():
        for u in units:
            links[u] |= units - {u}
    return links, shared, {k: names[k] for k in shared}


def assess_unit(u: Any, events_by_uid: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    members = [m for m in (u.profile or {}).get("member_event_uids", []) if m in events_by_uid]
    by_class: Dict[str, List[str]] = defaultdict(list)
    ev_markers: Dict[str, Dict[str, str]] = {}
    for uid in members:
        m = event_markers(events_by_uid[uid])
        if m:
            ev_markers[uid] = m
            for k in m:
                by_class[k].append(uid)
    return {"unit_id": u.unit_id, "scope": u.scope, "size": u.size, "tier": novelty_tier(u.novelty_summary),
            "classes": set(by_class), "events_by_class": dict(by_class), "event_markers": ev_markers,
            "is_noise": u.scope != "cluster", "novelty_max": (u.novelty_summary or {}).get("max") or 0.0}


def supplemental_noise_units(noise_units: Sequence[Dict[str, Any]], analysed_ids: Set[str],
                             events_by_uid: Dict[str, Dict[str, Any]], novelty_by_uid: Dict[str, Dict[str, Any]],
                             limit: int = 40) -> List[Any]:
    """Noise review units that did NOT get full semantic analysis (outside --noise-review-top-n) but whose own
    events carry a strong or medium observable indicator. They are surfaced as findings WITHOUT ATT&CK analysis,
    so a significant indicator is never hidden only because its unit ranked low by novelty."""
    out = []
    for ru in noise_units:
        if ru["unit_id"] in analysed_ids:
            continue
        uids = [x for x in ru["event_uids"] if x in events_by_uid]
        if not any(any(MARKERS[k]["strength"] in ("strong", "medium") for k in event_markers(events_by_uid[x])) for x in uids):
            continue
        rows = [novelty_by_uid[x] for x in uids if x in novelty_by_uid]
        nov: Dict[str, Any] = {}
        if rows:
            top = sorted(rows, key=lambda r: (-r["novelty_score"], r["event_uid"]))[:5]
            nov = {"max": round(max(r["novelty_score"] for r in rows), 4),
                   "mean": round(sum(r["novelty_score"] for r in rows) / len(rows), 4),
                   "label_counts": dict(Counter(r["novelty_label"] for r in rows)),
                   "top_events": [{"event_uid": r["event_uid"], "novelty_score": r["novelty_score"],
                                   "novelty_label": r["novelty_label"]} for r in top]}
        prof = build_unit_profile(ru["unit_id"], ru["scope"], [events_by_uid[x] for x in uids], uids[:3]).as_dict()
        out.append(SimpleNamespace(unit_id=ru["unit_id"], scope=ru["scope"], size=len(uids), profile=prof,
                                   novelty_summary=nov, mapping={}, retrieval={}, report={}, supplemental=True,
                                   priority=nov.get("max", 0.0)))
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- grouping
def _components(nodes: Sequence[str], links: Dict[str, Set[str]]) -> List[List[str]]:
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in nodes:
        for b in links.get(a, ()):
            if b in parent:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    groups: Dict[str, List[str]] = defaultdict(list)
    for n in nodes:
        groups[find(n)].append(n)
    return [sorted(g) for g in groups.values()]


# --------------------------------------------------------------------------- finding construction
def _merge_counter(items: Iterable[Sequence[Sequence[Any]]], n: int) -> List[Tuple[Any, int]]:
    c: Counter = Counter()
    for lst in items:
        for k, v in lst or []:
            c[k] += v
    return c.most_common(n)


def _time_span(units: Sequence[Any]) -> Tuple[Optional[str], Optional[str]]:
    lo = [u.profile.get("temporal_span", {}).get("earliest") for u in units]
    hi = [u.profile.get("temporal_span", {}).get("latest") for u in units]
    lo, hi = [x for x in lo if x], [x for x in hi if x]
    return (min(lo) if lo else None, max(hi) if hi else None)


def _short_time(ts: Optional[str]) -> str:
    return ts[:19].replace("T", " ") + " UTC" if ts else "time unavailable"


def _evidence_entries(units: Sequence[Any], assessments: Dict[str, Dict[str, Any]], events_by_uid: Dict[str, Dict[str, Any]],
                      prov: Dict[str, Dict[str, Any]], idx: EvidenceIndex, novelty_by_uid: Dict[str, Dict[str, Any]],
                      k: int = 6) -> List[Dict[str, Any]]:
    cand: Dict[str, Tuple] = {}
    for u in units:
        a = assessments[u.unit_id]
        for uid, m in a["event_markers"].items():
            w = sum(_STRENGTH_RANK[MARKERS[c]["strength"]] for c in m)
            launch = 1 if event_view(events_by_uid[uid])["parent"] else 0       # records that name the launching process explain most
            cand[uid] = (w + launch, (novelty_by_uid.get(uid) or {}).get("novelty_score") or 0.0)
        for te in (u.novelty_summary or {}).get("top_events", [])[:5]:
            cand.setdefault(te["event_uid"], (0, te["novelty_score"]))
        for uid in (u.profile or {}).get("representative_event_uids", [])[:2]:
            cand.setdefault(uid, (0, (novelty_by_uid.get(uid) or {}).get("novelty_score") or 0.0))
    ranked = sorted((uid for uid in cand if uid in events_by_uid),
                    key=lambda uid: (-cand[uid][0], -cand[uid][1], events_by_uid[uid].get("timestamp") or "", uid))[:k]
    ranked.sort(key=lambda uid: (events_by_uid[uid].get("timestamp") or "", uid))
    out = []
    for uid in ranked:
        e = events_by_uid[uid]
        v = event_view(e)
        nr = novelty_by_uid.get(uid) or {}
        m = {}
        for u in units:
            m.update(assessments[u.unit_id]["event_markers"].get(uid, {}))
        out.append({"ref": idx.ref(uid), "event_uid": uid, "timestamp": v["timestamp"], "host": v["host"],
                    "user": v["user"], "event_id": v["event_id"], "description": v["description"], "summary": describe_event(e),
                    "pid": v["pid"], "parent": v["parent"], "command": _clip(v["command"], 160) if v["command"] else None, "sha256": v["sha256"], "source": (prov.get(uid) or {}).get("source_ref") or "input log",
                    "novelty_score": nr.get("novelty_score"), "novelty_label": nr.get("novelty_label"),
                    "indicators": sorted(m), "indicator_detail": m,
                    "unit_ids": [u.unit_id for u in units if uid in (u.profile or {}).get("member_event_uids", [])]})
    return out


def _technique_entries(units: Sequence[Any], evidence_uids: Sequence[str], events_by_uid, idx: EvidenceIndex,
                       cand_thr: float, event_scorer: Optional[Callable], max_techniques: int = 3) -> Tuple[List[Dict[str, Any]], str]:
    best: Dict[str, Dict[str, Any]] = {}
    for u in units:
        for c in (u.mapping or {}).get("top_k_techniques", []):
            if c.get("mapping_status") != "CANDIDATE":
                continue
            cur = best.get(c["technique_id"])
            if cur is None or c["similarity_score"] > cur["unit_similarity"]:
                refs = [r for r in (u.retrieval or {}).get("references", []) if r["technique_id"] == c["technique_id"]]
                det = next((r for r in refs if r.get("passage_type") == "detection"), None)
                best[c["technique_id"]] = {
                    "technique_id": c["technique_id"], "name": c["technique_name"],
                    "tactics": c.get("tactics") or ([c["tactic"]] if c.get("tactic") else []),
                    "unit_similarity": c["similarity_score"], "status": c["mapping_status"], "from_unit": u.unit_id,
                    "reference_ids": [r["ref_id"] for r in refs],
                    "detection_guidance": _clip(re.sub(r"^Detection strategy:\s*", "", det["text"]), 260) if det else None}
    chosen = sorted(best.values(), key=lambda t: (-t["unit_similarity"], t["technique_id"]))[:max_techniques]
    if not chosen:
        weak = any(c.get("mapping_status") == "WEAK_MATCH" for u in units for c in (u.mapping or {}).get("top_k_techniques", []))
        analysed = any(u.mapping for u in units)
        note = ("No ATT&CK technique reached the candidate threshold" + (" (only weak similarity)" if weak else "") +
                "; this does not mean the activity is benign." if analysed else
                "No ATT&CK analysis was run for this unit (it ranked outside --noise-review-top-n); it is listed because of its observable indicator.")
        return [], note
    scores: Dict[str, Dict[str, float]] = {}
    if event_scorer is not None and evidence_uids:
        try:
            scores = event_scorer(list(evidence_uids), [t["technique_id"] for t in chosen]) or {}
        except Exception:                                                        # presentation aid only: never break the report
            scores = {}
    for t in chosen:
        sup = sorted(((uid, s.get(t["technique_id"], 0.0)) for uid, s in scores.items()), key=lambda x: (-x[1], x[0]))
        sup = [(uid, sc) for uid, sc in sup if sc >= cand_thr][:3]
        t["event_support"] = [{"ref": idx.ref(uid), "event_level_similarity": round(sc, 4)} for uid, sc in sup]
        t["context_only"] = not sup
        t["support_basis"] = ("event-level text similarity at or above the candidate threshold (ranking signal, not proof)"
                              if sup else "unit-level narrative similarity only; no single event reaches the candidate threshold"
                              if scores else "unit-level narrative similarity only; event-level check was not available")
        t["fit"] = "weak"

        t["why_candidate"] = (f"The unit narrative has cosine similarity {t['unit_similarity']} to ATT&CK {t['technique_id']} "
                              f"(uncalibrated; status CANDIDATE).")
        t["limitations"] = ("Similarity is a text-overlap lead, not evidence that the technique was used."
                            + (" Unit-level context only: no individual event supports it specifically." if t["context_only"] else ""))
    return chosen, ""


def _missing_evidence(units, ev_entries, ctx_trunc: bool, has_cmd: bool, neighbours_have_proc_create: bool,
                      has_user: bool) -> List[Dict[str, Any]]:
    out = []
    if not neighbours_have_proc_create:
        out.append("No process-creation record (e.g. Sysmon EventID 1) for the acting process was found among the unit or its "
                   "graph neighbours, so the parent process and command line are unknown.")
    if not has_cmd:
        out.append("No command-line or script text was recorded for these events.")
    if not has_user:
        out.append("No user account is recorded on the key events, so the responsible account cannot be identified.")
    if ctx_trunc:
        out.append("Graph traversal hit its event cap; the neighbourhood examined is partial.")
    if any(getattr(u, "supplemental", False) for u in units):
        out.append("This unit was not given semantic ATT&CK analysis or graph retrieval (it ranked outside --noise-review-top-n).")
    return [{"text": t, "source": "deterministic"} for t in out]


def _benign(classes: Set[str], is_noise: bool) -> List[Dict[str, Any]]:
    out = [MARKERS[k]["benign"][0].upper() + MARKERS[k]["benign"][1:] + "." for k in sorted(classes, key=lambda k: -_STRENGTH_RANK[MARKERS[k]["strength"]])]
    if is_noise:
        out.append("Rare but legitimate administration or software activity can fall outside every dense cluster.")
    if not out:
        out.append("No observed indicator separates this from rare but routine activity (maintenance, a new application, a one-off admin task).")
    return [{"text": t, "source": "deterministic"} for t in out[:4]]


def _next_steps(f: Dict[str, Any]) -> List[Dict[str, Any]]:
    ev = f["evidence"]
    refs = [e["ref"] for e in ev[:3]]
    host = (f["hosts"] or ["the host"])[0]
    steps: List[Dict[str, Any]] = []

    def add(key, action, why, evid, strengthens, weakens):
        steps.append({"key": key, "action": action, "why": why, "evidence": evid, "strengthens_if": strengthens,
                      "weakens_if": weakens, "source": "deterministic", "finding": f["id"]})

    classes = {c for e in ev for c in e["indicators"]}
    first_with = lambda c: next((e for e in ev if c in e["indicators"]), None)
    add("confirm_records",
        f"Open the source records for {', '.join(refs)} ({'; '.join(sorted({e['source'] for e in ev[:3]}))}) and confirm the user, process, path and timestamps shown above.",
        "Everything in this finding rests on these entries; confirming them rules out parsing or normalization artefacts.",
        refs, "Fields match this summary and show a consistent account / process across the events.",
        "Records show known test, admin or deployment tooling, or the fields differ from the summary.")
    ev_c = first_with("control_chars") or first_with("script_like_extension")
    if ev_c:
        det = ev_c["indicator_detail"].get("control_chars") or ev_c["indicator_detail"].get("script_like_extension")
        pid = ev_c["pid"]
        if ev_c["parent"]:
            add("review_launch",
                f"Review how {det} was launched on {host}: parent {_basename(ev_c['parent'])}"
                + (f", user {ev_c['user']}" if ev_c["user"] else "") + (f", process id {pid}" if pid else "")
                + f" at {_short_time(ev_c['timestamp'])}; check whether that parent and the user normally start such files.",
                "The launch record names the parent process, but whether this launch is expected for the host and account is not established.",
                [ev_c["ref"]], "The parent / user is unexpected for this host, or the command line stages or launches the file.",
                "The launch matches a known installer, user action or approved change.")
        else:
            add("find_creator",
                f"Find which process created or started {det} on {host}: look for process-creation and file-creation records"
                + (f" for process id {pid}" if pid else "") + f" around {_short_time(ev_c['timestamp'])}, and read the parent process and command line.",
                "The events name the file but no parent process or command line was recorded, so how it got there is unknown.",
                [ev_c["ref"]], "The creating process is unexpected for this host or its command line stages / launches the file.",
                "The creator is a known installer, service or admin tool covered by an approved change.")
        add("check_file",
            "If the file still exists, hash it and compare against reputation sources, and confirm its real on-disk name and extension.",
            "A disguised or screensaver-type executable cannot be judged from the log line alone.",
            [ev_c["ref"]], "The file is an unsigned or unknown executable, or its true extension differs from the displayed name.",
            "The file is a signed, known-good component.")
    ev_e = first_with("encoded_command")
    if ev_e:
        add("decode_command", f"Decode and read the command text in {ev_e['ref']}, then look for follow-on process, file or network events from the same process id.",
            "Encoded or download-style commands hide their purpose from the log line.", [ev_e["ref"]],
            "The decoded command downloads, stages or executes unexpected content.", "The decoded command is a documented management task.")
    ev_r = first_with("persistence_registry")
    if ev_r:
        add("check_registry", f"Identify the process that wrote {ev_r['indicator_detail']['persistence_registry']} ({ev_r['ref']}) and check what file the value points to and whether a software install or change explains it.",
            "Autostart locations are a common way to keep code running, but installers write them too.", [ev_r["ref"]],
            "The value points to an unfamiliar or user-writable file and no change record exists.", "A matching installer or update ran at the same time.")
    for rel in f["related"][:1]:
        add("review_related", f"Review the events of {rel['unit_id']} ({rel['relation']}"
            + (f"; e.g. {', '.join(rel['refs'][:2])}" if rel.get("refs") else "") + ") for the surrounding sequence.",
            "Related activity may show what happened before or after the key events.", rel.get("refs", [])[:2],
            "The related events continue the same process / host activity in a suspicious direction.", "The related events are routine for the host.")
    ev_n = first_with("external_destination")
    if ev_n and len(steps) < 4:
        add("check_network", f"Check DNS, proxy and firewall logs for {ev_n['indicator_detail']['external_destination']} from {host} around {_short_time(ev_n['timestamp'])}: which process owned the connection, and is the destination an expected service?",
            "The event shows the connection but not the destination's identity or purpose.", [ev_n["ref"]],
            "The destination is unexpected for this host and the owning process is unusual.", "The destination is a known update, cloud or DNS service.")
    if not classes:
        top = (f["activity"] or ["the observed activity"])[0]
        add("baseline", f"Check whether {host} shows {top} at other times or on peer hosts, and whether a change window or admin task explains this occurrence.",
            "With no observed indicator, the only signal is that this activity is rare in the dataset.", refs,
            "Nothing comparable appears elsewhere and no change record exists.", "The same activity recurs on peers or matches a known task.")
    return steps[:5]


def _best(ratings: Iterable[Tuple[str, str]]) -> Tuple[str, str]:
    return min(ratings, key=lambda r: _PRIO_RANK[r[0]])


def build_findings(units: Sequence[Any], events_by_uid: Dict[str, Dict[str, Any]], prov: Dict[str, Dict[str, Any]],
                   novelty_by_uid: Dict[str, Dict[str, Any]], idx: EvidenceIndex, *, max_findings: int, cand_thr: float,
                   event_scorer: Optional[Callable] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_id = {u.unit_id: u for u in units}
    pid_links, host_links = _unit_links(units)
    assess = {u.unit_id: assess_unit(u, events_by_uid) for u in units}
    art_links, shared_art, art_names = _artifact_links(assess)
    rated = {uid: rate_priority(a["tier"], a["classes"], a["is_noise"], False) for uid, a in assess.items()}
    cands = [uid for uid, (p, _) in rated.items() if p != "Informational"]
    cset = set(cands)
    links = {u: (pid_links.get(u, set()) | art_links.get(u, set())) & cset for u in cands}
    groups = _components(cands, links)

    def group_rating(g: List[str]) -> Tuple[str, str]:
        classes = set().union(*[assess[x]["classes"] for x in g])
        carriers = sum(1 for x in g if any(MARKERS[c]["strength"] != "weak" for c in assess[x]["classes"]))
        own = rate_priority(max(assess[x]["tier"] for x in g), classes, all(assess[x]["is_noise"] for x in g), carriers >= 2)
        return _best([own] + [rated[x] for x in g])

    gr = {tuple(g): group_rating(g) for g in groups}

    def gkey(g):
        return (_PRIO_RANK[gr[tuple(g)][0]],
                -sum(_STRENGTH_RANK[MARKERS[c]["strength"]] for c in set().union(*[assess[x]["classes"] for x in g])),
                -max(assess[x]["novelty_max"] for x in g), g[0])

    ordered = sorted(groups, key=gkey)
    unit_to_idx = {uid: gi for gi, g in enumerate(ordered, start=1) for uid in g}
    findings: List[Dict[str, Any]] = []
    for gi, g in enumerate(ordered, start=1):
        fid = f"F{gi}"
        if gi > max_findings:
            continue
        us = [by_id[x] for x in g]
        prio, why_prio = gr[tuple(g)]
        classes: Set[str] = set().union(*[assess[x]["classes"] for x in g])
        status, status_reason = status_for(prio, classes)
        ev = _evidence_entries(us, assess, events_by_uid, prov, idx, novelty_by_uid)
        hosts = [h for h, _ in _merge_counter([list(u.profile.get("hostname_distribution", {}).items()) for u in us], 3)]
        lo, hi = _time_span(us)
        activity = [d for d, _ in _merge_counter([u.profile.get("event_descriptions") for u in us], 3)]
        images = [_basename(p) for p, _ in _merge_counter([u.profile.get("common_source_images") for u in us], 3) if p]
        size = sum(u.size for u in us)
        # ---- merge basis (why several units are ONE finding)
        basis: List[str] = []
        if len(g) > 1:
            if any((pid_links.get(x, set()) & set(g)) for x in g):
                basis.append("their events share a process id")
            arts = sorted({art_names[k] for k, owners in shared_art.items() if len(owners & set(g)) > 1})
            if arts:
                basis.append("they carry the same indicator artefact (" + "; ".join(_clip(a, 70) for a in arts[:2]) + ")")
        # ---- title: from the event that carries the strongest indicator
        lead_ev = max(ev, key=lambda e: (sum(_STRENGTH_RANK[MARKERS[c]["strength"]] for c in e["indicators"]), e["novelty_score"] or 0.0)) if ev else None
        dom = max(classes, key=lambda c: (_STRENGTH_RANK[MARKERS[c]["strength"]], c)) if classes else None
        lead = _clip(lead_ev["description"], 45) if lead_ev else (activity[0] if activity else "mixed events")
        host_s = f" on {(lead_ev['host'] if lead_ev and lead_ev['host'] else (hosts[0] if hosts else ''))}".rstrip() if (hosts or lead_ev) else ""
        if dom == "control_chars":
            title = f"File name with hidden bidirectional control characters{host_s}"
        elif dom == "script_like_extension":
            title = f"Script- or screensaver-type file activity{host_s}"
        elif dom == "encoded_command":
            title = f"Encoded or download-style command text{host_s}"
        elif dom == "persistence_registry":
            title = f"Registry change at an autostart-style location{host_s}"
        elif dom:
            title = {"external_destination": "Connection to a public IP address", "staging_path": "Activity in a staging-style path"}.get(dom, dom.replace("_", " ")) + host_s
        else:
            kind = "Unclustered unusual activity" if all(u.scope != "cluster" for u in us) else "Unusual behaviour cluster"
            title = f"{kind}: {lead}{host_s}"
        # ---- related units (cross-referenced, NOT merged)
        rel_out: List[Dict[str, Any]] = []
        seen_rel: Set[str] = set()

        def add_rel(other: str, relation: str, n: int) -> None:
            if other in g or other in seen_rel:
                return
            seen_rel.add(other)
            tr = next((t for u in us for t in (u.retrieval or {}).get("traversal_events", []) if t.get("unit_id") == other), None)
            oi = unit_to_idx.get(other)
            rel_out.append({"unit_id": other, "finding": f"F{oi}" if oi and oi <= max_findings else None,
                            "candidate": oi is not None, "relation": relation, "graph_links": n,
                            "refs": [idx.ref(tr["event_uid"])] if tr else []})

        for uid in g:
            for other in sorted(pid_links.get(uid, ())):
                add_rel(other, RELATIONS["pid"], 1)
        for uid in g:
            for other, n in sorted(host_links.get(uid, {}).items(), key=lambda kv: (-kv[1], kv[0])):
                if other in unit_to_idx:
                    add_rel(other, RELATIONS["host_time"], n)
        f = {"id": fid, "title": title, "priority": prio, "priority_reason": why_prio, "status": status, "status_reason": status_reason,
             "unit_ids": g, "merged": len(g) > 1, "merge_basis": basis,
             "scope": sorted({u.scope for u in us}), "size": size, "hosts": hosts, "time_span": {"earliest": lo, "latest": hi},
             "novelty": {"max": max((u.novelty_summary or {}).get("max") or 0.0 for u in us),
                         "labels": dict(sum((Counter((u.novelty_summary or {}).get("label_counts") or {}) for u in us), Counter())),
                         "meaning": "unusualness within this dataset - not a maliciousness verdict"},
             "activity": activity, "processes": images,
             "indicators": [{"class": c, "strength": MARKERS[c]["strength"], "description": MARKERS[c]["text"],
                             "events": sorted({idx.ref(x) for y in g for x in assess[y]["events_by_class"].get(c, [])})[:6],
                             "event_count": sum(len(assess[y]["events_by_class"].get(c, [])) for y in g)}
                            for c in sorted(classes, key=lambda c: (-_STRENGTH_RANK[MARKERS[c]["strength"]], c))],
             "evidence": ev, "related": rel_out[:3]}
        # ---- ATT&CK (specific evidence only)
        f["attack"], f["attack_note"] = _technique_entries(us, [e["event_uid"] for e in ev], events_by_uid, idx, cand_thr, event_scorer)
        # ---- deterministic narrative blocks
        f["observed"] = _observed(f, us)
        f["why_it_matters"] = _why(f, us, classes)
        trunc = any((u.retrieval or {}).get("truncated") for u in us)
        has_cmd = any((u.profile or {}).get("command_texts") for u in us)
        nb_events = [t for u in us for t in (u.retrieval or {}).get("traversal_events", [])]
        proc_create = (any(t.get("event_id") == 1 for t in nb_events) or
                       any(int(eid) == 1 for u in us for eid, _ in (u.profile or {}).get("dominant_event_ids", []) if str(eid).isdigit()) or
                       any(events_by_uid.get(e["event_uid"], {}).get("event_id") == 1 for e in ev) or
                       any(e["parent"] for e in ev))
        has_user = any(e["user"] for e in ev)
        f["missing_evidence"] = _missing_evidence(us, ev, trunc, has_cmd, proc_create, has_user)
        f["benign_explanations"] = _benign(classes, all(u.scope != "cluster" for u in us))
        f["next_steps"] = _next_steps(f)
        f["analysis"] = {"source": "deterministic",
                         "what_happened": {"text": " ".join(f["observed"][:2]), "evidence": [e["ref"] for e in ev[:3]]},
                         "why_unusual": {"text": " ".join(f["why_it_matters"][:2]), "evidence": [f"U:{x}" for x in g[:2]]},
                         "security_relevance": {"text": _hedged_relevance(f), "kind": "hypothesis"},
                         "conclusion": {"supported": _supported(f), "uncertain": _uncertain(f)}}
        f["evidence_refs"] = sorted({e["ref"] for e in ev} | {f"U:{x}" for x in g[:6]} | {f"TC:{t['technique_id']}" for t in f["attack"]})
        findings.append(f)
    meta = {"candidates": len(cands), "groups": len(ordered), "shown": len(findings),
            "not_shown": [{"finding": f"F{i}", "units": g, "priority": gr[tuple(g)][0]} for i, g in enumerate(ordered, start=1) if i > max_findings],
            "informational_units": sorted(uid for uid in rated if uid not in cset),
            "priority_by_unit": {uid: p for uid, (p, _) in rated.items()}}
    return findings, meta


def _unit_list(ids: Sequence[str], n: int = 5) -> str:
    return ", ".join(ids[:n]) + (f" (+{len(ids) - n} more)" if len(ids) > n else "")


def _observed(f: Dict[str, Any], us: Sequence[Any]) -> List[str]:
    span = f["time_span"]
    scope = {"cluster": "behaviour cluster", "noise_group": "unclustered (noise) group", "noise_event": "unclustered (noise) event"}
    kinds = ", ".join(sorted({scope[x] for x in f["scope"]}))
    where = f" on {', '.join(f['hosts'])}" if f["hosts"] else ""
    when = f" between {_short_time(span['earliest'])} and {_short_time(span['latest'])}" if span["earliest"] else ""
    n_ind = max((i["event_count"] for i in f["indicators"]), default=0)
    ind_s = f" (up to {n_ind} carry an observed indicator)" if n_ind and len(f["unit_ids"]) > 1 else ""
    out = [f"{f['size']} event(s){ind_s} in {len(f['unit_ids'])} unit(s) ({kinds}: {_unit_list(f['unit_ids'])}){where}{when}."]
    if f["activity"]:
        out.append("Logged activity types: " + "; ".join(_clip(a, 60) for a in f["activity"]) +
                   (f". Main process: {', '.join(f['processes'][:2])}." if f["processes"] else "."))
    show = [e for e in f["evidence"] if e["indicators"]][:2] or f["evidence"][:1]
    for e in show:
        out.append(f"{e['ref']} ({_short_time(e['timestamp'])}): {e['summary']}.")
    return out


def _why(f: Dict[str, Any], us: Sequence[Any], classes: Set[str]) -> List[str]:
    out = [f"Highest event novelty {f['novelty']['max']} ({', '.join(f'{k}={v}' for k, v in sorted(f['novelty']['labels'].items()))}): "
           f"unusual compared with the rest of this dataset - a statistical measure, not a verdict."]
    if all(u.scope != "cluster" for u in us):
        sig = Counter()
        for u in us:
            sig.update((u.profile or {}).get("noise_signals") or {})
        out.append("Not part of any behaviour cluster" + (" (" + ", ".join(k.replace("_", " ") for k, _ in sig.most_common(3)) + ")" if sig else "") +
                   ": measured properties, not causes.")
    for i in f["indicators"]:
        out.append(f"Observed indicator ({i['strength']}): {i['description']} - {i['event_count']} event(s), e.g. {', '.join(i['events'][:2])}.")
    if f["merged"]:
        out.append(f"{len(f['unit_ids'])} units are treated as one finding because " + " and ".join(f["merge_basis"]) +
                   ("; a reused process id could link unrelated activity." if any("process id" in b for b in f["merge_basis"]) else "."))
    return out


def _hedged_relevance(f: Dict[str, Any]) -> str:
    if f["indicators"]:
        return ("If the indicator reflects intent, this could be an attempt to disguise or stage a file; the log alone cannot show intent, "
                "and the benign explanations below have not been ruled out.")
    return "Unusual activity of this kind is worth a baseline check; nothing observed indicates harm."


def _supported(f: Dict[str, Any]) -> str:
    if f["indicators"]:
        return "The listed events occurred and carry the stated indicator(s); the activity is unusual for this dataset."
    return "The activity is statistically unusual for this dataset."


def _uncertain(f: Dict[str, Any]) -> str:
    return ("Intent, the responsible account and whether anything followed are not established; ATT&CK candidates are text-similarity leads only."
            if f["attack"] else "Intent, the responsible account and whether anything followed are not established.")


# --------------------------------------------------------------------------- run-level sections
def build_overall(findings: List[Dict[str, Any]], meta: Dict[str, Any], limits: List[str]) -> Dict[str, Any]:
    if not findings:
        return {"significant_patterns": [], "common_explanation": "No finding met the reporting criteria.",
                "can_conclude": ["No unit combined unusual behaviour with an observable indicator."],
                "cannot_conclude": ["The absence of a finding does not show the environment is clean."],
                "investigate_first": None, "most_useful_additional_evidence": []}
    top = findings[0]
    sig = [f"{f['id']} [{f['priority']}] {f['title']}" for f in findings[:3]]
    merged = [f for f in findings if f["merged"]]
    host_rel = [(f["id"], r["unit_id"]) for f in findings for r in f["related"] if "same host" in r["relation"]]
    cross = []
    idx_of_unit = {u: f["id"] for f in findings for u in f["unit_ids"]}
    for f in findings:
        for r in f["related"]:
            o = idx_of_unit.get(r["unit_id"])
            if o and o != f["id"] and sorted([f["id"], o]) not in cross:
                cross.append(sorted([f["id"], o]))
    if merged:
        common = (f"{', '.join(m['id'] for m in merged)} combine units whose events share a process id, which is direct evidence of a shared process (not of intent). ")
    else:
        common = "No findings share a process id, so they are treated as independent. "
    if cross:
        common += "Findings " + "; ".join(f"{a} and {b}" for a, b in cross[:3]) + " are temporally adjacent on the same host; that may reflect a shared cause or coincidence and is not evidence of one campaign."
    elif host_rel:
        common += "Some findings sit near other activity on the same host within 300 s; no common cause is established."
    missing = Counter(m["text"] for f in findings for m in f["missing_evidence"])
    top_missing = [t for t, _ in missing.most_common(3)]
    return {"significant_patterns": sig,
            "common_explanation": common.strip(),
            "can_conclude": [f"{f['id']}: {f['analysis']['conclusion']['supported']}" for f in findings[:3]],
            "cannot_conclude": ["Whether any activity is malicious, who performed it, or whether it led to compromise, persistence, lateral movement or data loss.",
                                "That any ATT&CK technique was used: candidates are uncalibrated text-similarity leads."],
            "investigate_first": {"finding": top["id"], "why": f"highest priority ({top['priority']}: {top['priority_reason']})"},
            "most_useful_additional_evidence": top_missing}


def build_plan(findings: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    steps = []
    for f in findings:
        for s in f["next_steps"]:
            steps.append(dict(s, finding=f["id"], priority=f["priority"]))
    steps.sort(key=lambda s: (_PRIO_RANK[s["priority"]], 0 if s["key"] != "confirm_records" else 1, s["finding"]))
    out, seen = [], set()
    for s in steps:
        k = (s["finding"], s["key"])
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    out = dedupe_texts(out, key="action", sim=0.9)[:limit]
    for i, s in enumerate(out, start=1):
        s["order"] = i
    return out


def build_executive(findings, meta, overview, run_info, mode_note: str) -> Dict[str, Any]:
    counts = Counter(f["priority"] for f in findings)
    top = findings[0] if findings else None
    if not findings:
        assess = "No unit met the reporting criteria: nothing combined unusual behaviour with an observable indicator. This does not show the data is clean."
    elif counts.get("Critical") or counts.get("High"):
        assess = (f"{len(findings)} finding(s) warrant investigation, led by {top['id']} ({top['priority']}). "
                  "They rest on unusual behaviour plus directly observed indicators; no incident is confirmed.")
    else:
        assess = (f"{len(findings)} finding(s) are worth review, none with strong corroborated indicators. They are mostly statistically unusual "
                  "activity; no incident is confirmed.")
    reasons = [f"{f['id']} [{f['priority']}]: {f['title']} - {f['priority_reason']}." for f in findings[:3]]
    limits = list(run_info.get("limitations", []))
    return {"assessment": assess, "events_analysed": overview.get("events"),
            "units_reviewed": overview.get("units_analysed"), "clusters": overview.get("clusters"),
            "noise_events": overview.get("noise_events"), "findings": len(findings),
            "findings_by_priority": {p: counts[p] for p in PRIORITIES if counts.get(p)},
            "further_candidates_not_shown": len(meta.get("not_shown", [])),
            "highest_priority": ({"finding": top["id"], "priority": top["priority"], "title": top["title"], "status": top["status"]} if top else None),
            "main_reasons": reasons[:3], "evidence_limitations": limits[:5], "analysis_mode": mode_note}


def build_analyst_report(units: Sequence[Any], events_by_uid: Dict[str, Dict[str, Any]], prov: Dict[str, Dict[str, Any]],
                         novelty_by_uid: Dict[str, Dict[str, Any]], noise_units: Sequence[Dict[str, Any]], run_info: Dict[str, Any],
                         overview: Dict[str, Any], *, max_findings: int = 5, candidate_threshold: float = 0.35,
                         event_scorer: Optional[Callable] = None) -> Dict[str, Any]:
    """Deterministic analyst report (schema ANALYST_SCHEMA_VERSION). The LLM step may enrich it afterwards."""
    idx = EvidenceIndex(events_by_uid)
    analysed = {u.unit_id for u in units}
    extra = supplemental_noise_units(noise_units, analysed, events_by_uid, novelty_by_uid)
    findings, meta = build_findings(list(units) + extra, events_by_uid, prov, novelty_by_uid, idx, max_findings=max_findings,
                                    cand_thr=candidate_threshold, event_scorer=event_scorer)
    limits = []
    if run_info.get("embedder", {}).get("is_standin"):
        limits.append("ATT&CK similarity came from the offline lexical stand-in embedder (word overlap); treat technique leads as development-only.")
    limits.append("ATT&CK similarity scores and thresholds are uncalibrated text-similarity leads, not probabilities or proof of a technique.")
    limits.append("Novelty measures unusualness within this dataset, not maliciousness.")
    unanalysed = (overview.get("noise_units_total") or 0) - (overview.get("noise_units_analysed") or 0)
    if unanalysed > 0:
        limits.append(f"{unanalysed} noise unit(s) outside --noise-review-top-n received no ATT&CK analysis; they were scanned only for observable indicators"
                      + (f" ({len(extra)} surfaced)." if extra else "."))
    if findings and not any(e.get("user") for f in findings for e in f["evidence"]):
        limits.append("No user account is recorded on the key events.")
    if any(any(m["text"].startswith("No process-creation") for m in f["missing_evidence"]) for f in findings):
        limits.append("Process-creation / command-line records were not found for the key events, so parent processes are unknown.")
    run_info = dict(run_info, limitations=limits)
    analyst = {
        "schema_version": ANALYST_SCHEMA_VERSION,
        "run_id": run_info.get("run_id"),
        "analysis_mode": {"mode": "deterministic", "backend": None, "reason": "no LLM analysis applied"},
        "executive_summary": None,
        "findings": findings,
        "overall_assessment": build_overall(findings, meta, limits),
        "investigation_plan": build_plan(findings),
        "prioritization": {"criteria": PRIORITY_CRITERIA, "indicator_classes": {k: {"strength": v["strength"], "meaning": v["text"]} for k, v in MARKERS.items()},
                           "grouping_rule": "units are merged into one finding only when their events share a process id; shared host, time window, process name or ATT&CK candidate never merges",
                           "selection": f"up to {max_findings} findings, ordered by priority, indicator strength, then novelty", **meta},
        "evidence_index": {e["ref"]: {"event_uid": e["event_uid"], "source": e["source"], "timestamp": e["timestamp"], "host": e["host"]}
                           for f in findings for e in f["evidence"]},
        "limitations": limits,
        "disclaimer": ("This report lists leads for analyst review. Priorities are triage rules, novelty is statistical unusualness, and ATT&CK "
                       "candidates are uncalibrated text similarity; none of them confirms an incident, and no actor attribution is made."),
    }
    analyst["executive_summary"] = build_executive(findings, meta, overview, run_info, "deterministic")
    return analyst


def refresh_derived(analyst: Dict[str, Any], overview: Dict[str, Any], run_info: Dict[str, Any], mode_note: str) -> None:
    """Recompute run-level sections after findings changed (e.g. LLM text merged in)."""
    findings = analyst["findings"]
    meta = analyst["prioritization"]
    run_info = dict(run_info, limitations=analyst["limitations"])
    analyst["overall_assessment"] = build_overall(findings, meta, analyst["limitations"])
    analyst["investigation_plan"] = build_plan(findings)
    analyst["executive_summary"] = build_executive(findings, meta, overview, run_info, mode_note)
