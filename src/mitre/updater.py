"""
src/mitre/updater.py
---------------------
Explicit, versioned update of the local MITRE ATT&CK corpus cache.

    download (only when asked)  ->  parse STIX  ->  validate  ->  save
    cache/mitre/<domain>-v<ATT&CK version>-<hash8>.json  +  cache/mitre/CURRENT

Design rules:
  * Nothing in a normal analysis run calls this module's network code.
  * A failed or invalid update NEVER touches the existing cache and NEVER
    "falls back" to a smaller corpus -- it raises CorpusUpdateError.
  * Older cached versions are kept; CURRENT points at the active one.
"""

import hashlib
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from src.mitre.corpus import (CorpusError, MitreCorpus, compute_content_hash, compute_stats,
                              validate_techniques)
from src.mitre.stix_parser import parse_bundle

DEFAULT_STIX_URL = ("https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
                    "master/enterprise-attack/enterprise-attack.json")
DEFAULT_CACHE_DIR = os.path.join("cache", "mitre")
CURRENT_POINTER = "CURRENT"
MAX_DOWNLOAD_BYTES = 400 * 1024 * 1024
USER_AGENT = "subhany-cag-system/1.0 (mitre corpus updater)"


class CorpusUpdateError(Exception):
    """A requested corpus update failed. The existing cache is unchanged."""


@dataclass
class UpdateOutcome:
    corpus: MitreCorpus
    path: str
    changed: bool                       # False => identical content was already cached
    previous_label: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------- download
def download_bundle(url: str, dest_path: str, timeout: float = 60.0, max_bytes: int = MAX_DOWNLOAD_BYTES,
                    progress: Optional[Callable[[int, Optional[int]], None]] = None) -> int:
    """Stream `url` to `dest_path`. Returns the number of bytes written."""
    if not url.lower().startswith("https://"):
        raise CorpusUpdateError(f"Refusing non-HTTPS source URL: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            status = getattr(resp, "status", 200)
            if status != 200:
                raise CorpusUpdateError(f"Download failed: HTTP {status} from {url}")
            total = resp.headers.get("Content-Length")
            total = int(total) if total and total.isdigit() else None
            if total is not None and total > max_bytes:
                raise CorpusUpdateError(f"Download refused: {total} bytes exceeds the {max_bytes}-byte limit.")
            written = 0
            with open(dest_path, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        raise CorpusUpdateError(f"Download aborted: exceeded the {max_bytes}-byte limit.")
                    out.write(chunk)
                    if progress:
                        progress(written, total)
            if total is not None and written != total:
                raise CorpusUpdateError(f"Download truncated: received {written} of {total} bytes.")
            return written
    except CorpusUpdateError:
        raise
    except urllib.error.HTTPError as e:
        raise CorpusUpdateError(f"Download failed: HTTP {e.code} {e.reason} from {url}")
    except urllib.error.URLError as e:
        raise CorpusUpdateError(
            f"Could not reach {url}: {e.reason}. Check your internet/proxy settings, or download the STIX file "
            f"manually and run: python scripts/update_mitre_corpus.py --from-file <path>")
    except (TimeoutError, OSError) as e:
        raise CorpusUpdateError(f"Download failed ({type(e).__name__}): {e}")


# --------------------------------------------------------------------------- build
def build_corpus(bundle: object, source_url: Optional[str], source_kind: str,
                 retrieved_at: Optional[str] = None, source_sha256: Optional[str] = None) -> MitreCorpus:
    try:
        parsed = parse_bundle(bundle)
    except ValueError as e:
        raise CorpusUpdateError(str(e))

    rep = validate_techniques(parsed.techniques)
    if not rep.ok:
        raise CorpusUpdateError("Parsed corpus failed validation, cache NOT updated: " + "; ".join(rep.errors))

    stats = compute_stats(parsed.techniques)
    col = parsed.collection
    meta = {
        "domain": "enterprise-attack",
        "source_kind": source_kind,                       # "download" | "local-file"
        "source_url": source_url,
        "source_sha256": source_sha256,
        "collection_name": col.get("name"),
        "collection_stix_id": col.get("stix_id"),
        "attack_version": col.get("attack_version"),
        "attack_spec_version": col.get("attack_spec_version"),
        "collection_modified": col.get("modified"),
        "retrieved_at": retrieved_at or _now(),
        "content_hash": compute_content_hash(parsed.techniques),
        "stats": stats,
        "parse_problems": parsed.problems,
        "validation_warnings": rep.warnings,
    }
    return MitreCorpus(techniques=parsed.techniques, meta=meta)


# --------------------------------------------------------------------------- cache management
def corpus_filename(corpus: MitreCorpus) -> str:
    return corpus.version_label + ".json"


def list_cached(cache_dir: str = DEFAULT_CACHE_DIR) -> List[Dict[str, object]]:
    if not os.path.isdir(cache_dir):
        return []
    current = read_current_pointer(cache_dir)
    rows = []
    for name in sorted(os.listdir(cache_dir)):
        if not name.endswith(".json"):
            continue
        p = os.path.join(cache_dir, name)
        try:
            with open(p, "r", encoding="utf-8") as f:
                meta = (json.load(f).get("meta") or {})
        except (OSError, json.JSONDecodeError):
            rows.append({"file": name, "current": name == current, "error": "unreadable"})
            continue
        rows.append({"file": name, "current": name == current, "attack_version": meta.get("attack_version"),
                     "retrieved_at": meta.get("retrieved_at"),
                     "active": (meta.get("stats") or {}).get("active")})
    return rows


def read_current_pointer(cache_dir: str) -> Optional[str]:
    p = os.path.join(cache_dir, CURRENT_POINTER)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return f.read().strip() or None


def set_current(cache_dir: str, filename: str) -> None:
    if not os.path.exists(os.path.join(cache_dir, filename)):
        raise CorpusUpdateError(f"No cached corpus named {filename} in {cache_dir}")
    tmp = os.path.join(cache_dir, CURRENT_POINTER + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(filename + "\n")
    os.replace(tmp, os.path.join(cache_dir, CURRENT_POINTER))


def load_current(cache_dir: str = DEFAULT_CACHE_DIR) -> MitreCorpus:
    """Used by analysis runs. Raises CorpusError with an actionable message
    when no full corpus has been cached yet -- there is NO silent fallback."""
    name = read_current_pointer(cache_dir)
    if not name:
        raise CorpusError(
            f"No MITRE ATT&CK corpus is cached in '{cache_dir}'. Run once (needs internet):\n"
            f"    python scripts/update_mitre_corpus.py\n"
            f"Analysis runs are offline afterwards. (Not falling back to any smaller corpus.)")
    return MitreCorpus.load(os.path.join(cache_dir, name))


def update_corpus(cache_dir: str = DEFAULT_CACHE_DIR, url: str = DEFAULT_STIX_URL,
                  from_file: Optional[str] = None, force: bool = False, keep_raw: bool = False,
                  timeout: float = 60.0, progress: Optional[Callable[[int, Optional[int]], None]] = None
                  ) -> UpdateOutcome:
    os.makedirs(cache_dir, exist_ok=True)
    previous = read_current_pointer(cache_dir)

    raw_tmp = None
    try:
        if from_file:
            if not os.path.exists(from_file):
                raise CorpusUpdateError(f"--from-file not found: {from_file}")
            src_path, source_url, kind = from_file, None, "local-file"
        else:
            raw_tmp = os.path.join(cache_dir, f"download-{int(time.time())}.tmp")
            download_bundle(url, raw_tmp, timeout=timeout, progress=progress)
            src_path, source_url, kind = raw_tmp, url, "download"

        h = hashlib.sha256()
        with open(src_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        try:
            with open(src_path, "r", encoding="utf-8") as f:
                bundle = json.load(f)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            raise CorpusUpdateError(f"Source is not valid JSON ({type(e).__name__}): {e}")

        corpus = build_corpus(bundle, source_url, kind, source_sha256=h.hexdigest())
        del bundle

        fname = corpus_filename(corpus)
        dest = os.path.join(cache_dir, fname)
        changed = not os.path.exists(dest) or force
        if changed:
            corpus.save(dest)
        else:
            corpus = MitreCorpus.load(dest)       # identical content already cached; keep its original metadata
        set_current(cache_dir, fname)
        if keep_raw and raw_tmp:
            os.replace(raw_tmp, os.path.join(cache_dir, fname.replace(".json", ".stix.json")))
            raw_tmp = None
        warnings = list(corpus.meta.get("validation_warnings") or [])
        return UpdateOutcome(corpus=corpus, path=dest, changed=changed,
                             previous_label=previous, warnings=warnings)
    finally:
        if raw_tmp and os.path.exists(raw_tmp):
            try:
                os.remove(raw_tmp)
            except OSError:
                pass


def load_corpus_for_run(cache_dir: str = DEFAULT_CACHE_DIR, corpus_file: Optional[str] = None) -> MitreCorpus:
    """Resolve the corpus an analysis run should use.
    corpus_file: explicit path. A versioned cache file is loaded as-is; the
    old hand-written 15-entry JSON shape is accepted ONLY here (explicit),
    and is labelled legacy_curated."""
    if corpus_file:
        if not os.path.exists(corpus_file):
            raise CorpusError(f"Corpus file not found: {corpus_file}")
        try:
            with open(corpus_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise CorpusError(f"Corpus file {corpus_file} is unreadable or corrupt: {e}")
        if isinstance(payload, dict) and "meta" in payload:
            return MitreCorpus.from_payload(payload, corpus_file)       # versioned cache file; errors propagate
        return MitreCorpus.from_legacy_curated_file(corpus_file)         # old hand-written shape, explicit only
    return load_current(cache_dir)
