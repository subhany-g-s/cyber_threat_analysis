"""
scripts/update_mitre_corpus.py
--------------------------------
Explicitly download / ingest the official MITRE ATT&CK Enterprise STIX data
and write a versioned local corpus cache. This is the ONLY place the
project touches the network for MITRE data; normal analysis runs are offline.

Windows examples:
    python scripts/update_mitre_corpus.py                  # download latest, validate, cache, set as current
    python scripts/update_mitre_corpus.py --list           # show cached versions
    python scripts/update_mitre_corpus.py --from-file C:\\Downloads\\enterprise-attack.json
    python scripts/update_mitre_corpus.py --use enterprise-attack-v19.2-3fa91c2b.json
    python scripts/update_mitre_corpus.py --info           # statistics of the current cache

After updating, rebuild technique embeddings (optional; analysis runs also
build them on demand):
    python scripts/rebuild_technique_embeddings.py
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.mitre.corpus import CorpusError
from src.mitre.updater import (DEFAULT_CACHE_DIR, DEFAULT_STIX_URL, CorpusUpdateError, list_cached,
                               load_current, set_current, update_corpus)


def _print_stats(corpus):
    m, s = corpus.meta, corpus.stats()
    print(f"  Corpus version : {corpus.version_label}")
    print(f"  ATT&CK release : {m.get('attack_version')}  (STIX spec {m.get('attack_spec_version')})")
    print(f"  Source         : {m.get('source_url') or m.get('source_kind')}")
    print(f"  Retrieved at   : {m.get('retrieved_at')}")
    print(f"  Content hash   : {corpus.content_hash}")
    print(f"  Records        : {s['total_records']}  (active {s['active']}, revoked {s['revoked']}, deprecated {s['deprecated']})")
    print(f"  Active         : {s['active_techniques']} techniques + {s['active_subtechniques']} sub-techniques "
          f"({s['matchable_active']} matchable)")
    print(f"  With detection : {s['active_with_detection_info']} active entries carry ATT&CK detection info")
    print(f"  Per tactic     : {s['active_per_tactic']}")
    print(f"  Per platform   : {s['active_per_platform']}")
    probs = m.get("parse_problems") or []
    if probs:
        print(f"  Parse problems : {len(probs)} (first: {probs[0]})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Update the local MITRE ATT&CK Enterprise corpus cache (explicit, versioned).")
    ap.add_argument("--cache-dir", default=os.path.join(PROJECT_ROOT, DEFAULT_CACHE_DIR))
    ap.add_argument("--url", default=DEFAULT_STIX_URL, help="HTTPS URL of an ATT&CK Enterprise STIX 2.1 bundle.")
    ap.add_argument("--from-file", default=None, help="Ingest a STIX bundle you downloaded yourself (no network).")
    ap.add_argument("--force", action="store_true", help="Rewrite the cache file even if identical content is cached.")
    ap.add_argument("--keep-raw", action="store_true", help="Also keep the raw ~50 MB STIX download next to the cache.")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--list", action="store_true", help="List cached corpus versions and exit.")
    ap.add_argument("--info", action="store_true", help="Print statistics for the current cache and exit.")
    ap.add_argument("--use", default=None, metavar="FILE", help="Switch CURRENT to an already-cached corpus file.")
    ap.add_argument("--json", action="store_true", help="Machine-readable output.")
    args = ap.parse_args(argv)

    try:
        if args.list:
            rows = list_cached(args.cache_dir)
            if args.json:
                print(json.dumps(rows, indent=2))
            elif not rows:
                print(f"No cached corpus in {args.cache_dir}. Run: python scripts/update_mitre_corpus.py")
            for r in rows:
                print(f"{'*' if r.get('current') else ' '} {r['file']}  ATT&CK v{r.get('attack_version')}  "
                      f"retrieved {r.get('retrieved_at')}  active={r.get('active')}")
            return 0
        if args.info:
            _print_stats(load_current(args.cache_dir))
            return 0
        if args.use:
            set_current(args.cache_dir, args.use)
            print(f"CURRENT -> {args.use}")
            return 0

        last = {"t": 0}

        def progress(done, total):
            if sys.stderr.isatty() and done - last["t"] >= (4 << 20):
                last["t"] = done
                pct = f" {100 * done // total}%" if total else ""
                sys.stderr.write(f"\r  downloaded {done / 1e6:.1f} MB{pct}")
                sys.stderr.flush()

        src = args.from_file or args.url
        print(f"Updating MITRE ATT&CK corpus from {src} ...")
        out = update_corpus(args.cache_dir, url=args.url, from_file=args.from_file, force=args.force,
                            keep_raw=args.keep_raw, timeout=args.timeout, progress=progress)
        if sys.stderr.isatty():
            sys.stderr.write("\n")
        print(("Wrote new corpus cache: " if out.changed else "Already up to date (identical content cached): ")
              + os.path.relpath(out.path))
        if out.previous_label and out.previous_label != os.path.basename(out.path):
            print(f"  (previous CURRENT was {out.previous_label}; it is kept on disk)")
        _print_stats(out.corpus)
        for w in out.warnings[:10]:
            print(f"  WARNING: {w}")
        print("Next: python scripts/rebuild_technique_embeddings.py   (optional - runs build them on demand)")
        return 0
    except (CorpusUpdateError, CorpusError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        print("The existing corpus cache (if any) was left untouched.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
