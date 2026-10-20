"""
src/service/pipeline.py
-------------------------
PipelineService: the single, UI-independent implementation of the whole run.
The CLI (scripts/run_pipeline.py) and a future Streamlit dashboard both call

    result = PipelineService(PipelineConfig(input_path=...), on_stage=callback).run()

and read everything from the returned RunResult. Intermediate data live in
memory (RunResult.artifacts); nothing is written to disk unless
`config.export_dir` is set.

Ten stages, each timed and reported with metrics / warnings / an explanation:
  1 parsing   2 normalization   3 graph   4 GAE   5 clustering   6 novelty
  7 profiling (+ noise review)  8 MITRE mapping   9 GraphRAG   10 report

Failure policy: a stage that cannot run reports FAILED with a clear message,
never a traceback. Stages 1-6 are hard dependencies (later stages are SKIPPED).
Stage 8 (mapping) and 9 (retrieval) degrade: the run continues and the report
states what was unavailable.
"""

import os
import platform
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.clustering import cluster_embeddings as cluster_mod
from src.graph.graph_builder import build_event_graph, compute_graph_stats
from src.graph.normalizer import normalize_records
from src.mapping.embedders import EmbedderUnavailable, LexicalCorpusEmbedder, SBertEmbedder
from src.mapping.semantic_mapper import SemanticMitreMapper, TechniqueCorpus
from src.mapping.text_prep import PREP_VERSION, technique_embedding_text
from src.mitre.corpus import CorpusError
from src.mitre.updater import load_corpus_for_run
from src.noise.review import review_noise
from src.novelty import novelty_detection as novelty_mod
from src.profiling.cluster_profiler import build_cluster_profiles, build_unit_profile
from src.service.adapters import InputError, ingest
from src.service.config import PipelineConfig
from src.service.gae_stage import GaeError, load_precomputed, run_gae
from src.service.result import (FAILED, OK, SKIPPED, STAGE_NAMES, WARN, RunArtifacts, RunResult, StageResult,
                                UnitResult)

KNOWN_ERRORS = (InputError, GaeError, CorpusError, EmbedderUnavailable)


class _StageAbort(Exception):
    """Raised inside a stage body to fail it with a message (no traceback shown)."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class PipelineService:
    def __init__(self, config: PipelineConfig, on_stage: Optional[Callable[[StageResult], None]] = None,
                 embedder=None, llm_client=None, neo4j_store=None):
        self.cfg = config
        self.on_stage = on_stage
        self._embedder = embedder              # injected (tests / UI); None => built from config
        self._llm = llm_client
        self._neo4j = neo4j_store              # injected store (tests); None => from env when enabled
        self.res = RunResult(config=config.to_dict())
        self.art = RunArtifacts()
        self.res.artifacts = self.art
        self.res.stages = [StageResult(i + 1, n) for i, n in enumerate(STAGE_NAMES)]
        self._t0 = 0.0
        self._ctx: Dict[str, Any] = {}

    # ------------------------------------------------------------------ paths
    def _resolve(self, path: Optional[str]) -> Optional[str]:
        if not path or os.path.isabs(path):
            return path
        if os.path.exists(path):
            return os.path.abspath(path)
        cand = os.path.join(self.cfg.project_root, path)
        return cand if os.path.exists(cand) else path

    # ------------------------------------------------------------------ driver
    def run(self) -> RunResult:
        res = self.res
        self._t0 = time.time()
        res.metadata = {"started_at": _now(), "python": sys.version.split()[0], "platform": platform.platform(),
                        "input_path": self.cfg.input_path}
        stage_fns = [self._s1_parse, self._s2_normalize, self._s3_graph, self._s4_gae, self._s5_cluster,
                     self._s6_novelty, self._s7_profile, self._s8_map, self._s9_graphrag, self._s10_report]
        hard_failed = None
        for n, fn in enumerate(stage_fns, start=1):
            st = res.stages[n - 1]
            if hard_failed is not None:
                st.status = SKIPPED
                st.explanation = f"Skipped because stage {hard_failed} failed."
                self._emit(st)
                continue
            t = time.time()
            try:
                fn(st)
                if st.status == SKIPPED:
                    st.status = WARN if st.warnings else OK
            except _StageAbort as e:
                st.status, st.error = FAILED, str(e)
            except KNOWN_ERRORS as e:
                st.status, st.error = FAILED, str(e)
            except Exception as e:                       # unexpected: keep the run alive, keep the evidence
                st.status = FAILED
                st.error = f"Unexpected {type(e).__name__}: {e}"
                res.metadata.setdefault("tracebacks", []).append({"stage": n, "traceback": traceback.format_exc()[-1500:]})
            st.elapsed_s = round(time.time() - t, 3)
            if st.status == FAILED:
                res.errors.append(f"Stage {n} ({st.name}): {st.error}")
                if n <= 7:
                    hard_failed = n
            for w in st.warnings:
                res.warnings.append(f"Stage {n}: {w}")
            self._emit(st)
        res.metadata["finished_at"] = _now()
        res.metadata["total_seconds"] = round(time.time() - self._t0, 3)
        if self.cfg.export_dir and res.report:
            from src.service.export import export_run
            try:
                res.metadata["exported"] = export_run(res, self.cfg.export_dir)
            except Exception as e:
                res.warnings.append(f"Export failed: {type(e).__name__}: {e}")
        return res

    def _emit(self, st: StageResult) -> None:
        if self.on_stage:
            try:
                self.on_stage(st)
            except Exception:                            # a UI callback must never break the run
                pass

    # ================================================================ stage 1
    def _s1_parse(self, st: StageResult) -> None:
        cfg = self.cfg
        ing = ingest(self._resolve(cfg.input_path), cfg.input_format)
        self._ctx["ingest"] = ing
        s = ing.stats
        self.res.parsing = {
            "path": ing.path, "format": ing.format, "source_id": ing.source_id, "sha256": ing.sha256,
            "file_size_bytes": ing.file_size, "total_lines": s["total_lines"], "blank_lines": s["blank_lines"],
            "valid_records": s["valid_records"], "invalid_records": s["invalid_records"],
            "invalid_samples": ing.invalid_samples,
            "valid_share_pct": round(100.0 * s["valid_records"] / max(s["total_lines"] - s["blank_lines"], 1), 2),
        }
        self.res.metadata["source_id"] = ing.source_id
        st.metrics = dict(self.res.parsing)
        st.warnings += ing.warnings
        st.explanation = ("The input file was read with the explicit adapter for its format; malformed records are counted "
                          "and skipped (never repaired) and every valid record keeps its source line for provenance.")

    # ================================================================ stage 2
    def _s2_normalize(self, st: StageResult) -> None:
        ing = self._ctx["ingest"]
        events, nstats = normalize_records(ing.records)
        if not events:
            raise _StageAbort("Normalization produced no events.")
        seen, unique, dups = set(), [], 0
        # normalize_records preserves order 1:1 except skipped records; align provenance by re-normalising lazily
        for i, ev in enumerate(events):
            if ev.event_uid in seen:
                dups += 1
                continue
            seen.add(ev.event_uid)
            unique.append(ev)
            line = ing.record_lines[i] if i < len(ing.record_lines) else None
            self.art.provenance_by_uid[ev.event_uid] = {
                "source_id": ing.source_id, "line": line,
                "source_ref": f"{os.path.basename(ing.path)}:{'row' if ing.format == 'csv' else 'line'} {line}"}
        self.art.events = unique
        self.art.events_by_uid = {e.event_uid: e.as_dict() for e in unique}
        n = len(unique)
        no_ts = sum(1 for e in unique if e.timestamp_epoch is None)
        no_host = sum(1 for e in unique if not e.hostname)
        eid = Counter(str(e.event_id) for e in unique)
        self.res.normalization = {
            **nstats, "unique_events": n, "duplicate_records_collapsed": dups,
            "missing_timestamp": no_ts, "missing_hostname": no_host,
            "event_id_distribution": dict(eid.most_common(12)), "distinct_event_ids": len(eid),
            "event_uid_scheme": "uuid5 over the canonical raw record (deterministic, stable across runs/machines)",
            "provenance_recorded": len(self.art.provenance_by_uid)}
        st.metrics = dict(self.res.normalization)
        if dups:
            st.warnings.append(f"{dups} byte-identical record(s) collapsed into one event (same deterministic event_uid).")
        if no_ts:
            st.warnings.append(f"{no_ts} event(s) have no parseable timestamp (excluded from temporal edges).")
        if no_host:
            st.warnings.append(f"{no_host} event(s) have no hostname (excluded from host/time edges).")
        st.explanation = ("Each raw record becomes one common-schema event with a deterministic event_uid; missing fields stay "
                          "null (nothing is invented) and the original record is preserved on the event.")

    # ================================================================ stage 3
    def _s3_graph(self, st: StageResult) -> None:
        G = build_event_graph(self.art.events)
        gs = compute_graph_stats(G)
        self.art.graph = G
        import networkx as nx
        comps = sorted((len(c) for c in nx.connected_components(G.to_undirected())), reverse=True)
        self.res.graph = {**gs.as_dict(), "largest_component": comps[0] if comps else 0,
                          "component_sizes_top5": comps[:5], "temporal_window_s": 300,
                          "edge_scheme": "chronological chain per process id / per host window (O(n) edges, not cliques)"}
        # events_per_host can be long; keep top 6 in the metrics
        self.res.graph["events_per_host"] = dict(sorted(gs.events_per_host.items(), key=lambda kv: -kv[1])[:6])
        self.res.graph["events_per_eventid"] = dict(sorted(gs.events_per_eventid.items(), key=lambda kv: -kv[1])[:8])
        st.metrics = dict(self.res.graph)
        if gs.num_edges == 0:
            st.warnings.append("The event graph has no edges: embeddings will reflect event attributes only.")
        st.explanation = ("Events are nodes; SAME_PID links events sharing a process id and SAME_HOST_TEMPORAL links events on the "
                          "same host within 300 s, each as a time-ordered chain so the graph stays sparse and auditable.")

    # ================================================================ stage 4
    def _s4_gae(self, st: StageResult) -> None:
        cfg = self.cfg
        G = self.art.graph
        if cfg.precomputed_embeddings:
            out = load_precomputed(G, self._resolve(cfg.precomputed_embeddings), self._resolve(cfg.precomputed_mapping))
        else:
            out = run_gae(G, self._resolve(cfg.checkpoint_path), self._resolve(cfg.feature_schema_path))
        self.art.embeddings, self.art.row_to_uid = out.Z, out.row_to_uid
        self.res.gae = dict(out.info)
        st.metrics = dict(out.info)
        st.warnings += out.warnings
        st.explanation = ("The pretrained graph autoencoder is validated against its feature schema and run in inference mode only "
                          "(no retraining) to give every event a latent embedding that mixes its attributes with its graph context.")

    # ================================================================ stage 5
    def _s5_cluster(self, st: StageResult) -> None:
        cfg, Z = self.cfg, self.art.embeddings
        n = Z.shape[0]
        mcs, ms = cfg.min_cluster_size, cfg.min_samples
        if n < max(mcs, 2):
            labels = np.full(n, -1, dtype=int)
            probs = np.zeros(n)
            medoids, algo = {}, "none (dataset smaller than min_cluster_size)"
            st.warnings.append(f"Only {n} events (< min_cluster_size={mcs}): no clustering was possible; every event goes "
                               f"through the noise-review path.")
        else:
            labels, probs, medoids, algo = cluster_mod.cluster_embeddings(
                Z, min_cluster_size=mcs, min_samples=min(ms, n), algorithm=cfg.cluster_algorithm)
        summary = cluster_mod.summarize_clusters(labels, probs)
        quality = cluster_mod.compute_cluster_quality_metrics(Z, labels) if summary["number_of_clusters"] >= 2 else \
            {"status": "NOT_COMPUTED", "reason": "fewer than 2 clusters"}
        self.art.labels, self.art.probabilities = labels, probs
        self._ctx["medoids"] = medoids
        sizes = sorted(summary["cluster_sizes"].items(), key=lambda kv: (-kv[1], kv[0]))
        self.res.clustering = {
            "algorithm": algo, "min_cluster_size": mcs, "min_samples": ms, "total_events": summary["total_events"],
            "clusters": summary["number_of_clusters"], "noise_events": summary["number_of_noise_points"],
            "noise_share_pct": summary["percentage_noise"], "clustered_share_pct": summary["percentage_clustered"],
            "cluster_sizes": {str(k): v for k, v in sorted(summary["cluster_sizes"].items())},
            "largest_clusters": sizes[:5], "smallest_clusters": sizes[-5:] if sizes else [],
            "mean_membership_probability": summary["mean_membership_probability_clustered"],
            "quality": quality,
            "noise_representative_example": None}
        st.metrics = {k: v for k, v in self.res.clustering.items() if k != "cluster_sizes"}
        if summary["number_of_clusters"] == 0:
            st.warnings.append("No clusters formed; all events are noise and will be reviewed individually.")
        elif summary["percentage_noise"] >= 40:
            st.warnings.append(f"{summary['percentage_noise']}% of events are noise - they are preserved and reviewed separately.")
        st.explanation = ("HDBSCAN groups events whose embeddings sit in the same dense region; points in no dense region are labelled "
                          "noise (cluster -1). Noise is not discarded: it gets its own review path in stage 7.")

    # ================================================================ stage 6
    def _s6_novelty(self, st: StageResult) -> None:
        cfg, Z, G = self.cfg, self.art.embeddings, self.art.graph
        r2u, labels, probs = self.art.row_to_uid, self.art.labels, self.art.probabilities
        rows_in = [{"event_uid": r2u[i], "membership_probability": float(probs[i])} for i in range(len(r2u))]
        density = novelty_mod.compute_density_score(rows_in)
        emb = novelty_mod.compute_embedding_distance_score(Z, r2u, rows_in, self._ctx["medoids"])
        recon = novelty_mod.compute_reconstruction_score(Z, r2u, G)
        rarity = novelty_mod.compute_graph_rarity_score(G)
        rows = novelty_mod.combine_novelty_scores(density, emb, recon, rarity, cfg.novelty_weights)
        rows.sort(key=lambda r: r["event_uid"])                      # deterministic order (combine uses a set)
        cl = {r2u[i]: int(labels[i]) for i in range(len(r2u))}
        for r in rows:
            r["cluster_id"] = cl.get(r["event_uid"])
            r["is_noise"] = r["cluster_id"] == -1
            r["novelty_label"] = novelty_mod.label_novelty(r["novelty_score"], tuple(cfg.novelty_thresholds))
        rows.sort(key=lambda r: (-r["novelty_score"], r["event_uid"]))
        self.art.novelty_rows = rows
        counts = Counter(r["novelty_label"] for r in rows)
        scores = np.array([r["novelty_score"] for r in rows]) if rows else np.zeros(1)
        noise_scores = [r["novelty_score"] for r in rows if r["is_noise"]]
        clus_scores = [r["novelty_score"] for r in rows if not r["is_noise"]]
        self.res.novelty = {
            "weights": dict(cfg.novelty_weights), "thresholds": list(cfg.novelty_thresholds),
            "label_counts": {k: counts.get(k, 0) for k in ("NORMAL", "LOW_NOVELTY", "SUSPICIOUS", "HIGH_NOVELTY")},
            "score_mean": round(float(scores.mean()), 4), "score_max": round(float(scores.max()), 4),
            "mean_score_noise_events": round(float(np.mean(noise_scores)), 4) if noise_scores else None,
            "mean_score_clustered_events": round(float(np.mean(clus_scores)), 4) if clus_scores else None,
            "top_events": [{"event_uid": r["event_uid"], "novelty_score": r["novelty_score"], "novelty_label": r["novelty_label"],
                            "cluster_id": r["cluster_id"], "is_noise": r["is_noise"]} for r in rows[:10]],
            "signals": ["density (1 - HDBSCAN membership)", "distance to nearest cluster medoid",
                        "GAE decoder reconstruction error", "graph structural rarity"],
            "meaning": "statistical/structural unusualness within this dataset - NOT a maliciousness verdict"}
        st.metrics = {k: v for k, v in self.res.novelty.items() if k != "top_events"}
        st.explanation = ("Four independent signals (density, distance to the nearest cluster medoid, GAE reconstruction error, graph "
                          "rarity) are averaged into a 0-1 score. High novelty means unusual for this dataset, not malicious.")

    # ================================================================ stage 7
    def _s7_profile(self, st: StageResult) -> None:
        cfg, art = self.cfg, self.art
        r2u, labels, probs = art.row_to_uid, art.labels, art.probabilities
        nov = {r["event_uid"]: r for r in art.novelty_rows}
        assignments = [{"event_uid": r2u[i], "cluster_id": int(labels[i]), "membership_probability": float(probs[i])}
                       for i in range(len(r2u))]
        medoid_uid = {cid: r2u[row] for cid, row in self._ctx["medoids"].items()}
        profiles = build_cluster_profiles(assignments, art.events_by_uid, medoid_uid)

        def nsum(uids: List[str]) -> Dict[str, Any]:
            rs = [nov[u] for u in uids if u in nov]
            if not rs:
                return {}
            sc = [r["novelty_score"] for r in rs]
            top = sorted(rs, key=lambda r: (-r["novelty_score"], r["event_uid"]))[:5]
            return {"mean": round(float(np.mean(sc)), 4), "max": round(float(np.max(sc)), 4),
                    "label_counts": dict(Counter(r["novelty_label"] for r in rs)),
                    "top_events": [{"event_uid": r["event_uid"], "novelty_score": r["novelty_score"],
                                    "novelty_label": r["novelty_label"]} for r in top]}

        def seeds(prof, nsm) -> List[str]:
            out = list(prof.representative_event_uids)
            for e in nsm.get("top_events", [])[:3]:
                if e["event_uid"] not in out:
                    out.append(e["event_uid"])
            return out

        units: List[UnitResult] = []
        for p in profiles:
            nsm = nsum(p.member_event_uids)
            u = UnitResult(unit_id=p.unit_id, scope="cluster", cluster_id=p.cluster_id, size=p.cluster_size,
                           narrative=p.text, mapping_text=p.mapping_text, profile=p.as_dict(), novelty_summary=nsm,
                           priority=nsm.get("max", 0.0))
            u.profile["seed_event_uids"] = seeds(p, nsm)
            units.append(u)
        self.art.profiles = list(profiles)

        # ---- noise review: all noise events kept; top-N review units analysed semantically ----
        builder = None
        try:
            from src.gae.gae_features import NodeFeatureBuilder
            sp = self._resolve(cfg.feature_schema_path)
            if sp and os.path.exists(sp) and not cfg.precomputed_embeddings:
                builder = NodeFeatureBuilder.load(sp)
        except Exception:
            builder = None
        review = review_noise(r2u, labels, art.graph, art.events_by_uid, art.novelty_rows, cfg.min_cluster_size,
                              cfg.min_samples, schema_builder=builder, group_min_size=cfg.noise_group_min_size)
        sel = review["units"][: cfg.noise_review_top_n]
        ev_by_uid_noise = {e["event_uid"]: e for e in review["events"]}
        for ru in sel:
            members = [art.events_by_uid[x] for x in ru["event_uids"]]
            reps = ru["event_uids"][:3]
            p = build_unit_profile(ru["unit_id"], ru["scope"], members, reps)
            nsm = nsum(ru["event_uids"])
            sig = Counter(s["signal"] for x in ru["event_uids"] for s in ev_by_uid_noise[x]["signals"]
                          if s["signal"] != "no_dense_region")
            u = UnitResult(unit_id=ru["unit_id"], scope=ru["scope"], cluster_id=-1, size=ru["size"], narrative=p.text,
                           mapping_text=p.mapping_text, profile=p.as_dict(), novelty_summary=nsm,
                           priority=nsm.get("max", 0.0))
            u.profile["seed_event_uids"] = seeds(p, nsm)
            u.profile["noise_signals"] = dict(sig)
            u.profile["adjacent_clusters"] = ru["adjacent_clusters"]
            units.append(u)
            self.art.profiles.append(p)
        self.res.units = units
        self.res.noise = {
            "stats": review["stats"], "explanation": review["explanation"], "events": review["events"],
            "units": [{k: v for k, v in ru.items()} for ru in review["units"]],
            "units_analysed": [u.unit_id for u in units if u.scope != "cluster"],
            "review_top_n": cfg.noise_review_top_n}
        n_clusters = sum(1 for u in units if u.scope == "cluster")
        st.metrics = {
            "cluster_profiles": n_clusters, "noise_events_preserved": review["stats"]["noise_events"],
            "noise_review_units_total": len(review["units"]), "noise_review_units_analysed": len(sel),
            "noise_groups": review["stats"]["groups"], "noise_single_events": review["stats"]["single_event_units"],
            "noise_events_with_clustered_neighbors": review["stats"]["with_clustered_neighbors"],
            "noise_signal_counts": review["stats"]["signal_counts"],
            "representative_events": {u.unit_id: u.profile["representative_event_uids"][:3] for u in units[:8]}}
        if review["stats"]["noise_events"] and not sel:
            st.warnings.append("Noise events exist but --noise-review-top-n is 0: none received semantic analysis.")
        st.explanation = ("Each cluster is summarised from its member events into a fixed-template narrative. Noise is NOT clustered "
                          "or hidden: it is ranked by novelty, grouped only where events share a process id, and the top units are "
                          "profiled as event-level evidence labelled 'noise'.")

    # ================================================================ stage 8
    def _build_embedder(self, mitre) -> Any:
        if self._embedder is not None:
            return self._embedder
        cfg = self.cfg
        if cfg.embedder == "lexical":
            return LexicalCorpusEmbedder(technique_embedding_text(t) for t in mitre.active())
        return SBertEmbedder(cfg.embedding_model)

    def _s8_map(self, st: StageResult) -> None:
        cfg = self.cfg
        mitre = load_corpus_for_run(self._resolve(cfg.mitre_cache_dir) or cfg.mitre_cache_dir,
                                    self._resolve(cfg.corpus_file))
        self.art.mitre_corpus = mitre
        embedder = self._build_embedder(mitre)
        cache_dir = self._resolve(cfg.mitre_cache_dir) or cfg.mitre_cache_dir
        tcorpus = TechniqueCorpus.from_mitre(mitre, embedder, cache_dir, platforms=list(cfg.platforms) or None,
                                             rebuild=cfg.rebuild_embeddings)
        mapper = SemanticMitreMapper(tcorpus, embedder, top_k=cfg.top_k, min_similarity=cfg.candidate_threshold,
                                     weak_similarity=cfg.weak_threshold)
        self._ctx.update(mapper=mapper, tcorpus=tcorpus, embedder=embedder)
        units = self.res.units
        vecs = mapper.embed_texts([u.mapping_text or u.narrative for u in units]) if units else np.zeros((0, 1))
        self._ctx["unit_vectors"] = {u.unit_id: vecs[i] for i, u in enumerate(units)}
        outcomes = Counter()
        by_scope = {"cluster": Counter(), "noise_group": Counter(), "noise_event": Counter()}
        for i, u in enumerate(units):
            reps = u.profile["representative_event_uids"]
            m = mapper.map_vector(u.unit_id, u.scope, u.cluster_id, u.narrative, u.mapping_text or u.narrative, vecs[i], reps)
            u.mapping = m.as_dict()
            outcomes[m.overall_outcome] += 1
            by_scope[u.scope][m.overall_outcome] += 1
        meta = tcorpus.meta
        is_standin = bool(getattr(embedder, "is_standin", False))
        self.res.mapping = {
            "corpus_version": meta["corpus_version"], "corpus_hash": meta["content_hash"],
            "attack_version": meta["attack_version"], "legacy_curated_corpus": meta.get("legacy_curated", False),
            "corpus_source": mitre.meta.get("source_url") or mitre.meta.get("source_kind"),
            "corpus_retrieved_at": mitre.meta.get("retrieved_at"),
            "corpus_stats": mitre.stats(), "platform_filter": meta["platform_filter"],
            "techniques_active_total": meta["active_total"], "techniques_compared": meta["applicable"],
            "model_id": meta["model_id"], "is_standin_embedder": is_standin, "text_preprocessing": PREP_VERSION,
            "embedding_cache": meta["cache"], "top_k": cfg.top_k, "candidate_threshold": cfg.candidate_threshold,
            "weak_threshold": cfg.weak_threshold, "outcomes": dict(outcomes),
            "outcomes_by_scope": {k: dict(v) for k, v in by_scope.items()},
            "units_mapped": len(units),
            "similarity_semantics": "cosine similarity between text embeddings - not a probability, not calibrated confidence",
            "thresholds_calibrated_against_labels": False}
        st.metrics = {k: v for k, v in self.res.mapping.items() if k != "corpus_stats"}
        if mitre.meta.get("legacy_curated"):
            st.warnings.append("Using the LEGACY hand-written corpus passed via --corpus-file: it is NOT the full ATT&CK knowledge base.")
        if is_standin:
            st.warnings.append("Offline LEXICAL stand-in embedder in use: scores measure word overlap, not meaning, and are not "
                               "comparable to SBERT. Development/testing only.")
        st.warnings.append("Similarity thresholds are uncalibrated (no labelled ground truth available); treat candidates as leads.")
        st.explanation = ("Each cluster / noise-unit narrative and every active ATT&CK technique description are embedded with the same "
                          "model and preprocessing; cosine similarity ranks Top-K candidates. Weak or absent similarity is reported as such - "
                          "no technique is forced onto a unit.")

    # ================================================================ stage 9
    def _s9_graphrag(self, st: StageResult) -> None:
        from src.graphrag import schema as GS
        from src.graphrag.builder import build_knowledge_graph, make_run_id
        from src.graphrag.neo4j_store import Neo4jGraphStore, Neo4jUnavailable
        from src.graphrag.retriever import GraphRetriever, RetrievalSettings
        from src.graphrag.store import LocalGraphStore
        from src.ontology.contract import ContractError, load_ontology_document
        from src.ontology.resolution import ingest_ontology, resolve_ontology

        cfg, art, res = self.cfg, self.art, self.res
        ing = self._ctx["ingest"]
        ck_fp = ""
        try:
            ck = self._resolve(cfg.checkpoint_path)
            if ck and os.path.exists(ck):
                import hashlib
                ck_fp = hashlib.sha256(open(ck, "rb").read()).hexdigest()[:12]
        except OSError:
            pass
        mp = res.mapping or {}
        res.run_id = make_run_id(ing.sha256, {"mcs": cfg.min_cluster_size, "ms": cfg.min_samples, "algo": cfg.cluster_algorithm,
                                             "corpus": mp.get("corpus_hash", ""), "model": mp.get("model_id", ""),
                                             "top_k": cfg.top_k, "thr": (cfg.candidate_threshold, cfg.weak_threshold),
                                             "plat": tuple(cfg.platforms), "noise_n": cfg.noise_review_top_n}, ck_fp)

        local = LocalGraphStore()
        unit_dicts = [{"unit_id": u.unit_id, "scope": u.scope, "cluster_id": u.cluster_id, "size": u.size,
                       "narrative": u.narrative, "member_event_uids": u.profile["member_event_uids"],
                       "representative_event_uids": u.profile["representative_event_uids"], "mapping": u.mapping or None}
                      for u in res.units]
        labels_by_uid = {art.row_to_uid[i]: int(art.labels[i]) for i in range(len(art.row_to_uid))}
        novelty_by_uid = {r["event_uid"]: r for r in art.novelty_rows}
        local.set_run(res.run_id)
        stats = build_knowledge_graph(local, res.run_id, art.events_by_uid, art.provenance_by_uid, art.graph, labels_by_uid,
                                      novelty_by_uid, unit_dicts, art.mitre_corpus)

        # ---- optional ontology-module data ----
        onto = {"supplied": False}
        if cfg.ontology_path:
            onto = {"supplied": True, "path": cfg.ontology_path}
            try:
                doc, v = load_ontology_document(self._resolve(cfg.ontology_path))
                onto.update(valid=v.ok, errors=v.errors[:10], warnings=v.warnings[:10], entities=v.entities,
                            relationships=v.relationships)
                if v.ok:
                    onto["ingested"] = ingest_ontology(local, doc)
                    onto["resolution"] = resolve_ontology(doc, art.events_by_uid, local)
                    onto["contract_version"] = doc.get("contract_version")
                    onto["producer"] = (doc.get("producer") or {}).get("name")
                else:
                    st.warnings.append(f"Ontology file rejected by contract validation ({len(v.errors)} error(s)); "
                                       f"first: {v.errors[0]}")
            except ContractError as e:
                onto.update(valid=False, errors=[str(e)])
                st.warnings.append(str(e))
        res.ontology = onto

        # ---- choose retrieval backend: Neo4j (optional) or the local graph ----
        store, mode_note = local, "local in-memory graph (NetworkX)"
        neo = None
        if self._neo4j is not None or cfg.neo4j_enabled:
            try:
                neo = self._neo4j or Neo4jGraphStore.from_env()
                sync = neo.sync_from(local)
                neo.set_run(res.run_id)
                store, mode_note = neo, f"Neo4j ({neo.database})"
                onto["neo4j_sync"] = sync
            except Neo4jUnavailable as e:
                st.warnings.append(f"Neo4j requested but unavailable - retrieval is running in LOCAL mode. Reason: {e}")
                mode_note = "local in-memory graph (Neo4j unavailable)"
        art.knowledge_graph = local

        settings = RetrievalSettings(depth=cfg.retrieval_depth, max_events=cfg.max_traversal_events, hub_limit=cfg.hub_limit,
                                     related_units_k=cfg.related_units_k, semantic_reference_k=cfg.semantic_reference_k)
        unit_vectors = self._ctx.get("unit_vectors", {})
        unit_meta = {u.unit_id: {"scope": u.scope, "size": u.size, "narrative": u.narrative} for u in res.units}
        retriever = GraphRetriever(store, settings, res.run_id, technique_corpus=self._ctx.get("tcorpus"),
                                   mitre_corpus=art.mitre_corpus, unit_vectors=unit_vectors, unit_meta=unit_meta)
        contributed = failed = 0
        agg = Counter()
        fallback_local = None
        for u in res.units:
            unit = {"unit_id": u.unit_id, "scope": u.scope, "size": u.size, "member_event_uids": u.profile["member_event_uids"],
                    "seed_event_uids": u.profile["seed_event_uids"],
                    "candidates": (u.mapping or {}).get("top_k_techniques", []), "vector": unit_vectors.get(u.unit_id)}
            try:
                ctx = retriever.retrieve(unit)
            except Neo4jUnavailable as e:
                if fallback_local is None:
                    st.warnings.append(f"Neo4j retrieval failed mid-run; switching to LOCAL mode. Reason: {e}")
                    fallback_local = GraphRetriever(local, settings, res.run_id, technique_corpus=self._ctx.get("tcorpus"),
                                                    mitre_corpus=art.mitre_corpus, unit_vectors=unit_vectors, unit_meta=unit_meta)
                    mode_note = "local in-memory graph (Neo4j failed during retrieval)"
                ctx = fallback_local.retrieve(unit)
            except Exception as e:
                failed += 1
                from src.graphrag.retriever import RetrievedContext
                ctx = RetrievedContext(unit_id=u.unit_id, scope=u.scope, retrieval_mode=store.backend,
                                       warnings=[f"retrieval failed: {type(e).__name__}: {e}"])
                ctx.graph_contribution = {"status": "retrieval_failed"}
            u.retrieval = ctx.as_dict()
            gc = ctx.graph_contribution
            if gc.get("status") == "graph_traversal_contributed":
                contributed += 1
            for k in ("traversal_events", "cross_unit_events", "entity_bridged_events", "paths", "related_units_via_graph",
                      "graph_derived_references"):
                agg[k] += gc.get(k, 0)
        if failed:
            st.warnings.append(f"Retrieval failed for {failed} unit(s); their reports contain no graph evidence.")
        res.graphrag = {
            "retrieval_mode": store.backend if fallback_local is None else "local-networkx",
            "mode_note": mode_note, "run_id": res.run_id, "graph": stats,
            "settings": {"depth": cfg.retrieval_depth, "max_traversal_events": cfg.max_traversal_events,
                         "hub_limit": cfg.hub_limit, "related_units_k": cfg.related_units_k,
                         "semantic_reference_k": cfg.semantic_reference_k},
            "units_retrieved": len(res.units), "units_where_graph_contributed": contributed,
            "units_without_graph_evidence": len(res.units) - contributed - failed,
            "totals": dict(agg), "ontology": onto.get("supplied", False),
            "genuine_graphrag": contributed > 0,
            "claim": ("graph traversal contributed evidence beyond seed-event lookup for "
                      f"{contributed}/{len(res.units)} units" if contributed else
                      "NO unit received evidence from graph traversal - retrieval here is not GraphRAG")}
        st.metrics = {k: v for k, v in res.graphrag.items() if k != "graph"} | {"graph_nodes": stats["nodes"],
                                                                                "graph_edges": stats["edges"],
                                                                                "nodes_by_kind": stats["nodes_by_kind"],
                                                                                "edges_by_type": stats["edges_by_type"]}
        if not contributed and res.units:
            st.warnings.append("Graph traversal added no evidence for any unit; do not describe this retrieval as GraphRAG.")
        st.explanation = ("A heterogeneous knowledge graph (events, entities, clusters / noise units, ATT&CK candidates, reference passages) "
                          "is built; for each unit retrieval starts from its seed events, walks the event graph and shared entities, "
                          "follows links to neighbouring units, and adds semantic matches and ATT&CK reference passages - all with provenance.")

    # ================================================================ stage 10

    def _resolve_llm(self, st: StageResult):
        cfg = self.cfg

        if self._llm is not None:
            return self._llm, None

        if cfg.llm == "none":
            return None, "LLM disabled (--llm none)"

        if cfg.llm == "gemini":
            from src.rag.llm_clients import GeminiLLMClient
            try:
                return GeminiLLMClient(), None
            except Exception as e:
                why = f"Gemini unavailable: {type(e).__name__}: {e}"
                st.warnings.append(why)
                return None, why

        from src.rag.llm_clients import OllamaLLMClient

        c = OllamaLLMClient(
            model=cfg.ollama_model,
            host=cfg.ollama_host,
            timeout=cfg.ollama_timeout,
            num_ctx=cfg.ollama_num_ctx,
        )
        ok, why = c.available()

        if ok:
            return c, None

        st.warnings.append(
            f"LLM not used: {why}. The report is the deterministic fallback. "
            "To enable Ollama, start 'ollama serve' and pull the configured model; "
            "'--llm none' disables LLM analysis."
        )
        return None, why


    def _make_event_scorer(self):
        """Event-level text similarity to ATT&CK techniques, used ONLY to tell which events individually resemble a
        candidate technique (technique-specific evidence). Same embedder / corpus / preprocessing as stage 8; the
        stage-8 mapping and its ranking are not touched."""
        mapper, tcorpus = self._ctx.get("mapper"), self._ctx.get("tcorpus")
        if mapper is None or tcorpus is None:
            return None
        events = self.art.events_by_uid
        row = {t["technique_id"]: i for i, t in enumerate(tcorpus.techniques)}

        def score(uids, tids):
            valid = [u for u in uids if u in events]
            if not valid:
                return {}
            texts = [build_unit_profile(u, "noise_event", [events[u]], [u]).mapping_text for u in valid]
            vecs = mapper.embed_texts(texts)
            return {u: {t: float(np.dot(tcorpus.vectors[row[t]], v)) for t in tids if t in row} for u, v in zip(valid, vecs)}
        return score

    def _s10_report(self, st: StageResult) -> None:
        from src.rag.evidence_report import DETERMINISTIC_BACKEND, EvidenceReportBuilder
        from src.rag.finding_llm import apply_llm_analysis, mode_note
        from src.rag.findings import build_analyst_report, refresh_derived
        cfg, res, art = self.cfg, self.res, self.art
        mp = res.mapping or {}
        builder = EvidenceReportBuilder(
            {"run_id": res.run_id, "corpus_version": mp.get("corpus_version"),
             "embedder": {"model_id": mp.get("model_id"), "is_standin": mp.get("is_standin_embedder")},
             "ontology_present": bool((res.ontology or {}).get("supplied") and (res.ontology or {}).get("valid")),
             "platform_filter": mp.get("platform_filter"), "thresholds": (cfg.candidate_threshold, cfg.weak_threshold)},
            art.row_to_uid, art.provenance_by_uid)
        # ---- 1. full per-unit technical record (deterministic; the "technical appendix")
        reports = []
        for u in res.units:
            unit = {"unit_id": u.unit_id, "scope": u.scope, "cluster_id": u.cluster_id, "size": u.size, "profile": u.profile,
                    "novelty": u.novelty_summary, "mapping": u.mapping or None, "retrieval": u.retrieval or {},
                    "noise_signals": (u.profile or {}).get("noise_signals")}
            u.report = builder.build_unit(unit)
            reports.append(u.report)
        noise_units = (res.noise or {}).get("units", [])
        overview = {
            "events": res.normalization.get("unique_events"), "clusters": res.clustering.get("clusters"),
            "noise_events": res.clustering.get("noise_events"),
            "noise_units_analysed": sum(1 for u in res.units if u.scope != "cluster"), "noise_units_total": len(noise_units),
            "units_analysed": len(res.units), "novelty_label_counts": res.novelty.get("label_counts"),
            "high_novelty_events": (res.novelty.get("label_counts") or {}).get("HIGH_NOVELTY", 0),
            "technique_outcomes": mp.get("outcomes"), "corpus_version": mp.get("corpus_version"),
            "graphrag_claim": (res.graphrag or {}).get("claim"), "retrieval_mode": (res.graphrag or {}).get("mode_note")}
        # ---- 2. short prioritized analyst report (deterministic), then optional validated LLM narrative
        novelty_by_uid = {r["event_uid"]: r for r in art.novelty_rows}
        analyst = build_analyst_report(res.units, art.events_by_uid, art.provenance_by_uid, novelty_by_uid, noise_units, builder.run,
                                       overview, max_findings=cfg.report_top_n, candidate_threshold=cfg.candidate_threshold,
                                       event_scorer=self._make_event_scorer())
        llm, why = self._resolve_llm(st)
        info = {"backend": None, "attempted": 0, "accepted": 0, "rejected": 0, "not_attempted": 0, "unavailable_reason": why, "is_mock": False}
        if llm is not None:

            unit_retrievals = {
                u.unit_id: (u.retrieval or {})
                for u in res.units
            }

            info = apply_llm_analysis(
                analyst,
                llm,
                max_findings=cfg.report_top_n,
                unit_retrievals=unit_retrievals,
            )

            if info["unavailable_reason"]:
                st.warnings.append(f"LLM stopped after a service failure ({info['unavailable_reason']}); remaining findings use the deterministic text.")
            for f in analyst["findings"]:
                for p in (f.get("llm") or {}).get("problems", [])[:1]:
                    if not f["llm"]["accepted"] and f["llm"]["attempted"] and not info["unavailable_reason"]:
                        st.warnings.append(f"LLM analysis for {f['id']} rejected: {p}")
        else:
            analyst["analysis_mode"] = {"mode": "deterministic", "backend": None, "accepted": 0, "rejected": 0, "not_attempted": len(analyst["findings"]),
                                        "reason": why}
        note = mode_note(analyst)
        refresh_derived(analyst, overview, builder.run, note)
        mode = analyst["analysis_mode"]["mode"]
        llm_info = {"requested": cfg.llm, "used": mode in ("llm-assisted", "llm-partial", "mock"), "mode": mode,
                    "backend": getattr(llm, "backend_name", None) if llm else None,
                    "findings_with_accepted_analysis": info["accepted"], "findings_with_rejected_analysis": info["rejected"],
                    "findings_not_attempted": info["not_attempted"], "is_mock": mode == "mock",
                    "unavailable_reason": analyst["analysis_mode"].get("reason") if mode == "deterministic" else None,
                    "generated_by": (analyst["analysis_mode"]["backend"] if mode != "deterministic" else DETERMINISTIC_BACKEND)}
        res.report = builder.build_run_report(reports, [{"unit_id": u.unit_id, "priority": u.priority} for u in res.units],
                                              overview, llm_info, cfg.report_top_n)
        res.report["status_banner"] = note
        res.analyst_report = analyst
        issues = res.report["validation"]["issues"]
        st.metrics = {"units_reported": len(reports), "llm": llm_info, "validation_passed": res.report["validation"]["passed"],
                      "validation_issue_count": res.report["validation"]["issue_count"],
                      "findings": len(analyst["findings"]), "findings_by_priority": analyst["executive_summary"]["findings_by_priority"],
                      "analysis_mode": mode}
        if issues:
            st.warnings.append(f"Report evidence validation found {res.report['validation']['issue_count']} issue(s); first: {issues[0]}")
        st.explanation = ("Every fact, candidate, reference and path is assembled deterministically with evidence ids (technical appendix). "
                          "Units are then grouped into a short list of prioritized findings with documented rules; an LLM, when available, adds "
                          "narrative per finding that is rejected if it cites unknown events, techniques, paths or evidence, or over-claims.")
