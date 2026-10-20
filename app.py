import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Cyber Threat Analysis Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).resolve().parent
ACTIVE_OUTPUT_DIR = st.session_state.get("active_output_dir")
OUTPUT_DIR = Path(ACTIVE_OUTPUT_DIR) if ACTIVE_OUTPUT_DIR else None

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.35rem; padding-bottom: 2rem; }
    [data-testid="stMetric"] {
        background: rgba(100, 116, 139, 0.08);
        padding: 14px 16px;
        border: 1px solid rgba(100, 116, 139, 0.22);
        border-radius: 12px;
    }
    .hero {
        padding: 1.35rem 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(100, 116, 139, .25);
        background: linear-gradient(115deg, rgba(14, 116, 144, .18), rgba(30, 41, 59, .06));
        margin-bottom: 1rem;
    }
    .hero-kicker {
        text-transform: uppercase;
        letter-spacing: .12em;
        font-size: .72rem;
        color: #38bdf8;
        font-weight: 700;
    }
    .hero h1 { margin: .25rem 0 .4rem 0; }
    .muted { color: var(--text-color); opacity: .72; font-size: .9rem; }
    div[data-testid="stVerticalBlock"] div[data-testid="stAlert"] { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Safe file readers and schema helpers
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def read_json(path_string: str) -> Any:
    path = Path(path_string)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def read_csv(path_string: str) -> pd.DataFrame:
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def find_dicts(obj: Any, depth: int = 0):
    """Yield nested dictionaries without assuming one export schema."""
    if depth > 12:
        return
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from find_dicts(value, depth + 1)
    elif isinstance(obj, list):
        for value in obj:
            yield from find_dicts(value, depth + 1)


def find_records(obj: Any) -> list[dict]:
    """Find likely finding records in an exported JSON document."""
    candidates, seen = [], set()
    for item in find_dicts(obj):
        keys = {str(k).lower() for k in item}
        has_identity = any(
            k in keys for k in ("finding_id", "findingid", "finding_key", "rule_id", "id")
        )
        has_content = any(
            k in keys for k in ("narrative", "description", "summary", "title", "evidence")
        )
        if has_identity and has_content:
            marker = json.dumps(item, sort_keys=True, default=str)
            if marker not in seen:
                seen.add(marker)
                candidates.append(item)
    return candidates


def get_value(record: dict, *names: str):
    lowered = {str(k).lower(): v for k, v in record.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def read_text(name: str):
    if OUTPUT_DIR is None:
        return None
    path = OUTPUT_DIR / name
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8-sig")
    except OSError:
        return None


def as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def nested_value(obj: Any, *path: str):
    current = obj
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def available_count(value: Any) -> int | None:
    """Return a count only when the export actually provides countable data."""
    if isinstance(value, (list, dict, str)):
        return len(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return None


def _text_value(value: Any, fallback: str = "") -> str:
    if isinstance(value, (str, int, float)):
        return str(value).strip()
    return fallback


def extract_graph(data_sources: list[Any], max_nodes: int = 70, max_edges: int = 100):
    """Best-effort graph extraction from explicit node/edge records and retrieval paths."""
    nodes: dict[str, dict] = {}
    edges: list[tuple[str, str, str]] = []
    seen_edges = set()

    def add_node(node_id: Any, label: Any = None, kind: Any = None):
        if node_id is None:
            return
        key = str(node_id).strip()
        if not key:
            return
        if key not in nodes and len(nodes) < max_nodes:
            nodes[key] = {"label": _text_value(label, key)[:90] or key,
                          "kind": _text_value(kind, "entity") or "entity"}

    def add_edge(source: Any, target: Any, relation: Any = "related to"):
        if source is None or target is None:
            return
        src, dst = str(source).strip(), str(target).strip()
        if not src or not dst or src == dst:
            return
        add_node(src)
        add_node(dst)
        rel = _text_value(relation, "related to")[:70] or "related to"
        key = (src, dst, rel)
        if key not in seen_edges and len(edges) < max_edges:
            seen_edges.add(key)
            edges.append(key)

    def walk(obj: Any, depth: int = 0):
        if depth > 14 or len(edges) >= max_edges:
            return
        if isinstance(obj, dict):
            # Explicit graph node records
            oid = next((obj[k] for k in ("node_id", "entity_id", "event_id", "cluster_id", "id", "key") if k in obj and isinstance(obj[k], (str, int, float))), None)
            label = next((obj[k] for k in ("label", "name", "title", "entity", "event_type", "type") if k in obj and isinstance(obj[k], (str, int, float))), None)
            if oid is not None and any(k in obj for k in ("node_type", "entity_type", "event_type", "label", "name", "node_id", "entity_id")):
                add_node(oid, label, obj.get("node_type") or obj.get("entity_type") or obj.get("event_type") or obj.get("type"))
            # Explicit edge/relation records
            source = next((obj[k] for k in ("source", "src", "from", "subject", "source_id", "source_node") if k in obj and isinstance(obj[k], (str, int, float))), None)
            target = next((obj[k] for k in ("target", "dst", "to", "object", "target_id", "target_node") if k in obj and isinstance(obj[k], (str, int, float))), None)
            relation = next((obj[k] for k in ("relation", "relationship", "predicate", "edge_type", "type", "label") if k in obj and isinstance(obj[k], (str, int, float))), "related to")
            if source is not None and target is not None:
                add_edge(source, target, relation)
            # Retrieval traces can contain ordered paths
            for path_key in ("paths", "path", "traversal_path", "graph_path"):
                path = obj.get(path_key)
                if isinstance(path, list):
                    ids = []
                    for part in path:
                        if isinstance(part, dict):
                            pid = next((part[k] for k in ("node_id", "entity_id", "event_id", "id", "key", "name", "label") if k in part and isinstance(part[k], (str, int, float))), None)
                            plabel = part.get("label") or part.get("name")
                            if pid is not None:
                                add_node(pid, plabel, part.get("type") or part.get("entity_type"))
                                ids.append(str(pid))
                        elif isinstance(part, (str, int, float)):
                            add_node(part)
                            ids.append(str(part))
                    for a, b in zip(ids, ids[1:]):
                        add_edge(a, b, "traversal")
            for value in obj.values():
                walk(value, depth + 1)
        elif isinstance(obj, list):
            for value in obj:
                walk(value, depth + 1)

    for source in data_sources:
        walk(source)
    return nodes, edges



def analyze_uploaded_log(uploaded_file) -> dict:
    """Perform transparent, local first-pass analysis on an uploaded log file."""
    raw = uploaded_file.getvalue()
    text_content = raw.decode("utf-8", errors="replace")
    lines = text_content.splitlines()
    nonempty = [line for line in lines if line.strip()]

    patterns = {
        "IPv4 addresses": re.compile(r"(?<![\d.])(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\d.])"),
        "URLs": re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE),
        "Email addresses": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
        "File hashes (MD5/SHA1/SHA256)": re.compile(r"\b(?:[a-fA-F0-9]{32}|[a-fA-F0-9]{40}|[a-fA-F0-9]{64})\b"),
        "Domain-like strings": re.compile(r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+(?:com|net|org|io|co|in|ru|cn|uk|info|biz|xyz|top|site|online|local)\b", re.IGNORECASE),
    }
    indicators = {}
    for label, pattern in patterns.items():
        found = []
        seen = set()
        for match in pattern.findall(text_content):
            value = match.rstrip(".,;:)")
            if value not in seen:
                seen.add(value)
                found.append(value)
        indicators[label] = found

    severity_patterns = {
        "Critical": re.compile(r"\b(critical|emergency|severity\s*[=:]\s*critical|level\s*[=:]\s*critical)\b", re.I),
        "High": re.compile(r"\b(high|severe|fatal|malware|ransomware|privilege escalation|data exfiltration|command and control|\bc2\b)\b", re.I),
        "Medium": re.compile(r"\b(medium|warning|suspicious|unauthorized|failed login|authentication failure|blocked)\b", re.I),
        "Low/Informational": re.compile(r"\b(info|informational|debug|notice|success|accepted)\b", re.I),
    }
    severity_counts = Counter()
    matched_lines = []
    for number, line in enumerate(lines, 1):
        matched = [name for name, pattern in severity_patterns.items() if pattern.search(line)]
        if matched:
            # Keep the strongest matching category for a single line.
            severity = next((x for x in ("Critical", "High", "Medium", "Low/Informational") if x in matched), matched[0])
            severity_counts[severity] += 1
            matched_lines.append({"Line": number, "Severity signal": severity, "Log entry": line[:1500]})

    # Try structured parsing for JSON and CSV; preserve plain-text logs regardless.
    structured_rows = None
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".json":
        try:
            parsed = json.loads(text_content)
            if isinstance(parsed, list):
                structured_rows = pd.json_normalize(parsed)
            elif isinstance(parsed, dict):
                for candidate in ("events", "records", "logs", "data", "results"):
                    if isinstance(parsed.get(candidate), list):
                        structured_rows = pd.json_normalize(parsed[candidate])
                        break
                if structured_rows is None:
                    structured_rows = pd.json_normalize(parsed)
        except json.JSONDecodeError:
            pass
    elif suffix == ".csv":
        try:
            import io
            structured_rows = pd.read_csv(io.BytesIO(raw), low_memory=False)
        except Exception:
            pass

    return {
        "filename": uploaded_file.name,
        "size_bytes": len(raw),
        "line_count": len(lines),
        "nonempty_lines": len(nonempty),
        "lines": lines,
        "indicators": indicators,
        "severity_counts": dict(severity_counts),
        "matched_lines": matched_lines,
        "structured_rows": structured_rows,
    }

def graphviz_source(nodes: dict[str, dict], edges: list[tuple[str, str, str]]) -> str:
    """Create safe DOT source for Streamlit's built-in graph renderer."""
    import json as _json
    lines = ["digraph ThreatGraph {", "rankdir=LR;", 'graph [bgcolor="transparent", overlap=false, splines=true];',
             'node [shape=box, style="rounded,filled", fillcolor="#102a43", fontcolor="#e6f6ff", color="#38bdf8", fontname="Arial"];',
             'edge [color="#67e8f9", fontcolor="#cbd5e1", fontname="Arial", fontsize=9];']
    for node_id, info in nodes.items():
        nid = _json.dumps(node_id)
        label = _json.dumps(info.get("label", node_id))
        lines.append(f"{nid} [label={label}];")
    for src, dst, relation in edges:
        lines.append(f"{_json.dumps(src)} -> {_json.dumps(dst)} [label={_json.dumps(relation)}];")
    lines.append("}")
    return "\n".join(lines)


# --------------------------------------------------
# Load exported pipeline data
# --------------------------------------------------
# Only load outputs produced by the current user's uploaded-log run.
if OUTPUT_DIR and OUTPUT_DIR.exists():
    run_data = read_json(str(OUTPUT_DIR / "run_result.json"))
    analyst_data = read_json(str(OUTPUT_DIR / "analyst_report.json"))
    ontology_data = read_json(str(OUTPUT_DIR / "ontology_findings.json"))
    evidence_data = read_json(str(OUTPUT_DIR / "evidence_report.json"))
    profiles_data = read_json(str(OUTPUT_DIR / "cluster_profiles_v2.json"))
    noise_data = read_json(str(OUTPUT_DIR / "noise_review.json"))
    mappings_data = read_json(str(OUTPUT_DIR / "semantic_mitre_mappings.json"))
    rag_data = read_json(str(OUTPUT_DIR / "rag_ready_clusters.json"))
    clusters_df = read_csv(str(OUTPUT_DIR / "embedding_clusters.csv"))
    novelty_df = read_csv(str(OUTPUT_DIR / "novelty_scores.csv"))
    report_md = read_text("analyst_report.md")
    evidence_md = read_text("evidence_report.md")
    evidence_txt = read_text("evidence_report.txt")
    findings = find_records(analyst_data) or find_records(ontology_data)
else:
    run_data = analyst_data = ontology_data = evidence_data = None
    profiles_data = noise_data = mappings_data = rag_data = None
    clusters_df = pd.DataFrame()
    novelty_df = pd.DataFrame()
    report_md = evidence_md = evidence_txt = None
    findings = []

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("🛡️ Cyber Threat Analysis")
st.sidebar.caption("Threat Intelligence & GraphRAG Analysis")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Threat Findings",
        "Event & Cluster Explorer",
        "GraphRAG Knowledge Graph",
        "MITRE ATT&CK",
        "Evidence & Reports",
        "Upload & Analyze Logs",
        "Raw Data Inspector",
    ],
)

st.sidebar.divider()
st.sidebar.write("**Analysis:** Cyber threat investigation")
st.sidebar.write(f"**Active run:** `{OUTPUT_DIR.name if OUTPUT_DIR else 'No run yet'}`")
st.sidebar.caption("No results are shown until uploaded logs are processed by the backend.")

if page != "Upload & Analyze Logs" and not OUTPUT_DIR:
    st.title("Cyber Threat Analysis Dashboard")
    st.info("No analysis results yet. Open **Upload & Analyze Logs**, upload files, and run the backend analysis to populate this dashboard.")
    st.stop()
if OUTPUT_DIR and not OUTPUT_DIR.exists() and page != "Upload & Analyze Logs":
    st.info("No output exports found for this run. Upload logs and run backend analysis first.")
    st.stop()


# --------------------------------------------------
# Overview — unified dashboard
# --------------------------------------------------
if page == "Overview":
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Security Operations · Intelligence Overview</div>
          <h1>Cybersecurity Intelligence Dashboard</h1>
          <div class="muted">A consolidated view of event analysis, anomaly scoring,
          ATT&CK candidate mapping, GraphRAG retrieval, and evidence-grounded findings.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    run_obj = as_dict(run_data)
    run_status = (
        run_obj.get("status")
        or run_obj.get("run_status")
        or run_obj.get("analysis_mode")
        or ("Export loaded" if run_data is not None else "Unavailable")
    )
    run_id = run_obj.get("run_id") or nested_value(run_obj, "metadata", "run_id")
    graphrag = as_dict(run_obj.get("graphrag"))
    mapping = as_dict(run_obj.get("mapping"))
    novelty = as_dict(run_obj.get("novelty"))
    clustering = as_dict(run_obj.get("clustering"))
    normalization = as_dict(run_obj.get("normalization"))
    ontology = as_dict(run_obj.get("ontology"))

    # Prefer actual exported counts; do not manufacture missing metrics.
    event_count = (
        normalization.get("unique_events")
        or normalization.get("valid_events")
        or normalization.get("events_processed")
    )
    if event_count is None:
        event_count = available_count(clusters_df) if not clusters_df.empty else None

    cluster_count = clustering.get("clusters")
    if cluster_count is None and not clusters_df.empty:
        for col in ("cluster_id", "cluster", "label"):
            if col in clusters_df.columns:
                cluster_count = int(clusters_df[col].nunique())
                break

    novelty_count = len(novelty_df) if not novelty_df.empty else None
    finding_count = len(findings) if findings else None
    graph_units = graphrag.get("units_where_graph_contributed")
    graph_total = graphrag.get("units_retrieved")
    graph_claim = graphrag.get("claim")
    graph_mode = graphrag.get("mode_note") or graphrag.get("retrieval_mode")
    technique_count = mapping.get("techniques_compared")
    if technique_count is None:
        technique_count = mapping.get("units_mapped")

    st.subheader("Run at a glance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pipeline status", str(run_status)[:32])
    m2.metric("Events processed", f"{event_count:,}" if isinstance(event_count, (int, float)) else "Unavailable")
    m3.metric("Threat findings", f"{finding_count:,}" if finding_count is not None else "Unavailable")
    m4.metric("Novelty records", f"{novelty_count:,}" if novelty_count is not None else "Unavailable")

    with st.expander("Run metadata", expanded=False):
        if run_id:
            st.write("**Run ID:**", run_id)
        if isinstance(run_data, dict):
            st.json(
                {k: v for k, v in run_data.items() if k in
                 ("metadata", "stages", "errors", "warnings", "status", "run_id", "analysis_mode")},
                expanded=False,
            )
        else:
            st.info("Run metadata was not found in run_result.json.")

    st.divider()
    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("Finding priority distribution")
        if findings:
            priorities = [
                str(get_value(r, "priority", "severity", "risk_level") or "Unspecified")
                for r in findings
            ]
            severity_df = (
                pd.Series(priorities, name="Priority")
                .value_counts()
                .rename_axis("Priority")
                .reset_index(name="Findings")
            )
            chart_col, table_col = st.columns([1.1, 1])
            with chart_col:
                st.bar_chart(severity_df.set_index("Priority"), height=260)
            with table_col:
                st.dataframe(severity_df, hide_index=True, use_container_width=True)
        else:
            st.info("No finding records were detected in the available analyst/ontology exports.")

    with right:
        st.subheader("Analysis modules")
        module_rows = [
            ("Clustering", "Available" if clustering or not clusters_df.empty else "No data"),
            ("Novelty detection", "Available" if novelty or not novelty_df.empty else "No data"),
            ("MITRE ATT&CK mapping", "Available" if mapping or mappings_data is not None else "No data"),
            ("GraphRAG retrieval", "Available" if graphrag else "No data"),
            ("Ontology", "Validated" if ontology.get("valid") is True else
             "Supplied / status unknown" if ontology.get("supplied") else
             "No data" if not ontology_data else "Export available"),
        ]
        for label, status in module_rows:
            c1, c2 = st.columns([2, 1])
            c1.write(label)
            if status == "Available" or status == "Validated":
                c2.success(status)
            elif status == "No data":
                c2.warning(status)
            else:
                c2.info(status)

    st.divider()
    st.subheader("GraphRAG contribution")
    g1, g2, g3 = st.columns(3)
    g1.metric(
        "Units with graph contribution",
        f"{graph_units}/{graph_total}" if isinstance(graph_units, (int, float)) and isinstance(graph_total, (int, float)) else "Unavailable",
    )
    g2.metric("ATT&CK techniques compared", f"{technique_count:,}" if isinstance(technique_count, (int, float)) else "Unavailable")
    g3.metric("Cluster count", f"{cluster_count:,}" if isinstance(cluster_count, (int, float)) else "Unavailable")

    if graph_claim:
        st.caption(str(graph_claim))
    elif graph_mode:
        st.caption(f"Retrieval mode: {graph_mode}")
    else:
        st.info("GraphRAG metrics are not present in run_result.json. Check that the latest pipeline exports were loaded.")

    st.caption(
        "ATT&CK similarity results are investigative leads, not proof of malicious activity. "
        "Graph relationships provide context and do not independently establish causation."
    )

    st.divider()
    st.subheader("Recent findings")
    if findings:
        preview = []
        for idx, record in enumerate(findings[:8], start=1):
            preview.append({
                "Finding": get_value(record, "title", "name", "finding_id", "finding_key", "rule_id") or f"Finding {idx}",
                "Priority": get_value(record, "priority", "severity", "risk_level") or "Unspecified",
                "Summary": get_value(record, "summary", "narrative", "description") or "No summary available",
                "Status": get_value(record, "status", "analysis_status") or "Unspecified",
            })
        st.dataframe(pd.DataFrame(preview), hide_index=True, use_container_width=True)
        st.caption("Open the Threat Findings page from the sidebar to inspect each finding.")
    else:
        st.info("No findings are available to preview yet.")

    st.subheader("Export health")
    export_names = [
        "analyst_report.json", "analyst_report.md", "cluster_profiles_v2.json",
        "embedding_clusters.csv", "evidence_report.json", "evidence_report.md",
        "evidence_report.txt", "noise_review.json", "novelty_scores.csv",
        "ontology_findings.json", "rag_ready_clusters.json", "run_result.json",
        "semantic_mitre_mappings.json",
    ]
    export_df = pd.DataFrame([
        {
            "File": name,
            "Status": "Available" if (OUTPUT_DIR / name).exists() else "Missing",
            "Size (KB)": round((OUTPUT_DIR / name).stat().st_size / 1024, 1)
            if (OUTPUT_DIR / name).exists() else 0,
        }
        for name in export_names
    ])
    st.dataframe(export_df, hide_index=True, use_container_width=True)


# --------------------------------------------------
# Threat Findings
# --------------------------------------------------
elif page == "Threat Findings":
    st.title("Threat Findings")
    st.caption("Findings loaded from the exported analyst or ontology JSON.")

    if not findings:
        st.warning("No finding records were detected automatically. Inspect the actual JSON structure in Raw Data Inspector.")
    else:
        priorities = sorted({
            str(get_value(r, "priority", "severity", "risk_level") or "Unspecified")
            for r in findings
        })
        selected_priorities = st.multiselect("Filter by priority", priorities, default=priorities)
        visible = [
            r for r in findings
            if str(get_value(r, "priority", "severity", "risk_level") or "Unspecified")
            in selected_priorities
        ]
        st.write(f"Showing **{len(visible)}** of **{len(findings)}** findings.")
        for index, record in enumerate(visible, 1):
            title = get_value(record, "title", "name", "finding_id", "finding_key", "rule_id") or f"Finding {index}"
            severity = get_value(record, "priority", "severity", "risk_level") or "Unspecified"
            with st.expander(f"{severity} | {title}", expanded=index == 1):
                st.json(record, expanded=False)
                st.download_button(
                    "Download this finding",
                    data=json.dumps(record, indent=2, ensure_ascii=False, default=str),
                    file_name=f"finding_{index}.json",
                    mime="application/json",
                    key=f"finding_download_{index}",
                )
    if report_md:
        st.divider()
        with st.expander("Read the complete analyst report"):
            st.markdown(report_md)


# --------------------------------------------------
# Event & Cluster Explorer
# --------------------------------------------------
elif page == "Event & Cluster Explorer":
    st.title("Event & Cluster Explorer")
    tab1, tab2, tab3 = st.tabs(["Embedding Clusters", "Novelty Scores", "Cluster Profiles"])

    with tab1:
        st.subheader("Embedding cluster data")
        if clusters_df.empty:
            st.info("No readable cluster CSV data was found.")
        else:
            st.caption(f"{len(clusters_df):,} rows")
            search = st.text_input("Search cluster records", key="cluster_search")
            filtered = clusters_df
            if search:
                mask = filtered.astype(str).apply(
                    lambda col: col.str.contains(search, case=False, na=False)
                ).any(axis=1)
                filtered = filtered[mask]
            st.dataframe(filtered.head(500), use_container_width=True)
            st.download_button(
                "Download cluster CSV", clusters_df.to_csv(index=False),
                "embedding_clusters.csv", "text/csv",
            )

    with tab2:
        st.subheader("Novelty scores")
        if novelty_df.empty:
            st.info("No readable novelty CSV data was found.")
        else:
            st.dataframe(novelty_df.head(500), use_container_width=True)
            numeric_cols = novelty_df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Numeric column to chart", numeric_cols)
                st.line_chart(novelty_df[selected_col].reset_index(drop=True))
            st.download_button(
                "Download novelty CSV", novelty_df.to_csv(index=False),
                "novelty_scores.csv", "text/csv",
            )

    with tab3:
        st.subheader("Cluster profiles")
        if profiles_data is None:
            st.info("Cluster profile JSON is unavailable.")
        else:
            st.json(profiles_data, expanded=False)


# --------------------------------------------------
# GraphRAG Knowledge Graph
# --------------------------------------------------
elif page == "GraphRAG Knowledge Graph":
    st.title("GraphRAG Knowledge Graph")
    st.caption("Explore entities, events, references, and relationships recovered from the available GraphRAG and evidence exports.")
    graph_nodes, graph_edges = extract_graph([run_data, evidence_data, analyst_data, ontology_data, profiles_data, mappings_data, rag_data])
    if graph_nodes and graph_edges:
        st.caption(f"Showing {len(graph_nodes)} nodes and {len(graph_edges)} relationships (bounded preview).")
        st.graphviz_chart(graphviz_source(graph_nodes, graph_edges), use_container_width=True)
        left, right = st.columns(2)
        with left:
            st.subheader("Nodes")
            node_rows = [{"Node": key, "Label": val.get("label"), "Type": val.get("kind")} for key, val in graph_nodes.items()]
            st.dataframe(pd.DataFrame(node_rows), use_container_width=True, hide_index=True)
        with right:
            st.subheader("Edges")
            st.dataframe(pd.DataFrame(edges, columns=["Source", "Target", "Relationship"]), use_container_width=True, hide_index=True)
        graph_json = {"nodes": [{"id": k, **v} for k, v in graph_nodes.items()],
                      "edges": [{"source": a, "target": b, "relationship": rel} for a, b, rel in graph_edges]}
        st.download_button("Download displayed graph JSON", json.dumps(graph_json, indent=2, ensure_ascii=False),
                           "graphrag_knowledge_graph.json", "application/json")
    else:
        st.warning("No explicit node/edge records or ordered GraphRAG paths were found in the loaded exports.")
        st.write("The dashboard will not invent relationships. Check the retrieval trace in the evidence JSON, or export GraphRAG nodes and edges from the retrieval stage to populate this view.")
        if evidence_data is not None:
            with st.expander("Inspect evidence/retrieval export"):
                st.json(evidence_data, expanded=False)

    st.divider()
    st.subheader("GraphRAG-supported final report")
    st.caption("The report below is loaded from the generated analyst report export; the evidence report provides the retrieval/evidence trace used for review.")
    if report_md:
        st.markdown(report_md)
        st.download_button("Download final analyst report", report_md, "analyst_report.md", "text/markdown", key="graph_report_download")
    else:
        st.info("analyst_report.md is not present in the selected export directory.")
    if evidence_md:
        with st.expander("Review GraphRAG evidence report"):
            st.markdown(evidence_md)
            st.download_button("Download evidence report", evidence_md, "evidence_report.md", "text/markdown", key="graph_evidence_download")


# --------------------------------------------------
# MITRE ATT&CK
# --------------------------------------------------
elif page == "MITRE ATT&CK":
    st.title("Semantic MITRE ATT&CK Mapping")
    st.warning(
        "Similarity-based mappings are candidate leads. The pipeline reports uncalibrated similarity thresholds."
    )
    if mappings_data is None:
        st.info("MITRE mapping JSON is unavailable.")
    else:
        st.json(mappings_data, expanded=False)
        st.download_button(
            "Download MITRE mapping JSON",
            json.dumps(mappings_data, indent=2, ensure_ascii=False, default=str),
            "semantic_mitre_mappings.json",
            "application/json",
        )


# --------------------------------------------------
# Evidence & Reports
# --------------------------------------------------
elif page == "Evidence & Reports":
    st.title("Evidence & Reports")
    if isinstance(run_data, dict):
        with st.expander("Pipeline run details"):
            st.json(run_data, expanded=False)

    tab1, tab2, tab3 = st.tabs(["Analyst Report", "Evidence Report", "Evidence JSON"])
    with tab1:
        if report_md:
            st.markdown(report_md)
            st.download_button("Download Analyst Report", report_md, "analyst_report.md", "text/markdown")
        else:
            st.info("Analyst Markdown report not found.")
    with tab2:
        if evidence_md:
            st.markdown(evidence_md)
            st.download_button("Download Evidence Report", evidence_md, "evidence_report.md", "text/markdown")
        elif evidence_txt:
            st.text(evidence_txt)
            st.download_button("Download Evidence Text", evidence_txt, "evidence_report.txt", "text/plain")
        else:
            st.info("Evidence report is unavailable.")
    with tab3:
        if evidence_data is None:
            st.info("Evidence JSON is unavailable.")
        else:
            st.caption("Large export: expand only the relevant sections.")
            st.json(evidence_data, expanded=False)
            st.download_button(
                "Download Evidence JSON",
                json.dumps(evidence_data, indent=2, ensure_ascii=False, default=str),
                "evidence_report.json",
                "application/json",
            )


# --------------------------------------------------
# Upload & Analyze Logs — submit uploads to backend
# --------------------------------------------------
elif page == "Upload & Analyze Logs":
    import io
    import os
    import subprocess
    import sys
    import time

    st.title("Upload Logs and Run Threat Analysis")
    st.caption("No data is displayed initially. Uploaded logs are submitted to the backend, and the dashboard is populated only after the run succeeds.")
    st.info("The project must contain run_pipeline.py beside app.py. JSON and CSV records are preserved; plain-text logs are converted into message records for the pipeline.")

    uploaded_files = st.file_uploader(
        "Upload security log files", type=["log", "txt", "json", "csv"],
        accept_multiple_files=True,
        help="Upload the logs you want this run to analyze. Previous sample exports are not loaded automatically.",
    )
    llm_choice = st.selectbox("Analysis LLM", ["gemini"], index=0)
    run_name = st.text_input("Run name (optional)", value="uploaded_logs")
    if uploaded_files:
        st.dataframe(pd.DataFrame([
            {"File": f.name, "Size (KB)": round(f.size / 1024, 1), "Format": Path(f.name).suffix.lower()}
            for f in uploaded_files
        ]), hide_index=True, use_container_width=True)

    if st.button("Send logs to backend and analyze", type="primary", disabled=not uploaded_files):
        pipeline_script = PROJECT_DIR / "run_pipeline.py"
        if not pipeline_script.exists():
            st.error(f"Backend entry point not found: {pipeline_script}. Set PROJECT_DIR to the backend project root or keep app.py beside run_pipeline.py.")
        else:
            try:
                events, manifest = [], []
                for uploaded in uploaded_files:
                    raw = uploaded.getvalue()
                    content = raw.decode("utf-8-sig", errors="replace")
                    suffix = Path(uploaded.name).suffix.lower()
                    manifest.append({"filename": uploaded.name, "size_bytes": len(raw)})
                    if suffix == ".json":
                        # Accept standard JSON, JSON Lines (JSONL), and concatenated JSON objects.
                        try:
                            parsed = json.loads(content)
                            rows = parsed if isinstance(parsed, list) else next(
                                (parsed[k] for k in ("events", "records", "logs", "data", "results")
                                 if isinstance(parsed, dict) and isinstance(parsed.get(k), list)),
                                [parsed] if isinstance(parsed, dict) else None,
                            )
                            if rows is None:
                                raise ValueError(f"{uploaded.name}: JSON must contain an object or list of records.")
                        except json.JSONDecodeError as first_error:
                            rows = []
                            # First try one JSON object per non-empty line.
                            line_rows = []
                            line_parse_ok = True
                            for line in content.splitlines():
                                if not line.strip():
                                    continue
                                try:
                                    line_rows.append(json.loads(line))
                                except json.JSONDecodeError:
                                    line_parse_ok = False
                                    break
                            if line_parse_ok and line_rows:
                                rows = line_rows
                            else:
                                # Also support multiple JSON values concatenated without a surrounding array.
                                decoder = json.JSONDecoder()
                                position = 0
                                values = []
                                try:
                                    while position < len(content):
                                        while position < len(content) and content[position].isspace():
                                            position += 1
                                        if position >= len(content):
                                            break
                                        value, position = decoder.raw_decode(content, position)
                                        values.append(value)
                                    if values:
                                        rows = values
                                    else:
                                        raise first_error
                                except json.JSONDecodeError:
                                    raise ValueError(
                                        f"{uploaded.name} is not a single JSON document, JSON Lines file, or sequence of JSON objects. "
                                        f"Original parser error: {first_error}"
                                    ) from first_error
                        for idx, row in enumerate(rows, 1):
                            item = dict(row) if isinstance(row, dict) else {"message": json.dumps(row, ensure_ascii=False) if isinstance(row, (list, dict)) else str(row)}
                            item.setdefault("_source_file", uploaded.name)
                            item.setdefault("_source_record", idx)
                            events.append(item)
                    elif suffix == ".csv":
                        frame = pd.read_csv(io.BytesIO(raw), low_memory=False)
                        for idx, row in enumerate(frame.to_dict(orient="records"), 1):
                            row.setdefault("_source_file", uploaded.name)
                            row.setdefault("_source_record", idx)
                            events.append(row)
                    else:
                        for idx, line in enumerate(content.splitlines(), 1):
                            if line.strip():
                                events.append({"message": line, "raw_log": line, "_source_file": uploaded.name, "_source_record": idx})

                if not events:
                    st.error("No non-empty records found in the uploaded files.")
                else:
                    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", run_name).strip("_") or "uploaded_logs"
                    run_stamp = time.strftime("%Y%m%d_%H%M%S")
                    export_dir = PROJECT_DIR / "outputs" / f"{safe_name}_{run_stamp}"
                    input_dir = PROJECT_DIR / "data" / "dashboard_uploads"
                    input_dir.mkdir(parents=True, exist_ok=True)
                    input_path = input_dir / f"{safe_name}_{run_stamp}.json"
                    input_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
                    st.write(f"Submitting **{len(events):,} records** from **{len(manifest)} file(s)** to the backend.")
                    with st.spinner("Backend is running threat analysis, GraphRAG retrieval, and report generation..."):
                        result = subprocess.run(
                            [sys.executable, str(pipeline_script), "--input", str(input_path), "--llm", llm_choice, "--export-dir", str(export_dir)],
                            cwd=str(PROJECT_DIR), capture_output=True, text=True, timeout=3600, env=os.environ.copy()
                        )
                    with st.expander("Backend execution log", expanded=result.returncode != 0):
                        st.code((result.stdout or "")[-12000:] or "(No standard output)")
                        if result.stderr:
                            st.text("Standard error")
                            st.code(result.stderr[-12000:])
                    if result.returncode != 0:
                        st.error("Backend pipeline failed. The dashboard will not display results for this run.")
                    elif not (export_dir / "run_result.json").exists():
                        st.error("Backend finished but run_result.json was not found in the expected output directory. Verify the backend's export behavior.")
                    else:
                        st.session_state["active_output_dir"] = str(export_dir)
                        st.session_state["active_upload_manifest"] = manifest
                        st.success("Analysis complete. Dashboard pages will now display the results from these uploaded logs.")
                        st.rerun()
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON in uploaded file: {exc}")
            except subprocess.TimeoutExpired:
                st.error("Backend analysis exceeded the one-hour timeout.")
            except Exception as exc:
                st.error(f"Could not submit logs to the backend: {exc}")

    if not OUTPUT_DIR:
        st.divider()
        st.info("Waiting for uploaded logs. No old exports or sample data are being displayed.")
    else:
        st.caption(f"Active results: `{OUTPUT_DIR.name}`")


# --------------------------------------------------
# Raw Data Inspector
# --------------------------------------------------
elif page == "Raw Data Inspector":
    st.title("Raw Data Inspector")
    st.caption("Inspect exported JSON to troubleshoot unfamiliar schemas.")
    json_files = [
        "analyst_report.json", "cluster_profiles_v2.json", "evidence_report.json",
        "noise_review.json", "ontology_findings.json", "rag_ready_clusters.json",
        "run_result.json", "semantic_mitre_mappings.json",
    ]
    selected_file = st.selectbox("Choose JSON export", json_files)
    selected_data = read_json(str(OUTPUT_DIR / selected_file))
    if selected_data is None:
        st.error("Could not load this JSON file.")
    else:
        if isinstance(selected_data, dict):
            st.write("**Top-level keys**")
            st.code("\n".join(map(str, selected_data.keys())))
        elif isinstance(selected_data, list):
            st.write(f"Top-level list with {len(selected_data)} items")
            if selected_data and isinstance(selected_data[0], dict):
                st.write("**First item keys**")
                st.code("\n".join(map(str, selected_data[0].keys())))
        st.json(selected_data, expanded=False)
        st.download_button(
            "Download selected JSON",
            json.dumps(selected_data, indent=2, ensure_ascii=False, default=str),
            selected_file,
            "application/json",
        )
