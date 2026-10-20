import json
import os
import subprocess
import sys

import pytest

from src.cli import main as cli
from src.cli.render import render_stage
from src.service.result import StageResult
from tests.conftest import ROOT


def args(small_input, mitre_cache, *extra):
    return ["--input", small_input, "--embedder", "lexical", "--llm", "none", "--min-cluster-size", "8", "--min-samples", "4",
            "--mitre-cache-dir", mitre_cache, "--noise-review-top-n", "6", *extra]


def test_help_lists_windows_commands(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert e.value.code == 0
    for needle in ("--input", "--input-format", "--export-dir", "--neo4j", "--ollama-model", "--rebuild-embeddings",
                   "update_mitre_corpus.py", "rebuild_technique_embeddings.py", "python run_pipeline.py --help"):
        assert needle in out


def test_full_cli_run_prints_ten_numbered_stages_ascii_only(small_input, mitre_cache, capsys):
    rc = cli.main(args(small_input, mitre_cache))
    out = capsys.readouterr().out
    assert rc == 0
    pos = [out.index(f"[{i}/10]") for i in range(1, 11)]
    assert pos == sorted(pos)
    assert out.count("      Status:") == 10 and "Elapsed:" in out and "What this stage did:" in out
    assert "RUN SUMMARY" in out and "EVIDENCE VALIDATION: passed" in out and "none (default; use --export-dir)" in out
    out.encode("ascii")                                                        # safe for cp437/cp1252 consoles
    assert "noise_events: " in out and "clusters: " in out


def test_cli_values_come_from_the_run_not_from_templates(small_input, mitre_cache, capsys):
    cli.main(args(small_input, mitre_cache, "--json"))
    data = json.loads(capsys.readouterr().out)
    assert data["parsing"]["valid_records"] == 400 and data["ok"] is True
    assert len(data["stages"]) == 10 and data["clustering"]["noise_events"] > 0


def test_cli_failure_exit_code_and_message(tmp_path, mitre_cache, capsys):
    rc = cli.main(["--input", str(tmp_path / "missing.jsonl"), "--embedder", "lexical", "--mitre-cache-dir", mitre_cache])
    out = capsys.readouterr().out
    assert rc == 1 and "[FAIL]" in out and "not found" in out and out.count("Status: [SKIP]") == 9


def test_cli_rejects_bad_thresholds(capsys):
    assert cli.main(["--weak-threshold", "0.6", "--candidate-threshold", "0.3"]) == 2
    assert "weak-threshold" in capsys.readouterr().err
    assert cli.main(["--top-k", "0"]) == 2


def test_cli_export_flag_writes_files(small_input, mitre_cache, tmp_path, capsys):
    out_dir = tmp_path / "exp"
    assert cli.main(args(small_input, mitre_cache, "--export-dir", str(out_dir), "--quiet", "--no-report")) == 0
    assert (out_dir / "run_result.json").exists() and (out_dir / "evidence_report.md").exists()


def test_render_stage_handles_nasty_text():
    st = StageResult(number=1, name="Input validation and parsing", status="WARN", warnings=["bad \u202e name \x07 \u00e9"],
                     metrics={"path": "C:\\x\u202e.scr", "valid_records": 3})
    text = render_stage(st)
    text.encode("ascii")
    assert "\\u202e" in text or "?" in text and "\u202e" not in text


@pytest.mark.parametrize("script", ["run_pipeline.py", os.path.join("scripts", "run_pipeline.py")])
def test_entry_points_run_as_scripts(script):
    r = subprocess.run([sys.executable, os.path.join(ROOT, script), "--help"], capture_output=True, text=True, cwd="/")
    assert r.returncode == 0 and "SUBHANY" in r.stdout


@pytest.mark.parametrize("script", ["update_mitre_corpus.py", "rebuild_technique_embeddings.py", "evaluate_mapping.py"])
def test_helper_scripts_have_help(script):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", script), "--help"], capture_output=True, text=True, cwd="/")
    assert r.returncode == 0 and "usage" in r.stdout.lower()


def test_run_result_keys_are_stable_for_a_future_dashboard(small_run):
    d = small_run.to_dict()
    assert {"run_id", "metadata", "config", "stages", "parsing", "normalization", "graph", "gae", "clustering", "novelty",
            "mapping", "graphrag", "ontology", "units", "noise", "report", "warnings", "errors", "ok"} <= set(d)
    assert {"number", "name", "status", "elapsed_s", "explanation", "metrics", "warnings", "error"} <= set(d["stages"][0])
