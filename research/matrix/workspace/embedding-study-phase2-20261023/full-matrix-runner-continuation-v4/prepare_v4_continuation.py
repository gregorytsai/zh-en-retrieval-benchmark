#!/usr/bin/env python3
"""Strictly re-audit frozen B01/B02 sources and seed a new v4 sequence output.

This is an offline preparation step: it does not launch a server or inference.
It keeps raw v3 runs immutable, writes a derived B02 combined ranking into the
new output root, and seeds the controller to continue at B03.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
V3_SOURCE = PROJECT / "full-matrix-runner-continuation-v3"
V3_OUTPUT = PROJECT / "run-full-matrix-continuation-v3"
FREEZE = PROJECT / "CONTINUATION_V4_FREEZE_MANIFEST.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stable_summary(summary: dict) -> dict:
    return {key: summary.get(key) for key in (
        "status", "batch", "embedder", "dataset_ids", "summaries", "rows", "errors",
        "accepted_gpu_attempts", "candidate_hashes_verified")}


def prepare(workspace: Path, output: Path, plan: Path) -> dict:
    workspace = workspace.resolve()
    output = output.resolve()
    plan = plan.resolve()
    if not FREEZE.is_file():
        raise RuntimeError(f"Frozen v4 manifest is required: {FREEZE}")
    freeze = read(FREEZE)
    if freeze.get("status") != "v4_frozen_before_resume":
        raise RuntimeError("v4 freeze manifest is not in a launchable frozen state")
    for name, expected in freeze.get("runtime_sha256", {}).items():
        path = HERE / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen v4 source hash mismatch: {name}")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite nonempty v4 output: {output}")
    if not (V3_OUTPUT / "sequence-status.json").is_file():
        raise RuntimeError("Preserved v3 sequence-status.json is missing")
    old_state = read(V3_OUTPUT / "sequence-status.json")
    if not (old_state.get("phase") == "stopped" and old_state.get("batch") == "B02-B" and
            old_state.get("error") == "'reranker'" and
            old_state.get("progress", {}).get("phase") == "complete"):
        raise RuntimeError("v3 does not match the expected post-audit ranking serialization failure")

    # Load the old auditor from its immutable source snapshot. That auditor
    # checks the exact v3 measurement-code hashes used by accepted commits.
    sys.path.insert(0, str(V3_SOURCE))
    import batch_progress as v3_progress  # noqa: E402
    import run_sequence as v3_sequence  # noqa: E402
    from matrix_spec import RUN_BATCHES  # noqa: E402

    # Measurement and guard code must be byte-identical across versions; only
    # the reporting/controller layer is repaired in v4.
    measurement_modules = (
        "core.py", "guarded_engine.py", "matrix_spec.py", "run_full_matrix.py",
        "traffic_guard.py", "server_helper.py", "pilot_forecast.py",
    )
    for name in measurement_modules:
        if sha256_file(V3_SOURCE / name) != sha256_file(HERE / name):
            raise RuntimeError(f"Measurement/guard module changed across v3/v4: {name}")

    # Strictly validate B01 against its original manifest/audit, current data,
    # binary, models, source snapshot, committed rows, and candidate evidence.
    output.mkdir(parents=True, exist_ok=True)
    b01_carryover = v3_sequence.validate_b01_carryover(
        PROJECT, workspace, output)

    # Re-audit every accepted B02 subbatch using the exact v3 audit code. This
    # reads source artifacts only; it does not write into the frozen v3 output.
    subbatch_reports = {}
    subbatch_sources = []
    for batch in ("B02-A", "B02-B"):
        home = V3_OUTPUT / batch / "segment-01" / "official"
        stored_audit = read(home / "subbatch-completion-audit.json")
        fresh_audit = v3_progress.audit_completed(batch, home, workspace)
        if stable_summary(stored_audit) != stable_summary(fresh_audit):
            raise RuntimeError(f"Stored and freshly recomputed {batch} audit differ")
        subbatch_reports[batch] = fresh_audit
        subbatch_sources.append({
            "batch": batch,
            "run": str(home.resolve()),
            "stored_audit_sha256": sha256_file(home / "subbatch-completion-audit.json"),
            "manifest_sha256": sha256_file(home / "run-manifest.json"),
            "committed_sha256": sha256_file(home / "committed.json"),
            "summary_sha256": sha256_file(home / "summary.json"),
            "rows_sha256": sha256_file(home / "query-results.jsonl"),
            "traffic_sha256": sha256_file(home / "traffic-events.jsonl"),
            "fresh_audit": fresh_audit,
        })

    combined_source = V3_OUTPUT / "B02" / "combined"
    old_combined_audit = read(combined_source / "batch-completion-audit.json")
    if (old_combined_audit.get("status"), old_combined_audit.get("summaries"),
            old_combined_audit.get("rows"), old_combined_audit.get("errors"),
            old_combined_audit.get("accepted_gpu_attempts"),
            old_combined_audit.get("candidate_hashes_verified")) != (
            "passed", 24, 10452, 0, 24, True):
        raise RuntimeError("Stored B02 combined audit is not complete/eligible")
    for source in old_combined_audit.get("split_sources", []):
        source_home = Path(source["run"])
        if any(sha256_file(source_home / name) != source[digest_key] for name, digest_key in (
                ("subbatch-completion-audit.json", "audit_sha256"),
                ("summary.json", "summary_sha256"),
                ("query-results.jsonl", "rows_sha256"))):
            raise RuntimeError(f"B02 combined source hash changed: {source_home}")
    if len(old_combined_audit.get("split_sources", [])) != 2:
        raise RuntimeError("B02 combined audit does not identify exactly two source subbatches")

    # Reconstruct B02 into a new, versioned output root. Symlinks are read-only
    # provenance links to v3 official runs; the aggregator writes only under
    # the new B02/combined directory.
    for batch in ("B02-A", "B02-B"):
        (output / batch).symlink_to(V3_OUTPUT / batch, target_is_directory=True)
    v4_progress = load_module("v4_batch_progress_for_prepare", HERE / "batch_progress.py")
    v4_sequence = load_module("v4_run_sequence_for_prepare", HERE / "run_sequence.py")
    for batch in ("B02-A", "B02-B"):
        official_home = output / batch / "segment-01" / "official"
        started_epoch = read(official_home / "run-manifest.json")["original_started_epoch"]
        v4_sequence.update_plan(plan, output, batch, "complete",
                                official_home, started_epoch)
    merged = v4_progress.aggregate_b02_splits(output, workspace, plan)
    if not merged or (merged.get("status"), merged.get("summaries"), merged.get("rows"), merged.get("errors")) != (
            "passed", 24, 10452, 0):
        raise RuntimeError("v4 could not reconstruct a complete B02 combined audit")

    # Re-check the newly generated merged files before allowing sequence resume.
    combined = output / "B02" / "combined"
    summary = read(combined / "summary.json")
    rows = [json.loads(line) for line in (combined / "query-results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    key = lambda row: (row["dataset"], row["embedding"], row["reranker"], str(row["query_id"]))
    if len(summary) != 24 or len(rows) != 10452 or len({key(row) for row in rows}) != 10452:
        raise RuntimeError("v4 B02 combined rows are incomplete or duplicated")
    old_ranking = combined_source / "ranking.json"
    if old_ranking.is_file() and sha256_file(combined / "ranking.json") == sha256_file(old_ranking):
        # Normally v3 has no ranking file. If one exists, the new version must
        # produce its own file rather than silently carrying it forward.
        raise RuntimeError("Unexpected reuse of a v3 ranking artifact")

    carryover_audit = {
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "offline strict re-audit; no inference or rerun",
        "v3_source_root": str(V3_OUTPUT.resolve()),
        "v3_source_sequence_sha256": sha256_file(V3_OUTPUT / "sequence-status.json"),
        "v3_combined_b02_audit_sha256": sha256_file(combined_source / "batch-completion-audit.json"),
        "v4_freeze_manifest_sha256": sha256_file(FREEZE),
        "measurement_modules_unchanged_sha256": {name: sha256_file(HERE / name) for name in measurement_modules},
        "b01": {
            "status": "passed", "summaries": 24, "rows": 10452,
            "carryover_manifest_sha256": sha256_file(output / "carryover-manifest.json"),
            "source": b01_carryover["batches"]["B01"],
        },
        "b02_subbatches": subbatch_sources,
        "b02_rebuilt_combined": {
            "status": "passed", "summaries": 24, "rows": 10452,
            "unique_rows": 10452, "errors": 0,
            "accepted_gpu_attempts": 24, "candidate_hashes_verified": True,
            "summary_sha256": sha256_file(combined / "summary.json"),
            "rows_sha256": sha256_file(combined / "query-results.jsonl"),
            "ranking_sha256": sha256_file(combined / "ranking.json"),
            "audit_sha256": sha256_file(combined / "batch-completion-audit.json"),
            "source_subbatches": [str((output / name).resolve()) for name in ("B02-A", "B02-B")],
            "ranking_source": "recomputed from the preserved v3 B02-A/B accepted summaries; not a measurement",
        },
        "combined_official_totals": {"summaries": 48, "rows": 20904, "unique_rows": 20904, "errors": 0},
        "excluded": {"pilot_rows": True, "v3_partial_attempts": True},
    }
    dump(output / "v4-carryover-audit.json", carryover_audit)

    # Snapshot the exact launchable source and seed only already audited official
    # batches. B03 remains the next and only batch the controller may launch.
    snapshot = output / "controller-source-snapshot"
    snapshot.mkdir()
    for source in sorted(HERE.glob("*.py")):
        shutil.copy2(source, snapshot / source.name)
    from matrix_spec import RUN_BATCHES as V4_RUN_BATCHES  # noqa: E402
    state = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "stopped",
        "resume_reason": "v4 ranking/plan-progress fix; strict B01+B02 official carryover passed",
        "completed_batches": ["B01", "B02"],
        "completed_subbatches": ["B02-A", "B02-B"],
        "schedule": [list(item) for item in V4_RUN_BATCHES],
        "carryover_manifest": str((output / "carryover-manifest.json").resolve()),
        "carryover_verified": True,
        "carryover_sha256": sha256_file(output / "carryover-manifest.json"),
        "v4_carryover_audit": str((output / "v4-carryover-audit.json").resolve()),
        "v4_carryover_audit_sha256": sha256_file(output / "v4-carryover-audit.json"),
        "batch": "B02", "embedder": "qwen-8b", "stage": "official", "segment": "01",
        "accepted_summaries": 24, "accepted_rows": 10452,
        "progress": {"phase": "complete", "summaries": 24, "query_rows": 10452, "errors": 0},
        "run_dir": str(combined.resolve()),
        "controller_pid": None, "child_pid": None, "deadline_epoch": None,
    }
    dump(output / "sequence-status.json", state)
    return {
        "status": "prepared_for_b03",
        "output": str(output),
        "b01": "24 summaries / 10452 rows validated and reused",
        "b02": "24 summaries / 10452 rows freshly re-audited and ranking rebuilt offline",
        "official_carryover_totals": {"summaries": 48, "rows": 20904, "unique_rows": 20904, "errors": 0},
        "next_batch": "B03",
        "source_snapshot": str(snapshot),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="Workspace root containing llama.cpp/models")
    parser.add_argument("--output", type=Path, required=True, help="New v4 output root; must be empty or absent")
    parser.add_argument("--plan", type=Path, required=True, help="Updated full-matrix plan file")
    args = parser.parse_args()
    print(json.dumps(prepare(args.workspace, args.output, args.plan), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
