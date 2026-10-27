#!/usr/bin/env python3
"""Bounded continuation controller for B02-A/B and B03-B07.

B01 is reused only after verifying its stored completion audit and immutable
manifest. Each new pilot has a fixed 2h budget; each official segment has its
own immutable 12h cap. A failed/incomplete stage stops the sequence and keeps
committed work; this controller never loops retries or resets deadlines.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from matrix_spec import BATCHES, RUN_BATCHES, BATCH_DATASETS, DATASET_COUNTS, RERANKERS

HERE = Path(__file__).resolve().parent
CHILD = None


def read(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def validate_b01_carryover(project: Path, workspace: Path, output: Path) -> dict:
    """Reuse only B01's already-passed full completion audit; never use pilot rows."""
    from matrix_spec import MODEL_REGISTRY, sha256_json
    from run_full_matrix import _combined_dataset_sha256

    home = project / "run-full-matrix" / "B01" / "official"
    audit_path = home / "batch-completion-audit.json"
    manifest_path = home / "run-manifest.json"
    summary_path = home / "summary.json"
    rows_path = home / "query-results.jsonl"
    for path in (audit_path, manifest_path, summary_path, rows_path,
                 home / "committed.json", home / "progress.json"):
        if not path.is_file():
            raise RuntimeError(f"B01 carryover evidence is missing: {path}")
    audit, manifest = read(audit_path), read(manifest_path)
    if (audit.get("status"), audit.get("batch"), audit.get("embedder"),
            audit.get("summaries"), audit.get("rows"), audit.get("errors"),
            audit.get("accepted_gpu_attempts"), audit.get("candidate_hashes_verified")) != (
            "passed", "B01", "qwen-4b", 24, 10452, 0, 24, True):
        raise RuntimeError("B01 completion audit does not qualify for strict carryover")
    if manifest.get("run_kind") != "official" or manifest.get("batch_id") != "B01":
        raise RuntimeError("B01 source is not the audited official run")
    if manifest.get("dataset_ids") != list(DATASET_COUNTS):
        raise RuntimeError("B01 source does not cover the frozen six-dataset order")
    if manifest.get("dataset_sha256") != _combined_dataset_sha256(
            project / "datasets.json", project / "datasets-batch2.json"):
        raise RuntimeError("B01 source data manifest hash differs from the frozen inputs")
    expected_models = {name: spec.sha256 for name, spec in MODEL_REGISTRY.items()}
    if manifest.get("model_sha256") != expected_models:
        raise RuntimeError("B01 source model pins differ from the current frozen registry")
    if manifest.get("binary_sha256") != "5f7bee49eaf08bf77b3cc523d88d385750d6b7da8479775b32bc9be030464ae5":
        raise RuntimeError("B01 source binary pin differs from the frozen llama-server")
    if manifest.get("effective_configs") != [
            "qwen-4b__standalone", "qwen-4b__bge-reranker",
            "qwen-4b__qwen-reranker-0.6b", "qwen-4b__qwen-reranker-4b"]:
        raise RuntimeError("B01 source settings/configuration differ from the continuation matrix")
    progress = read(home / "progress.json", {})
    if progress.get("phase") != "complete" or progress.get("summaries") != 24 or progress.get("query_rows") != 10452 or progress.get("errors") != 0:
        raise RuntimeError("B01 progress does not agree with its passed completion audit")
    summary = read(summary_path)
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    key = lambda row: (row.get("dataset"), row.get("embedding"), row.get("reranker"), str(row.get("query_id")))
    if len(summary) != 24 or len(rows) != 10452 or len({key(row) for row in rows}) != 10452:
        raise RuntimeError("B01 main summary/row keys fail carryover uniqueness checks")
    if any(row.get("status") != "ok" or row.get("embedding") != "qwen-4b" for row in rows):
        raise RuntimeError("B01 carryover contains failed rows or a different embedder")

    snapshot = home / "source-snapshot"
    for name, digest in manifest.get("code_sha256", {}).items():
        path = snapshot / name
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"B01 frozen source snapshot mismatch: {name}")
    entry = {
        "status": "passed", "batch": "B01", "embedder": "qwen-4b",
        "run": str(home.resolve()), "summaries": 24, "rows": 10452,
        "unique_rows": 10452, "errors": 0, "accepted_gpu_attempts": 24,
        "candidate_hashes_verified": True,
        "manifest_sha256": sha256_file(manifest_path),
        "audit_sha256": sha256_file(audit_path),
        "summary_sha256": sha256_file(summary_path),
        "rows_sha256": sha256_file(rows_path),
        "dataset_sha256": manifest["dataset_sha256"],
        "binary_sha256": manifest["binary_sha256"],
        "model_sha256": manifest["model_sha256"],
        "source_code_sha256": manifest["code_sha256"],
        "started": manifest["original_started_epoch"],
        "finished": progress["time"],
        "eligibility_scope": "B01 only; reused from the original audit-passed official run",
    }
    old_pilot = project / "run-full-matrix" / "B02" / "pilot" / "pilot-decision.json"
    if old_pilot.is_file():
        decision = read(old_pilot)
        entry["excluded_pilot_evidence"] = {
            "run": str(old_pilot.parent.resolve()), "status": decision.get("status"),
            "eligible": decision.get("eligible"),
            "forecast_hours": (decision.get("forecast") or {}).get("conservative_official_hours"),
            "excluded_from_official_rows": True,
        }
    carry = {"schema_version": 1, "created_at": stamp(), "batches": {"B01": entry}}
    target = output / "carryover-manifest.json"
    if target.exists() and read(target) != carry:
        # Timestamps differ on a controller resume; compare stable fields rather than rewrite provenance.
        previous = read(target)
        if previous.get("batches", {}).get("B01") != entry:
            raise RuntimeError("Existing continuation carryover manifest differs from current B01 evidence")
        return previous
    dump(target, carry)
    return carry


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _official_homes(output: Path, batch: str) -> list[Path]:
    return sorted((output / batch).glob("segment-*/official"), key=lambda path: path.parent.name)


def _latest_official_home(output: Path, batch: str) -> Path:
    homes = _official_homes(output, batch)
    return homes[-1] if homes else output / batch / "segment-01" / "official"


def _progress_row_indices(lines: list[str], batch_name: str) -> list[int]:
    """Find a batch only in the plan's progress table, not scope tables above it."""
    header = next((i for i, line in enumerate(lines)
                  if line.strip().startswith("| 批次 | Embedder ID | 狀態 | 摘要 |")), None)
    if header is None:
        # Small unit-test plans may contain only the progress table without a header.
        candidate_indices = range(len(lines))
    else:
        candidate_indices = []
        for index in range(header + 1, len(lines)):
            line = lines[index].strip()
            if not line:
                break
            if not line.startswith("|"):
                break
            candidate_indices.append(index)
    return [i for i in candidate_indices
            if lines[i].strip().startswith(f"| {batch_name} |")]


def update_plan(plan: Path, output: Path, batch: str, state: str, home: Path, started=None):
    """Show official accepted work only; pilots never increment matrix counts."""
    from matrix_spec import BATCHES
    if batch in ("B02-A", "B02-B"):
        from matrix_spec import BATCH_DATASETS, DATASET_COUNTS, RUN_BATCHES
        batch_name = batch
        embedder = dict(RUN_BATCHES)[batch]
        expected_summaries = len(BATCH_DATASETS[batch]) * 4
        expected_rows = sum(DATASET_COUNTS[name]["queries"] for name in BATCH_DATASETS[batch]) * 4
        official_home = _latest_official_home(output, batch)
        committed = {} if state in ("pilot", "pilot_complete") else read(official_home / "committed.json", {})
        summaries = len(committed)
        rows = sum(int(item.get("queries", 0)) for item in committed.values())
        audit_path = official_home / "subbatch-completion-audit.json"
        audit_passed = audit_path.is_file() and read(audit_path).get("status") == "passed"
        if audit_passed:
            label, summaries, rows = "稽核通過（B02子批）", expected_summaries, expected_rows
        elif state == "pilot":
            label, summaries, rows = "Pilot驗證中（正式0）", 0, 0
        elif state == "pilot_complete":
            label, summaries, rows = "Pilot稽核通過；正式待啟動", 0, 0
        elif state == "partial_budget_exhausted":
            label = "保留部分提交；期限到，待相容續段"
        else:
            label = {"official": "正式執行中", "blocked": "保留部分提交，待檢查",
                     "complete": "稽核通過（B02子批）"}.get(state, state)
        run_path = str(home.resolve())
    else:
        path = _latest_official_home(output, batch)
        committed = read(path / "committed.json", {}) if state != "pilot" else {}
        summaries = len(committed)
        rows = sum(int(item.get("queries", 0)) for item in committed.values())
        label = {"pilot": "Pilot 驗證中（正式0）", "official": "正式執行中",
                 "blocked": "保留部分提交，待稽核/續批", "complete": "稽核通過"}.get(state, state)
        run_path = str(home)
        batch_name = batch
        embedder = dict(BATCHES).get(batch)
        if embedder is None:
            embedder = dict(RUN_BATCHES).get(batch)
        if embedder is None:
            raise RuntimeError(f"Unknown batch in progress updater: {batch}")
        audit = path / "batch-completion-audit.json"
        if audit.is_file() and read(audit, {}).get("status") == "passed":
            label, summaries, rows = "稽核通過", 24, 10452
    timestamp = datetime.fromtimestamp(started, ZoneInfo("Asia/Taipei")).strftime("%m/%d %H:%M") if started else "—"
    with plan.with_suffix(".progress.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lines = plan.read_text(encoding="utf-8").splitlines()
        matches = _progress_row_indices(lines, batch_name)
        if len(matches) != 1:
            raise RuntimeError(f"Progress table must contain exactly one {batch_name} row")
        if batch in ("B02-A", "B02-B"):
            from matrix_spec import BATCH_DATASETS, DATASET_COUNTS
            expected_summaries = len(BATCH_DATASETS[batch]) * 4
            expected_rows = sum(DATASET_COUNTS[name]["queries"] for name in BATCH_DATASETS[batch]) * 4
        else:
            expected_summaries, expected_rows = 24, 10452
        cells = [cell.strip() for cell in lines[matches[0]].strip().strip("|").split("|")]
        if (len(cells) >= 5 and cells[2] in ("稽核通過", "稽核通過（B02子批）") and
                cells[3] == f"{expected_summaries}/{expected_summaries}" and
                cells[4] == f"{expected_rows}/{expected_rows}"):
            return
        lines[matches[0]] = (f"| {batch_name} | {embedder} | {label} | "
                             f"{summaries}/{expected_summaries} | {rows}/{expected_rows} | "
                             f"{timestamp} 起 | {run_path} |")
        temp = plan.with_suffix(".progress.tmp")
        temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temp.replace(plan)


def stage_home(output: Path, batch: str, kind: str, segment: str = "01") -> Path:
    return output / batch / f"segment-{segment}" / kind


def stage_command(args, batch, kind, resume, segment="01", continue_from=None, pilot_evidence=None):
    command = [sys.executable, str(HERE / "run_full_matrix.py"), "--workspace", str(args.workspace),
               "--output", str(args.output), "--batch", batch, "--execute",
               "--plan-path", str(args.plan), "--segment", str(segment)]
    if kind == "pilot":
        command.append("--pilot")
    if continue_from is not None:
        command.extend(["--continue-compatible-from", str(continue_from)])
    if pilot_evidence is not None and kind == "official":
        command.extend(["--pilot-evidence", str(pilot_evidence)])
    if resume:
        command.append("--resume")
    return command


def stop_owned_child(child: subprocess.Popen, grace_seconds: int = 30) -> None:
    """Stop only the process group created for this controller-owned stage."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=10)


def stop(signum, _frame):
    global CHILD
    if CHILD is not None and CHILD.poll() is None:
        stop_owned_child(CHILD)
        CHILD = None
    raise SystemExit(128 + signum)


def run_stage(args, batch, kind, state, segment="01", continue_from=None, pilot_evidence=None):
    global CHILD
    home = stage_home(args.output, batch, kind, segment)
    progress = read(home / "progress.json", {})
    audit_name = "subbatch-completion-audit.json" if batch in ("B02-A", "B02-B") else "batch-completion-audit.json"
    if kind == "official" and read(home / audit_name, {}).get("status") == "passed":
        return {"status": "complete", "home": home,
                "committed": read(home / "committed.json", {})}
    if kind == "pilot" and progress.get("phase") == "pilot_complete":
        decision = read(home / "pilot-decision.json", {})
        if decision.get("evidence_audit_passed") is True:
            return
        raise RuntimeError(f"{batch}/pilot completed but its evidence audit is not clean")
    manifest = read(home / "run-manifest.json")
    resume = manifest is not None
    if resume and time.time() >= float(manifest["deadline_epoch"]):
        if (kind == "official" and progress.get("phase") == "stopped" and
                (home / "fatal-errors.jsonl").is_file()):
            committed = read(home / "committed.json", {})
            if committed:
                state.update(batch=batch, stage=kind, segment=segment, phase="partial_budget_exhausted",
                             accepted_summaries=len(committed), accepted_rows=sum(int(v.get("queries", 0)) for v in committed.values()),
                             deadline_epoch=manifest["deadline_epoch"], run_dir=str(home))
                update_plan(args.plan, args.output, batch, "partial_budget_exhausted", home,
                            manifest.get("original_started_epoch"))
                return {"status": "partial_budget_exhausted", "committed": committed,
                        "home": home, "deadline_epoch": manifest["deadline_epoch"]}
        raise TimeoutError(f"{batch}/{kind} original deadline expired; a new authorized segment is required")
    if manifest and progress.get("phase") == "blocked_preflight":
        raise RuntimeError(f"{batch}/{kind} blocked by token preflight; preserve artifacts and inspect")
    command = stage_command(args, batch, kind, resume, segment, continue_from, pilot_evidence)
    state.update(batch=batch, embedder=dict(RUN_BATCHES)[batch], stage=kind, segment=segment,
                 phase="launching", updated_at=stamp(), run_dir=str(home),
                 accepted_summaries=len(read(home / "committed.json", {})),
                 fixed_budget_hours=2 if kind == "pilot" else 12,
                 continuation_source=str(continue_from) if continue_from else None)
    dump(args.output / "sequence-status.json", state)
    # Validate/update the intended progress row before launching any child.
    # A bad plan must never leave an unmonitored runner process behind.
    manifest_started = manifest.get("original_started_epoch") if manifest else None
    update_plan(args.plan, args.output, batch, kind, home, manifest_started)
    with (args.output / f"{batch}-{kind}-console.log").open("a", encoding="utf-8") as log:
        CHILD = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, close_fds=True,
                                 start_new_session=True)
        state.update(phase="running", child_pid=CHILD.pid, updated_at=stamp())
        dump(args.output / "sequence-status.json", state)
        last_doc_update = 0.0
        try:
            while CHILD.poll() is None:
                progress = read(home / "progress.json", {})
                manifest = read(home / "run-manifest.json", {})
                committed = read(home / "committed.json", {})
                state.update(progress=progress, deadline_epoch=manifest.get("deadline_epoch"),
                             accepted_summaries=len(committed),
                             accepted_rows=sum(int(item.get("queries", 0)) for item in committed.values()),
                             updated_at=stamp())
                dump(args.output / "sequence-status.json", state)
                if time.monotonic() - last_doc_update >= 30:
                    update_plan(args.plan, args.output, batch, kind, home, manifest.get("original_started_epoch"))
                    last_doc_update = time.monotonic()
                time.sleep(5)
        except BaseException:
            # The child is this controller's own run_full_matrix process. Ask
            # its SIGTERM handler to unwind the engine; the dedicated process
            # group contains only this controller-owned runner/server.
            if CHILD is not None and CHILD.poll() is None:
                stop_owned_child(CHILD)
            raise
        finally:
            if CHILD is not None:
                code = CHILD.wait() if CHILD.poll() is None else CHILD.returncode
                CHILD = None
            else:
                code = 1
    if code:
        progress = read(home / "progress.json", {})
        deadline = read(home / "deadline.json", {})
        if deadline.get("deadline_epoch") and time.time() >= float(deadline["deadline_epoch"]):
            stage_state = "partial_budget_exhausted"
        else:
            stage_state = "blocked"
        update_plan(args.plan, args.output, batch, stage_state, home,
                    read(home / "run-manifest.json", {}).get("original_started_epoch"))
        if stage_state == "partial_budget_exhausted" and kind == "official":
            committed = read(home / "committed.json", {})
            state.update(phase=stage_state, accepted_summaries=len(committed),
                         accepted_rows=sum(int(value.get("queries", 0)) for value in committed.values()),
                         updated_at=stamp())
            dump(args.output / "sequence-status.json", state)
            return {"status": stage_state, "committed": committed,
                    "home": home, "deadline_epoch": deadline.get("deadline_epoch")}
        raise RuntimeError(f"{batch}/{kind} exited {code}; status={stage_state}; committed evidence preserved")
    terminal = read(home / "progress.json", {}).get("phase")
    if terminal != ("pilot_complete" if kind == "pilot" else "complete"):
        raise RuntimeError(f"{batch}/{kind} has no expected completion marker: {terminal}")
    if kind == "official" and not (home / audit_name).is_file():
        raise RuntimeError(f"{batch} official has no successful completion audit")
    update_plan(args.plan, args.output, batch, "pilot_complete" if kind == "pilot" else "complete",
                home, read(home / "run-manifest.json", {}).get("original_started_epoch"))
    return {"status": "complete", "home": home,
            "committed": read(home / "committed.json", {})}


def run_batch_stages(args, batch: str, state: dict) -> Path:
    """Run one fixed pilot and, only on deadline exhaustion, capped continuation segments."""
    first_pilot = stage_home(args.output, batch, "pilot", "01")
    run_stage(args, batch, "pilot", state, "01")
    decision_path = first_pilot / "pilot-decision.json"
    decision = read(decision_path, {})
    if decision.get("evidence_audit_passed") is not True:
        raise RuntimeError(f"{batch} pilot evidence audit did not pass")

    segment_number = 1
    previous_home = None
    previous_accepted = 0
    while True:
        segment = f"{segment_number:02d}"
        result = run_stage(args, batch, "official", state, segment,
                           continue_from=previous_home,
                           pilot_evidence=decision_path)
        if result["status"] == "complete":
            return result["home"]
        imported_or_accepted = result["committed"]
        accepted = len(imported_or_accepted)
        if previous_home is not None and accepted <= previous_accepted:
            raise RuntimeError(f"{batch} segment {segment} exhausted its budget without new accepted work; stopping to avoid no-progress retries")
        if segment_number == 1 and accepted == 0:
            raise RuntimeError(f"{batch} official segment reached its deadline with no accepted work; stopping for review")
        previous_home = result["home"]
        previous_accepted = accepted
        segment_number += 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--execute", action="store_true", help="Explicitly authorize the guarded serial sequence")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("run_sequence is live; pass --execute explicitly after offline validation")
    args.workspace = args.workspace.resolve()
    project = args.workspace / "embedding-study-phase2-20261023"
    args.output = (args.output or project / "run-full-matrix-continuation-v4").resolve()
    args.plan = (args.plan or project / "FULL_MATRIX_PLAN_ZH_TW.md").resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (project / ".full-matrix-sequence.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another full-matrix sequence controller is active") from exc
        previous = read(args.output / "sequence-status.json")
        if previous and not args.resume:
            raise RuntimeError("Continuation state already exists; use --resume without changing any stage deadline")
        if previous and previous.get("child_pid") and previous.get("phase") == "running":
            try:
                os.kill(int(previous["child_pid"]), 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("Previous continuation runner may still be alive; do not start a duplicate")
        carry = validate_b01_carryover(project, args.workspace, args.output)
        state = previous or {
            "created_at": stamp(), "completed_batches": ["B01"],
            "completed_subbatches": [], "schedule": list(RUN_BATCHES),
            "carryover_manifest": str((args.output / "carryover-manifest.json").resolve()),
            "carryover_verified": True,
        }
        if "B01" not in state.get("completed_batches", []):
            state.setdefault("completed_batches", []).insert(0, "B01")
        state.update(controller_pid=os.getpid(), phase="running", updated_at=stamp(),
                     carryover_verified=True, carryover_sha256=sha256_file(args.output / "carryover-manifest.json"))
        dump(args.output / "sequence-status.json", state)
        try:
            for batch, _embedder in RUN_BATCHES:
                if batch in state.get("completed_subbatches", []):
                    continue
                official_home = run_batch_stages(args, batch, state)
                if batch == "B02-B":
                    from batch_progress import aggregate_b02_splits
                    combined = aggregate_b02_splits(args.output, args.workspace, args.plan)
                    if not combined:
                        raise RuntimeError("B02 split official runs finished but their combined audit did not pass")
                    state.setdefault("completed_batches", []).append("B02")
                elif batch not in ("B02-A",):
                    state.setdefault("completed_batches", []).append(batch)
                state.setdefault("completed_subbatches", []).append(batch)
                state.update(batch_complete=batch, current_official_run=str(official_home),
                             updated_at=stamp(), phase="batch_complete")
                dump(args.output / "sequence-status.json", state)
            if "B02" not in state.get("completed_batches", []):
                raise RuntimeError("B02 requires both B02-A and B02-B accepted/audited before completion")
            from batch_progress import aggregate_global
            report = aggregate_global(args.output, args.workspace, args.output / "carryover-manifest.json")
            state.update(phase="complete", updated_at=stamp(), expected_summaries=168,
                         expected_rows=73164, global_audit=report)
            dump(args.output / "sequence-status.json", state)
        except BaseException as exc:
            state.update(phase="stopped", error=str(exc), updated_at=stamp())
            dump(args.output / "sequence-status.json", state)
            raise


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    main()
