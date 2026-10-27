#!/usr/bin/env python3
"""Fail-closed Phase 2 runner with one global lock and immutable wall deadline."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import platform
import shutil
import signal
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import core
from traffic_guard import Interference, TrafficGuard


PHASE2_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PHASE2_DIR / "run-phase2"
DEFAULT_DATASETS = PHASE2_DIR / "datasets.json"
GLOBAL_LOCK = PHASE2_DIR.parent / ".phase2-benchmark.lock"
EXPECTED_QUERY_ROWS = 10452
EXPECTED_SUMMARIES = 24
DEFAULT_WALL_HOURS = 12.0


class PreflightBlockedError(RuntimeError):
    pass


class DeadlineWatchdog:
    """Kill only the currently bound runner-owned server at the original deadline."""
    def __init__(self, deadline: float, period: float = 0.2):
        self.deadline = deadline
        self.period = period
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.server = None
        self.terminating_process = None
        self.terminate_started = None
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def bind(self, server):
        with self.lock:
            self.server = server

    def unbind(self):
        with self.lock:
            self.server = None

    def loop(self):
        while not self.stop_event.wait(self.period):
            with self.lock:
                now = time.time()
                process = self.server.process if self.server is not None else self.terminating_process
                if now >= self.deadline and process is not None and process.poll() is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                    if self.terminating_process is not process:
                        self.terminating_process = process
                        self.terminate_started = now
                if (self.terminating_process is not None and self.terminating_process.poll() is None and
                        self.terminate_started is not None and now - self.terminate_started >= 2):
                    try:
                        self.terminating_process.kill()
                    except ProcessLookupError:
                        pass

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5)


class GuardedStudy(core.Study):
    def __init__(self, *args, guard: TrafficGuard, watchdog: DeadlineWatchdog, **kwargs):
        super().__init__(*args, **kwargs)
        self.guard = guard
        self.watchdog = watchdog
        # core.Study stores its callback as an instance attribute named
        # `check_time`, which otherwise shadows this subclass implementation.
        self.check_time = self.check_deadline_and_guard

    def check_deadline_and_guard(self):
        if time.time() >= self.deadline:
            raise TimeoutError("Original 12-hour wall-clock deadline reached, including traffic pauses")
        self.guard.check()

    @contextmanager
    def server(self, name: str, model: Path, mode: str, pooling: str = "last"):
        self.check_time()
        server = core.Server(model, mode, name, self.deadline, pooling)
        self.guard.bind(server)
        self.watchdog.bind(server)
        try:
            with server:
                self.check_time()
                command = server._command(server.port)
                log_path = Path(command[command.index("--log-file") + 1])
                offloads = re.findall(r"offloaded\s+(\d+)/(\d+) layers to GPU",
                                      log_path.read_text(errors="replace"))
                if not offloads or int(offloads[-1][0]) == 0:
                    raise RuntimeError(f"Model has no actual GPU offload evidence: {name}")
                core.append(self.out / "server-events.jsonl", {
                    "time": core.stamp(), "model": name, "load_ms": server.load_ms,
                    "command": command, "gpu_offload": list(map(int, offloads[-1])),
                    "props": core.base.get_json(server.url + "/props"),
                })
                yield server
        finally:
            server.stop()
            self.guard.unbind()
            self.watchdog.unbind()
            core.append(self.out / "server-events.jsonl", {
                "time": core.stamp(), "model": name, "event": "stopped",
                "sampled_peak_rss_mb": server.peak_rss_mb,
            })


def parse_deadline(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("Deadline must include a timezone, for example +00:00")
    return parsed.timestamp()


def parse_args(argv=None):
    raise RuntimeError("Preparation build: use run_full_matrix.py for offline checks. Live entry is disabled pending explicit execution authorization and pilot-gate review.")


@contextmanager
def benchmark_lock():
    GLOBAL_LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = GLOBAL_LOCK.open("a+")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another Phase 2 benchmark or pilot holds the global lock: {GLOBAL_LOCK}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "acquired_at": core.stamp()}) + "\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()


def build_config_ids(args):
    requested = [key for key, value in core.PILOT_CONFIGS.items() if value[0] == args.embedder]
    if len(requested) != 4:
        raise ValueError("One registered embedder batch with four settings is required")
    return requested, list(requested)


def build_dataset_ids(args: argparse.Namespace) -> list[str]:
    from matrix_spec import BATCH_DATASETS
    registered = list(BATCH_DATASETS.get(args.batch_id, ()))
    result = list(getattr(args, "dataset_ids", None) or registered)
    if result != registered:
        raise ValueError("Dataset scope must exactly match the fixed batch/subbatch registry")
    unknown = sorted(set(result) - set(core.DATASET_COUNTS))
    if unknown:
        raise ValueError(f"Unknown pilot dataset IDs {unknown}; allowed: {sorted(core.DATASET_COUNTS)}")
    if len(set(result)) != len(result):
        raise ValueError("Pilot dataset IDs must be unique")
    return result


def build_selection(datasets: dict[str, dict[str, Any]], dataset_ids: list[str],
                    pilot: bool, pilot_queries: int | None) -> dict[str, list[str]]:
    selection = {}
    for name in dataset_ids:
        data = datasets[name]
        if pilot:
            # Historical manifests contain hand-picked pilot IDs. The new
            # matrix uses one deterministic length-quantile rule across all six.
            pilot_data = {k: v for k, v in data.items() if k != "pilot_query_ids"}
            selection[name] = core.select_pilot_ids(pilot_data, requested_n=pilot_queries)
        else:
            selection[name] = [query["id"] for query in data["queries"]]
    return selection


def job_list(dataset_ids: list[str], config_ids: list[str]) -> list[tuple[str, str, str, str]]:
    configs = [core.PILOT_CONFIGS[value] for value in config_ids]
    jobs = []
    for dataset in dataset_ids:
        for embedding, reranker in configs:
            jobs.append((dataset, embedding, reranker, f"{dataset}--{embedding}--{reranker}"))
    # Standalone candidate caches must be committed before reranker jobs.
    return sorted(jobs, key=lambda row: (row[2] != "standalone", row[1], row[0], row[2]))


def model_registry() -> dict[str, Path]:
    return {**{name: spec[0] for name, spec in core.EMBEDDERS.items()}, **core.RERANKERS}


def environment_hashes(dataset_path: Path, dataset_sha256: str) -> dict[str, Any]:
    from matrix_spec import MODEL_REGISTRY
    model_paths = model_registry()
    missing = [str(path) for path in [core.BINARY, *model_paths.values()] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required benchmark materials missing: " + ", ".join(missing))
    code_paths = {
        "guarded_engine.py": Path(__file__).resolve(),
        "matrix_spec.py": PHASE2_DIR / "matrix_spec.py",
        "run_full_matrix.py": PHASE2_DIR / "run_full_matrix.py",
        "batch_progress.py": PHASE2_DIR / "batch_progress.py",
        "run_study.py": Path(core.__file__).resolve(),
        "traffic_guard.py": PHASE2_DIR / "traffic_guard.py",
        "legacy_run_benchmark.py": core.LEGACY_BENCHMARK,
    }
    model_hashes = {name: core.sha256_file(path) for name, path in sorted(model_paths.items())}
    for name, digest in model_hashes.items():
        spec = MODEL_REGISTRY[name]
        if digest != spec.sha256 or model_paths[name].stat().st_size != spec.bytes:
            raise RuntimeError(f"Model pin mismatch: {name}")
    binary_hash = core.sha256_file(core.BINARY)
    if binary_hash != "5f7bee49eaf08bf77b3cc523d88d385750d6b7da8479775b32bc9be030464ae5":
        raise RuntimeError("llama-server binary differs from the approved environment")
    for name in ("pilot_forecast.py", "run_sequence.py"):
        path = PHASE2_DIR / name
        if path.exists():
            code_paths[name] = path
    return {
        "dataset_sha256": dataset_sha256,
        "binary_sha256": binary_hash,
        "model_sha256": model_hashes,
        "code_sha256": {name: core.sha256_file(path) for name, path in sorted(code_paths.items())},
    }


def selection_record(datasets: dict[str, dict[str, Any]], selections: dict[str, list[str]],
                     dataset_ids: list[str], run_kind: str, dataset_sha256: str) -> dict[str, Any]:
    entries = {}
    for name in dataset_ids:
        data = datasets[name]
        by_id = {query["id"]: query for query in data["queries"]}
        entries[name] = {
            "grade_mode": data["grade_mode"], "dataset_metadata": data["metadata"],
            "query_ids": selections[name],
            "queries": [{"id": qid, **core.query_metadata(by_id[qid])} for qid in selections[name]],
        }
    return {"run_kind": run_kind, "dataset_sha256": dataset_sha256, "selection_policy": "all-manifest-queries" if run_kind == "official" else "deterministic-short-mid-long-query-length-quantiles", "datasets": entries}


def qrels_record(datasets: dict[str, dict[str, Any]], selections: dict[str, list[str]],
                 dataset_ids: list[str], dataset_sha256: str) -> dict[str, Any]:
    return {
        "dataset_sha256": dataset_sha256,
        "datasets": {
            name: {
                "grade_mode": datasets[name]["grade_mode"],
                "qrels": {qid: datasets[name]["qrels"][qid] for qid in selections[name]},
            }
            for name in dataset_ids
        },
    }


def cache_integrity(home: Path, model_name: str, dataset_name: str,
                    dataset_sha256: str, model_sha256: str, query_ids: list[str],
                    binary_sha256: str, code_sha256: dict[str, str]) -> None:
    cache_dir = home / "cache" / model_name / dataset_name
    manifest, _ = verified_cache_manifest(cache_dir, query_ids)
    if (manifest.get("dataset_sha256") != dataset_sha256 or manifest.get("model_sha256") != model_sha256 or
            manifest.get("binary_sha256") != binary_sha256 or manifest.get("code_sha256") != code_sha256):
        raise RuntimeError(f"Accepted candidate cache does not match frozen run inputs: {cache_dir}")


def verified_cache_manifest(cache_dir: Path, query_ids: list[str]) -> tuple[dict[str, Any], str]:
    manifest_path = cache_dir / "cache-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Accepted candidate cache is missing: {cache_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("query_ids") != query_ids:
        raise RuntimeError(f"Accepted candidate cache query set differs from the frozen selection: {cache_dir}")
    for filename in ("corpus.npy", "queries.npy", "candidates.json"):
        if not (cache_dir / filename).is_file():
            raise RuntimeError(f"Accepted candidate cache is incomplete: {cache_dir / filename}")
        if manifest.get("files_sha256", {}).get(filename) != core.sha256_file(cache_dir / filename):
            raise RuntimeError(f"Accepted candidate cache file hash mismatch: {cache_dir / filename}")
    cache_spec = {key: value for key, value in manifest.items() if key not in ("cache_key_sha256", "files_sha256")}
    if manifest.get("cache_key_sha256") != core.sha256_json(cache_spec):
        raise RuntimeError(f"Accepted candidate cache key is invalid: {cache_dir}")
    return manifest, core.sha256_file(manifest_path)


def accepted_embedder_cache(home: Path, embedding: str, dataset_name: str, query_ids: list[str],
                          committed: dict[str, Any], dataset_sha256: str,
                          model_hashes: dict[str, str], binary_sha256: str,
                          code_sha256: dict[str, str]) -> str:
    source_key = f"{dataset_name}--{embedding}--standalone"
    if source_key not in committed:
        raise RuntimeError(f"Reranking requires an accepted standalone {embedding} result first: {source_key}")
    standalone_job = (dataset_name, embedding, "standalone", source_key)
    validate_accepted_attempt(home, standalone_job, committed, query_ids)
    standalone_attempt = (home / committed[source_key]["attempt"]).resolve()
    accepted_cache_dir = standalone_attempt / "cache" / embedding / dataset_name
    accepted_manifest, accepted_manifest_sha = verified_cache_manifest(accepted_cache_dir, query_ids)
    cache_integrity(home, embedding, dataset_name, dataset_sha256,
                    model_hashes[embedding], query_ids, binary_sha256, code_sha256)
    run_manifest, run_manifest_sha = verified_cache_manifest(
        home / "cache" / embedding / dataset_name, query_ids)
    if accepted_manifest != run_manifest or accepted_manifest_sha != run_manifest_sha:
        raise RuntimeError(f"Reranking cache is not the cache from accepted standalone attempt {source_key}")
    return run_manifest_sha


def validate_saved_manifest(home: Path, expected: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    path = home / "run-manifest.json"
    if not path.is_file():
        raise RuntimeError(f"Cannot resume: missing run manifest {path}")
    saved = json.loads(path.read_text(encoding="utf-8"))
    immutable_keys = (
        "batch_id", "run_kind", "dataset_file_name", "dataset_sha256", "binary_sha256", "model_sha256", "code_sha256",
        "dataset_ids", "requested_configs", "effective_configs", "selection_sha256",
        "selected_qrels_sha256", "deadline_record_sha256", "original_started_epoch", "deadline_epoch", "wall_hours",
        "segment_id", "continuation_source",
    )
    for key in immutable_keys:
        if saved.get(key) != expected.get(key):
            raise RuntimeError(f"Resume rejected: immutable run field changed: {key}")
    if args.deadline:
        requested_deadline = parse_deadline(args.deadline)
        if abs(requested_deadline - float(saved["deadline_epoch"])) > 1:
            raise RuntimeError("Resume rejected: --deadline cannot move or reset the original wall deadline")
    if args.wall_hours is not None and abs(float(saved["wall_hours"]) - args.wall_hours) > 1e-9:
        raise RuntimeError("Resume rejected: --wall-hours differs from the original wall-clock budget")
    return saved


def immutable_manifest_fields(run_kind: str, args: argparse.Namespace, hashes: dict[str, Any],
                              dataset_ids: list[str], requested_configs: list[str], effective_configs: list[str],
                              selection_record_value: dict[str, Any], selected_qrels_sha256: str,
                              deadline_record_sha256: str, started_epoch: float,
                              deadline_epoch: float, wall_hours: float) -> dict[str, Any]:
    return {
        "batch_id": args.batch_id, "run_kind": run_kind, "dataset_file_name": args.datasets_file.name,
        **hashes, "dataset_ids": dataset_ids, "requested_configs": requested_configs,
        "effective_configs": effective_configs, "selection_sha256": core.sha256_json(selection_record_value),
        "selected_qrels_sha256": selected_qrels_sha256, "deadline_record_sha256": deadline_record_sha256,
        "original_started_epoch": started_epoch, "deadline_epoch": deadline_epoch,
        "wall_hours": wall_hours,
        "segment_id": str(getattr(args, "segment", "01")),
        "continuation_source": (str(Path(args.continue_compatible_from).resolve())
                                if getattr(args, "continue_compatible_from", None) else None),
    }


def append_event(home: Path, kind: str, **extra) -> None:
    core.append(home / "traffic-events.jsonl", {"time": core.stamp(), "event": kind, **extra})


def _safe_relative_attempt(home: Path, relative: str, required_parent: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts or not value.parts or value.parts[0] != required_parent:
        raise RuntimeError(f"Unsafe accepted artifact path: {relative}")
    resolved = (home / value).resolve()
    try:
        resolved.relative_to((home / required_parent).resolve())
    except ValueError as exc:
        raise RuntimeError(f"Accepted artifact path escapes {required_parent}: {relative}") from exc
    if not resolved.is_dir() or any(path.is_symlink() for path in resolved.rglob("*")):
        raise RuntimeError(f"Accepted artifact directory is missing or contains symlinks: {relative}")
    return resolved


def _tree_hashes(home: Path) -> dict[str, str]:
    return {path.relative_to(home).as_posix(): core.sha256_file(path)
            for path in sorted(home.rglob("*")) if path.is_file() and not path.is_symlink()}


def _rewrite_copied_log_paths(attempt: Path, old_attempt: Path) -> None:
    path = attempt / "server-events.jsonl"
    if not path.is_file():
        raise RuntimeError(f"Accepted attempt is missing server-events.jsonl: {attempt}")
    events = core.rows(path)
    for item in events:
        command = item.get("command")
        if not isinstance(command, list) or "--log-file" not in command:
            continue
        index = command.index("--log-file") + 1
        old = Path(command[index])
        try:
            relative = old.resolve().relative_to(old_attempt.resolve())
        except ValueError:
            continue
        command[index] = str((attempt / relative).resolve())
    temp = path.with_suffix(".jsonl.tmp")
    temp.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in events), encoding="utf-8")
    temp.replace(path)


def _verify_gpu_attempt(attempt: Path) -> None:
    events = [item for item in core.rows(attempt / "server-events.jsonl") if isinstance(item.get("command"), list)]
    if len(events) != 1:
        raise RuntimeError(f"Expected exactly one server load record in {attempt}")
    command = events[0]["command"]
    if "--n-gpu-layers" not in command or command[command.index("--n-gpu-layers") + 1] != "all":
        raise RuntimeError(f"GPU-all setting is missing in {attempt}")
    if "--log-file" not in command:
        raise RuntimeError(f"GPU log path is missing in {attempt}")
    log = Path(command[command.index("--log-file") + 1])
    if not log.is_file():
        raise RuntimeError(f"GPU log is missing in {attempt}: {log}")
    matches = re.findall(r"offloaded\s+(\d+)/(\d+) layers to GPU", log.read_text(errors="replace"))
    if not matches or int(matches[-1][0]) <= 0:
        raise RuntimeError(f"Actual GPU offload evidence is missing in {attempt}")


def _verify_guard_interval(events: list[dict[str, Any]], *, marker_start: str,
                           marker_finish: str, identity_key: str, identity: str,
                           attempt: str,
                           traffic_epoch: int) -> dict[str, Any]:
    starts = [index for index, item in enumerate(events)
              if item.get("event") == marker_start and item.get(identity_key) == identity
              and item.get("attempt") == attempt]
    finishes = [index for index, item in enumerate(events)
                if item.get("event") == marker_finish and item.get(identity_key) == identity
                and item.get("attempt") == attempt]
    if len(starts) != 1 or len(finishes) != 1 or starts[0] >= finishes[0]:
        raise RuntimeError(f"Missing/duplicate guard markers for accepted work: {identity}")
    marker_start_index, marker_finish_index = starts[0], finishes[0]
    guard_starts = [index for index in range(marker_start_index)
                    if events[index].get("event") == "guard_unit_started"]
    guard_finishes = [index for index in range(marker_start_index + 1, marker_finish_index)
                      if events[index].get("event") == "guard_unit_finished"]
    if not guard_starts or not guard_finishes:
        raise RuntimeError(f"Guard snapshots are missing for accepted work: {identity}")
    guard_start_index = guard_starts[-1]
    guard_finish_index = guard_finishes[0]
    if not (guard_start_index < marker_start_index < guard_finish_index < marker_finish_index):
        raise RuntimeError(f"Guard interval ordering is invalid for accepted work: {identity}")
    start_event, finish_event = events[guard_start_index], events[guard_finish_index]
    if (start_event.get("traffic_epoch") != traffic_epoch or finish_event.get("traffic_epoch") != traffic_epoch or
            not isinstance(start_event.get("snapshot"), dict) or not isinstance(finish_event.get("snapshot"), dict)):
        raise RuntimeError(f"Guard epoch/snapshot differs for accepted work: {identity}")
    start_snapshot, finish_snapshot = start_event["snapshot"], finish_event["snapshot"]
    if (start_event.get("quiet_seconds") != 30 or start_event.get("poll_period_seconds") != 1 or
            start_snapshot != finish_snapshot or not isinstance(start_snapshot.get("workers"), dict) or
            not isinstance(start_snapshot.get("routers"), dict) or
            any(worker.get("busy") is not False for worker in start_snapshot["workers"].values())):
        raise RuntimeError(f"Guard snapshots are not identical, idle, and schema-valid: {identity}")
    if any(item.get("event") == "external_activity"
           for item in events[guard_start_index + 1:guard_finish_index]):
        raise RuntimeError(f"External traffic was recorded inside accepted work: {identity}")
    return {"started": start_event, "finished": finish_event}


def _validate_source_terminal(source: Path, source_manifest: dict[str, Any]) -> dict[str, Any]:
    deadline_path = source / "deadline.json"
    progress_path = source / "progress.json"
    fatal_path = source / "fatal-errors.jsonl"
    if not deadline_path.is_file() or not progress_path.is_file() or not fatal_path.is_file():
        raise RuntimeError("Continuation source must have a deadline, stopped progress, and terminal error record")
    deadline = json.loads(deadline_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    errors = core.rows(fatal_path)
    if (source_manifest.get("deadline_record_sha256") != core.sha256_json(deadline) or
            deadline.get("deadline_epoch") != source_manifest.get("deadline_epoch") or
            not 0 < float(deadline.get("wall_hours", 0)) <= 12):
        raise RuntimeError("Continuation source deadline record is inconsistent or was changed")
    if progress.get("phase") != "stopped" or float(deadline["deadline_epoch"]) > time.time():
        raise RuntimeError("Continuation source must be stopped after its own original deadline has expired")
    if not errors:
        raise RuntimeError("Continuation source has no terminal error evidence")
    last = errors[-1]
    error_time = last.get("time")
    try:
        terminal_after_deadline = datetime.fromisoformat(str(error_time).replace("Z", "+00:00")).timestamp() >= float(deadline["deadline_epoch"])
    except (TypeError, ValueError):
        terminal_after_deadline = False
    if not (terminal_after_deadline or "deadline" in str(last.get("error", "")).lower() or
            "wall-clock" in str(last.get("error", "")).lower()):
        raise RuntimeError("Continuation source did not stop because its fixed wall-clock budget expired")
    return {"deadline": deadline, "progress": progress, "terminal_error": last}


def import_compatible_commits(source: Path, target: Path, run_manifest: dict[str, Any],
                              datasets: dict[str, dict[str, Any]], selections: dict[str, list[str]],
                              jobs: list[tuple[str, str, str, str]]) -> dict[str, Any]:
    """Carry only independently revalidated official commits into a new capped run."""
    source = source.resolve()
    target = target.resolve()
    if source == target or not source.is_dir():
        raise RuntimeError("Continuation source must be a distinct existing official run directory")
    manifest_path = source / "run-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Continuation source run-manifest.json is missing")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    terminal = _validate_source_terminal(source, source_manifest)
    immutable_match = (
        "batch_id", "run_kind", "dataset_file_name", "dataset_sha256", "binary_sha256",
        "model_sha256", "code_sha256", "dataset_ids", "requested_configs", "effective_configs",
        "selection_sha256", "selected_qrels_sha256",
    )
    for key in immutable_match:
        if source_manifest.get(key) != run_manifest.get(key):
            raise RuntimeError(f"Continuation source differs in immutable input/config field: {key}")
    if source_manifest.get("run_kind") != "official":
        raise RuntimeError("Pilot or non-official evidence cannot be continued as official work")
    source_selection = json.loads((source / "query-selection.json").read_text(encoding="utf-8"))
    target_selection = json.loads((target / "query-selection.json").read_text(encoding="utf-8"))
    source_qrels = json.loads((source / "qrels.json").read_text(encoding="utf-8"))
    target_qrels = json.loads((target / "qrels.json").read_text(encoding="utf-8"))
    if (source_selection != target_selection or source_qrels != target_qrels or
            core.sha256_json(source_selection) != run_manifest["selection_sha256"] or
            core.sha256_json(source_qrels) != run_manifest["selected_qrels_sha256"]):
        raise RuntimeError("Continuation selection/qrels differ from the frozen current run")
    source_jobs = job_list(list(run_manifest["dataset_ids"]), list(run_manifest["effective_configs"]))
    source_committed = verify_committed_set(source, source_jobs)
    if not source_committed:
        raise RuntimeError("Continuation source contains no accepted commits; refusing a no-progress segment loop")
    source_events = core.rows(source / "traffic-events.jsonl")
    accepted_events_by_combo: dict[str, list[dict[str, Any]]] = {}
    for item in source_events:
        if item.get("event") == "combo_accepted":
            accepted_events_by_combo.setdefault(item.get("combo"), []).append(item)
    copied_commits: dict[str, Any] = {}
    attempt_provenance = {}
    destination_cache_hashes = {}
    model_hashes = run_manifest["model_sha256"]
    for job in source_jobs:
        dataset, embedding, reranker, key = job
        if key not in source_committed:
            continue
        query_ids = selections[dataset]
        source_record = source_committed[key]
        source_attempt = _safe_relative_attempt(source, source_record.get("attempt", ""), "attempts")
        rows = validate_accepted_attempt(source, job, source_committed, query_ids)
        validity = json.loads((source_attempt / "validity.json").read_text(encoding="utf-8"))
        traffic_epoch = validity.get("traffic_epoch")
        if not isinstance(traffic_epoch, int):
            raise RuntimeError(f"Accepted commit has no integer traffic epoch: {key}")
        guard_evidence = _verify_guard_interval(
            source_events, marker_start="combo_started", marker_finish="combo_work_finished",
            identity_key="combo", identity=key, attempt=source_record["attempt"],
            traffic_epoch=traffic_epoch,
        )
        accepted_events = accepted_events_by_combo.get(key, [])
        if accepted_events and (len(accepted_events) != 1 or
                                accepted_events[0].get("attempt") != source_record["attempt"]):
            raise RuntimeError(f"Accepted traffic ledger is duplicated or points at another attempt: {key}")
        if not accepted_events:
            guard_evidence["accepted_ledger_recovered"] = True
        if reranker == "standalone":
            gate = json.loads((source_attempt / "gate-embedding.json").read_text(encoding="utf-8"))
            if not gate.get("repeatability") or len(gate.get("fixtures", [])) != 10:
                raise RuntimeError(f"Standalone embedding fixture gate is incomplete: {key}")
            source_cache = source / "cache" / embedding / dataset
            source_attempt_cache = source_attempt / "cache" / embedding / dataset
            source_cache_manifest, cache_sha = verified_cache_manifest(source_cache, query_ids)
            attempt_manifest, attempt_sha = verified_cache_manifest(source_attempt_cache, query_ids)
            if source_cache_manifest != attempt_manifest or cache_sha != attempt_sha:
                raise RuntimeError(f"Accepted standalone cache differs from its committed attempt: {key}")
            cache_integrity(source, embedding, dataset, datasets[dataset]["_dataset_sha256"],
                            model_hashes[embedding], query_ids, source_manifest["binary_sha256"],
                            source_manifest["code_sha256"])
        else:
            gate = json.loads((source_attempt / "gate-reranker.json").read_text(encoding="utf-8"))
            if gate.get("status") != "passed" or len(gate.get("fixtures", [])) != 10 or gate.get("timing_included") is not False:
                raise RuntimeError(f"Reranker fixture gate is incomplete: {key}")
            source_standalone = f"{dataset}--{embedding}--standalone"
            if source_standalone not in source_committed:
                raise RuntimeError(f"Reranker commit lacks an accepted standalone source: {key}")
            inp = json.loads((source_attempt / "cache-input.json").read_text(encoding="utf-8"))
            _, cache_sha = verified_cache_manifest(source / "cache" / embedding / dataset, query_ids)
            if (inp.get("source_commit_key") != source_standalone or
                    inp.get("query_ids") != query_ids or inp.get("cache_manifest_sha256") != cache_sha):
                raise RuntimeError(f"Reranker candidate provenance differs from the standalone cache: {key}")
        _verify_gpu_attempt(source_attempt)
        source_attempt_hashes = _tree_hashes(source_attempt)
        target_attempt_rel = Path("attempts") / f"imported-{key}-{time.time_ns()}"
        target_attempt = target / target_attempt_rel
        shutil.copytree(source_attempt, target_attempt)
        _rewrite_copied_log_paths(target_attempt, source_attempt)
        if reranker == "standalone":
            target_cache = target / "cache" / embedding / dataset
            if target_cache.exists():
                raise RuntimeError(f"Continuation target cache already exists: {target_cache}")
            target_cache.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source / "cache" / embedding / dataset, target_cache)
            target_cache_manifest, target_cache_sha = verified_cache_manifest(target_cache, query_ids)
            if target_cache_manifest != source_cache_manifest or target_cache_sha != cache_sha:
                raise RuntimeError(f"Imported cache failed strict hash verification: {key}")
            destination_cache_hashes[dataset] = target_cache_sha
        # Revalidate copied artifacts after path rewriting, especially row and cache provenance.
        copied_record = {**source_record, "attempt": target_attempt_rel.as_posix(),
                         "imported_from": {"run": str(source), "attempt": source_record["attempt"]}}
        copied = validate_accepted_attempt(target, job, {key: copied_record}, query_ids)
        if copied != rows:
            raise RuntimeError(f"Imported result rows changed while copying {key}")
        copied_commits[key] = copied_record
        attempt_provenance[key] = {
            "source_attempt": source_record["attempt"], "source_attempt_files_sha256": source_attempt_hashes,
            "source_guard_interval": guard_evidence,
        }

    if not copied_commits:
        raise RuntimeError("No compatible accepted work could be imported")
    # Imported commits may only be a dependency-closed subset: rerankers require
    # their standalone result and exact cache from this same frozen embedder/data.
    for key in copied_commits:
        dataset, embedding, reranker = key.split("--", 2)
        if reranker != "standalone" and f"{dataset}--{embedding}--standalone" not in copied_commits:
            raise RuntimeError(f"Continuation import is not dependency-closed: {key}")

    # Reuse only successful preflight audits for the exact dataset scope.
    imported_preflight = {"model_ids": [], "models": {}}
    source_preflight_path = source / "preflight.json"
    if source_preflight_path.is_file():
        source_preflight = json.loads(source_preflight_path.read_text(encoding="utf-8"))
        from matrix_spec import RUN_BATCHES
        expected_models = [dict(RUN_BATCHES)[run_manifest["batch_id"]], *core.RERANKERS]
        if source_preflight.get("model_ids") != expected_models:
            raise RuntimeError("Continuation source preflight model matrix is not the fixed four-model order")
        for model, item in source_preflight.get("models", {}).items():
            if item.get("status") == "blocked":
                raise RuntimeError(f"Continuation source has an input-token blocker for {model}")
            if item.get("status") != "passed":
                continue
            relative = item.get("attempt", "")
            attempt = _safe_relative_attempt(source, relative, "preflight-attempts")
            validity = json.loads((attempt / "validity.json").read_text(encoding="utf-8"))
            audits = core.rows(attempt / "token-audit.jsonl")
            if validity.get("status") != "passed" or len(audits) != len(selections) or {row.get("dataset") for row in audits} != set(selections):
                raise RuntimeError(f"Full-corpus preflight evidence is incomplete for {model}")
            if any(row.get("status") != "ok" or row.get("truncated") is not False for row in audits):
                raise RuntimeError(f"Full-corpus preflight contains a truncation/block for {model}")
            _verify_gpu_attempt(attempt)
            matching_audit = [event for event in source_events
                              if event.get("event") == "token_audit_passed" and event.get("model") == model]
            if len(matching_audit) != 1:
                raise RuntimeError(f"Preflight pass traffic-ledger record is missing or duplicated: {model}")
            # New runner writes explicit markers; absence means the prior attempt
            # cannot be independently linked to guard snapshots and is not reusable.
            guard_evidence = _verify_guard_interval(source_events, marker_start="preflight_started",
                                                    marker_finish="preflight_finished", identity_key="model",
                                                    identity=model, attempt=relative,
                                                    traffic_epoch=int(item.get("traffic_epoch", -1)))
            target_attempt_rel = Path(relative)
            target_attempt = target / target_attempt_rel
            if target_attempt.exists():
                raise RuntimeError(f"Preflight import target already exists: {target_attempt}")
            shutil.copytree(attempt, target_attempt)
            _rewrite_copied_log_paths(target_attempt, attempt)
            imported_preflight["model_ids"].append(model)
            imported_preflight["models"][model] = dict(item)
            attempt_provenance[f"preflight:{model}"] = {"source_attempt": relative,
                "source_attempt_files_sha256": _tree_hashes(attempt),
                "source_guard_interval": guard_evidence}

    if imported_preflight["models"]:
        expected_ids = expected_models
        # Preserve the registered order and allow the next segment to audit only
        # models not yet passed in the source.
        imported_preflight["model_ids"] = expected_ids
        core.dump(target / "preflight.json", imported_preflight)
    core.dump(target / "committed.json", copied_commits)
    source_files = {name: core.sha256_file(source / name) for name in
                    ("run-manifest.json", "deadline.json", "progress.json", "committed.json",
                     "query-selection.json", "qrels.json", "traffic-events.jsonl")}
    source_import = source / "continuation-import.json"
    if source_import.is_file():
        source_files["continuation-import.json"] = core.sha256_file(source_import)
    imported = {
        "status": "passed", "source_run": str(source), "source_manifest_sha256": source_files["run-manifest.json"],
        "source_deadline_sha256": source_files["deadline.json"], "source_progress_sha256": source_files["progress.json"],
        "source_committed_sha256": source_files["committed.json"], "source_traffic_ledger_sha256": source_files["traffic-events.jsonl"],
        "source_continuation_import_sha256": source_files.get("continuation-import.json"),
        "source_terminal": terminal, "imported_commits": sorted(copied_commits),
        "imported_preflight_models": sorted(imported_preflight["models"]),
        "attempt_provenance": attempt_provenance, "candidate_cache_sha256": destination_cache_hashes,
        "imported_at": core.stamp(),
    }
    core.dump(target / "continuation-import.json", imported)
    for key, record in copied_commits.items():
        guard = attempt_provenance[key]["source_guard_interval"]
        epoch = guard["started"]["traffic_epoch"]
        append_event(target, "guard_unit_started", traffic_epoch=epoch,
                     quiet_seconds=guard["started"].get("quiet_seconds", 30),
                     poll_period_seconds=guard["started"].get("poll_period_seconds", 1),
                     snapshot=guard["started"]["snapshot"], inherited=True,
                     source_run=str(source), source_traffic_ledger_sha256=source_files["traffic-events.jsonl"])
        append_event(target, "combo_started", combo=key, attempt=record["attempt"],
                     inherited=True, source_run=str(source))
        append_event(target, "guard_unit_finished", traffic_epoch=epoch,
                     snapshot=guard["finished"]["snapshot"], inherited=True,
                     source_run=str(source), source_traffic_ledger_sha256=source_files["traffic-events.jsonl"])
        append_event(target, "combo_work_finished", combo=key, attempt=record["attempt"],
                     traffic_epoch=epoch, inherited=True, source_run=str(source))
        append_event(target, "combo_accepted", combo=key, attempt=record["attempt"], inherited=True,
                     source_run=str(source), source_traffic_ledger_sha256=source_files["traffic-events.jsonl"])
    for model in imported_preflight["models"]:
        guard = attempt_provenance[f"preflight:{model}"]["source_guard_interval"]
        epoch = guard["started"]["traffic_epoch"]
        relative = imported_preflight["models"][model]["attempt"]
        append_event(target, "guard_unit_started", traffic_epoch=epoch,
                     quiet_seconds=guard["started"].get("quiet_seconds", 30),
                     poll_period_seconds=guard["started"].get("poll_period_seconds", 1),
                     snapshot=guard["started"]["snapshot"], inherited=True,
                     source_run=str(source), source_traffic_ledger_sha256=source_files["traffic-events.jsonl"])
        append_event(target, "preflight_started", model=model, attempt=relative, inherited=True,
                     source_run=str(source))
        append_event(target, "guard_unit_finished", traffic_epoch=epoch,
                     snapshot=guard["finished"]["snapshot"], inherited=True,
                     source_run=str(source), source_traffic_ledger_sha256=source_files["traffic-events.jsonl"])
        append_event(target, "preflight_finished", model=model, attempt=relative,
                     traffic_epoch=epoch, inherited=True, source_run=str(source))
        append_event(target, "token_audit_passed", model=model, inherited=True, source_run=str(source))
    return imported


def guarded_progress(home: Path, started_epoch: float, phase: str, **extra) -> None:
    value = {"time": core.stamp(), "phase": phase, "elapsed_seconds": max(0.0, time.time() - started_epoch), **extra}
    core.dump(home / "progress.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def wait_for_idle(home: Path, started_epoch: float, deadline_epoch: float,
                  guard: TrafficGuard, name: str, committed: int, total: int) -> None:
    guarded_progress(home, started_epoch, "waiting_for_idle", unit=name, committed=committed,
                     required_quiet_seconds=30)
    while True:
        if time.time() >= deadline_epoch:
            raise TimeoutError("Original wall-clock deadline reached while waiting for production traffic")
        if not guard.ready():
            time.sleep(1)
            continue
        try:
            guard.begin()
        except Interference:
            # A fresh /metrics poll may invalidate readiness between checks.
            continue
        if time.time() >= deadline_epoch:
            raise TimeoutError("Original wall-clock deadline reached before guarded work began")
        return


def run_preflight_model(home: Path, started_epoch: float, deadline_epoch: float, model_name: str,
                        model_kind: str, datasets: dict[str, dict[str, Any]], dataset_ids: list[str],
                        guard: TrafficGuard, watchdog: DeadlineWatchdog,
                        preflight: dict[str, Any]) -> None:
    previous = preflight["models"].get(model_name)
    if previous and previous.get("status") in ("passed", "blocked"):
        return
    model_path = model_registry()[model_name]
    bge_template = core.verify_bge_reranker_model(model_path) if model_name == "bge-reranker" else None
    attempt_count = 0
    while True:
        if time.time() >= deadline_epoch:
            raise TimeoutError("Original wall-clock deadline reached during input-token preflight")
        wait_for_idle(home, started_epoch, deadline_epoch, guard, f"token-audit:{model_name}",
                      sum(item.get("status") in ("passed", "blocked") for item in preflight["models"].values()),
                      len(preflight["model_ids"]))
        attempt_count += 1
        attempt = home / "preflight-attempts" / f"{model_name}--{time.time_ns()}"
        attempt.mkdir(parents=True)
        attempt_rel = str(attempt.relative_to(home))
        append_event(home, "preflight_started", model=model_name, attempt=attempt_rel)
        core.base.ROOT = attempt
        issues = []
        study = GuardedStudy(attempt, {}, "preflight", deadline_epoch,
                             lambda: None, lambda phase, **extra: guarded_progress(home, started_epoch, phase, **extra),
                             guard=guard, watchdog=watchdog)
        mode = "embedding" if model_kind == "embedding" else "rerank"
        pooling = core.EMBEDDERS[model_name][1] if model_kind == "embedding" else "last"
        model_spec = core.EMBEDDERS.get(model_name, ())
        try:
            with study.server(f"preflight-{model_name}", model_path, mode, pooling) as server:
                for dataset_name in dataset_ids:
                    dataset = datasets[dataset_name]
                    record = core.audit_token_limits(
                        server, dataset_name, dataset, model_name, model_kind, model_spec,
                        attempt / "token-audit.jsonl", study.check_time,
                        bge_template=bge_template,
                    )
                    issues.extend({"dataset": dataset_name, **issue} for issue in record["issues"])
            guard.finish()
            append_event(home, "preflight_finished", model=model_name, attempt=attempt_rel,
                         traffic_epoch=guard.epoch)
            status = "blocked" if issues else "passed"
            core.dump(attempt / "validity.json", {"status": status, "issues": issues})
            preflight["models"][model_name] = {
                "status": status, "attempt": str(attempt.relative_to(home)), "issues": issues,
                "accepted_at": core.stamp(), "attempt_count": attempt_count,
                "traffic_epoch": guard.epoch,
            }
            core.dump(home / "preflight.json", preflight)
            append_event(home, f"token_audit_{status}", model=model_name, issues=len(issues))
            if issues:
                guarded_progress(home, started_epoch, "preflight_blocker", model=model_name,
                                 issue_count=len(issues), issues=issues[:20])
            else:
                guarded_progress(home, started_epoch, "preflight_model_complete", model=model_name,
                                 models_complete=sum(value.get("status") in ("passed", "blocked") for value in preflight["models"].values()),
                                 models_total=len(preflight["model_ids"]))
            return
        except BaseException as exc:
            try:
                guard.check()
                contaminated = isinstance(exc, Interference)
            except Interference:
                contaminated = True
            core.dump(attempt / "validity.json", {
                "status": "invalidated" if contaminated else "failed", "reason": str(exc),
            })
            if contaminated and isinstance(exc, Exception):
                append_event(home, "token_audit_invalidated", model=model_name, reason=str(exc))
                guarded_progress(home, started_epoch, "paused_external_traffic", unit=f"token-audit:{model_name}", reason=str(exc))
                continue
            raise


def ensure_preflight(home: Path, started_epoch: float, deadline_epoch: float,
                     datasets: dict[str, dict[str, Any]], dataset_ids: list[str], config_ids: list[str],
                     guard: TrafficGuard, watchdog: DeadlineWatchdog) -> None:
    model_ids = []
    for config_id in config_ids:
        embedding, reranker = core.PILOT_CONFIGS[config_id]
        if embedding not in model_ids:
            model_ids.append(embedding)
        if reranker != "standalone" and reranker not in model_ids:
            model_ids.append(reranker)
    preflight_path = home / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.is_file() else {
        "model_ids": model_ids, "models": {},
    }
    if preflight.get("model_ids") != model_ids:
        raise RuntimeError("Preflight model matrix differs from the frozen run manifest")
    for model_name in model_ids:
        run_preflight_model(home, started_epoch, deadline_epoch, model_name,
                            "embedding" if model_name in core.EMBEDDERS else "reranker",
                            datasets, dataset_ids, guard, watchdog, preflight)
    blockers = {name: value["issues"] for name, value in preflight["models"].items() if value.get("status") == "blocked"}
    if blockers:
        core.dump(home / "preflight-blockers.json", blockers)
        raise PreflightBlockedError(f"Token preflight found inputs at or above the 4096-token limit: {list(blockers)}")


def load_committed(home: Path) -> dict[str, Any]:
    path = home / "committed.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def copy_accepted_cache(home: Path, attempt: Path, embedding: str, dataset: str) -> None:
    source = attempt / "cache" / embedding / dataset
    destination = home / "cache" / embedding / dataset
    if not source.is_dir():
        raise RuntimeError(f"Standalone attempt did not produce a cache: {source}")
    if destination.exists() or destination.is_symlink():
        standalone_key = f"{dataset}--{embedding}--standalone"
        committed = load_committed(home)
        if standalone_key in committed:
            old = json.loads((destination / "cache-manifest.json").read_text(encoding="utf-8"))
            new = json.loads((source / "cache-manifest.json").read_text(encoding="utf-8"))
            if old != new:
                raise RuntimeError(f"Committed cache conflicts with accepted attempt and cannot be replaced: {destination}")
            verified_cache_manifest(destination, old.get("query_ids", []))
            return
        # A run-level cache is copied immediately before the standalone attempt's
        # validity/commit writes. If the process crashed in that window, preserve
        # the orphan for audit, then let the complete standalone unit be rerun.
        orphan_root = home / "orphan-caches"
        orphan_root.mkdir(parents=True, exist_ok=True)
        orphan = orphan_root / f"{embedding}--{dataset}--{time.time_ns()}"
        destination.rename(orphan)
        append_event(home, "orphan_cache_preserved", embedding=embedding,
                     dataset=dataset, path=str(orphan.relative_to(home)),
                     committed_standalone=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp-{time.time_ns()}")
    shutil.copytree(source, temporary)
    try:
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_accepted_attempt(home: Path, job: tuple[str, str, str, str], committed: dict[str, Any],
                              expected_query_ids: list[str]) -> list[dict[str, Any]]:
    dataset, embedding, reranker, key = job
    record = committed[key]
    attempt = (home / record["attempt"]).resolve()
    try:
        attempt.relative_to((home / "attempts").resolve())
    except ValueError as exc:
        raise RuntimeError(f"Committed attempt path escapes attempts directory: {record['attempt']}") from exc
    validity = json.loads((attempt / "validity.json").read_text(encoding="utf-8"))
    if validity.get("status") != "accepted":
        raise RuntimeError(f"Committed attempt {key} is not marked accepted")
    result = core.rows(attempt / "query-results.jsonl")
    if len(result) != len(expected_query_ids) or {row["query_id"] for row in result} != set(expected_query_ids):
        raise RuntimeError(f"Committed attempt {key} does not contain the frozen query set")
    if any(row["status"] != "ok" for row in result):
        raise RuntimeError(f"Committed attempt {key} contains an invalid result row")
    if any(row["dataset"] != dataset or row["embedding"] != embedding or row["reranker"] != reranker for row in result):
        raise RuntimeError(f"Committed attempt {key} has mismatched model/dataset metadata")
    return result


def consolidate(home: Path, jobs: list[tuple[str, str, str, str]], committed: dict[str, Any],
                selections: dict[str, list[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows = []
    index_rows = []
    for job in jobs:
        dataset, embedding, _, key = job
        if key not in committed:
            continue
        all_rows.extend(validate_accepted_attempt(home, job, committed, selections[dataset]))
        if job[2] == "standalone":
            attempt = home / committed[key]["attempt"]
            index_rows.extend(core.rows(attempt / "index-stats.jsonl"))
        if job[2] != "standalone":
            source_key = f"{dataset}--{embedding}--standalone"
            if source_key not in committed:
                raise RuntimeError(f"Committed reranker job has no accepted standalone source: {source_key}")
            standalone_job = (dataset, embedding, "standalone", source_key)
            validate_accepted_attempt(home, standalone_job, committed, selections[dataset])
            standalone_attempt = (home / committed[source_key]["attempt"]).resolve()
            _, source_manifest_sha = verified_cache_manifest(
                standalone_attempt / "cache" / embedding / dataset, selections[dataset])
            _, run_manifest_sha = verified_cache_manifest(
                home / "cache" / embedding / dataset, selections[dataset])
            if source_manifest_sha != run_manifest_sha:
                raise RuntimeError(f"Committed reranker {key} did not use its accepted standalone cache")
            rerank_attempt = (home / committed[key]["attempt"]).resolve()
            input_path = rerank_attempt / "cache-input.json"
            if not input_path.is_file():
                raise RuntimeError(f"Committed reranker {key} is missing cache provenance")
            cache_input = json.loads(input_path.read_text(encoding="utf-8"))
            if (cache_input.get("source_commit_key") != source_key or
                    cache_input.get("cache_manifest_sha256") != run_manifest_sha or
                    cache_input.get("query_ids") != selections[dataset]):
                raise RuntimeError(f"Committed reranker {key} cache provenance does not match accepted standalone source")
    temp = home / "query-results.jsonl.tmp"
    with temp.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(home / "query-results.jsonl")
    temp = home / "index-stats.jsonl.tmp"
    with temp.open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(home / "index-stats.jsonl")
    summary = core.summarize(all_rows)
    core.dump(home / "summary.json", summary)
    core.write_report(home, summary, "pilot" if home.name == "pilot" else "official")
    return all_rows, summary


def run_job(home: Path, started_epoch: float, deadline_epoch: float,
            datasets: dict[str, dict[str, Any]], selections: dict[str, list[str]],
            job: tuple[str, str, str, str], committed: dict[str, Any], total_jobs: int,
            guard: TrafficGuard, watchdog: DeadlineWatchdog, dataset_sha256: str,
            model_hashes: dict[str, str], jobs: list[tuple[str, str, str, str]]) -> None:
    dataset_name, embedding, reranker_name, key = job
    if key in committed:
        return
    data = dict(datasets[dataset_name])
    queries = core.select_by_ids(data, selections[dataset_name])
    attempt_count = 0
    while True:
        if time.time() >= deadline_epoch:
            raise TimeoutError("Original wall-clock deadline reached while waiting for production traffic")
        wait_for_idle(home, started_epoch, deadline_epoch, guard, key, len(committed), total_jobs)
        attempt_count += 1
        attempt = home / "attempts" / f"{key}--{time.time_ns()}"
        attempt.mkdir(parents=True)
        core.base.ROOT = attempt
        study = GuardedStudy(attempt, data, dataset_name, deadline_epoch,
                             lambda: None, lambda phase, **extra: guarded_progress(home, started_epoch, phase, **extra),
                             guard=guard, watchdog=watchdog)
        append_event(home, "combo_started", combo=key, attempt=str(attempt.relative_to(home)))
        try:
            if reranker_name == "standalone":
                result = study.standalone(embedding, queries, attempt / "cache")
            else:
                cache_manifest_sha = accepted_embedder_cache(
                    home, embedding, dataset_name, selections[dataset_name], committed,
                    data["_dataset_sha256"], model_hashes, data["_binary_sha256"],
                    data["_code_sha256"],
                )
                core.dump(attempt / "cache-input.json", {
                    "source_commit_key": f"{dataset_name}--{embedding}--standalone",
                    "cache_manifest_sha256": cache_manifest_sha,
                    "query_ids": selections[dataset_name],
                })
                result = study.rerank(reranker_name, queries, home / "cache", embedding)
            guard.finish()
            append_event(home, "combo_work_finished", combo=key,
                         attempt=str(attempt.relative_to(home)), traffic_epoch=guard.epoch)
            if len(result) != len(queries) or len({row["query_id"] for row in result}) != len(queries):
                raise RuntimeError(f"Attempt {key} returned an incomplete or duplicate query set")
            if any(row["status"] != "ok" for row in result):
                raise RuntimeError(f"Attempt {key} contains a non-ok query result")
            if reranker_name == "standalone":
                copy_accepted_cache(home, attempt, embedding, dataset_name)
            # The attempt becomes accepted before the run-level pointer is committed.
            # A crash between these writes leaves an auditable orphan, not a main-table row.
            core.dump(attempt / "validity.json", {"status": "accepted", "traffic_epoch": guard.epoch})
            committed[key] = {
                "attempt": str(attempt.relative_to(home)), "accepted_at": core.stamp(),
                "queries": len(result), "attempt_count": attempt_count,
            }
            core.dump(home / "committed.json", committed)
            consolidate(home, jobs, committed, selections)
            append_event(home, "combo_accepted", combo=key, attempt=str(attempt.relative_to(home)),
                         committed=len(committed), total=total_jobs)
            guarded_progress(home, started_epoch, "combo_complete", combo=key,
                             committed=len(committed), total=total_jobs)
            return
        except BaseException as exc:
            if key in committed:
                # Once committed, never downgrade an accepted attempt because a later
                # progress/report write failed; resume will revalidate the pointer.
                raise
            try:
                guard.check()
                contaminated = isinstance(exc, Interference)
            except Interference:
                contaminated = True
            core.dump(attempt / "validity.json", {
                "status": "invalidated" if contaminated else "failed", "reason": str(exc),
            })
            if contaminated and isinstance(exc, Exception):
                append_event(home, "combo_invalidated", combo=key, reason=str(exc))
                guarded_progress(home, started_epoch, "paused_external_traffic", combo=key, reason=str(exc))
                continue
            raise


def verify_committed_set(home: Path, jobs: list[tuple[str, str, str, str]]) -> dict[str, Any]:
    committed = load_committed(home)
    expected_keys = {job[3] for job in jobs}
    unknown = set(committed) - expected_keys
    if unknown:
        raise RuntimeError(f"Run contains committed combos outside its frozen matrix: {sorted(unknown)}")
    return committed


def set_up_manifest(home: Path, args: argparse.Namespace, datasets: dict[str, dict[str, Any]],
                    dataset_sha256: str, dataset_ids: list[str], requested_configs: list[str],
                    effective_configs: list[str], selections: dict[str, list[str]], qrels_sha256: str,
                    started_epoch: float) -> tuple[dict[str, Any], float]:
    run_kind = "pilot" if args.pilot else "official"
    selection_value = selection_record(datasets, selections, dataset_ids, run_kind, dataset_sha256)
    hashes = environment_hashes(args.datasets_file, dataset_sha256)
    aliases = {"run_study.py": "core.py", "legacy_run_benchmark.py": "server_helper.py"}
    snapshot = home / "source-snapshot"
    snapshot.mkdir(exist_ok=True)
    for name, digest in hashes["code_sha256"].items():
        source = PHASE2_DIR / aliases.get(name, name)
        target = snapshot / name
        if target.exists() and core.sha256_file(target) != digest:
            raise RuntimeError(f"Conflicting frozen source snapshot: {name}")
        if not target.exists():
            shutil.copy2(source, target)
    if args.deadline:
        deadline_epoch = parse_deadline(args.deadline)
        if deadline_epoch <= started_epoch:
            raise ValueError("--deadline must be in the future")
        wall_hours = (deadline_epoch - started_epoch) / 3600
        if wall_hours > DEFAULT_WALL_HOURS:
            raise ValueError("--deadline cannot exceed the hard 12-hour limit")
    else:
        wall_hours = args.wall_hours if args.wall_hours is not None else DEFAULT_WALL_HOURS
        if not math.isfinite(wall_hours) or not 0 < wall_hours <= (2 if args.pilot else 12):
            raise ValueError("Pilot/official wall budget must be finite and within 2h/12h")
        deadline_epoch = started_epoch + wall_hours * 3600
    deadline_record = {
        "original_started_epoch": started_epoch, "deadline_epoch": deadline_epoch,
        "wall_hours": wall_hours, "created_at": core.stamp(),
    }
    core.dump(home / "deadline.json", deadline_record)
    deadline_record_sha256 = core.sha256_json(deadline_record)
    immutable = immutable_manifest_fields(run_kind, args, hashes, dataset_ids, requested_configs,
                                          effective_configs, selection_value, qrels_sha256,
                                          deadline_record_sha256, started_epoch,
                                          deadline_epoch, wall_hours)
    run_manifest = {
        **immutable, "created_at": core.stamp(), "deadline_utc": datetime.fromtimestamp(deadline_epoch, timezone.utc).isoformat(),
        "matrix_summary_count": len(job_list(dataset_ids, effective_configs)),
        "matrix_full_query_rows": (sum(core.DATASET_COUNTS[name]["queries"] for name in dataset_ids) * len(effective_configs)
                                    if not args.pilot else None),
        "traffic_policy": "fail-closed; /metrics polling every second; 30 seconds continuously quiet; whole dataset×model combo retry",
        "runner_policy": "continuation-v2; one global lock; fixed full-query scope per registered batch; 4096-token preflight; log verbosity 4; GPU layers all; ten fixed positive/negative reranker fixture warmups per reranker server, all excluded from measured rows; forecast is planning-only; immutable per-run 12-hour official cap",
        "production_modified": False,
        "runtime_environment": {"python": sys.version, "executable": sys.executable,
                                "platform": platform.platform(), "numpy": core.np.__version__},
    }
    core.dump(home / "run-manifest.json", run_manifest)
    core.dump(home / "query-selection.json", selection_value)
    return run_manifest, deadline_epoch


def load_matrix_inputs(args):
    from matrix_spec import validate_and_merge_manifests
    pinned = {"first": "bc0657a77090f86b081b378e3f369653dc08f6049da78487b7dc0c134e7adad1",
              "second": "fdf802cc03e9059783e5a2cba0785437253f07841ec6956246d5884461e7ba3d"}
    observed = {"first": core.sha256_file(args.datasets_file), "second": core.sha256_file(args.datasets_batch2)}
    if observed != pinned:
        raise RuntimeError("Frozen six-section material changed; a new approved data version is required")
    first = json.loads(args.datasets_file.read_text())
    second = json.loads(args.datasets_batch2.read_text())
    datasets = validate_and_merge_manifests(first, second)
    return datasets, core.sha256_json(observed)

def run(args: argparse.Namespace) -> None:
    if not getattr(args, "execute", False):
        raise RuntimeError("Execution requires explicit --execute authorization")
    from matrix_spec import RUN_BATCHES
    core.configure(args.workspace)
    args.embedder = dict(RUN_BATCHES)[args.batch_id]
    global GLOBAL_LOCK
    GLOBAL_LOCK = args.workspace.resolve() / "embedding-study-phase2-20261023/.phase2-benchmark.lock"
    run_kind = "pilot" if args.pilot else "official"
    home = args.output.resolve() / run_kind
    if args.validate_data_only:
        datasets, digest = load_matrix_inputs(args)
        dataset_ids = build_dataset_ids(args)
        summary = {name: {"corpus": len(data["corpus"]), "queries": len(data["queries"]),
                          "qrels": sum(map(len, data["qrels"].values())), "grade_mode": data["grade_mode"]}
                   for name, data in datasets.items() if name in dataset_ids}
        print(json.dumps({"datasets_sha256": digest, "datasets": summary,
                          "expected_summaries": len(dataset_ids) * 4,
                          "expected_rows": sum(len(datasets[name]["queries"]) for name in dataset_ids) * 4},
                         ensure_ascii=False, indent=2))
        return

    args.datasets_file = args.datasets_file.resolve()
    args.output = args.output.resolve()
    requested_configs, effective_configs = build_config_ids(args)
    dataset_ids = build_dataset_ids(args)
    started_epoch = time.time()
    with benchmark_lock():
        datasets, dataset_sha256 = load_matrix_inputs(args)
        if not set(dataset_ids) <= set(datasets):
            raise ValueError(f"Selected datasets are absent from manifest: {sorted(set(dataset_ids) - set(datasets))}")
        selections = build_selection(datasets, dataset_ids, args.pilot, args.pilot_queries)
        qrels_value = qrels_record(datasets, selections, dataset_ids, dataset_sha256)
        selected_qrels_sha256 = core.sha256_json(qrels_value)
        args.output.mkdir(parents=True, exist_ok=True)
        if args.resume:
            if not home.is_dir():
                raise RuntimeError(f"Cannot resume because the {run_kind} output directory does not exist: {home}")
            saved = json.loads((home / "run-manifest.json").read_text(encoding="utf-8")) if (home / "run-manifest.json").exists() else None
            if saved is None:
                raise RuntimeError(f"Cannot resume without run-manifest.json in {home}")
            expected_hashes = environment_hashes(args.datasets_file, dataset_sha256)
            selection_value = selection_record(datasets, selections, dataset_ids, run_kind, dataset_sha256)
            deadline_path = home / "deadline.json"
            if not deadline_path.is_file():
                raise RuntimeError("Cannot resume without the original deadline.json")
            deadline_record = json.loads(deadline_path.read_text(encoding="utf-8"))
            deadline_record_sha256 = core.sha256_json(deadline_record)
            requested_started = float(deadline_record["original_started_epoch"])
            requested_deadline = float(deadline_record["deadline_epoch"])
            wall_hours = float(deadline_record["wall_hours"])
            if abs((requested_deadline - requested_started) - wall_hours * 3600) > 1:
                raise RuntimeError("Original deadline record is internally inconsistent")
            if not math.isfinite(wall_hours) or wall_hours <= 0 or wall_hours > DEFAULT_WALL_HOURS:
                raise RuntimeError("Resume rejected: original deadline exceeds the hard 12-hour wall limit")
            expected = immutable_manifest_fields(run_kind, args, expected_hashes, dataset_ids,
                                                 requested_configs, effective_configs, selection_value,
                                                 selected_qrels_sha256, deadline_record_sha256,
                                                 requested_started, requested_deadline, wall_hours)
            saved = validate_saved_manifest(home, expected, args)
            run_manifest = saved
            deadline_epoch = float(saved["deadline_epoch"])
            saved_qrels = json.loads((home / "qrels.json").read_text(encoding="utf-8")) if (home / "qrels.json").is_file() else None
            if saved_qrels is None or core.sha256_json(saved_qrels) != selected_qrels_sha256:
                raise RuntimeError("Resume rejected: selected qrels artifact is missing or changed")
        else:
            if home.exists() and any(home.iterdir()):
                raise RuntimeError(f"Output already exists; use --resume for the same run: {home}")
            home.mkdir(parents=True, exist_ok=True)
            run_manifest, deadline_epoch = set_up_manifest(
                home, args, datasets, dataset_sha256, dataset_ids, requested_configs,
                effective_configs, selections, selected_qrels_sha256, started_epoch,
            )
            core.dump(home / "qrels.json", qrels_value)
        started_epoch = float(run_manifest["original_started_epoch"])
        if time.time() >= deadline_epoch:
            raise TimeoutError("Original wall-clock deadline already reached; resume cannot reset it")

        model_hashes = run_manifest["model_sha256"]
        binary_sha256 = run_manifest["binary_sha256"]
        code_hashes = run_manifest["code_sha256"]
        for name in datasets:
            datasets[name]["_dataset_sha256"] = core.sha256_json(datasets[name])
            datasets[name]["_model_hashes"] = model_hashes
            datasets[name]["_binary_sha256"] = binary_sha256
            datasets[name]["_code_sha256"] = code_hashes
        core.base.ROOT = home
        jobs = job_list(dataset_ids, effective_configs)
        if args.continue_compatible_from is not None and not args.resume:
            source_home = Path(args.continue_compatible_from).expanduser().resolve()
            import_compatible_commits(source_home, home, run_manifest, datasets, selections, jobs)
        committed = verify_committed_set(home, jobs)
        selection_value = selection_record(datasets, selections, dataset_ids, run_kind, dataset_sha256)
        core.dump(home / "query-selection.json", selection_value)
        if (home / "run-manifest.json").exists():
            core.append(home / "resume-history.jsonl", {
                "resumed_at": core.stamp(), "committed_before_resume": len(committed),
                "original_deadline": run_manifest["deadline_utc"],
            }) if args.resume else None

        event_lock = threading.Lock()
        def event(kind: str, **extra):
            with event_lock:
                append_event(home, kind, **extra)

        guard = TrafficGuard(event, idle_seconds=30, period=1)
        watchdog = DeadlineWatchdog(deadline_epoch)
        guard.start()
        watchdog.start()
        try:
            if committed:
                consolidate(home, jobs, committed, selections)
                # A crash can occur after an atomic commit and before the
                # acceptance event is appended. Revalidate before recovering
                # that ledger edge; never rerun or discard an accepted unit.
                recorded = {row.get("combo") for row in core.rows(home / "traffic-events.jsonl")
                            if row.get("event") == "combo_accepted"}
                for key in committed.keys() - recorded:
                    append_event(home, "combo_accepted", combo=key, recovered_from_commit=True,
                                 attempt=committed[key]["attempt"], committed=len(committed), total=len(jobs))
            ensure_preflight(home, started_epoch, deadline_epoch, datasets, dataset_ids,
                             effective_configs, guard, watchdog)
            for job in jobs:
                if job[3] in committed:
                    continue
                run_job(home, started_epoch, deadline_epoch, datasets, selections, job,
                        committed, len(jobs), guard, watchdog, dataset_sha256, model_hashes, jobs)
            all_rows, summary = consolidate(home, jobs, committed, selections)
            core.dump(home / "paired-confidence.json", {
                "status": "deferred", "reason": "optional analysis after full matrix; no extra inference required"})
            expected_queries = sum(len(selections[name]) for name in dataset_ids) * len(effective_configs)
            if args.pilot:
                if len(summary) != len(jobs) or any(
                    row["queries"] != len(selections.get(row["dataset"], [])) or row["errors"]
                    for row in summary
                ):
                    raise RuntimeError("Pilot result set does not match its frozen pilot matrix")
                guarded_progress(home, started_epoch, "pilot_complete", summaries=len(summary), query_rows=len(all_rows), errors=0)
            else:
                expected_summaries = len(dataset_ids) * len(effective_configs)
                expected_query_rows = sum(core.DATASET_COUNTS[name]["queries"] for name in dataset_ids) * len(effective_configs)
                if len(summary) != expected_summaries or len(all_rows) != expected_query_rows:
                    raise RuntimeError(f"Official run incomplete: summaries={len(summary)}, rows={len(all_rows)}; expected={expected_summaries}/{expected_query_rows}")
                if any(row["queries"] != core.DATASET_COUNTS[row["dataset"]]["queries"] or row["errors"] for row in summary):
                    raise RuntimeError("Official run has incomplete or erroneous dataset/model summaries")
                if expected_queries != expected_query_rows:
                    raise RuntimeError("Official manifest query selection no longer matches the registered batch scope")
                guarded_progress(home, started_epoch, "complete", summaries=len(summary), query_rows=len(all_rows), errors=0)
                from batch_progress import update_completed
                update_completed(args.batch_id, home, args.workspace, args.plan_path)
        except PreflightBlockedError as exc:
            core.append(home / "fatal-errors.jsonl", {"time": core.stamp(), "type": type(exc).__name__, "error": str(exc)})
            guarded_progress(home, started_epoch, "blocked_preflight", reason=str(exc))
            raise
        except BaseException as exc:
            core.append(home / "fatal-errors.jsonl", {"time": core.stamp(), "type": type(exc).__name__, "error": str(exc)})
            guarded_progress(home, started_epoch, "stopped", reason=str(exc), committed=len(committed))
            raise
        finally:
            watchdog.close()
            guard.close()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("Termination requested"))
    main()
