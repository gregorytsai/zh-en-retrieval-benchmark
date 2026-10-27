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
    if not args.pilot:
        return list(core.DATASET_COUNTS)
    result = list(args.pilot_datasets or ["acord-eng-eng"])
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
    }


def append_event(home: Path, kind: str, **extra) -> None:
    core.append(home / "traffic-events.jsonl", {"time": core.stamp(), "event": kind, **extra})


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
            status = "blocked" if issues else "passed"
            core.dump(attempt / "validity.json", {"status": status, "issues": issues})
            preflight["models"][model_name] = {
                "status": status, "attempt": str(attempt.relative_to(home)), "issues": issues,
                "accepted_at": core.stamp(), "attempt_count": attempt_count,
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
            append_event(home, "combo_accepted", combo=key, committed=len(committed), total=total_jobs)
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
        "matrix_summary_count": EXPECTED_SUMMARIES if not args.pilot else len(job_list(dataset_ids, effective_configs)),
        "matrix_full_query_rows": EXPECTED_QUERY_ROWS if not args.pilot else None,
        "traffic_policy": "fail-closed; /metrics polling every second; 30 seconds continuously quiet; whole dataset×model combo retry",
        "runner_policy": "one global lock; no v2 dependency; full official query lists; 4096-token preflight; log verbosity 4; GPU layers all; ten fixed positive/negative reranker fixture warmups per reranker server, all excluded from measured rows",
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
    from matrix_spec import BATCHES
    core.configure(args.workspace)
    args.embedder = dict(BATCHES)[args.batch_id]
    global GLOBAL_LOCK
    GLOBAL_LOCK = args.workspace.resolve() / "embedding-study-phase2-20261023/.phase2-benchmark.lock"
    run_kind = "pilot" if args.pilot else "official"
    home = args.output.resolve() / run_kind
    if args.validate_data_only:
        datasets, digest = load_matrix_inputs(args)
        summary = {name: {"corpus": len(data["corpus"]), "queries": len(data["queries"]),
                          "qrels": sum(map(len, data["qrels"].values())), "grade_mode": data["grade_mode"]}
                   for name, data in datasets.items()}
        print(json.dumps({"datasets_sha256": digest, "datasets": summary,
                          "expected_summaries": EXPECTED_SUMMARIES, "expected_rows": EXPECTED_QUERY_ROWS},
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
                                 committed=len(committed), total=len(jobs))
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
                if len(summary) != EXPECTED_SUMMARIES or len(all_rows) != EXPECTED_QUERY_ROWS:
                    raise RuntimeError(f"Official run incomplete: summaries={len(summary)}, rows={len(all_rows)}")
                if any(row["queries"] != core.DATASET_COUNTS[row["dataset"]]["queries"] or row["errors"] for row in summary):
                    raise RuntimeError("Official run has incomplete or erroneous dataset/model summaries")
                if expected_queries != EXPECTED_QUERY_ROWS:
                    raise RuntimeError("Official manifest query selection no longer matches the frozen single-embedder target")
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
