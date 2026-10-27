"""Fail-closed pilot audit and conservative per-embedder runtime forecast.

This module reads only completed runner artifacts. It never starts a model,
server, HTTP request, lock, or benchmark.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from matrix_spec import BATCHES, DATASET_COUNTS, RERANKERS, SUMMARIES_PER_EMBEDDER, sha256_json


PILOT_QUERIES_PER_DATASET = 3
PILOT_ROWS = len(DATASET_COUNTS) * PILOT_QUERIES_PER_DATASET * 4
MAX_OFFICIAL_FORECAST_HOURS = 10.0
QUERY_WORK_MULTIPLIER = 1.25
FIXED_SAFETY_SECONDS = 600.0


class PilotGateError(RuntimeError):
    """Pilot evidence is incomplete, changed, or not eligible for official work."""


def length_quantile_selection_issues(datasets: dict[str, dict[str, Any]],
                                     selections: dict[str, list[str]]) -> list[str]:
    """Confirm each frozen ID set spans short, middle, and long query lengths."""
    issues = []
    for dataset in DATASET_COUNTS:
        records = datasets[dataset]["queries"]
        ordered = sorted(
            records,
            key=lambda row: (
                len(row["text"]),
                hashlib.sha256(str(row["id"]).encode()).hexdigest(),
                str(row["id"]),
            ),
        )
        observed = list(map(str, selections.get(dataset, [])))
        if len(observed) != PILOT_QUERIES_PER_DATASET or len(set(observed)) != PILOT_QUERIES_PER_DATASET:
            issues.append(f"{dataset}: pilot selection is not three unique IDs")
            continue
        rank = {str(row["id"]): index / max(1, len(ordered) - 1) for index, row in enumerate(ordered)}
        if not set(observed) <= set(rank):
            issues.append(f"{dataset}: pilot selection contains an unknown query ID")
            continue
        positions = [rank[query_id] for query_id in observed]
        if not (positions[0] <= 0.10 and 0.35 <= positions[1] <= 0.65 and positions[2] >= 0.90):
            issues.append(f"{dataset}: pilot IDs do not span short/middle/long query-length strata")
        lengths = [len(next(row["text"] for row in ordered if str(row["id"]) == query_id))
                   for query_id in observed]
        if lengths != sorted(lengths) or len(set(lengths)) != PILOT_QUERIES_PER_DATASET:
            issues.append(f"{dataset}: pilot query text lengths are not strictly short/middle/long")
    return issues


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _forecast_code_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise PilotGateError(f"Invalid non-negative finite {label}: {value!r}")
    return float(value)


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise PilotGateError(f"Required pilot evidence is missing: {path}")


def _expected_keys(embedder: str) -> set[str]:
    return {
        f"{dataset}--{embedder}--{reranker}"
        for dataset in DATASET_COUNTS
        for reranker in ("standalone", *RERANKERS)
    }


def _check_gpu_log(attempt: Path) -> Path:
    event_path = attempt / "server-events.jsonl"
    events = _jsonl(event_path)
    starts = [event for event in events if event.get("command")]
    if len(starts) != 1:
        raise PilotGateError(f"Expected one measured server event in {event_path}")
    command = starts[0]["command"]
    if not isinstance(command, list) or "--n-gpu-layers" not in command:
        raise PilotGateError(f"Server command has no explicit GPU-layer setting: {event_path}")
    if command[command.index("--n-gpu-layers") + 1] != "all":
        raise PilotGateError(f"Server did not request all GPU layers: {event_path}")
    if "--log-file" not in command:
        raise PilotGateError(f"Server command is missing its evidence log path: {event_path}")
    log_path = Path(command[command.index("--log-file") + 1])
    if not log_path.is_absolute():
        log_path = attempt / log_path
    log_path = log_path.resolve()
    try:
        log_path.relative_to(attempt.resolve())
    except ValueError as exc:
        raise PilotGateError(f"GPU log escaped accepted attempt directory: {log_path}") from exc
    _require_file(log_path)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    offloads = re.findall(r"offloaded\s+(\d+)/(\d+) layers to GPU", text, flags=re.IGNORECASE)
    if not offloads or int(offloads[-1][0]) <= 0 or int(offloads[-1][1]) <= 0:
        raise PilotGateError(f"Actual GPU offload evidence is missing or zero: {log_path}")
    return log_path


def _verify_candidate_cache(home: Path, dataset: str, embedder: str,
                            query_ids: list[str], committed: dict[str, Any],
                            rows: list[dict[str, Any]]) -> list[Path]:
    standalone_key = f"{dataset}--{embedder}--standalone"
    standalone_attempt = home / committed[standalone_key]["attempt"]
    source = standalone_attempt / "cache" / embedder / dataset
    run_cache = home / "cache" / embedder / dataset
    source_manifest_path = source / "cache-manifest.json"
    run_manifest_path = run_cache / "cache-manifest.json"
    _require_file(source_manifest_path)
    _require_file(run_manifest_path)
    source_manifest = _json(source_manifest_path)
    run_manifest = _json(run_manifest_path)
    if source_manifest != run_manifest:
        raise PilotGateError(f"{dataset}: accepted run cache differs from the standalone attempt cache")
    if source_manifest.get("query_ids") != query_ids or source_manifest.get("candidate_k") != 50:
        raise PilotGateError(f"{dataset}: candidate manifest query selection or k is wrong")
    unsigned = {key: value for key, value in source_manifest.items() if key not in ("cache_key_sha256", "files_sha256")}
    if source_manifest.get("cache_key_sha256") != sha256_json(unsigned):
        raise PilotGateError(f"{dataset}: candidate cache key hash is invalid")
    cache_files = []
    for filename, digest in source_manifest.get("files_sha256", {}).items():
        for directory in (source, run_cache):
            path = directory / filename
            _require_file(path)
            if sha256_file(path) != digest:
                raise PilotGateError(f"{dataset}: candidate cache file hash mismatch: {path}")
            cache_files.append(path)

    candidates = _json(source / "candidates.json")
    if len(candidates) != len(query_ids) or [str(row.get("query_id")) for row in candidates] != list(map(str, query_ids)):
        raise PilotGateError(f"{dataset}: candidate cache is incomplete or out of query order")
    candidate_by_query: dict[str, list[str]] = {}
    for candidate in candidates:
        ids = candidate.get("ids")
        if not isinstance(ids, list) or len(ids) != 50 or len(set(map(str, ids))) != 50:
            raise PilotGateError(f"{dataset}: candidate pool is not exactly 50 unique IDs")
        candidate_by_query[str(candidate["query_id"])] = list(map(str, ids))

    for query_id, ids in candidate_by_query.items():
        config_rows = [row for row in rows if row.get("dataset") == dataset and str(row.get("query_id")) == query_id]
        if len(config_rows) != 4:
            raise PilotGateError(f"{dataset}/{query_id}: expected standalone plus three reranker rows")
        for row in config_rows:
            if row.get("reranker") == "standalone":
                if row.get("final_ids") != ids:
                    raise PilotGateError(f"{dataset}/{query_id}: standalone result differs from frozen top-50 pool")
            else:
                if row.get("candidate_ids_sha256") != sha256_json(ids):
                    raise PilotGateError(f"{dataset}/{query_id}: reranker candidate input hash differs")
                if set(map(str, row.get("final_ids", []))) != set(ids) or len(row.get("final_ids", [])) != 50:
                    raise PilotGateError(f"{dataset}/{query_id}: reranker changed the frozen top-50 set")
    for reranker in RERANKERS:
        key = f"{dataset}--{embedder}--{reranker}"
        attempt = home / committed[key]["attempt"]
        cache_input_path = attempt / "cache-input.json"
        _require_file(cache_input_path)
        cache_input = _json(cache_input_path)
        if (cache_input.get("source_commit_key") != standalone_key or
                cache_input.get("query_ids") != query_ids or
                cache_input.get("cache_manifest_sha256") != sha256_file(run_manifest_path)):
            raise PilotGateError(f"{key}: reranker does not reference the accepted standalone cache")
        cache_files.append(cache_input_path)
    cache_files.extend([source_manifest_path, run_manifest_path, source / "candidates.json", run_cache / "candidates.json"])
    return cache_files


def _artifact_hashes(home: Path, embedder: str) -> dict[str, str]:
    fixed = (
        "run-manifest.json", "progress.json", "query-selection.json", "summary.json",
        "query-results.jsonl", "committed.json", "preflight.json", "index-stats.jsonl",
        "traffic-events.jsonl",
    )
    paths: set[Path] = {home / name for name in fixed}
    for path in paths:
        _require_file(path)

    committed = _json(home / "committed.json")
    for key, record in committed.items():
        relative = Path(record.get("attempt", ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise PilotGateError(f"Unsafe accepted attempt path for {key}")
        attempt = home / relative
        for name in ("validity.json", "server-events.jsonl", "cache-input.json"):
            if name != "cache-input.json" or not key.endswith("--standalone"):
                paths.add(attempt / name)
        reranker = key.rsplit("--", 1)[-1]
        paths.add(attempt / ("gate-embedding.json" if reranker == "standalone" else "gate-reranker.json"))
        try:
            paths.add(_check_gpu_log(attempt))
        except PilotGateError:
            raise

    preflight = _json(home / "preflight.json")
    for item in preflight.get("models", {}).values():
        relative = Path(item.get("attempt", ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise PilotGateError("Unsafe preflight attempt path")
        paths.add(home / relative / "validity.json")
        paths.add(home / relative / "token-audit.jsonl")
        paths.add(home / relative / "server-events.jsonl")
        paths.add(_check_gpu_log(home / relative))

    selection = _json(home / "query-selection.json")
    query_ids_by_dataset = {
        dataset: list(map(str, selection.get("datasets", {}).get(dataset, {}).get("query_ids", [])))
        for dataset in DATASET_COUNTS
    }
    rows = _jsonl(home / "query-results.jsonl")
    for dataset in DATASET_COUNTS:
        paths.update(_verify_candidate_cache(home, dataset, embedder, query_ids_by_dataset[dataset], committed, rows))

    result = {}
    resolved_home = home.resolve()
    for original_path in sorted(paths):
        path = original_path.resolve()
        _require_file(path)
        try:
            label = path.relative_to(resolved_home).as_posix()
        except ValueError as exc:
            raise PilotGateError("Pilot evidence path escaped its output directory") from exc
        result[label] = sha256_file(path)
    return result


def audit_and_forecast(home: Path, batch_id: str, embedder: str) -> dict[str, Any]:
    """Audit a completed 72-row pilot and return a frozen decision record.

    Official time is estimated as the actual pilot wall time (which includes
    full-corpus index and token-preflight work) plus 1.25× the additional
    measured query work required to expand 3 pilot queries to each dataset's
    full query count, plus a ten-minute fixed margin.
    """
    issues: list[str] = []
    expected_embedder = dict(BATCHES).get(batch_id)
    if expected_embedder is None or expected_embedder != embedder:
        issues.append(f"Batch/embedder mismatch: {batch_id} expects {expected_embedder}, got {embedder}")

    required = (
        "run-manifest.json", "progress.json", "query-selection.json", "summary.json",
        "query-results.jsonl", "committed.json", "preflight.json", "index-stats.jsonl",
    )
    missing = [name for name in required if not (home / name).is_file()]
    if missing:
        return {
            "schema_version": 1, "status": "blocked", "eligible": False,
            "batch_id": batch_id, "embedder": embedder,
            "issues": [*issues, f"Missing pilot artifacts: {missing}"],
            "forecast": None,
        }

    manifest = _json(home / "run-manifest.json")
    progress = _json(home / "progress.json")
    selection = _json(home / "query-selection.json")
    summaries = _json(home / "summary.json")
    rows = _jsonl(home / "query-results.jsonl")
    committed = _json(home / "committed.json")
    preflight = _json(home / "preflight.json")
    indexes = _jsonl(home / "index-stats.jsonl")

    if manifest.get("batch_id") != batch_id or manifest.get("run_kind") != "pilot":
        issues.append("Run manifest is not this batch's pilot")
    if manifest.get("dataset_ids") != list(DATASET_COUNTS):
        issues.append("Pilot manifest does not cover the frozen six datasets in order")
    if manifest.get("wall_hours") != 2.0:
        issues.append("Pilot did not use the fixed two-hour wall-clock budget")
    if progress.get("phase") != "pilot_complete":
        issues.append(f"Pilot progress phase is not pilot_complete: {progress.get('phase')!r}")
    if progress.get("summaries") != SUMMARIES_PER_EMBEDDER or progress.get("query_rows") != PILOT_ROWS or progress.get("errors") != 0:
        issues.append("Pilot final progress does not report 24 summaries, 72 rows, and zero errors")
    if len(summaries) != SUMMARIES_PER_EMBEDDER or len(rows) != PILOT_ROWS:
        issues.append(f"Pilot has {len(summaries)} summaries / {len(rows)} rows; expected 24 / 72")

    expected_summary_keys = {
        (dataset, embedder, reranker)
        for dataset in DATASET_COUNTS
        for reranker in ("standalone", *RERANKERS)
    }
    observed_summary_keys = {
        (item.get("dataset"), item.get("embedding"), item.get("reranker")) for item in summaries
    }
    if observed_summary_keys != expected_summary_keys:
        issues.append("Pilot summary dataset/configuration matrix differs from the frozen 6×4 matrix")
    if any(item.get("queries") != PILOT_QUERIES_PER_DATASET or item.get("errors") != 0 for item in summaries):
        issues.append("Each pilot summary must have exactly 3 queries and zero errors")
    if any(row.get("status") != "ok" or row.get("embedding") != embedder for row in rows):
        issues.append("Pilot contains a failed row or a different embedder")

    selected = selection.get("datasets", {})
    if selection.get("run_kind") != "pilot" or set(selected) != set(DATASET_COUNTS):
        issues.append("Frozen pilot query selection does not cover all six datasets")
    if selection.get("selection_policy") != "deterministic-short-mid-long-query-length-quantiles":
        issues.append("Pilot query selection policy is not the frozen length-stratified policy")
    expected_query_ids: dict[str, set[str]] = {}
    for dataset in DATASET_COUNTS:
        query_ids = selected.get(dataset, {}).get("query_ids", [])
        if len(query_ids) != PILOT_QUERIES_PER_DATASET or len(set(query_ids)) != PILOT_QUERIES_PER_DATASET:
            issues.append(f"{dataset}: pilot must use three unique frozen length-stratified query IDs")
        expected_query_ids[dataset] = set(map(str, query_ids))
        for reranker in ("standalone", *RERANKERS):
            group = [row for row in rows if row.get("dataset") == dataset and row.get("reranker") == reranker]
            ids = [str(row.get("query_id")) for row in group]
            if len(ids) != PILOT_QUERIES_PER_DATASET or set(ids) != expected_query_ids[dataset]:
                issues.append(f"{dataset}/{reranker}: measured rows differ from frozen pilot query IDs")

    expected_commits = _expected_keys(embedder)
    if set(committed) != expected_commits:
        issues.append("Pilot accepted commits do not exactly cover all 24 configurations")
    traffic_events = _jsonl(home / "traffic-events.jsonl")
    accepted_combos = [event.get("combo") for event in traffic_events if event.get("event") == "combo_accepted"]
    if set(accepted_combos) != expected_commits or len(accepted_combos) != len(expected_commits):
        issues.append("Fail-closed traffic ledger does not contain exactly one acceptance per pilot combo")
    passed_audits = {event.get("model") for event in traffic_events if event.get("event") == "token_audit_passed"}
    if passed_audits != {embedder, *RERANKERS}:
        issues.append("Fail-closed traffic ledger is missing one or more full-corpus token-audit passes")
    for key, record in committed.items():
        relative = Path(record.get("attempt", ""))
        if relative.is_absolute() or ".." in relative.parts:
            issues.append(f"{key}: accepted attempt path is unsafe")
            continue
        attempt = home / relative
        validity_path = attempt / "validity.json"
        gate_file = attempt / ("gate-embedding.json" if key.endswith("--standalone") else "gate-reranker.json")
        if not validity_path.is_file() or _json(validity_path).get("status") != "accepted":
            issues.append(f"{key}: accepted attempt validity is missing or not accepted")
        if not gate_file.is_file():
            issues.append(f"{key}: required embedding/reranker fixture gate is missing")
        elif key.endswith("--standalone"):
            gate = _json(gate_file)
            if not gate.get("repeatability") or len(gate.get("fixtures", [])) != 10:
                issues.append(f"{key}: ten-fixture embedding/repeatability gate did not pass")
        else:
            gate = _json(gate_file)
            if gate.get("status") != "passed" or len(gate.get("fixtures", [])) != 10 or gate.get("timing_included") is not False:
                issues.append(f"{key}: ten-fixture reranker warmup gate is incomplete")
        try:
            _check_gpu_log(attempt)
        except (OSError, PilotGateError) as exc:
            issues.append(f"{key}: GPU offload audit failed: {exc}")

    reranker_models = (embedder, *RERANKERS)
    if preflight.get("model_ids") != list(reranker_models):
        issues.append("Full-corpus token preflight model list is incomplete or reordered")
    if set(preflight.get("models", {})) != set(reranker_models):
        issues.append("Full-corpus token preflight does not contain all four models")
    for model in reranker_models:
        item = preflight.get("models", {}).get(model, {})
        if item.get("status") != "passed":
            issues.append(f"{model}: full-corpus token preflight did not pass")
            continue
        relative = Path(item.get("attempt", ""))
        if relative.is_absolute() or ".." in relative.parts:
            issues.append(f"{model}: unsafe preflight artifact path")
            continue
        audit_path = home / relative / "token-audit.jsonl"
        if not audit_path.is_file():
            issues.append(f"{model}: token-audit.jsonl is missing")
            continue
        validity_path = home / relative / "validity.json"
        if not validity_path.is_file() or _json(validity_path).get("status") != "passed":
            issues.append(f"{model}: token-audit attempt validity is missing or not passed")
        try:
            _check_gpu_log(home / relative)
        except (OSError, PilotGateError) as exc:
            issues.append(f"{model}: token-audit GPU offload audit failed: {exc}")
        audit = _jsonl(audit_path)
        if len(audit) != len(DATASET_COUNTS) or {row.get("dataset") for row in audit} != set(DATASET_COUNTS):
            issues.append(f"{model}: token audit does not cover every dataset exactly once")
        if any(row.get("status") != "ok" or row.get("truncated") is not False for row in audit):
            issues.append(f"{model}: token audit contains a block or truncation")

    index_by_dataset = {row.get("dataset"): row for row in indexes}
    if set(index_by_dataset) != set(DATASET_COUNTS) or len(indexes) != len(DATASET_COUNTS):
        issues.append("Pilot did not produce exactly one full-corpus index timing for each dataset")
    for dataset, expected in DATASET_COUNTS.items():
        if dataset in index_by_dataset:
            if index_by_dataset[dataset].get("corpus_docs") != expected["corpus"]:
                issues.append(f"{dataset}: pilot index did not use the full frozen corpus")
            try:
                _finite_nonnegative(index_by_dataset[dataset].get("corpus_ms"), f"{dataset} corpus_ms")
            except PilotGateError as exc:
                issues.append(str(exc))

    try:
        pilot_elapsed = _finite_nonnegative(progress.get("elapsed_seconds"), "pilot elapsed_seconds")
        standalone_ms = 0.0
        candidate_ms_by_dataset: dict[str, list[float]] = {}
        rerank_ms_by_dataset: dict[tuple[str, str], list[float]] = {}
        for row in rows:
            dataset, reranker = row.get("dataset"), row.get("reranker")
            if reranker == "standalone":
                elapsed_ms = _finite_nonnegative(row.get("embedding_ms"), "embedding_ms") + _finite_nonnegative(row.get("retrieval_ms"), "retrieval_ms")
                candidate_ms_by_dataset.setdefault(str(dataset), []).append(elapsed_ms)
                standalone_ms += elapsed_ms
            else:
                key = (str(dataset), str(reranker))
                elapsed_ms = _finite_nonnegative(row.get("rerank_ms"), "rerank_ms")
                rerank_ms_by_dataset.setdefault(key, []).append(elapsed_ms)

        if any(not values or max(values) <= 0 for values in candidate_ms_by_dataset.values()):
            raise PilotGateError("Pilot standalone query timings are missing or all zero")
        if set(candidate_ms_by_dataset) != set(DATASET_COUNTS):
            raise PilotGateError("Pilot standalone query timings do not cover all six datasets")
        if len(rerank_ms_by_dataset) != len(DATASET_COUNTS) * len(RERANKERS):
            raise PilotGateError("Pilot reranker timings do not cover all 18 dataset/config groups")
        if any(not values or max(values) <= 0 for values in rerank_ms_by_dataset.values()):
            raise PilotGateError("Pilot reranker timings are missing or all zero")

        pilot_query_compute_s = standalone_ms / 1000.0 + sum(sum(values) for values in rerank_ms_by_dataset.values()) / 1000.0
        full_candidate_s = 0.0
        full_rerank_s = 0.0
        per_dataset: dict[str, Any] = {}
        for dataset, counts in DATASET_COUNTS.items():
            sample_candidate_ms = candidate_ms_by_dataset.get(dataset, [])
            candidate_s = (max(sample_candidate_ms) * counts["queries"]) / 1000.0
            rerank_s_by_name = {}
            for reranker in RERANKERS:
                sample_ms = rerank_ms_by_dataset.get((dataset, reranker), [])
                projected_s = (max(sample_ms) * counts["queries"]) / 1000.0 if sample_ms else 0.0
                rerank_s_by_name[reranker] = projected_s
                full_rerank_s += projected_s
            full_candidate_s += candidate_s
            per_dataset[dataset] = {
                "pilot_candidate_sample_max_ms": max(sample_candidate_ms),
                "candidate_stage_seconds": candidate_s,
                "rerank_sample_max_ms": {name: max(rerank_ms_by_dataset[(dataset, name)]) for name in RERANKERS},
                "rerank_seconds": rerank_s_by_name,
            }

        projected_query_compute_s = full_candidate_s + full_rerank_s
        incremental_query_s = max(0.0, projected_query_compute_s - pilot_query_compute_s)
        safety_margin_s = max(FIXED_SAFETY_SECONDS, pilot_elapsed * 0.15)
        conservative_s = pilot_elapsed + QUERY_WORK_MULTIPLIER * incremental_query_s + safety_margin_s
        forecast = {
            "pilot_elapsed_seconds": pilot_elapsed,
            "pilot_measured_query_compute_seconds": pilot_query_compute_s,
            "projected_full_query_compute_seconds": projected_query_compute_s,
            "projected_full_candidate_stage_seconds": full_candidate_s,
            "projected_full_rerank_seconds": full_rerank_s,
            "incremental_query_work_seconds": incremental_query_s,
            "incremental_work_multiplier": QUERY_WORK_MULTIPLIER,
            "sampling_policy": "For each dataset and configuration, extrapolate the maximum of its 3 length-stratified pilot query timings to the full query count; apply the 1.25 multiplier only to incremental query work. Three samples are not a statistically stable p95.",
            "fixed_safety_margin_seconds": safety_margin_s,
            "conservative_official_seconds": conservative_s,
            "conservative_official_hours": conservative_s / 3600.0,
            "official_gate_hours": MAX_OFFICIAL_FORECAST_HOURS,
            "per_dataset": per_dataset,
            "traffic_wait_note": "Pilot wall time includes observed guarded waits; future external traffic wait remains uncertain and is bounded only by the immutable 12-hour official deadline.",
        }
    except PilotGateError as exc:
        forecast = None
        issues.append(str(exc))

    if forecast and forecast["conservative_official_hours"] > MAX_OFFICIAL_FORECAST_HOURS:
        issues.append(
            f"Conservative official forecast {forecast['conservative_official_hours']:.2f}h exceeds the 10h launch gate"
        )

    source_environment = {
        key: manifest.get(key)
        for key in ("dataset_sha256", "binary_sha256", "model_sha256", "code_sha256")
    }
    source_environment["pilot_forecast_sha256"] = _forecast_code_sha256()
    try:
        artifact_hashes = _artifact_hashes(home, embedder)
    except (OSError, ValueError, PilotGateError) as exc:
        artifact_hashes = {}
        issues.append(f"Could not fingerprint all pilot evidence: {exc}")

    eligible = not issues and forecast is not None and forecast["conservative_official_hours"] <= MAX_OFFICIAL_FORECAST_HOURS
    return {
        "schema_version": 1,
        "status": "approved" if eligible else "blocked",
        "eligible": eligible,
        "batch_id": batch_id,
        "embedder": embedder,
        "pilot_home": str(home.resolve()),
        "expected_pilot_summaries": SUMMARIES_PER_EMBEDDER,
        "expected_pilot_rows": PILOT_ROWS,
        "issues": issues,
        "source_environment": source_environment,
        "artifact_sha256": artifact_hashes,
        "forecast": forecast,
    }


def verify_official_gate(decision_path: Path, pilot_home: Path, batch_id: str,
                         embedder: str, current_environment: dict[str, Any]) -> dict[str, Any]:
    """Require an unchanged, approved pilot and identical pinned environment."""
    if not decision_path.is_file():
        raise PilotGateError(f"Official run requires the completed pilot decision: {decision_path}")
    decision = _json(decision_path)
    if decision.get("status") != "approved" or decision.get("eligible") is not True:
        raise PilotGateError("Official run blocked: pilot decision is not approved")
    if decision.get("batch_id") != batch_id or decision.get("embedder") != embedder:
        raise PilotGateError("Official run blocked: pilot decision belongs to another batch/embedder")
    if Path(decision.get("pilot_home", "")).resolve() != pilot_home.resolve():
        raise PilotGateError("Official run blocked: pilot output path differs from decision")
    forecast = decision.get("forecast")
    if not isinstance(forecast, dict):
        raise PilotGateError("Official run blocked: pilot forecast is missing")
    hours = _finite_nonnegative(forecast.get("conservative_official_hours"), "conservative official forecast")
    if hours > MAX_OFFICIAL_FORECAST_HOURS:
        raise PilotGateError(f"Official run blocked: {hours:.2f}h forecast exceeds 10h gate")
    expected_environment = decision.get("source_environment")
    current_subset = {key: current_environment.get(key) for key in ("dataset_sha256", "binary_sha256", "model_sha256", "code_sha256")}
    current_subset["pilot_forecast_sha256"] = _forecast_code_sha256()
    if expected_environment != current_subset:
        raise PilotGateError("Official run blocked: pilot data/model/binary/code hashes differ from current environment")
    current_artifacts = _artifact_hashes(pilot_home, embedder)
    if not current_artifacts or current_artifacts != decision.get("artifact_sha256"):
        raise PilotGateError("Official run blocked: pilot evidence changed after its decision was written")
    return decision


def write_decision(home: Path, decision: dict[str, Any]) -> Path:
    path = home / "pilot-decision.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    return path
