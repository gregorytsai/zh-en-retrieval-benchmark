#!/usr/bin/env python3
"""Read-only integrity and summary audit for the guarded v3 benchmark.

Run from this directory with:
    python3 audit_v3.py

The script does not start inference, modify benchmark artifacts, or contact
local model servers. It prints a JSON audit and descriptive statistics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DIRECTIONS = ("eng-zho", "zho-zho", "eng-eng")
CORE_EMBEDDERS = ("qwen-0.6b", "qwen-4b", "qwen-8b", "bge-m3", "embeddinggemma-2")
STANDALONE_ONLY = ("f2llm-v2-0.6b", "granite-97m-r2")
RERANKERS = ("bge-reranker", "qwen-reranker-0.6b", "qwen-reranker-4b")
METRICS = ("top1", "mrr10", "ndcg10", "recall10", "recall50", "hit10", "candidate_recall50")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def qkey(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (row["config"], row["embedding"], row["reranker"], row["query_id"])


def group_key(row: tuple[str, str, str]) -> str:
    return "--".join(row[:3])


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def record_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {m: round(mean([float(r[m]) for r in rows]), 6) for m in METRICS}
    for field in ("embedding_ms", "retrieval_ms", "rerank_ms", "total_ms"):
        values = [float(r[field]) for r in rows if r.get(field) is not None]
        result[f"{field}_mean"] = round(mean(values), 3) if values else None
        result[f"{field}_p50"] = round(percentile(values, 0.50), 3) if values else None
        result[f"{field}_p95"] = round(percentile(values, 0.95), 3) if values else None
    result["queries"] = len(rows)
    result["errors"] = sum(r.get("status") != "ok" for r in rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("run-20261022-v3-guarded"), help="guarded run directory")
    args = parser.parse_args()
    run_dir = args.run.resolve()
    root = run_dir.parent

    progress = read_json(run_dir / "progress.json")
    summary = read_json(run_dir / "summary.json")
    committed = read_json(run_dir / "committed.json")
    main_rows = read_jsonl(run_dir / "query-results.jsonl")
    traffic = read_jsonl(run_dir / "traffic-events.jsonl")
    run_manifest = read_json(run_dir / "run-manifest.json")

    expected = {
        f"{direction}--{embedder}--{reranker}"
        for direction in DIRECTIONS
        for embedder in CORE_EMBEDDERS
        for reranker in ("standalone", *RERANKERS)
    }
    expected |= {
        f"{direction}--{embedder}--standalone"
        for direction in DIRECTIONS
        for embedder in STANDALONE_ONLY
    }

    summary_by_key = {group_key((r["config"], r["embedding"], r["reranker"])): r for r in summary}
    main_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    duplicate_main = []
    for row in main_rows:
        key = qkey(row)
        if key in main_by_key:
            duplicate_main.append(key)
        main_by_key[key] = row

    accepted_attempt_rows: list[dict[str, Any]] = []
    accepted_paths: set[str] = set()
    accepted_validity: dict[str, str] = {}
    accepted_gpu_logs = []
    gpu_by_model: Counter[str] = Counter()
    gpu_failures = []
    accepted_group_checks = []
    index_rows: list[dict[str, Any]] = []
    for key, entry in committed.items():
        rel = entry["attempt"]
        accepted_paths.add(rel)
        attempt_dir = run_dir / rel
        validity = read_json(attempt_dir / "validity.json")
        accepted_validity[key] = validity.get("status", "missing")
        rows = read_jsonl(attempt_dir / "query-results.jsonl")
        accepted_attempt_rows.extend(rows)
        index_path = attempt_dir / "index-stats.jsonl"
        if index_path.is_file():
            index_rows.extend(read_jsonl(index_path))
        expected_rows = [r for r in main_rows if "--".join((r["config"], r["embedding"], r["reranker"])) == key]
        is_equal = rows == expected_rows
        accepted_group_checks.append({"key": key, "queries": len(rows), "main_rows_equal": is_equal})

        for event in read_jsonl(attempt_dir / "server-events.jsonl"):
            command = event.get("command")
            if not command:
                continue
            model = event.get("model", "unknown")
            expected_binary = str(run_manifest["binary"])

            def flag_is(flag: str, expected_value: str) -> bool:
                return flag in command and command[command.index(flag) + 1] == expected_value

            command_ok = (
                command[0] == expected_binary
                and flag_is("--n-gpu-layers", "all")
                and flag_is("--log-verbosity", "4")
                and flag_is("--ctx-size", "8192")
                and flag_is("--parallel", "1")
                and flag_is("--threads", "8")
                and flag_is("--threads-batch", "8")
                and flag_is("--batch-size", "4096")
                and flag_is("--ubatch-size", "4096")
            )
            log_file = command[command.index("--log-file") + 1] if "--log-file" in command else None
            log_path = Path(log_file) if log_file else None
            if log_path and not log_path.is_file():
                # The server arguments preserve the original absolute path; allow
                # an archived run folder to be relocated while keeping its log.
                relocated = attempt_dir / "logs" / log_path.name
                if relocated.is_file():
                    log_path = relocated
            log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path and log_path.is_file() else ""
            found = re.findall(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", log_text)
            full_offload = bool(found) and all(int(a) == int(b) for a, b in found)
            good = command_ok and full_offload
            accepted_gpu_logs.append({
                "attempt": rel,
                "model": model,
                "log": str(log_path) if log_path else None,
                "offload": [f"{a}/{b}" for a, b in found],
                "command_ok": command_ok,
                "full_offload": full_offload,
            })
            if good:
                gpu_by_model[model] += 1
            else:
                gpu_failures.append({"attempt": rel, "model": model, "command_ok": command_ok, "offload": found})

    attempts_root = run_dir / "attempts"
    all_attempt_dirs = sorted(p for p in attempts_root.iterdir() if p.is_dir())
    invalid_attempts = []
    other_attempts = []
    for attempt_dir in all_attempt_dirs:
        rel = str(attempt_dir.relative_to(run_dir))
        validity_path = attempt_dir / "validity.json"
        validity = read_json(validity_path) if validity_path.is_file() else {"status": "missing"}
        if rel in accepted_paths:
            continue
        if validity.get("status") == "invalidated":
            invalid_attempts.append({
                "attempt": rel,
                "reason": validity.get("reason"),
                "query_rows": len(read_jsonl(attempt_dir / "query-results.jsonl")) if (attempt_dir / "query-results.jsonl").is_file() else 0,
            })
        else:
            other_attempts.append({"attempt": rel, "status": validity.get("status")})

    main_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in main_rows:
        main_groups[group_key((row["config"], row["embedding"], row["reranker"]))].append(row)
    standalone = {k: v for k, v in main_groups.items() if k.endswith("--standalone")}
    reranker_stats: dict[str, Any] = {}
    paired_delta: dict[str, Any] = {}
    for reranker in RERANKERS:
        rows = [r for r in main_rows if r["reranker"] == reranker]
        reranker_stats[reranker] = record_metrics(rows)
        deltas: dict[str, list[float]] = {m: [] for m in METRICS}
        for row in rows:
            base_key = "--".join((row["config"], row["embedding"], "standalone"))
            base_rows = {r["query_id"]: r for r in standalone.get(base_key, [])}
            base = base_rows.get(row["query_id"])
            if base is not None:
                for metric in METRICS:
                    deltas[metric].append(float(row[metric]) - float(base[metric]))
        paired_delta[reranker] = {
            metric: {"mean": round(mean(values), 6), "n": len(values)}
            for metric, values in deltas.items()
        }

    standalone_by_embedder: dict[str, Any] = {}
    for embedder in (*CORE_EMBEDDERS, *STANDALONE_ONLY):
        rows = [r for r in main_rows if r["embedding"] == embedder and r["reranker"] == "standalone"]
        standalone_by_embedder[embedder] = record_metrics(rows)

    index_by_embedder: dict[str, Any] = {}
    for embedder in (*CORE_EMBEDDERS, *STANDALONE_ONLY):
        parts = [r for r in index_rows if r.get("model") == embedder]
        docs = sum(int(r["corpus_docs"]) for r in parts)
        seconds = sum(float(r["corpus_ms"]) for r in parts) / 1000
        index_by_embedder[embedder] = {
            "directions": len(parts),
            "corpus_docs_total": docs,
            "corpus_seconds_total": round(seconds, 3),
            "weighted_docs_per_second": round(docs / seconds, 2) if seconds else None,
            "mean_corpus_seconds_per_direction": round(seconds / len(parts), 3) if parts else None,
        }

    core_embedding_by_reranker: dict[str, Any] = {}
    for embedder in CORE_EMBEDDERS:
        core_embedding_by_reranker[embedder] = {}
        for reranker in ("standalone", *RERANKERS):
            rows = [r for r in main_rows if r["embedding"] == embedder and r["reranker"] == reranker]
            core_embedding_by_reranker[embedder][reranker] = record_metrics(rows)

    confidence = read_json(run_dir / "paired-confidence.json")
    confidence_by_reranker: dict[str, Any] = {}
    for reranker in RERANKERS:
        records = [r for r in confidence if r.get("reranker") == reranker]
        lowers = [float(r["paired_bootstrap95"][0]) for r in records]
        uppers = [float(r["paired_bootstrap95"][1]) for r in records]
        confidence_by_reranker[reranker] = {
            "groups": len(records),
            "groups_ci_lower_above_zero": sum(lo > 0 for lo in lowers),
            "groups_ci_crosses_zero": sum(lo <= 0 <= hi for lo, hi in zip(lowers, uppers)),
            "minimum_ci_lower": min(lowers) if lowers else None,
            "maximum_ci_upper": max(uppers) if uppers else None,
            "interpretation": "per-group exploratory paired bootstrap; no multiple-comparison correction",
        }

    by_direction: dict[str, dict[str, Any]] = {}
    core_standalone_by_direction: dict[str, Any] = {}
    for direction in DIRECTIONS:
        by_direction[direction] = {}
        for reranker in ("standalone", *RERANKERS):
            rows = [r for r in main_rows if r["config"] == direction and r["reranker"] == reranker]
            by_direction[direction][reranker] = record_metrics(rows)
        core_rows = [
            r for r in main_rows
            if r["config"] == direction and r["reranker"] == "standalone" and r["embedding"] in CORE_EMBEDDERS
        ]
        core_standalone_by_direction[direction] = record_metrics(core_rows)

    validity_counts = Counter(accepted_validity.values())
    event_counts = Counter(e.get("event", "unknown") for e in traffic)
    expected_rows_by_key = Counter(group_key((r["config"], r["embedding"], r["reranker"])) for r in main_rows)
    checks = {
        "phase_complete": progress.get("phase") == "complete",
        "expected_matrix_66": expected == set(committed) == set(summary_by_key),
        "progress_counts_match": progress.get("summaries") == len(summary) == len(committed) == 66
        and progress.get("query_rows") == len(main_rows) == 6600,
        "all_groups_100_queries": len(expected_rows_by_key) == 66 and set(expected_rows_by_key.values()) == {100},
        "zero_errors": progress.get("errors") == 0
        and sum(int(r.get("errors", -1)) for r in summary) == 0
        and all(r.get("status") == "ok" for r in main_rows),
        "unique_main_keys": len(main_by_key) == len(main_rows) == 6600 and not duplicate_main,
        "accepted_validity_all_accepted": len(accepted_validity) == 66 and validity_counts == Counter({"accepted": 66}),
        "accepted_attempt_rows_exactly_main": len(accepted_attempt_rows) == 6600
        and all(c["queries"] == 100 and c["main_rows_equal"] for c in accepted_group_checks),
        "invalid_attempts_excluded": len(invalid_attempts) == 1 and not any(i["attempt"] in accepted_paths for i in invalid_attempts),
        "all_accepted_gpu_offload_verified": len(accepted_gpu_logs) == 66 and not gpu_failures,
        "one_external_activity_and_retry": event_counts.get("external_activity", 0) == 1
        and event_counts.get("combo_started", 0) == 67
        and event_counts.get("combo_accepted", 0) == 66,
        "no_unclassified_attempts": not other_attempts,
    }

    report = {
        "run": str(run_dir),
        "progress": progress,
        "checks": checks,
        "ok": all(checks.values()),
        "matrix": {
            "expected_groups": len(expected),
            "committed_groups": len(committed),
            "summary_groups": len(summary),
            "main_query_rows": len(main_rows),
            "unique_main_query_keys": len(main_by_key),
            "queries_per_group": sorted(set(expected_rows_by_key.values())),
            "errors": sum(int(r.get("errors", 0)) for r in summary),
            "accepted_validity_statuses": dict(validity_counts),
            "invalid_attempts": invalid_attempts,
            "unclassified_attempts": other_attempts,
            "traffic_event_counts": dict(event_counts),
            "gpu_log_count": len(accepted_gpu_logs),
            "gpu_offload_by_server_model": dict(gpu_by_model),
            "gpu_failures": gpu_failures,
        },
        "overall_reranker_metrics_and_latency": reranker_stats,
        "paired_delta_vs_same_embedder_standalone": paired_delta,
        "paired_top1_confidence_summary": confidence_by_reranker,
        "standalone_by_embedder": standalone_by_embedder,
        "corpus_indexing_by_embedder": index_by_embedder,
        "core_embedding_by_reranker": core_embedding_by_reranker,
        "core_standalone_by_direction": core_standalone_by_direction,
        "by_direction_and_reranker": by_direction,
    }
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
