#!/usr/bin/env python3
"""Read-only structural audit for the frozen Phase 2 official run."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"AUDIT FAILED: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("run-phase2-v2/official"))
    parser.add_argument("--datasets", type=Path, default=Path("datasets.json"))
    parser.add_argument("--data-validation", type=Path, default=Path("data-validation.json"))
    parser.add_argument("--output", type=Path, default=Path("FINAL_AUDIT.json"))
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    project = run_dir.parents[1]
    manifest = read_json(run_dir / "run-manifest.json")
    progress = read_json(run_dir / "progress.json")
    committed = read_json(run_dir / "committed.json")
    summaries = read_json(run_dir / "summary.json")
    selection = read_json(run_dir / "query-selection.json")
    qrels_data = read_json(run_dir / "qrels.json")
    preflight = read_json(run_dir / "preflight.json")
    deadline = read_json(run_dir / "deadline.json")
    traffic_events = read_jsonl(run_dir / "traffic-events.jsonl")
    rows = read_jsonl(run_dir / "query-results.jsonl")
    validation = read_json(args.data_validation)
    dataset_manifest_hash = sha256(args.datasets)

    require(progress.get("phase") == "complete", "progress phase is not complete")
    require(manifest.get("run_kind") == "official", "run is not official")
    require(manifest.get("production_modified") is False, "manifest says production was modified")
    require(dataset_manifest_hash == manifest.get("dataset_sha256"), "datasets.json SHA differs from run manifest")
    require(manifest.get("matrix_summary_count") == 24, "manifest expected-summary count is not 24")
    require(manifest.get("matrix_full_query_rows") == 13278, "manifest expected-row count is not 13,278")
    require(len(committed) == len(summaries) == 24, "accepted/summary count is not 24")
    require(sum(int(item.get("queries", 0)) for item in summaries) == 13278, "summary query count is not 13,278")
    require(sum(int(item.get("errors", 0)) for item in summaries) == 0, "summary errors are nonzero")
    require(len(rows) == 13278, "main query-results row count is not 13,278")
    require(all(row.get("status") == "ok" for row in rows), "main query-results has a non-ok row")

    workspace = project.parent
    code_paths = {
        "run_guarded.py": project / "run_guarded.py",
        "run_study.py": project / "run_study.py",
        "traffic_guard.py": project / "traffic_guard.py",
        "legacy_run_benchmark.py": workspace / "bilingual-retrieval-benchmark/run_benchmark.py",
    }
    binary_path = workspace / "llama.cpp-qwen38/build/bin/llama-server"
    model_paths = {
        "qwen-4b": workspace / "Qwen3-Embedding-4B.Q8_0.gguf",
        "qwen-8b": workspace / "embedding-study-20261022/models/Qwen3-Embedding-8B-Q8_0.gguf",
        "embeddinggemma-2": workspace / "embedding-study-20261022/models/embeddinggemma-2-Q8_0.gguf",
        "bge-reranker": workspace / "bge-reranker-v2-m3-q8_0.gguf",
        "qwen-reranker-0.6b": workspace / "qwen-reranker-rebuilt/Qwen3-Reranker-0.6B-Q8_0.gguf",
        "qwen-reranker-4b": workspace / "qwen-reranker-rebuilt/Qwen3-Reranker-4B-Q8_0.gguf",
    }
    actual_code_hashes = {name: sha256(path) for name, path in code_paths.items()}
    actual_model_hashes = {name: sha256(path) for name, path in model_paths.items()}
    actual_binary_hash = sha256(binary_path)
    require(actual_code_hashes == manifest["code_sha256"], "current code hashes differ from frozen run manifest")
    require(actual_model_hashes == manifest["model_sha256"], "current model hashes differ from frozen run manifest")
    require(actual_binary_hash == manifest["binary_sha256"], "current llama-server binary hash differs from run manifest")

    selected: dict[str, list[str]] = {}
    selection_metadata: dict[str, dict[str, Any]] = {}
    group_keys: list[tuple[str, str, str]] = []
    for dataset, item in selection["datasets"].items():
        ids = item["query_ids"]
        require(len(ids) == len(set(ids)), f"selection has duplicate query IDs: {dataset}")
        selected[dataset] = ids
        expected_counts = validation["datasets"][dataset]
        require(len(ids) == expected_counts["queries"], f"selection query count mismatch: {dataset}")
        query_metadata = item.get("queries", [])
        require(len(query_metadata) == len(ids), f"query metadata does not cover selection: {dataset}")
        require([q["id"] for q in query_metadata] == ids, f"query metadata order differs from selection: {dataset}")
        selection_metadata[dataset] = {
            "previous100_count": sum(q.get("previous100") is True for q in query_metadata),
            "cluster_count": len({q.get("cluster_id") or q["id"] for q in query_metadata}),
            "cluster_units": dict(Counter(q.get("cluster_unit") or "query_id_fallback" for q in query_metadata)),
            "cluster_limitation": item.get("dataset_metadata", {}).get("cluster_limitation"),
        }
        qrels = qrels_data["datasets"][dataset]["qrels"]
        require(set(qrels) == set(ids), f"qrels query keys do not match selection: {dataset}")
        for embedding, reranker in (
            ("embeddinggemma-2", "standalone"),
            ("qwen-4b", "standalone"),
            ("qwen-8b", "standalone"),
            ("qwen-4b", "bge-reranker"),
            ("qwen-4b", "qwen-reranker-0.6b"),
            ("qwen-4b", "qwen-reranker-4b"),
        ):
            group_keys.append((dataset, embedding, reranker))

    group_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    identities = []
    for row in rows:
        key = (row["dataset"], row["embedding"], row["reranker"])
        group_rows[key].append(row)
        identities.append((*key, row["query_id"]))
    require(len(identities) == len(set(identities)), "duplicate dataset/config/query composite keys in main rows")

    groups: list[dict[str, Any]] = []
    for dataset, embedding, reranker in group_keys:
        key = (dataset, embedding, reranker)
        group = group_rows.get(key, [])
        require(len(group) == len(selected[dataset]), f"wrong main-row count for {key}: {len(group)}")
        require({row["query_id"] for row in group} == set(selected[dataset]), f"query IDs differ from frozen selection for {key}")
        expected_width = 100 if reranker == "standalone" else 50
        require(all(len(row["final_ids"]) == expected_width for row in group), f"final_ids width mismatch for {key}")
        require(all(len(row["final_ids"]) == len(set(row["final_ids"])) for row in group), f"duplicate final IDs for {key}")

        commit_key = "--".join(key)
        commit = committed.get(commit_key)
        require(commit is not None, f"missing commit pointer: {commit_key}")
        attempt = run_dir / commit["attempt"]
        validity = read_json(attempt / "validity.json")
        require(validity.get("status") == "accepted", f"attempt validity not accepted: {commit_key}")
        attempt_rows = read_jsonl(attempt / "query-results.jsonl")
        require(len(attempt_rows) == len(selected[dataset]), f"accepted attempt row count mismatch: {commit_key}")
        require({row["query_id"] for row in attempt_rows} == set(selected[dataset]), f"accepted attempt query IDs mismatch: {commit_key}")
        main_by_id = {row["query_id"]: row for row in group}
        for attempt_row in attempt_rows:
            main_row = main_by_id[attempt_row["query_id"]]
            for field in set(attempt_row) & set(main_row):
                require(attempt_row[field] == main_row[field], f"accepted attempt/main-row mismatch for {commit_key}/{attempt_row['query_id']}/{field}")

        gpu_servers = []
        for event in read_jsonl(attempt / "server-events.jsonl"):
            command = event.get("command")
            if not command:
                continue
            require("--n-gpu-layers" in command and command[command.index("--n-gpu-layers") + 1] == "all",
                    f"server did not request all GPU layers: {commit_key}")
            log_path = Path(command[command.index("--log-file") + 1])
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            offloads = re.findall(r"offloaded\s+(\d+)/(\d+) layers to GPU", log_text)
            require(bool(offloads), f"no GPU offload evidence in server log: {log_path}")
            gpu_servers.append({
                "model": event.get("model"),
                "log": str(log_path.relative_to(project)) if log_path.is_relative_to(project) else str(log_path),
                "offloaded_layers": int(offloads[-1][0]),
                "total_layers": int(offloads[-1][1]),
            })
        expected_servers = 1
        require(len(gpu_servers) == expected_servers, f"wrong GPU-backed server count for {commit_key}: {len(gpu_servers)}")
        if reranker != "standalone":
            cache_input = read_json(attempt / "cache-input.json")
            source_key = f"{dataset}--qwen-4b--standalone"
            require(cache_input.get("source_commit_key") == source_key,
                    f"reranker did not use its accepted Qwen4B standalone embedding cache: {commit_key}")
            require(source_key in committed, f"reranker source embedding is not committed: {source_key}")

        fixture_gate = None
        if reranker != "standalone":
            fixture_gate = read_json(attempt / "gate-reranker.json")
            fixtures = fixture_gate.get("fixtures", [])
            require(fixture_gate.get("status") == "passed" and len(fixtures) == 10,
                    f"reranker warmup gate did not pass 10 fixtures: {commit_key}")
            require(fixture_gate.get("timing_included") is False, f"warmups counted in timed rows: {commit_key}")
            require(all(item.get("timing_included") is False and item.get("positive_score", 0) > item.get("negative_score", 0)
                        and item.get("order") == [0, 1] for item in fixtures),
                    f"reranker positive/negative fixture assertion failed: {commit_key}")

        metric_rows = group
        rerank_times = [float(row["rerank_ms"]) for row in metric_rows if row.get("rerank_ms") is not None]
        groups.append({
            "dataset": dataset, "embedding": embedding, "reranker": reranker,
            "queries": len(group), "errors": 0,
            "final_ids_per_query": expected_width,
            "top1": sum(float(row["top1"]) for row in group) / len(group),
            "mrr10": sum(float(row["mrr10"]) for row in group) / len(group),
            "ndcg10": sum(float(row["ndcg10"]) for row in group) / len(group),
            "rerank_ms_p50": quantile(rerank_times, 0.5),
            "rerank_ms_p95": quantile(rerank_times, 0.95),
            "accepted_attempt": commit["attempt"],
            "gpu_servers": gpu_servers,
            "reranker_fixture_gate": {
                "status": fixture_gate["status"], "fixtures": len(fixture_gate["fixtures"]),
                "timing_included": fixture_gate["timing_included"],
            } if fixture_gate else None,
        })

    expected_combo_keys = {"--".join(key) for key in group_keys}
    require(set(committed) == expected_combo_keys, "committed keys differ from frozen six-config matrix")

    preflight_models = preflight.get("models", {})
    require(len(preflight_models) == 6 and all(value.get("status") == "passed" for value in preflight_models.values()),
            "one or more of the six model preflights did not pass")
    preflight_audits: list[dict[str, Any]] = []
    for model, value in preflight_models.items():
        token_rows = read_jsonl(run_dir / value["attempt"] / "token-audit.jsonl")
        require(token_rows and all(item.get("status") == "ok" and not item.get("truncated", False) for item in token_rows),
                f"preflight token audit has a failure or truncation: {model}")
        preflight_audits.extend(token_rows)
    require(len(preflight_audits) == len(preflight_models) * len(selected),
            "preflight token audit does not cover every model × dataset pair")
    bge_acord = next(item for item in preflight_audits if item.get("model") == "bge-reranker" and item.get("dataset") == "acord-eng-eng")
    bge_bound = (int(bge_acord["max_document"]["tokens"])
                 + int(bge_acord["max_query"]["tokens"])
                 + int(bge_acord["reranker_special_token_overhead"]))
    require(bge_bound <= int(bge_acord["max_tokens"]), "BGE ACORD pair bound exceeds 4,096 tokens")

    qrel_stats = {}
    for dataset, item in qrels_data["datasets"].items():
        judgments = [int(grade) for per_query in item["qrels"].values() for grade in per_query.values()]
        grade_counts = dict(sorted(Counter(judgments).items()))
        qrel_stats[dataset] = {
            "grade_mode": item["grade_mode"], "queries": len(item["qrels"]),
            "judgments": len(judgments), "grade_counts": {str(key): value for key, value in grade_counts.items()},
        }
        require(item["grade_mode"] == validation["datasets"][dataset]["grade_mode"], f"grade mode mismatch: {dataset}")

    paired_path = run_dir / "paired-confidence.json"
    paired = read_json(paired_path)
    require(paired and all(item.get("bootstrap_replicates") == 2000 for item in paired), "paired CI bootstrap metadata mismatch")
    traffic_counts = dict(Counter(event.get("event", "unknown") for event in traffic_events))
    require(traffic_counts.get("combo_accepted") == 24, "traffic ledger does not show 24 accepted combinations")
    require(traffic_counts.get("token_audit_passed") == 6, "traffic ledger does not show six passed token audits")
    require(not any(event.get("event") in {"interference", "retry", "blocked", "fatal_error"} for event in traffic_events),
            "traffic ledger contains interference/retry/blocker/fatal event")

    start = float(manifest["original_started_epoch"])
    deadline_epoch = float(manifest["deadline_epoch"])
    require(abs(deadline_epoch - (start + float(manifest["wall_hours"]) * 3600)) < 0.01,
            "original deadline is not start + wall-hours")
    resume_history = read_jsonl(run_dir / "resume-history.jsonl")
    require(len(resume_history) == 1, "expected exactly one resume record for the interrupted official run")
    require(all(item.get("original_deadline") == manifest["deadline_utc"] for item in resume_history),
            "resume history shows a reset/changed original deadline")

    interrupted_attempt = "attempts/eng-zho--qwen-4b--qwen-reranker-4b--1792761710300535000"
    interrupted_file = run_dir / interrupted_attempt / "query-results.jsonl"
    snapshot_file = run_dir / "recovery-snapshots/20261023-2140/incomplete-attempt/eng-zho--qwen-4b--qwen-reranker-4b--1792761710300535000/query-results.jsonl"
    old_partial_rows = read_jsonl(interrupted_file)
    snapshot_rows = read_jsonl(snapshot_file)
    require(len(old_partial_rows) == len(snapshot_rows), "old partial attempt and recovery snapshot differ in row count")
    require(sha256(interrupted_file) == sha256(snapshot_file), "old partial attempt and recovery snapshot differ in bytes")
    rerun_commit = committed["eng-zho--qwen-4b--qwen-reranker-4b"]
    require(rerun_commit["attempt"] != interrupted_attempt, "committed pointer incorrectly references interrupted partial attempt")
    require(len(group_rows[("eng-zho", "qwen-4b", "qwen-reranker-4b")]) == 504,
            "main table does not contain exactly the full fresh rerun")

    summary_records = [{key: item.get(key) for key in (
        "dataset", "embedding", "reranker", "queries", "errors", "top1", "mrr10", "ndcg10",
        "recall50", "candidate_recall50", "judged_coverage10", "judged_coverage50",
        "embedding_ms_p50", "retrieval_ms_p50", "rerank_ms_p50", "rerank_ms_p95",
        "total_ms_p50", "total_ms_p95", "p5_grade1", "p5_grade2", "p5_grade3", "p5_grade4",
    )} for item in summaries]

    report = {
        "audit_status": "passed",
        "run_kind": "official",
        "completion_time_utc": progress.get("time"),
        "project_root": str(project),
        "matrix": {
            "datasets": {name: {"query_count": len(ids), "selected_ids_unique": len(set(ids)),
                                 "corpus_docs": validation["datasets"][name]["corpus"],
                                 "grade_mode": validation["datasets"][name]["grade_mode"],
                                 **selection_metadata[name]}
                         for name, ids in selected.items()},
            "config_count": 6, "summary_count": len(summaries), "main_query_rows": len(rows),
            "dataset_qualified_queries": len({(row["dataset"], row["query_id"]) for row in rows}),
            "unique_composite_keys": len(set(identities)), "non_ok_rows": 0,
            "group_counts": [{"dataset": d, "embedding": e, "reranker": r, "rows": len(group_rows[(d, e, r)])}
                             for d, e, r in group_keys],
        },
        "integrity": {
            "committed_count": len(committed), "all_committed_attempts_accepted": True,
            "all_attempt_query_sets_match_selection": True, "all_candidate_widths_valid": True,
            "all_summaries_error_free": True,
            "dataset_sha256": dataset_manifest_hash,
            "manifest_sha256": sha256(run_dir / "run-manifest.json"),
            "deadline_epoch": deadline_epoch, "deadline_utc": manifest["deadline_utc"],
            "original_started_epoch": start, "wall_hours": manifest["wall_hours"],
            "deadline_unchanged_on_resume": all(item.get("original_deadline") == manifest["deadline_utc"] for item in resume_history),
            "resume_history": resume_history,
            "code_sha256_actual_matches_manifest": actual_code_hashes,
            "model_sha256_actual_matches_manifest": actual_model_hashes,
            "binary_sha256_actual_matches_manifest": actual_binary_hash,
            "interrupted_attempt": interrupted_attempt,
            "interrupted_partial_rows": len(old_partial_rows),
            "interrupted_snapshot_sha256": sha256(snapshot_file),
            "committed_rerun_attempt": rerun_commit["attempt"],
            "committed_rerun_rows": len(group_rows[("eng-zho", "qwen-4b", "qwen-reranker-4b")]),
            "partial_rows_mixed_into_main": False,
        },
        "preflight": {
            "models": {name: value.get("status") for name, value in preflight_models.items()},
            "token_audit_records": len(preflight_audits), "all_inputs_within_limit_without_truncation": True,
            "acord_max_document_tokens_by_model": {
                item["model"]: item["max_document"]["tokens"]
                for item in preflight_audits if item.get("dataset") == "acord-eng-eng"
            },
            "bge_acord_max_document_tokens": bge_acord["max_document"]["tokens"],
            "bge_acord_max_query_tokens": bge_acord["max_query"]["tokens"],
            "bge_pair_special_token_overhead": bge_acord["reranker_special_token_overhead"],
            "bge_acord_conservative_pair_bound": bge_bound, "token_limit": bge_acord["max_tokens"],
            "bge_model_sha256_verified": bge_acord["bge_template_verification"]["model_sha256"],
        },
        "guard_and_gpu": {
            "traffic_policy": manifest["traffic_policy"], "runner_policy": manifest["runner_policy"],
            "traffic_event_counts": traffic_counts, "accepted_combinations": 24,
            "accepted_attempts_gpu_verified": len(groups),
            "measurement_server_processes_gpu_verified": sum(len(group["gpu_servers"]) for group in groups),
            "all_accepted_attempt_servers_requested_all_gpu_layers": True,
            "all_accepted_attempt_server_logs_show_offload": True,
            "reranker_fixture_gate_count": sum(group["reranker_fixture_gate"] is not None for group in groups),
            "reranker_fixture_instances": sum((group["reranker_fixture_gate"] or {}).get("fixtures", 0) for group in groups),
            "reranker_positive_over_negative_all_passed": True,
        },
        "qrels": qrel_stats,
        "paired_confidence": {
            "comparisons": len(paired), "bootstrap_replicates": 2000,
            "cluster_unit": "query metadata cluster_id; query_id fallback",
            "cluster_limitation": {name: item["dataset_metadata"].get("cluster_limitation")
                                   for name, item in selection["datasets"].items()
                                   if item.get("dataset_metadata", {}).get("cluster_limitation")},
        },
        "groups": groups,
        "summary_metrics": summary_records,
        "not_run": ["standalone 30-query x 3-repeat latency study", "MIRACL batch 2", "QASPER (excluded by decision)"],
    }

    output = args.output.resolve()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "audit_status": report["audit_status"], "summaries": len(summaries), "rows": len(rows),
        "accepted_attempts_gpu_verified": len(groups),
        "measurement_server_processes_gpu_verified": report["guard_and_gpu"]["measurement_server_processes_gpu_verified"],
        "output": str(output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
