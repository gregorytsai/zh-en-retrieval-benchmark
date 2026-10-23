#!/usr/bin/env python3
"""Read-only structural, hash, guard, token, GPU and result audit for MIRACL batch 2."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import run_guarded as runner
import run_study as core


SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent
EXPECTED_DATASETS = {
    "miracl-zh-zh": {"corpus": 3786, "queries": 200, "qrels": 1998, "grade_mode": "binary"},
    "miracl-en-en": {"corpus": 7921, "queries": 200, "qrels": 2091, "grade_mode": "binary"},
}
EXPECTED_CONFIGS = [
    "qwen-4b-standalone",
    "qwen-4b+qwen-reranker-0.6b",
    "qwen-4b+qwen-reranker-4b",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path):
    if not path.is_file():
        raise RuntimeError(f"Missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise RuntimeError(f"Missing required JSONL: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_evidence(attempt: Path, expected_count: int = 1) -> list[dict]:
    events = read_jsonl(attempt / "server-events.jsonl")
    servers = []
    for event in events:
        command = event.get("command")
        if not command:
            continue
        require("--n-gpu-layers" in command and command[command.index("--n-gpu-layers") + 1] == "all",
                f"Server did not request all GPU layers: {attempt}")
        require("--log-file" in command, f"Server command has no log-file setting: {attempt}")
        log_path = Path(command[command.index("--log-file") + 1])
        require(log_path.is_file(), f"Server log is missing: {log_path}")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        offloads = re.findall(r"offloaded\s+(\d+)/(\d+) layers to GPU", log_text)
        require(bool(offloads), f"No actual GPU offload evidence in server log: {log_path}")
        servers.append({
            "model": event.get("model"), "log": str(log_path.relative_to(PROJECT_ROOT))
            if log_path.is_relative_to(PROJECT_ROOT) else str(log_path),
            "offloaded_layers": int(offloads[-1][0]), "total_layers": int(offloads[-1][1]),
        })
    require(len(servers) == expected_count,
            f"Expected {expected_count} measurement server log(s), found {len(servers)}: {attempt}")
    return servers


def audit(run_dir: Path, datasets_path: Path, validation_path: Path, output_path: Path,
          expected_kind: str) -> dict:
    run_dir = run_dir.resolve()
    datasets_path = datasets_path.resolve()
    validation_path = validation_path.resolve()
    manifest = read_json(run_dir / "run-manifest.json")
    progress = read_json(run_dir / "progress.json")
    deadline = read_json(run_dir / "deadline.json")
    selection = read_json(run_dir / "query-selection.json")
    qrels_record = read_json(run_dir / "qrels.json")
    committed = read_json(run_dir / "committed.json")
    summary = read_json(run_dir / "summary.json")
    rows = read_jsonl(run_dir / "query-results.jsonl")
    traffic = read_jsonl(run_dir / "traffic-events.jsonl")
    validation = read_json(validation_path)
    datasets, dataset_digest = core.load_datasets(datasets_path)

    require(manifest.get("run_kind") == expected_kind, "Run kind differs from requested audit kind")
    require(expected_kind in ("pilot", "official"), f"Unsupported run kind: {expected_kind}")
    required_phase = "pilot_complete" if expected_kind == "pilot" else "complete"
    require(progress.get("phase") == required_phase, f"Run phase is not {required_phase}: {progress.get('phase')}")
    require(manifest.get("production_modified") is False, "Manifest says production was modified")
    require(set(datasets) == set(EXPECTED_DATASETS), f"Unexpected dataset IDs: {sorted(datasets)}")
    require(list(manifest.get("dataset_ids", [])) == list(EXPECTED_DATASETS), "Manifest dataset order differs from frozen MIRACL matrix")
    require(manifest.get("requested_configs") == EXPECTED_CONFIGS and manifest.get("effective_configs") == EXPECTED_CONFIGS,
            "Manifest configs are not the frozen A/E/F matrix")
    require(manifest.get("matrix_summary_count") == runner.EXPECTED_SUMMARIES == 6,
            "Manifest expected summary count is not six")
    if expected_kind == "official":
        require(manifest.get("matrix_full_query_rows") == runner.EXPECTED_QUERY_ROWS == 1200,
                "Manifest expected full query row count is not 1,200")
    else:
        require(manifest.get("matrix_full_query_rows") is None,
                "Pilot manifest must not label its sample as the full 1,200-row matrix")
    require(manifest.get("dataset_sha256") == dataset_digest == validation.get("datasets_sha256"),
            "Dataset SHA does not match validation and run manifest")
    require(dataset_digest == "fdf802cc03e9059783e5a2cba0785437253f07841ec6956246d5884461e7ba3d",
            "Batch-2 dataset SHA differs from the frozen MIRACL manifest")
    require(validation.get("status") == "draft_not_ready_for_inference" or
            validation.get("status") in ("validated", "ready_for_inference"),
            f"Unexpected validation status: {validation.get('status')}")

    observed_data = {}
    for name, expected in EXPECTED_DATASETS.items():
        data = datasets[name]
        observed = {
            "corpus": len(data["corpus"]), "queries": len(data["queries"]),
            "qrels": sum(map(len, data["qrels"].values())), "grade_mode": data["grade_mode"],
        }
        require(observed == expected, f"Frozen count mismatch for {name}: {observed}")
        require(data["metadata"].get("corpus_mode") == "all dev judged passages union; NOT full-corpus MIRACL",
                f"Corpus scope is not the frozen judged-pool subset: {name}")
        require(data["grade_mode"] == "binary", f"MIRACL grade mode is not binary: {name}")
        observed_data[name] = {**observed, "pilot_query_ids": data["pilot_query_ids"]}

    selected = {name: entry.get("query_ids", []) for name, entry in selection.get("datasets", {}).items()}
    require(set(selected) == set(EXPECTED_DATASETS), "Selection datasets differ from the frozen matrix")
    for name, data in datasets.items():
        ids = selected[name]
        require(len(ids) == len(set(ids)), f"Duplicate selected query IDs: {name}")
        if expected_kind == "official":
            expected_ids = [query["id"] for query in data["queries"]]
        else:
            expected_ids = list(data["pilot_query_ids"] or [])
        require(ids == expected_ids, f"Selection does not match frozen {expected_kind} query IDs: {name}")
    expected_rows = sum(len(ids) for ids in selected.values()) * len(EXPECTED_CONFIGS)
    require(expected_rows == (18 if expected_kind == "pilot" else 1200), "Selected query count differs from expected run size")
    require(selection.get("dataset_sha256") == dataset_digest, "Selection dataset SHA mismatch")
    require(manifest.get("selection_sha256") == core.sha256_json(selection), "Frozen selection hash mismatch")
    require(qrels_record.get("dataset_sha256") == dataset_digest, "Qrels artifact dataset SHA mismatch")
    require(manifest.get("selected_qrels_sha256") == core.sha256_json(qrels_record), "Frozen selected-qrels hash mismatch")

    current_code = {
        "run_guarded.py": Path(runner.__file__).resolve(),
        "run_study.py": Path(core.__file__).resolve(),
        "traffic_guard.py": runner.BATCH2_SOURCE_DIR / "traffic_guard.py",
        "legacy_run_benchmark.py": core.LEGACY_BENCHMARK,
    }
    code_hashes = {name: sha256(path) for name, path in current_code.items()}
    model_paths = runner.model_registry()
    model_hashes = {name: sha256(path) for name, path in sorted(model_paths.items())}
    binary_hash = sha256(core.BINARY)
    require(manifest.get("code_sha256") == code_hashes, "Current batch-2 code hashes differ from frozen run manifest")
    require(manifest.get("model_sha256") == model_hashes, "Current model hashes differ from frozen run manifest")
    require(manifest.get("binary_sha256") == binary_hash, "Current server binary hash differs from frozen run manifest")
    require(manifest.get("binary_sha256") == "5f7bee49eaf08bf77b3cc523d88d385750d6b7da8479775b32bc9be030464ae5",
            "llama-server binary differs from the Phase 2 validated binary")

    started = float(manifest["original_started_epoch"])
    deadline_epoch = float(manifest["deadline_epoch"])
    wall_hours = float(manifest["wall_hours"])
    require(wall_hours > 0 and wall_hours <= 12, "Wall-clock limit is outside the hard cap")
    require(abs(deadline_epoch - float(deadline["deadline_epoch"])) < 0.01, "Manifest/deadline deadline differs")
    require(abs(started - float(deadline["original_started_epoch"])) < 0.01, "Manifest/deadline start differs")
    require(abs(deadline_epoch - (started + wall_hours * 3600)) < 0.01, "Deadline is not original start + wall budget")
    if expected_kind == "official":
        require(abs(wall_hours - 12.0) < 1e-9, "Official run must retain the 12-hour wall budget")
    resume_history = read_jsonl(run_dir / "resume-history.jsonl") if (run_dir / "resume-history.jsonl").exists() else []
    require(all(item.get("original_deadline") == manifest.get("deadline_utc") for item in resume_history),
            "Resume history shows a changed deadline")

    group_rows: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    identities = []
    for row in rows:
        key = (row.get("dataset"), row.get("embedding"), row.get("reranker"))
        group_rows[key].append(row)
        identities.append((*key, row.get("query_id")))
    require(len(identities) == len(set(identities)), "Duplicate composite query keys in main results")
    require(len(rows) == expected_rows == progress.get("query_rows"), "Main row count differs from selection/progress")
    require(all(row.get("status") == "ok" for row in rows), "Main results contain non-ok rows")
    require(progress.get("summaries") == len(summary) == len(committed) == 6, "Summary/commit count is not six")
    require(progress.get("errors") == 0 and all(item.get("errors") == 0 for item in summary), "Run contains summary errors")

    expected_keys = set()
    group_audits = []
    for dataset in EXPECTED_DATASETS:
        for config in EXPECTED_CONFIGS:
            embedding, reranker_name = core.CONFIGS[config]
            key_tuple = (dataset, embedding, reranker_name)
            commit_key = "--".join(key_tuple)
            expected_keys.add(commit_key)
            group = group_rows.get(key_tuple, [])
            query_ids = selected[dataset]
            require(len(group) == len(query_ids), f"Wrong main row count for {commit_key}")
            require({row["query_id"] for row in group} == set(query_ids), f"Main query IDs differ for {commit_key}")
            width = 100 if reranker_name == "standalone" else 50
            require(all(len(row.get("final_ids", [])) == width for row in group), f"final_ids width mismatch for {commit_key}")
            require(all(len(row["final_ids"]) == len(set(row["final_ids"])) for row in group), f"Duplicate final IDs for {commit_key}")
            require(all(row.get("grade_mode") == "binary" for row in group), f"Non-binary row in {commit_key}")
            commit = committed.get(commit_key)
            require(commit is not None, f"Missing commit pointer: {commit_key}")
            attempt = (run_dir / commit["attempt"]).resolve()
            require(attempt.is_relative_to((run_dir / "attempts").resolve()), f"Attempt path escapes run directory: {commit_key}")
            require(read_json(attempt / "validity.json").get("status") == "accepted", f"Attempt not accepted: {commit_key}")
            attempt_rows = read_jsonl(attempt / "query-results.jsonl")
            require(len(attempt_rows) == len(query_ids), f"Accepted attempt row count mismatch: {commit_key}")
            require({row["query_id"] for row in attempt_rows} == set(query_ids), f"Attempt query IDs mismatch: {commit_key}")
            require(all(row.get("status") == "ok" for row in attempt_rows), f"Attempt has non-ok rows: {commit_key}")
            main_by_id = {row["query_id"]: row for row in group}
            for attempt_row in attempt_rows:
                main_row = main_by_id[attempt_row["query_id"]]
                for field in set(attempt_row) & set(main_row):
                    require(attempt_row[field] == main_row[field], f"Main/attempt mismatch: {commit_key}/{attempt_row['query_id']}/{field}")
            servers = gpu_evidence(attempt)
            fixture_summary = None
            if reranker_name != "standalone":
                fixture_gate = read_json(attempt / "gate-reranker.json")
                fixtures = fixture_gate.get("fixtures", [])
                require(fixture_gate.get("status") == "passed" and len(fixtures) == 10,
                        f"Reranker fixture gate failed: {commit_key}")
                require(fixture_gate.get("timing_included") is False and
                        all(item.get("timing_included") is False and item.get("order") == [0, 1] and
                            item.get("positive_score", 0) > item.get("negative_score", 0) for item in fixtures),
                        f"Reranker fixture ordering/timing policy failed: {commit_key}")
                fixture_summary = {"status": "passed", "fixtures": len(fixtures), "timing_included": False}
                source_key = f"{dataset}--qwen-4b--standalone"
                cache_input = read_json(attempt / "cache-input.json")
                require(cache_input.get("source_commit_key") == source_key, f"Reranker cache source mismatch: {commit_key}")
                require(cache_input.get("query_ids") == query_ids, f"Reranker cache query set mismatch: {commit_key}")
                source_commit = committed.get(source_key)
                require(source_commit is not None, f"Reranker has no accepted standalone cache source: {commit_key}")
                source_attempt = run_dir / source_commit["attempt"]
                cache_dir = source_attempt / "cache" / "qwen-4b" / dataset
                cache_manifest_path = cache_dir / "cache-manifest.json"
                require(cache_manifest_path.is_file(), f"Standalone cache manifest missing: {commit_key}")
                require(cache_input.get("cache_manifest_sha256") == sha256(cache_manifest_path),
                        f"Reranker did not use the accepted standalone cache: {commit_key}")
            group_audits.append({"dataset": dataset, "config": config, "rows": len(group),
                                 "accepted_attempt": commit["attempt"], "gpu_servers": servers,
                                 "reranker_fixture_gate": fixture_summary})
    require(set(committed) == expected_keys, "Committed keys differ from the frozen six-combination matrix")

    preflight = read_json(run_dir / "preflight.json")
    expected_models = ["qwen-4b", "qwen-reranker-0.6b", "qwen-reranker-4b"]
    require(preflight.get("model_ids") == expected_models, "Preflight model set differs from A/E/F")
    require(set(preflight.get("models", {})) == set(expected_models), "Preflight did not audit all used models")
    preflight_audits = []
    preflight_gpu = []
    for model in expected_models:
        record = preflight["models"][model]
        require(record.get("status") == "passed", f"Model token preflight did not pass: {model}")
        attempt = run_dir / record["attempt"]
        token_rows = read_jsonl(attempt / "token-audit.jsonl")
        require(len(token_rows) == len(EXPECTED_DATASETS) and
                {item.get("dataset") for item in token_rows} == set(EXPECTED_DATASETS),
                f"Token audit does not cover both MIRACL datasets: {model}")
        require(all(item.get("status") == "ok" and not item.get("truncated", False) for item in token_rows),
                f"Token audit failed or truncated input: {model}")
        preflight_audits.extend(token_rows)
        preflight_gpu.extend(gpu_evidence(attempt))
    require(len(preflight_audits) == 6, "Expected six model×dataset token audit records")

    summary_keys = {(item.get("dataset"), item.get("embedding"), item.get("reranker")) for item in summary}
    require(summary_keys == set(group_rows), "Summary groups and main result groups differ")
    require(all(item.get("queries") == len(selected[item["dataset"]]) and item.get("errors") == 0 for item in summary),
            "Summary counts/errors differ from frozen selection")

    event_counts = dict(Counter(event.get("event", "unknown") for event in traffic))
    require(event_counts.get("token_audit_passed") == 3, "Traffic ledger does not record all three model token audits")
    require(event_counts.get("combo_accepted") == 6, "Traffic ledger does not show six accepted combinations")
    require(not (run_dir / "fatal-errors.jsonl").exists() or not read_jsonl(run_dir / "fatal-errors.jsonl"),
            "Fatal errors were recorded")
    require(not any(event.get("event") in {"blocked", "fatal_error"} for event in traffic),
            "Traffic ledger contains a blocker/fatal event")

    paired = read_json(run_dir / "paired-confidence.json")
    require(len(paired) == 8 and all(item.get("bootstrap_replicates") == 2000 for item in paired),
            "Paired A/E and E/F bootstrap metadata is incomplete")
    require({item.get("comparison") for item in paired} == {"A_vs_E", "E_vs_F"},
            "Paired comparisons differ from the frozen A/E/F plan")

    report = {
        "audit_status": "passed", "run_kind": expected_kind,
        "completion_time_utc": progress.get("time"), "run_dir": str(run_dir),
        "matrix": {
            "datasets": observed_data, "config_ids": EXPECTED_CONFIGS,
            "summary_count": len(summary), "query_rows": len(rows),
            "dataset_qualified_queries": sum(len(ids) for ids in selected.values()),
            "unique_composite_keys": len(set(identities)), "non_ok_rows": sum(row.get("status") != "ok" for row in rows),
        },
        "integrity": {
            "dataset_sha256": dataset_digest, "validation_sha256": sha256(validation_path),
            "manifest_sha256": sha256(run_dir / "run-manifest.json"),
            "code_sha256_matches_manifest": code_hashes,
            "model_sha256_matches_manifest": model_hashes,
            "binary_sha256_matches_manifest": binary_hash,
            "deadline_utc": manifest.get("deadline_utc"), "wall_hours": wall_hours,
            "deadline_unchanged_on_resume": all(item.get("original_deadline") == manifest.get("deadline_utc") for item in resume_history),
            "resume_count": len(resume_history),
        },
        "preflight": {
            "models": expected_models, "model_dataset_audits": len(preflight_audits),
            "all_inputs_within_limit_without_truncation": True,
            "max_document_tokens_by_model_dataset": {
                f"{item.get('model')}:{item.get('dataset')}": item.get("max_document", {}).get("tokens")
                for item in preflight_audits
            },
            "gpu_preflight_servers": preflight_gpu,
        },
        "guard_and_gpu": {
            "traffic_policy": manifest.get("traffic_policy"), "runner_policy": manifest.get("runner_policy"),
            "traffic_event_counts": event_counts, "accepted_attempt_gpu_servers": sum(len(item["gpu_servers"]) for item in group_audits),
            "preflight_gpu_servers": len(preflight_gpu), "reranker_fixture_gates": sum(item["reranker_fixture_gate"] is not None for item in group_audits),
            "reranker_fixture_instances": sum((item["reranker_fixture_gate"] or {}).get("fixtures", 0) for item in group_audits),
            "all_reranker_positive_scores_exceeded_negatives": True,
        },
        "qrels": {
            name: {"grade_mode": data["grade_mode"], "queries": len(data["qrels"]),
                   "judgments": sum(map(len, data["qrels"].values())),
                   "positive_judgments": sum(grade > 0 for judgments in data["qrels"].values() for grade in judgments.values())}
            for name, data in datasets.items()
        },
        "paired_confidence": {"comparisons": len(paired), "bootstrap_replicates": 2000,
                              "cluster_unit": "query metadata cluster_id; query_id fallback"},
        "groups": group_audits,
        "summary_metrics": summary,
        "not_run": ["QASPER (excluded by user decision)", "30-query x 3-repeat latency study"],
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "run-phase2-batch2/official")
    parser.add_argument("--datasets", type=Path, default=SOURCE_DIR / "datasets.json")
    parser.add_argument("--data-validation", type=Path, default=SOURCE_DIR / "data-validation.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "run-phase2-batch2/FINAL_AUDIT.json")
    parser.add_argument("--expected-kind", choices=("pilot", "official"), default="official")
    args = parser.parse_args()
    report = audit(args.run_dir, args.datasets, args.data_validation, args.output, args.expected_kind)
    print(json.dumps({"audit_status": report["audit_status"], "run_kind": report["run_kind"],
                      "summaries": report["matrix"]["summary_count"], "rows": report["matrix"]["query_rows"],
                      "gpu_attempts": report["guard_and_gpu"]["accepted_attempt_gpu_servers"],
                      "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
