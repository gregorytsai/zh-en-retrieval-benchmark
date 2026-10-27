from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import guarded_engine
import matrix_spec
import pilot_forecast
import run_full_matrix


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8")


def _server_evidence(attempt: Path, label: str) -> None:
    log_path = attempt / "logs" / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("offloaded 4/4 layers to GPU\n", encoding="utf-8")
    _write_jsonl(attempt / "server-events.jsonl", [{
        "model": label,
        "command": ["llama-server", "--n-gpu-layers", "all", "--log-file", str(log_path)],
    }])


def make_complete_pilot(home: Path, batch: str = "B03", rerank_ms: float = 1.0,
                        elapsed_seconds: float = 1200.0) -> tuple[str, dict[str, str]]:
    embedder = dict(matrix_spec.RUN_BATCHES)[batch]
    dataset_ids = list(matrix_spec.BATCH_DATASETS[batch])
    rerankers = ("standalone", *matrix_spec.RERANKERS)
    selection = {
        dataset: [f"{dataset}-q{i}" for i in range(3)]
        for dataset in dataset_ids
    }
    selection_record = {
        "run_kind": "pilot",
        "selection_policy": "deterministic-short-mid-long-query-length-quantiles",
        "datasets": {
            dataset: {"query_ids": ids, "grade_mode": matrix_spec.DATASET_COUNTS[dataset]["grade_mode"]}
            for dataset, ids in selection.items()
        },
    }
    rows = []
    summaries = []
    committed = {}
    candidate_cache_paths = {}
    for dataset in dataset_ids:
        counts = matrix_spec.DATASET_COUNTS[dataset]
        ids = [f"d{i:02}" for i in range(50)]
        candidates = [{"query_id": qid, "ids": ids, "scores": [1.0] * 50} for qid in selection[dataset]]
        standalone_key = f"{dataset}--{embedder}--standalone"
        standalone_attempt = home / "attempts" / (standalone_key + "--attempt")
        cache = standalone_attempt / "cache" / embedder / dataset
        cache.mkdir(parents=True, exist_ok=True)
        _write_json(cache / "candidates.json", candidates)
        (cache / "corpus.npy").write_bytes(b"corpus-vector-fixture")
        (cache / "queries.npy").write_bytes(b"query-vector-fixture")
        file_hashes = {name: pilot_forecast.sha256_file(cache / name)
                       for name in ("corpus.npy", "queries.npy", "candidates.json")}
        cache_manifest = {
            "dataset": dataset, "embedding": embedder, "query_ids": selection[dataset],
            "candidate_k": 50, "normalized": True, "files_sha256": file_hashes,
        }
        cache_manifest["cache_key_sha256"] = matrix_spec.sha256_json(
            {key: value for key, value in cache_manifest.items() if key != "files_sha256"}
        )
        _write_json(cache / "cache-manifest.json", cache_manifest)
        run_cache = home / "cache" / embedder / dataset
        run_cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(cache, run_cache)
        candidate_cache_paths[dataset] = ids
        for name in ("validity.json", "gate-embedding.json"):
            _write_json(standalone_attempt / name, {
                "status": "accepted", "repeatability": True,
                "fixtures": [{} for _ in range(10)],
            })
        _server_evidence(standalone_attempt, "standalone")
        committed[standalone_key] = {
            "attempt": standalone_attempt.relative_to(home).as_posix(), "queries": 3,
        }

        for reranker in matrix_spec.RERANKERS:
            key = f"{dataset}--{embedder}--{reranker}"
            attempt = home / "attempts" / (key + "--attempt")
            _write_json(attempt / "validity.json", {"status": "accepted"})
            _write_json(attempt / "gate-reranker.json", {
                "status": "passed", "fixtures": [{} for _ in range(10)], "timing_included": False,
            })
            _write_json(attempt / "cache-input.json", {
                "source_commit_key": standalone_key,
                "cache_manifest_sha256": pilot_forecast.sha256_file(run_cache / "cache-manifest.json"),
                "query_ids": selection[dataset],
            })
            _server_evidence(attempt, reranker)
            committed[key] = {"attempt": attempt.relative_to(home).as_posix(), "queries": 3}

        for reranker in rerankers:
            summaries.append({"dataset": dataset, "embedding": embedder, "reranker": reranker,
                              "queries": 3, "errors": 0})
            for index, qid in enumerate(selection[dataset]):
                row = {
                    "dataset": dataset, "embedding": embedder, "reranker": reranker,
                    "query_id": qid, "status": "ok", "embedding_ms": 1.0,
                    "retrieval_ms": 1.0, "rerank_ms": None if reranker == "standalone" else rerank_ms,
                    "final_ids": ids if reranker == "standalone" else list(reversed(ids)),
                }
                if reranker != "standalone":
                    row["candidate_ids_sha256"] = matrix_spec.sha256_json(ids)
                rows.append(row)

    preflight_models = [embedder, *matrix_spec.RERANKERS]
    preflight = {"model_ids": preflight_models, "models": {}}
    for model in preflight_models:
        attempt = home / "preflight-attempts" / (model + "--attempt")
        _write_json(attempt / "validity.json", {"status": "passed"})
        _write_jsonl(attempt / "token-audit.jsonl", [
            {"dataset": dataset, "status": "ok", "truncated": False}
            for dataset in dataset_ids
        ])
        _server_evidence(attempt, "preflight-" + model)
        preflight["models"][model] = {
            "status": "passed", "attempt": attempt.relative_to(home).as_posix(),
        }

    events = [{"event": "combo_accepted", "combo": key} for key in committed]
    events.extend({"event": "token_audit_passed", "model": model} for model in preflight_models)
    _write_jsonl(home / "traffic-events.jsonl", events)
    _write_jsonl(home / "index-stats.jsonl", [
        {"dataset": dataset, "corpus_docs": matrix_spec.DATASET_COUNTS[dataset]["corpus"], "corpus_ms": 1000.0}
        for dataset in dataset_ids
    ])
    _write_json(home / "run-manifest.json", {
        "batch_id": batch, "run_kind": "pilot", "dataset_ids": dataset_ids,
        "wall_hours": 2.0, "dataset_sha256": "data", "binary_sha256": "binary",
        "model_sha256": {embedder: "model"}, "code_sha256": {"runner": "code"},
    })
    _write_json(home / "progress.json", {
        "phase": "pilot_complete", "summaries": len(dataset_ids) * 4,
        "query_rows": len(dataset_ids) * 3 * 4, "errors": 0,
        "elapsed_seconds": elapsed_seconds,
    })
    _write_json(home / "query-selection.json", selection_record)
    _write_json(home / "summary.json", summaries)
    _write_jsonl(home / "query-results.jsonl", rows)
    _write_json(home / "committed.json", committed)
    _write_json(home / "preflight.json", preflight)
    return embedder, {"dataset_sha256": "data", "binary_sha256": "binary",
                      "model_sha256": {embedder: "model"}, "code_sha256": {"runner": "code"}}


class PilotForecastTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "pilot"
        self.embedder, self.environment = make_complete_pilot(self.home)

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_pilot_requires_24_summaries_72_rows_and_emits_forecast(self):
        decision = pilot_forecast.audit_and_forecast(self.home, "B03", self.embedder)
        self.assertEqual(decision["status"], "approved")
        self.assertTrue(decision["evidence_audit_passed"])
        self.assertEqual(decision["expected_pilot_summaries"], 24)
        self.assertEqual(decision["expected_pilot_rows"], 72)
        self.assertLessEqual(decision["forecast"]["conservative_official_hours"], 10)
        self.assertEqual(decision["forecast"]["incremental_work_multiplier"], 1.25)
        self.assertIn("not a statistically stable p95", decision["forecast"]["sampling_policy"])

    def test_b02a_subset_pilot_audits_exactly_one_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "pilot"
            embedder, _ = make_complete_pilot(home, batch="B02-A")
            decision = pilot_forecast.audit_and_forecast(home, "B02-A", embedder)
        self.assertTrue(decision["evidence_audit_passed"], decision["issues"])
        self.assertEqual(decision["dataset_ids"], ["eng-eng"])
        self.assertEqual(decision["expected_pilot_summaries"], 4)
        self.assertEqual(decision["expected_pilot_rows"], 12)

    def test_forecast_over_ten_hours_is_planning_only_not_a_measurement_block(self):
        _write_jsonl(self.home / "query-results.jsonl", [
            {**row, "rerank_ms": 120_000.0} if row["reranker"] != "standalone" else row
            for row in [json.loads(line) for line in (self.home / "query-results.jsonl").read_text().splitlines()]
        ])
        decision = pilot_forecast.audit_and_forecast(self.home, "B03", self.embedder)
        self.assertEqual(decision["status"], "audited_planning_only")
        self.assertFalse(decision["eligible"])
        self.assertTrue(decision["evidence_audit_passed"])
        self.assertGreater(decision["forecast"]["conservative_official_hours"], 10)
        decision_path = pilot_forecast.write_decision(self.home, decision)
        verified = pilot_forecast.verify_official_gate(
            decision_path, self.home, "B03", self.embedder, self.environment,
            list(matrix_spec.BATCH_DATASETS["B03"]))
        self.assertTrue(verified["forecast_is_planning_only"])

    def test_incomplete_pilot_and_candidate_tampering_fail_closed(self):
        progress = json.loads((self.home / "progress.json").read_text())
        progress["query_rows"] = 71
        _write_json(self.home / "progress.json", progress)
        decision = pilot_forecast.audit_and_forecast(self.home, "B03", self.embedder)
        self.assertFalse(decision["eligible"])
        self.assertTrue(any("72 rows" in issue for issue in decision["issues"]))

    def test_legacy_handpicked_selection_policy_is_rejected(self):
        selection = json.loads((self.home / "query-selection.json").read_text())
        selection["selection_policy"] = "registered-pilot-ids-or-evenly-spaced-length-quantiles"
        _write_json(self.home / "query-selection.json", selection)
        decision = pilot_forecast.audit_and_forecast(self.home, "B03", self.embedder)
        self.assertFalse(decision["eligible"])
        self.assertTrue(any("selection policy" in issue for issue in decision["issues"]))

    def test_official_gate_binds_environment_and_unchanged_pilot_artifacts(self):
        decision = pilot_forecast.audit_and_forecast(self.home, "B03", self.embedder)
        decision_path = pilot_forecast.write_decision(self.home, decision)
        verified = pilot_forecast.verify_official_gate(decision_path, self.home, "B03", self.embedder, self.environment,
                                                       list(matrix_spec.BATCH_DATASETS["B03"]))
        self.assertEqual(verified["status"], "approved")
        with (self.home / "query-results.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
        with self.assertRaisesRegex(pilot_forecast.PilotGateError, "evidence changed"):
            pilot_forecast.verify_official_gate(decision_path, self.home, "B03", self.embedder, self.environment,
                                               list(matrix_spec.BATCH_DATASETS["B03"]))

    def test_official_gate_rejects_environment_hash_change(self):
        decision = pilot_forecast.audit_and_forecast(self.home, "B03", self.embedder)
        decision_path = pilot_forecast.write_decision(self.home, decision)
        changed = {**self.environment, "binary_sha256": "changed"}
        with self.assertRaisesRegex(pilot_forecast.PilotGateError, "hashes differ"):
            pilot_forecast.verify_official_gate(decision_path, self.home, "B03", self.embedder, changed,
                                               list(matrix_spec.BATCH_DATASETS["B03"]))


class PublicCliIntegrationTests(unittest.TestCase):
    def test_namespace_matches_guarded_engine_contract_and_isolates_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = run_full_matrix.build_engine_namespace(
                workspace=root, datasets_path=root / "datasets.json", miracl_path=root / "miracl.json",
                output_root=root / "out", selected_batch="B06", pilot=True, resume=True,
                plan_path=root / "plan.md",
            )
        self.assertTrue(args.execute)
        self.assertTrue(args.pilot)
        self.assertEqual(args.batch_id, "B06")
        self.assertEqual(args.embedder, "f2llm-v2-0.6b")
        self.assertEqual(args.output.name, "segment-01")
        self.assertEqual(args.dataset_ids, list(matrix_spec.BATCH_DATASETS["B06"]))
        self.assertEqual(args.wall_hours, 2.0)
        self.assertIsNone(args.deadline)
        self.assertEqual(args.pilot_queries, 3)
        self.assertEqual(args.pilot_datasets, list(matrix_spec.DATASET_COUNTS))
        self.assertTrue(args.resume)

    def test_public_pilot_cli_calls_guarded_engine_only_after_explicit_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "out"
            plan = root / "plan.md"
            with patch.object(run_full_matrix, "build_plan", return_value={}), \
                 patch.object(guarded_engine, "load_matrix_inputs", return_value=({}, "data")), \
                 patch.object(guarded_engine, "build_selection", return_value={}), \
                 patch.object(run_full_matrix, "length_quantile_selection_issues", return_value=[]), \
                 patch.object(guarded_engine, "run") as run_mock, \
                 patch.object(run_full_matrix, "audit_and_forecast", return_value={"eligible": False, "evidence_audit_passed": True}) as forecast_mock, \
                 patch.object(run_full_matrix, "write_decision") as write_mock:
                result = run_full_matrix.main([
                    "--workspace", str(root), "--output", str(output), "--plan-path", str(plan),
                    "--batch", "B03", "--execute", "--pilot",
                ])
            self.assertEqual(result, 0)
            run_mock.assert_called_once()
            engine_args = run_mock.call_args.args[0]
            self.assertEqual(engine_args.output, output.resolve() / "B03" / "segment-01")
            self.assertEqual(engine_args.plan_path, plan.resolve())
            self.assertEqual(engine_args.wall_hours, 2.0)
            forecast_mock.assert_called_once()
            write_mock.assert_called_once()

    def test_public_official_cli_requires_gate_before_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "out"
            with patch.object(run_full_matrix, "build_plan", return_value={}), \
                 patch.object(guarded_engine, "environment_hashes", return_value={"dataset_sha256": "d"}), \
                 patch.object(run_full_matrix, "_combined_dataset_sha256", return_value="d"), \
                 patch.object(run_full_matrix, "verify_official_gate", return_value={"status": "approved"}) as gate, \
                 patch.object(guarded_engine, "run") as run_mock:
                result = run_full_matrix.main([
                    "--workspace", str(root), "--output", str(output), "--batch", "B03", "--execute",
                ])
            self.assertEqual(result, 0)
            gate.assert_called_once()
            run_mock.assert_called_once()
            self.assertFalse((output / "B03" / "segment-01" / "pilot").exists())

    def test_execute_rejects_wall_budget_changes_and_resume_without_execute(self):
        for argv, message in (
            (["--execute", "--pilot", "--batch", "B03", "--wall-hours", "12"], "fixed 2-hour wall deadline"),
            (["--resume"], "--resume is valid only with --execute"),
        ):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                run_full_matrix.main(argv)
            self.assertIn(message, stderr.getvalue())

    def test_termination_handler_exits_so_engine_finally_can_run(self):
        for signum, exit_code in ((run_full_matrix.signal.SIGINT, 128 + run_full_matrix.signal.SIGINT),
                                  (run_full_matrix.signal.SIGTERM, 128 + run_full_matrix.signal.SIGTERM)):
            with self.assertRaises(SystemExit) as raised:
                run_full_matrix._handle_termination(signum, None)
            self.assertEqual(raised.exception.code, exit_code)


if __name__ == "__main__":
    unittest.main()
