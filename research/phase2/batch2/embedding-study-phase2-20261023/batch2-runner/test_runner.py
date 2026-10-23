"""Offline unit tests for the guarded run matrix and resume constraints."""
import argparse
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import run_guarded as runner
import run_study as core


class MatrixTests(unittest.TestCase):
    def test_official_matrix_is_fixed_at_six_summaries_and_1200_rows(self):
        jobs = runner.job_list(list(core.DATASET_COUNTS), list(core.CONFIGS))
        self.assertEqual(len(jobs), 6)
        self.assertEqual(sum(core.DATASET_COUNTS[name]["queries"] for name in core.DATASET_COUNTS) * 3, 1200)
        self.assertEqual(sum(job[2] == "standalone" for job in jobs), 2)
        self.assertEqual(sum(job[2] != "standalone" for job in jobs), 4)

    def test_default_pilot_is_both_miracl_sets_three_queries_and_three_configs(self):
        args = runner.parse_args(["--pilot"])
        dataset_ids = runner.build_dataset_ids(args)
        requested, effective = runner.build_config_ids(args)
        self.assertEqual(dataset_ids, list(core.DATASET_COUNTS))
        self.assertEqual(requested, list(core.CONFIGS))
        self.assertEqual(effective, list(core.CONFIGS))
        self.assertEqual(len(runner.job_list(dataset_ids, effective)), 6)

    def test_official_matrix_cannot_be_reduced_with_configs_flag(self):
        with self.assertRaises(SystemExit):
            runner.parse_args(["--configs", "qwen-4b-standalone"])

    def test_reranker_only_pilot_adds_qwen4b_candidate_dependency(self):
        args = runner.parse_args(["--pilot", "--configs", "qwen-4b+qwen-reranker-0.6b"])
        requested, effective = runner.build_config_ids(args)
        self.assertEqual(requested, ["qwen-4b+qwen-reranker-0.6b"])
        self.assertEqual(effective, ["qwen-4b-standalone", "qwen-4b+qwen-reranker-0.6b"])

    def test_cli_rejects_wall_hours_above_hard_cap_and_nonfinite_values(self):
        for value in ("12.01", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                runner.parse_args(["--wall-hours", value])

    def test_absolute_deadline_cannot_exceed_twelve_hours(self):
        args = argparse.Namespace(
            pilot=False, datasets_file=Path("datasets.json"),
            deadline="1970-01-01T13:00:01+00:00", wall_hours=None,
        )
        datasets = {"tiny": {"metadata": {}, "grade_mode": "binary", "queries": []}}
        hashes = {"dataset_sha256": "d", "binary_sha256": "b", "model_sha256": {}, "code_sha256": {}}
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, "environment_hashes", return_value=hashes):
            with self.assertRaisesRegex(ValueError, "hard 12-hour limit"):
                runner.set_up_manifest(Path(temp), args, datasets, "d", ["tiny"], ["cfg"], ["cfg"],
                                       {"tiny": []}, "qrels", started_epoch=1.0)


class IdleGateTests(unittest.TestCase):
    def test_begin_race_returns_to_idle_wait_instead_of_failing(self):
        class FakeGuard:
            def __init__(self):
                self.begin_calls = 0

            def ready(self):
                return True

            def begin(self):
                self.begin_calls += 1
                if self.begin_calls == 1:
                    raise runner.Interference("new /metrics poll invalidated readiness")

        with tempfile.TemporaryDirectory() as temp:
            guard = FakeGuard()
            runner.wait_for_idle(Path(temp), time.time(), time.time() + 30,
                                 guard, "test-combo", 0, 1)
            self.assertEqual(guard.begin_calls, 2)


class PreflightTests(unittest.TestCase):
    def test_blocker_does_not_skip_auditing_other_selected_models(self):
        observed = []

        def fake_audit(home, started, deadline, model_name, model_kind, datasets,
                       dataset_ids, guard, watchdog, preflight):
            observed.append(model_name)
            preflight["models"][model_name] = {
                "status": "blocked" if len(observed) == 1 else "passed",
                "issues": [{"kind": "fixture"}] if len(observed) == 1 else [],
            }

        with tempfile.TemporaryDirectory() as temp, patch.object(runner, "run_preflight_model", side_effect=fake_audit):
            with self.assertRaises(runner.PreflightBlockedError):
                runner.ensure_preflight(Path(temp), time.time(), time.time() + 120,
                                        {"miracl-zh-zh": {}}, ["miracl-zh-zh"],
                                        list(core.CONFIGS), object(), object())
        self.assertEqual(len(observed), 3)
        self.assertEqual(set(observed), set(runner.model_registry()))


class ResumeTests(unittest.TestCase):
    def test_resume_rejects_changed_deadline_or_hash(self):
        expected = {
            "run_kind": "official", "dataset_file_name": "datasets.json", "dataset_sha256": "d",
            "binary_sha256": "b", "model_sha256": {"qwen-4b": "m"}, "code_sha256": {"runner": "c"},
            "dataset_ids": ["eng-zho"], "requested_configs": ["qwen-4b-standalone"],
            "effective_configs": ["qwen-4b-standalone"], "selection_sha256": "s",
            "selected_qrels_sha256": "q", "deadline_record_sha256": "dl",
            "original_started_epoch": 100.0, "deadline_epoch": 200.0, "wall_hours": 1 / 36,
        }
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            (home / "run-manifest.json").write_text(json.dumps(expected), encoding="utf-8")
            args = argparse.Namespace(deadline=None, wall_hours=None)
            self.assertEqual(runner.validate_saved_manifest(home, expected, args), expected)
            changed = dict(expected, code_sha256={"runner": "changed"})
            with self.assertRaisesRegex(RuntimeError, "code_sha256"):
                runner.validate_saved_manifest(home, changed, args)
            args.deadline = "1970-01-01T00:03:30+00:00"
            with self.assertRaisesRegex(RuntimeError, "cannot move"):
                runner.validate_saved_manifest(home, expected, args)


if __name__ == "__main__":
    unittest.main()
