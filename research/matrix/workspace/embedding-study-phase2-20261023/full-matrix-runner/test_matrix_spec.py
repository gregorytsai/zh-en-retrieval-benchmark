from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matrix_spec as spec
import run_full_matrix


class MatrixShapeTests(unittest.TestCase):
    def test_frozen_matrix_totals_and_batch_size(self):
        self.assertEqual(len(spec.EMBEDDERS), 7)
        self.assertEqual(len(spec.RERANKERS), 3)
        self.assertEqual((spec.QUERY_ROWS, spec.SUMMARIES_PER_EMBEDDER, spec.ROWS_PER_EMBEDDER), (2613, 24, 10452))
        self.assertEqual((spec.TOTAL_SUMMARIES, spec.TOTAL_ROWS), (168, 73164))
        self.assertEqual(len(spec.enumerate_configs()), 28)
        self.assertEqual(len(spec.enumerate_configs(["f2llm-v2-0.6b", "granite-97m-r2"])), 8)
        self.assertEqual(spec.BATCHES, (
            ("B01", "qwen-4b"), ("B02", "qwen-8b"), ("B03", "embeddinggemma-2"),
            ("B04", "qwen-0.6b"), ("B05", "bge-m3"), ("B06", "f2llm-v2-0.6b"),
            ("B07", "granite-97m-r2"),
        ))
        for embedder in spec.EMBEDDERS:
            configs = spec.enumerate_configs([embedder])
            self.assertEqual([row["reranker"] for row in configs], ["standalone", *spec.RERANKERS])

    def test_unknown_or_duplicate_embedder_is_rejected(self):
        with self.assertRaises(spec.MatrixValidationError):
            spec.enumerate_configs(["missing"])
        with self.assertRaises(spec.MatrixValidationError):
            spec.enumerate_configs(["qwen-4b", "qwen-4b"])


class DataValidationTests(unittest.TestCase):
    def test_small_schema_validation_without_frozen_count_check(self):
        dataset = {
            "eng-zho": {
                "corpus": [{"id": "d1", "text": "text"}],
                "queries": [{"id": "q1", "text": "query"}],
                "qrels": {"q1": {"d1": 1}},
                "grade_mode": "binary",
            }
        }
        normalized = spec.validate_dataset_mapping(dataset, require_full=False)
        self.assertEqual(normalized["eng-zho"]["queries"][0]["id"], "q1")

    def test_grade_range_and_orphan_doc_are_rejected(self):
        dataset = {
            "eng-zho": {
                "corpus": [{"id": "d1", "text": "text"}],
                "queries": [{"id": "q1", "text": "query"}],
                "qrels": {"q1": {"d2": 1}},
                "grade_mode": "binary",
            }
        }
        with self.assertRaises(spec.MatrixValidationError):
            spec.validate_dataset_mapping(dataset, require_full=False)

    def test_manifest_merge_rejects_overlap(self):
        left = {"datasets": {"eng-zho": {}}}
        right = {"datasets": {"eng-zho": {}}}
        with self.assertRaises(spec.MatrixValidationError):
            spec.validate_and_merge_manifests(left, right)


class CandidateCacheTests(unittest.TestCase):
    def setUp(self):
        self.query_ids = ["q1"]
        self.candidates = [{"query_id": "q1", "ids": [f"d{i:02}" for i in range(50)]}]
        self.expected = {
            "dataset": "eng-zho", "embedder": "qwen-4b", "dataset_sha256": "data",
            "model_sha256": "model", "binary_sha256": "binary", "code_sha256": "code",
            "query_ids": self.query_ids, "candidate_k": 50, "tie_break": "score_desc_then_doc_id_asc",
            "prompt_spec": {"pooling": "last", "dimension": 2560},
        }

    def test_ties_are_sorted_by_doc_id(self):
        self.assertEqual(spec.rank_candidates({"z": 1.0, "a": 1.0, "c": 0.5}, 2), ["a", "z"])

    def test_rank_candidates_rejects_nonfinite_scores_and_candidate_cache_is_exactly_50(self):
        with self.assertRaises(spec.MatrixValidationError):
            spec.rank_candidates({"a": float("nan")})
        with self.assertRaises(spec.MatrixValidationError):
            spec.rank_candidates({"a": float("inf")})
        score_by_doc = {f"d{i:02}": 1.0 for i in range(51)}
        top50 = spec.rank_candidates(score_by_doc, 50)
        self.assertEqual(len(top50), 50)
        self.assertEqual(top50, sorted(top50))

    def test_cache_manifest_verifies_and_detects_candidate_order_changes(self):
        manifest = spec.build_cache_manifest(
            dataset="eng-zho", embedder="qwen-4b", data_sha256="data", model_sha256="model",
            binary_sha256="binary", code_sha256="code", query_ids=self.query_ids,
            candidates=self.candidates, prompt_spec={"pooling": "last", "dimension": 2560},
        )
        spec.verify_cache_manifest(manifest, self.expected, self.candidates)
        changed = [{"query_id": "q1", "ids": list(reversed(self.candidates[0]["ids"]))}]
        with self.assertRaises(spec.MatrixValidationError):
            spec.verify_cache_manifest(manifest, self.expected, changed)

    def test_candidate_pools_must_be_exactly_top50_unique_and_ordered(self):
        bad = [{"query_id": "q1", "ids": self.candidates[0]["ids"][:-1]}]
        with self.assertRaises(spec.MatrixValidationError):
            spec.candidate_pool_hash(self.query_ids, bad)


class MetricsAndEligibilityTests(unittest.TestCase):
    def test_binary_metrics_and_coverage(self):
        qrels = {"d0": 1, "d1": 0, "d2": 1}
        candidates = ["d0", "d1", *[f"x{i}" for i in range(48)]]
        row = spec.evaluate_ranking(["d1", "d0", "d2"], candidates, qrels, "binary")
        self.assertEqual(row["top1"], 0.0)
        self.assertEqual(row["mrr10"], 0.5)
        self.assertEqual(row["judged_coverage10"], 1.0)
        self.assertEqual(row["candidate_judged_coverage50"], 2 / 50)

    def test_graded_ndcg_uses_exp2_gain_and_preserves_grade_thresholds(self):
        qrels = {"g4": 4, "g2": 2, "g1": 1, "n": 0}
        row = spec.evaluate_ranking(["g2", "g4", "n"], ["g2", "g4", "g1", "n"], qrels, "graded")
        expected_dcg = 3 / 1 + 15 / 1.584962500721156
        expected_idcg = 15 / 1 + 3 / 1.584962500721156 + 1 / 2
        self.assertAlmostEqual(row["ndcg10"], expected_dcg / expected_idcg)
        self.assertEqual(row["p5_grade4"], 1 / 5)
        self.assertEqual(row["p5_grade2"], 2 / 5)

    def test_macro_requires_all_six_complete_summaries(self):
        summaries = [
            {"dataset": name, "queries": values["queries"], "errors": 0, "ndcg10": index / 10}
            for index, (name, values) in enumerate(spec.DATASET_COUNTS.items(), 1)
        ]
        complete = spec.equal_weight_macro(summaries)
        self.assertTrue(complete["eligible"])
        self.assertAlmostEqual(complete["macro_ndcg10_equal_weight"], sum(range(1, 7)) / 60)
        incomplete = spec.equal_weight_macro(summaries[:-1])
        self.assertFalse(incomplete["eligible"])
        self.assertIsNone(incomplete["macro_ndcg10_equal_weight"])
        errored = [*summaries[:-1], {**summaries[-1], "errors": 1}]
        self.assertFalse(spec.equal_weight_macro(errored)["eligible"])

    def test_macro_rejects_missing_summary_fields_and_invalid_ndcg(self):
        summaries = [
            {"dataset": name, "queries": values["queries"], "errors": 0, "ndcg10": 0.5}
            for name, values in spec.DATASET_COUNTS.items()
        ]
        invalid_cases = [
            ({"errors": None}, "missing errors"), ({"queries": None}, "queries=None"),
            ({"ndcg10": float("nan")}, "ndcg nan"), ({"ndcg10": float("inf")}, "ndcg inf"),
            ({"ndcg10": -0.01}, "ndcg negative"), ({"ndcg10": 1.01}, "ndcg above one"),
        ]
        for change, label in invalid_cases:
            changed = [dict(row) for row in summaries]
            changed[0].update(change)
            if label == "missing errors":
                del changed[0]["errors"]
            with self.subTest(change=label), self.assertRaises(spec.MatrixValidationError):
                spec.equal_weight_macro(changed)


class ProvenanceAndCliTests(unittest.TestCase):
    def test_resume_hash_envelope_rejects_changed_inputs(self):
        manifest = spec.build_run_manifest(
            workspace_label="portable-workspace", datasets_sha256={"a": "1"},
            model_sha256={"m": "2"}, binary_sha256="3", code_sha256={"runner": "4"},
            runtime={"candidate_k": 50},
        )
        spec.verify_resume_manifest(manifest, manifest)
        changed = {**manifest, "candidate_k": 100}
        with self.assertRaises(spec.MatrixValidationError):
            spec.verify_resume_manifest(manifest, changed)

    def test_execute_requires_fixed_batch_and_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                run_full_matrix.main(["--workspace", directory, "--output", str(output), "--execute"])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("--execute requires exactly one --batch", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_wall_limit_rejects_over_12h_and_nan(self):
        for value in ("13", "nan"):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                run_full_matrix.main(["--wall-hours", value])
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
