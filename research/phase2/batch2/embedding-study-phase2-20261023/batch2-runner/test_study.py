"""Offline unit tests for Phase 2 validation, metrics, and query selection."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_study as core


ROOT = Path(__file__).resolve().parent


class MetricsTests(unittest.TestCase):
    def test_binary_multiple_relevant_and_candidate_recall(self):
        result = core.metrics(["a", "x", "b"], ["a", "x", "b"], {"a": 1, "b": 1}, "binary")
        self.assertEqual(result["recall10"], 1)
        self.assertEqual(result["candidate_recall50"], 1)
        self.assertEqual(result["p5"], 0.4)
        self.assertEqual(result["hit10"], 1)

    def test_binary_judged_coverage_does_not_call_unjudged_irrelevant(self):
        result = core.metrics(["a", "u", "b", "v", "w"], ["a", "u", "b", "v", "w"],
                              {"a": 1, "b": 1, "x": 0}, "binary")
        self.assertAlmostEqual(result["judged_coverage10"], 2 / 5)
        self.assertAlmostEqual(result["judged_coverage50"], 2 / 5)

    def test_acord_keeps_grades_and_exp2_gain(self):
        qrels = {"low": 1, "high": 4, "zero": 0, "g3": 3, "g2": 2}
        result = core.metrics(["low", "high", "unjudged", "g3", "g2"],
                              ["low", "high", "unjudged", "g3", "g2"], qrels, "graded")
        self.assertLess(result["ndcg10"], 1)
        self.assertGreater(result["ndcg10"], 0)
        self.assertEqual(result["relevant_count"], 4)
        self.assertEqual(result["judged_count"], 5)
        self.assertEqual(result["p5_grade1"], 4 / 5)
        self.assertEqual(result["p5_grade2"], 3 / 5)
        self.assertEqual(result["p5_grade3"], 2 / 5)
        self.assertEqual(result["p5_grade4"], 1 / 5)
        self.assertEqual(result["judged_coverage10"], 4 / 5)

    def test_requires_at_least_one_positive(self):
        with self.assertRaises(ValueError):
            core.metrics(["a"], ["a"], {"a": 0}, "graded")


class RerankerGateTests(unittest.TestCase):
    def test_ten_fixture_pairs_must_rank_positive_higher_and_are_audited(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            study = core.Study(out, {}, "tiny", float("inf"), lambda: None, lambda *a, **k: None)
            with patch.object(core, "rerank", return_value=([0, 1], 987.0, [0.8, 0.2])) as mocked:
                study.gate_reranker(object(), "fixture-model")
            self.assertEqual(mocked.call_count, 10)
            report = json.loads((out / "gate-reranker.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["warmup_count"], 10)
            self.assertFalse(report["timing_included"])
            self.assertTrue(all(row["positive_score"] > row["negative_score"] for row in report["fixtures"]))

    def test_reranker_gate_fails_closed_when_negative_scores_higher(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            study = core.Study(out, {}, "tiny", float("inf"), lambda: None, lambda *a, **k: None)
            with patch.object(core, "rerank", return_value=([1, 0], 1.0, [0.1, 0.9])):
                with self.assertRaisesRegex(RuntimeError, "fixture gate failed"):
                    study.gate_reranker(object(), "fixture-model")
            report = json.loads((out / "gate-reranker.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "blocked")


class BGERerankerTokenAuditTests(unittest.TestCase):
    def metadata(self):
        return {
            "general.architecture": "bert",
            "tokenizer.ggml.model": "t5",
            "tokenizer.ggml.bos_token_id": 0,
            "tokenizer.ggml.eos_token_id": 2,
            "tokenizer.ggml.add_bos_token": True,
            "tokenizer.ggml.add_eos_token": True,
        }

    def test_pinned_fallback_metadata_derives_exact_three_token_pair_overhead(self):
        result = core.validate_bge_reranker_metadata(self.metadata())
        self.assertEqual(result["fallback_pair_overhead"], 3)
        self.assertEqual(result["fallback_pair_special_tokens"], {
            "bos": 1, "query_eos": 1, "document_eos": 1, "separator": 0,
        })

    def test_unexpected_sep_or_template_metadata_fails_closed(self):
        metadata = self.metadata()
        metadata["tokenizer.ggml.add_sep_token"] = True
        with self.assertRaisesRegex(RuntimeError, "metadata differs"):
            core.validate_bge_reranker_metadata(metadata)
        metadata = self.metadata()
        metadata["tokenizer.chat_template"] = "{query} {document}"
        with self.assertRaisesRegex(RuntimeError, "metadata differs"):
            core.validate_bge_reranker_metadata(metadata)

    def test_bge_tokenize_matches_rerank_raw_tokenization_flags(self):
        with patch.object(core, "request", return_value=({"tokens": [10, 11]}, 0.0)) as mocked:
            self.assertEqual(core.token_count(object(), "raw text", raw_reranker_input=True), 2)
        self.assertEqual(mocked.call_args.args[1:], ("/tokenize", {
            "content": "raw text", "add_special": False, "parse_special": False,
        }))

    def test_exact_three_specials_are_included_in_combined_context_gate(self):
        dataset = {
            "corpus": [{"id": "d", "text": "doc"}],
            "queries": [{"id": "q", "text": "query"}],
        }
        token_counts = {"doc": 6, "query": 2}
        bodies = []

        def fake_request(server, endpoint, body):
            bodies.append(body)
            return ({"tokens": list(range(token_counts[body["content"]]))}, 0.0)

        verified_template = {"fallback_pair_overhead": 3}
        with tempfile.TemporaryDirectory() as temp, patch.object(core, "request", side_effect=fake_request):
            record = core.audit_token_limits(
                object(), "tiny", dataset, "bge-reranker", "reranker", (), Path(temp) / "audit.jsonl",
                lambda: None, max_tokens=10, bge_template=verified_template,
            )
        self.assertEqual(record["status"], "blocked")  # 6 + 2 + 3 exceeds 10.
        self.assertEqual(record["issues"][0]["upper_bound"], 11)
        self.assertTrue(all(body == {"content": body["content"], "add_special": False, "parse_special": False}
                            for body in bodies))

    def test_bge_audit_refuses_unverified_template(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "requires a verified"):
                core.audit_token_limits(
                    object(), "tiny", {"corpus": [], "queries": []}, "bge-reranker", "reranker", (),
                    Path(temp) / "audit.jsonl", lambda: None,
                )


class DatasetTests(unittest.TestCase):
    def tiny_payload(self):
        return {"datasets": {"tiny": {
            "corpus": [{"id": "d1", "text": "Doc one"}, {"id": "d2", "text": "Doc two"}],
            "queries": [
                {"id": "q1", "text": "x", "metadata": {"cluster_id": "c1", "previous100": True}},
                {"id": "q2", "text": "xx"}, {"id": "q3", "text": "xxx"},
            ],
            "qrels": {"q1": {"d1": 1}, "q2": {"d2": 2}, "q3": {"d1": 0, "d2": 4}},
            "grade_mode": "graded", "pilot_query_ids": ["q1", "q2", "q3"],
        }}}

    def test_schema_normalizes_ids_and_preserves_query_metadata(self):
        result = core.validate_datasets(self.tiny_payload(), require_expected=False)["tiny"]
        self.assertEqual(result["qrels"]["q2"]["d2"], 2)
        self.assertEqual(core.query_metadata(result["queries"][0])["cluster_id"], "c1")
        self.assertTrue(core.query_metadata(result["queries"][0])["previous100"])

    def test_binary_schema_rejects_graded_values(self):
        payload = self.tiny_payload()
        payload["datasets"]["tiny"]["grade_mode"] = "binary"
        with self.assertRaises(core.DataValidationError):
            core.validate_datasets(payload, require_expected=False)

    def test_pilot_uses_registered_ids_or_evenly_spaced_length_quantiles(self):
        data = core.validate_datasets(self.tiny_payload(), require_expected=False)["tiny"]
        self.assertEqual(core.select_pilot_ids(data), ["q1", "q2", "q3"])
        data["pilot_query_ids"] = None
        self.assertEqual(core.select_pilot_ids(data, requested_n=3), ["q1", "q2", "q3"])

    def test_frozen_miracl_batch2_dataset_counts_and_schema(self):
        datasets, _ = core.load_datasets(ROOT / "datasets.json")
        self.assertEqual(list(datasets), list(core.DATASET_COUNTS))
        self.assertEqual(sum(len(data["queries"]) for data in datasets.values()) * 3, 1200)
        self.assertEqual(datasets["miracl-zh-zh"]["grade_mode"], "binary")
        self.assertEqual(datasets["miracl-en-en"]["grade_mode"], "binary")
        self.assertEqual(len(datasets["miracl-zh-zh"]["pilot_query_ids"]), 3)
        self.assertEqual(len(datasets["miracl-en-en"]["pilot_query_ids"]), 3)
        self.assertEqual(sum(map(len, datasets["miracl-zh-zh"]["qrels"].values())), 1998)
        self.assertEqual(sum(map(len, datasets["miracl-en-en"]["qrels"].values())), 2091)


if __name__ == "__main__":
    unittest.main()
