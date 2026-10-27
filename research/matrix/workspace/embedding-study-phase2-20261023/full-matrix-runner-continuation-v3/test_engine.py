from __future__ import annotations

import contextlib
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import core
import guarded_engine


class FakeStudy(core.Study):
    @contextlib.contextmanager
    def server(self, name, model, mode, pooling="last"):
        yield object()

    def check_time(self):
        return None

    def gate_embedding(self, server, query_prefix, document_prefix, dim):
        return None

    def gate_reranker(self, server, model_name):
        return None


class GuardStub:
    def check(self):
        return None


class EngineCoreIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = "eng-zho"
        self.embedder = "f2llm-v2-0.6b"  # deliberately exercises a non-Qwen-4B cache
        self.data = {
            "corpus": [{"id": f"d{i:02}", "text": f"document {i}"} for i in reversed(range(50))],
            "queries": [{"id": "q1", "text": "query text"}],
            "qrels": {"q1": {"d00": 1}},
            "grade_mode": "binary",
            "metadata": {},
            "_dataset_sha256": "dataset-sha",
            "_model_hashes": {self.embedder: "embedder-sha"},
            "_binary_sha256": "binary-sha",
            "_code_sha256": {"core.py": "core-sha"},
        }
        self.query = self.data["queries"]
        self.cache_root = self.root / "cache"
        self.standalone_out = self.root / "standalone"

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def fake_embed(server, texts, dim):
        # Identical vectors create intentional ties; the core must use doc-ID ordering.
        return np.ones((len(texts), dim), dtype=np.float32), 1.0

    @staticmethod
    def fake_token_count(server, text, *, raw_reranker_input=False):
        return 1

    def make_candidates(self):
        study = FakeStudy(self.standalone_out, self.data, self.dataset, time.time() + 600,
                          lambda: None, lambda *args, **kwargs: None)
        with patch.object(core, "embed", self.fake_embed), patch.object(core, "token_count", self.fake_token_count):
            rows = study.standalone(self.embedder, self.query, self.cache_root)
        self.assertEqual(len(rows), 1)
        return rows

    def fake_rerank(self, server, query, documents):
        scores = [float(index) for index in range(len(documents))]
        order = list(reversed(range(len(documents))))
        return order, 12.5, scores

    def test_non_qwen_embedder_candidate_cache_is_exact_top50_and_tie_breaks_by_doc_id(self):
        rows = self.make_candidates()
        expected_ids = [f"d{i:02}" for i in range(50)]
        self.assertEqual(rows[0]["final_ids"], expected_ids)
        self.assertEqual(rows[0]["k"], 50)
        cache_dir = self.cache_root / self.embedder / self.dataset
        cache = json.loads((cache_dir / "candidates.json").read_text())
        self.assertEqual(cache[0]["ids"], expected_ids)
        manifest = json.loads((cache_dir / "cache-manifest.json").read_text())
        self.assertEqual(manifest["embedding"], self.embedder)
        self.assertEqual(manifest["candidate_k"], 50)
        self.assertEqual(manifest["stored_final_ids"], 50)

    def test_all_three_rerankers_consume_same_non_qwen_candidate_pool(self):
        self.make_candidates()
        candidate_hashes = []
        for model_name in core.RERANKERS:
            out = self.root / model_name
            study = FakeStudy(out, self.data, self.dataset, time.time() + 600,
                              lambda: None, lambda *args, **kwargs: None)
            with patch.object(core, "rerank", self.fake_rerank):
                rows = study.rerank(model_name, self.query, self.cache_root, self.embedder)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["embedding"], self.embedder)
            self.assertEqual(rows[0]["reranker"], model_name)
            self.assertEqual(rows[0]["k"], 50)
            candidate_hashes.append(rows[0]["candidate_ids_sha256"])
        self.assertEqual(len(set(candidate_hashes)), 1)

    def test_candidate_file_tampering_is_rejected_before_reranker_call(self):
        self.make_candidates()
        candidate_file = self.cache_root / self.embedder / self.dataset / "candidates.json"
        candidate_file.write_text(candidate_file.read_text() + " ")
        study = FakeStudy(self.root / "rerank", self.data, self.dataset, time.time() + 600,
                          lambda: None, lambda *args, **kwargs: None)
        with patch.object(core, "rerank") as rerank_mock:
            with self.assertRaisesRegex(RuntimeError, "cache file hash changed"):
                study.rerank("qwen-reranker-0.6b", self.query, self.cache_root, self.embedder)
        rerank_mock.assert_not_called()

    def test_live_entrypoints_remain_disabled(self):
        with self.assertRaisesRegex(RuntimeError, "Execution requires explicit"):
            guarded_engine.run(type("Args", (), {"execute": False})())
        with self.assertRaisesRegex(RuntimeError, "Preparation build"):
            guarded_engine.parse_args(["--execute"])

    def test_expired_deadline_fails_before_guard_or_server(self):
        study = guarded_engine.GuardedStudy(
            self.root / "deadline", self.data, self.dataset, time.time() - 1,
            lambda: None, lambda *args, **kwargs: None,
            guard=GuardStub(), watchdog=object(),
        )
        with self.assertRaisesRegex(TimeoutError, "deadline"):
            study.check_time()

    def test_timezone_is_required_for_absolute_deadline(self):
        parsed = guarded_engine.parse_deadline("2026-10-24T00:00:00+08:00")
        self.assertAlmostEqual(parsed, 1792771200.0, delta=1.0)
        with self.assertRaises(Exception):
            guarded_engine.parse_deadline("2026-10-24T00:00:00")

    def test_uncommitted_cache_left_by_crash_is_preserved_then_replaced(self):
        home = self.root / "run"
        attempt = home / "attempts" / "new"
        source = attempt / "cache" / self.embedder / self.dataset
        destination = home / "cache" / self.embedder / self.dataset
        source.mkdir(parents=True)
        destination.mkdir(parents=True)
        (source / "cache-manifest.json").write_text('{"manifest":"fresh"}')
        (destination / "cache-manifest.json").write_text('{"manifest":"orphan"}')

        guarded_engine.copy_accepted_cache(home, attempt, self.embedder, self.dataset)

        self.assertEqual(json.loads((destination / "cache-manifest.json").read_text()), {"manifest": "fresh"})
        orphan_dirs = list((home / "orphan-caches").iterdir())
        self.assertEqual(len(orphan_dirs), 1)
        self.assertEqual(json.loads((orphan_dirs[0] / "cache-manifest.json").read_text()), {"manifest": "orphan"})

    def test_committed_cache_conflict_is_never_overwritten(self):
        home = self.root / "committed-run"
        attempt = home / "attempts" / "new"
        source = attempt / "cache" / self.embedder / self.dataset
        destination = home / "cache" / self.embedder / self.dataset
        source.mkdir(parents=True)
        destination.mkdir(parents=True)
        (source / "cache-manifest.json").write_text('{"manifest":"fresh"}')
        (destination / "cache-manifest.json").write_text('{"manifest":"committed"}')
        key = f"{self.dataset}--{self.embedder}--standalone"
        (home / "committed.json").write_text(json.dumps({key: {"attempt": "attempts/accepted"}}))

        with self.assertRaisesRegex(RuntimeError, "cannot be replaced"):
            guarded_engine.copy_accepted_cache(home, attempt, self.embedder, self.dataset)
        self.assertEqual(json.loads((destination / "cache-manifest.json").read_text()), {"manifest": "committed"})


if __name__ == "__main__":
    unittest.main()
