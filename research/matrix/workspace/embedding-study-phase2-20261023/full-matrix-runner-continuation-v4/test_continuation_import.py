from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import core
import guarded_engine
import matrix_spec


def put_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def put_jsonl(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8")


class CompatibleImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.source = root / "segment-01" / "official"
        self.target = root / "segment-02" / "official"
        self.source.mkdir(parents=True)
        self.target.mkdir(parents=True)
        self.dataset = "eng-eng"
        self.embedder = "qwen-8b"
        self.query_ids = ["q1"]
        self.internal_dataset_sha = "dataset-record-sha"
        self.model_hashes = {name: f"sha-{name}" for name in (self.embedder, *matrix_spec.RERANKERS)}
        self.binary_sha = "approved-binary-sha"
        self.code_hashes = {"guarded_engine.py": "frozen-code-sha"}
        self.selection = {
            "run_kind": "official", "dataset_sha256": "combined-data-sha",
            "selection_policy": "all-manifest-queries",
            "datasets": {self.dataset: {"query_ids": self.query_ids}},
        }
        self.qrels = {"dataset_sha256": "combined-data-sha", "datasets": {self.dataset: {"qrels": {"q1": {"d00": 1}}}}}
        self.deadline = {"original_started_epoch": time.time() - 13 * 3600,
                         "deadline_epoch": time.time() - 3600, "wall_hours": 12.0,
                         "created_at": "old"}
        self.manifest = {
            "batch_id": "B02-A", "run_kind": "official", "dataset_file_name": "datasets.json",
            "dataset_sha256": "combined-data-sha", "binary_sha256": self.binary_sha,
            "model_sha256": self.model_hashes, "code_sha256": self.code_hashes,
            "dataset_ids": [self.dataset],
            "requested_configs": [f"{self.embedder}__{name}" for name in ("standalone", *matrix_spec.RERANKERS)],
            "effective_configs": [f"{self.embedder}__{name}" for name in ("standalone", *matrix_spec.RERANKERS)],
            "selection_sha256": core.sha256_json(self.selection),
            "selected_qrels_sha256": core.sha256_json(self.qrels),
            "deadline_record_sha256": core.sha256_json(self.deadline),
            "original_started_epoch": self.deadline["original_started_epoch"],
            "deadline_epoch": self.deadline["deadline_epoch"], "wall_hours": 12.0,
            "segment_id": "01", "continuation_source": None,
        }
        self.target_manifest = {
            **self.manifest, "deadline_record_sha256": "new-deadline-sha",
            "original_started_epoch": time.time(), "deadline_epoch": time.time() + 12 * 3600,
            "segment_id": "02", "continuation_source": str(self.source),
        }
        self.datasets = {self.dataset: {
            "queries": [{"id": "q1", "text": "example query"}],
            "corpus": [{"id": f"d{i:02}", "text": f"document {i}"} for i in range(50)],
            "qrels": {"q1": {"d00": 1}}, "grade_mode": "binary", "metadata": {},
            "_dataset_sha256": self.internal_dataset_sha,
        }}
        self.selections = {self.dataset: self.query_ids}
        put_json(self.source / "run-manifest.json", self.manifest)
        put_json(self.source / "deadline.json", self.deadline)
        put_json(self.source / "progress.json", {"phase": "stopped", "time": "expired"})
        put_jsonl(self.source / "fatal-errors.jsonl", [{"time": "2099-01-01T00:00:00+00:00",
                                                       "type": "TimeoutError", "error": "Original deadline reached"}])
        put_json(self.source / "query-selection.json", self.selection)
        put_json(self.source / "qrels.json", self.qrels)
        self.key = f"{self.dataset}--{self.embedder}--standalone"
        self.attempt_rel = "attempts/accepted-standalone"
        self.attempt = self.source / self.attempt_rel
        self._write_cache(self.attempt / "cache" / self.embedder / self.dataset)
        self._write_cache(self.source / "cache" / self.embedder / self.dataset)
        put_json(self.attempt / "validity.json", {"status": "accepted", "traffic_epoch": 3})
        put_json(self.attempt / "gate-embedding.json", {"repeatability": True, "fixtures": [{} for _ in range(10)]})
        row = {"dataset": self.dataset, "embedding": self.embedder, "reranker": "standalone",
               "query_id": "q1", "status": "ok", "final_ids": [f"d{i:02}" for i in range(50)]}
        put_jsonl(self.attempt / "query-results.jsonl", [row])
        put_jsonl(self.attempt / "index-stats.jsonl", [{"dataset": self.dataset, "corpus_ms": 1.0}])
        log = self.attempt / "logs" / "server.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("offloaded 12/12 layers to GPU\n", encoding="utf-8")
        command = ["llama-server", "--n-gpu-layers", "all", "--log-file", str(log)]
        put_jsonl(self.attempt / "server-events.jsonl", [{"command": command}])
        put_json(self.source / "committed.json", {
            self.key: {"attempt": self.attempt_rel, "accepted_at": "old", "queries": 1, "attempt_count": 1}
        })
        put_jsonl(self.source / "traffic-events.jsonl", [
            {"event": "guard_unit_started", "traffic_epoch": 3, "quiet_seconds": 30,
             "poll_period_seconds": 1, "snapshot": {"workers": {}, "routers": {}}},
            {"event": "combo_started", "combo": self.key, "attempt": self.attempt_rel},
            {"event": "guard_unit_finished", "traffic_epoch": 3, "snapshot": {"workers": {}, "routers": {}}},
            {"event": "combo_work_finished", "combo": self.key, "attempt": self.attempt_rel, "traffic_epoch": 3},
            {"event": "combo_accepted", "combo": self.key, "attempt": self.attempt_rel},
        ])
        put_json(self.target / "run-manifest.json", self.target_manifest)
        put_json(self.target / "query-selection.json", self.selection)
        put_json(self.target / "qrels.json", self.qrels)

    def tearDown(self):
        self.temp.cleanup()

    def _write_cache(self, cache: Path):
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "corpus.npy").write_bytes(b"fixture corpus vectors")
        (cache / "queries.npy").write_bytes(b"fixture query vectors")
        put_json(cache / "candidates.json", [{"query_id": "q1", "ids": [f"d{i:02}" for i in range(50)],
                                              "scores": [1.0] * 50}])
        files = {name: core.sha256_file(cache / name) for name in
                 ("corpus.npy", "queries.npy", "candidates.json")}
        manifest = {"dataset": self.dataset, "embedding": self.embedder,
                    "dataset_sha256": self.internal_dataset_sha,
                    "model_sha256": self.model_hashes[self.embedder],
                    "binary_sha256": self.binary_sha, "code_sha256": self.code_hashes,
                    "query_ids": self.query_ids, "candidate_k": 50, "normalized": True,
                    "files_sha256": files}
        manifest["cache_key_sha256"] = core.sha256_json({k: v for k, v in manifest.items()
                                                         if k not in ("cache_key_sha256", "files_sha256")})
        put_json(cache / "cache-manifest.json", manifest)

    def test_imports_only_compatible_commit_and_preserves_guard_gpu_provenance(self):
        jobs = guarded_engine.job_list([self.dataset], self.manifest["effective_configs"])
        imported = guarded_engine.import_compatible_commits(
            self.source, self.target, self.target_manifest, self.datasets, self.selections, jobs)
        self.assertEqual(imported["status"], "passed")
        self.assertEqual(imported["imported_commits"], [self.key])
        committed = json.loads((self.target / "committed.json").read_text())
        self.assertIn(self.key, committed)
        self.assertTrue((self.target / committed[self.key]["attempt"] / "validity.json").is_file())
        events = core.rows(self.target / "traffic-events.jsonl")
        self.assertEqual({e["combo"] for e in events if e.get("event") == "combo_accepted"}, {self.key})
        self.assertTrue((self.target / "continuation-import.json").is_file())
        copied_cache = self.target / "cache" / self.embedder / self.dataset
        guarded_engine.verified_cache_manifest(copied_cache, self.query_ids)

    def test_candidate_cache_tampering_rejects_import(self):
        path = self.source / "cache" / self.embedder / self.dataset / "candidates.json"
        path.write_text(path.read_text() + " ", encoding="utf-8")
        jobs = guarded_engine.job_list([self.dataset], self.manifest["effective_configs"])
        with self.assertRaises(RuntimeError):
            guarded_engine.import_compatible_commits(
                self.source, self.target, self.target_manifest, self.datasets, self.selections, jobs)

    def test_new_run_cannot_import_pilot_or_unexpired_source(self):
        jobs = guarded_engine.job_list([self.dataset], self.manifest["effective_configs"])
        changed = {**self.manifest, "run_kind": "pilot"}
        put_json(self.source / "run-manifest.json", changed)
        with self.assertRaises(RuntimeError):
            guarded_engine.import_compatible_commits(
                self.source, self.target, self.target_manifest, self.datasets, self.selections, jobs)


if __name__ == "__main__":
    unittest.main()
