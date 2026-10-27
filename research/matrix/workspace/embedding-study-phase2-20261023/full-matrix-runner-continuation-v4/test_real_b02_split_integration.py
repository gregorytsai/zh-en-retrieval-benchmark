"""Offline end-to-end regression using the preserved, accepted B02-A/B artifacts."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from batch_progress import aggregate_b02_splits
from run_sequence import update_plan


PROJECT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = PROJECT / "run-full-matrix-continuation-v3"
PLAN_SOURCE = PROJECT / "FULL_MATRIX_PLAN_ZH_TW.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RealB02SplitIntegrationTests(unittest.TestCase):
    """Exercise real summary/row schema through merge, audit, and ranking."""

    def test_real_b02_splits_rebuild_combined_ranking_without_touching_v3(self):
        if not (SOURCE_ROOT / "B02-A/segment-01/official/subbatch-completion-audit.json").is_file():
            self.skipTest("preserved v3 B02-A/B official evidence is unavailable")
        source_combined = SOURCE_ROOT / "B02/combined"
        protected_sources = [
            source_combined / "batch-completion-audit.json",
            source_combined / "summary.json",
            source_combined / "query-results.jsonl",
        ]
        before = {path: sha256(path) for path in protected_sources}

        with tempfile.TemporaryDirectory(prefix="phase2-b02-combined-") as tmp:
            root = Path(tmp)
            run_root = root / "run"
            run_root.mkdir()
            for batch in ("B02-A", "B02-B"):
                (run_root / batch).symlink_to(SOURCE_ROOT / batch, target_is_directory=True)

            plan = root / "plan.md"
            plan.write_text(PLAN_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
            update_plan(plan, run_root, "B02-A", "complete", run_root / "B02-A/segment-01/official")
            update_plan(plan, run_root, "B02-B", "complete", run_root / "B02-B/segment-01/official")

            report = aggregate_b02_splits(run_root, PROJECT, plan)
            self.assertIsNotNone(report)
            self.assertEqual(
                {key: report[key] for key in ("status", "batch", "summaries", "rows", "errors",
                                               "accepted_gpu_attempts", "candidate_hashes_verified")},
                {"status": "passed", "batch": "B02", "summaries": 24, "rows": 10452,
                 "errors": 0, "accepted_gpu_attempts": 24, "candidate_hashes_verified": True},
            )

            combined = run_root / "B02/combined"
            ranking = json.loads((combined / "ranking.json").read_text(encoding="utf-8"))
            self.assertEqual(ranking["scope"],
                             "B02 only; merged from two independently audited disjoint dataset subbatches")
            configs = ranking["configurations"]
            self.assertEqual({row["reranker"] for row in configs}, {
                "standalone", "bge-reranker", "qwen-reranker-0.6b", "qwen-reranker-4b"})
            self.assertEqual(len(configs), 4)
            self.assertTrue(all(row["completion_audit_passed"] and row["eligible"] for row in configs))
            self.assertTrue(all(row["source_subbatches"] == ["B02-A", "B02-B"] for row in configs))

        self.assertEqual({path: sha256(path) for path in protected_sources}, before,
                         "offline reconstruction must not rewrite v3 evidence")


if __name__ == "__main__":
    unittest.main()
