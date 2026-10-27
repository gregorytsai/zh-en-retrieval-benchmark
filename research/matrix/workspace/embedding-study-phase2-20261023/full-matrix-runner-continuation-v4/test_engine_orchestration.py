import json
import shutil
import tempfile
import unittest
from pathlib import Path

import core
import guarded_engine as engine
import test_engine as engine_fixtures


class AcceptedAggregationTests(unittest.TestCase):
    def test_only_committed_index_stats_and_rows_are_consolidated(self):
        helper = engine_fixtures.EngineCoreIntegrationTests()
        helper.setUp()
        try:
            helper.make_candidates()
            home = helper.root / 'official'
            attempt = home / 'attempts' / 'accepted'
            shutil.copytree(helper.standalone_out, attempt)
            (attempt / 'validity.json').write_text(json.dumps({'status':'accepted'}))
            key = f'{helper.dataset}--{helper.embedder}--standalone'
            jobs = [(helper.dataset,helper.embedder,'standalone',key)]
            committed = {key:{'attempt':'attempts/accepted','queries':1}}
            # A larger uncommitted result must not influence corpus timing totals.
            other = home / 'attempts' / 'invalidated'
            other.mkdir()
            (other / 'index-stats.jsonl').write_text('{"corpus_ms":999999}\n')
            rows, summary = engine.consolidate(home,jobs,committed,{helper.dataset:['q1']})
            indexes = core.rows(home / 'index-stats.jsonl')
            self.assertEqual(len(rows),1)
            self.assertEqual(len(summary),1)
            self.assertEqual(len(indexes),1)
            self.assertEqual(indexes[0]['corpus_docs'],50)
            self.assertLess(indexes[0]['corpus_ms'],999999)
        finally:
            helper.tearDown()


if __name__=='__main__':
    unittest.main()
