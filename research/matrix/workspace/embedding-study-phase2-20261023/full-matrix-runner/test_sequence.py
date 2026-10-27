import argparse
import json
import tempfile
import unittest
from pathlib import Path

from run_sequence import stage_command, update_plan


class SequenceTests(unittest.TestCase):
    def test_stage_command_uses_one_fixed_batch_and_explicit_resume(self):
        args = argparse.Namespace(workspace=Path('/workspace'), output=Path('/out'), plan=Path('/plan'))
        pilot = stage_command(args, 'B01', 'pilot', False)
        official = stage_command(args, 'B02', 'official', True)
        self.assertIn('--execute', pilot)
        self.assertIn('--pilot', pilot)
        self.assertNotIn('--resume', pilot)
        self.assertIn('--resume', official)
        self.assertNotIn('--pilot', official)
        self.assertEqual(official[official.index('--batch')+1], 'B02')

    def test_pilot_never_counts_as_formal_rows_and_passed_row_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            plan = p / 'plan.md'
            plan.write_text('| B01 | qwen-4b | 待執行 | 0/24 | 0/10452 | — | — |\n')
            home = p / 'pilot'
            home.mkdir()
            (home / 'committed.json').write_text(json.dumps({'a':{'queries':3}}))
            update_plan(plan,'B01','pilot',home)
            self.assertIn('| 0/24 | 0/10452 |',plan.read_text())
            plan.write_text('| B01 | qwen-4b | 稽核通過 | 24/24 | 10452/10452 | done | path |\n')
            original = plan.read_text()
            update_plan(plan,'B01','official',home)
            self.assertEqual(plan.read_text(),original)

    def test_only_actual_committed_rows_enter_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            plan=p/'plan.md'
            plan.write_text('| B01 | qwen-4b | 待執行 | 0/24 | 0/10452 | — | — |\n')
            home=p/'official'
            home.mkdir()
            (home/'committed.json').write_text(json.dumps({'a':{'queries':504},'b':{'queries':200}}))
            update_plan(plan,'B01','official',home)
            self.assertIn('| 2/24 | 704/10452 |',plan.read_text())


if __name__=='__main__':
    unittest.main()
