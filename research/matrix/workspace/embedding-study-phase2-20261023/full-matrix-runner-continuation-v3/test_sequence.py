import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from run_sequence import main, run_stage, stage_command, stop_owned_child, update_plan


class SequenceTests(unittest.TestCase):
    def test_controller_launches_guarded_child_only_after_plan_row_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / 'plan.md'
            plan.write_text(
                '| B02-A | qwen-8b | waiting | 0/4 | 0/4592 | — | — |\n'
                '| B02-B | qwen-8b | waiting | 0/20 | 0/5860 | — | — |\n',
                encoding='utf-8')
            output = root / 'output'
            args = argparse.Namespace(output=output, workspace=root, plan=plan)
            child = type('ExitedChild', (), {'pid': 987654, 'returncode': 0,
                                               'poll': lambda self: 0,
                                               'wait': lambda self: 0})()

            def fake_popen(command, **kwargs):
                self.assertIn('--execute', command)
                self.assertIn('--pilot', command)
                self.assertTrue(kwargs.get('start_new_session'))
                self.assertIn('| B02-A | qwen-8b | Pilot驗證中（正式0） |', plan.read_text())
                home = output / 'B02-A' / 'segment-01' / 'pilot'
                home.mkdir(parents=True)
                (home / 'progress.json').write_text(
                    json.dumps({'phase': 'pilot_complete'}), encoding='utf-8')
                return child

            state = {}
            with patch('run_sequence.subprocess.Popen', side_effect=fake_popen) as popen:
                result = run_stage(args, 'B02-A', 'pilot', state, segment='01')

            popen.assert_called_once()
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(state['phase'], 'running')
            self.assertEqual(state['child_pid'], 987654)
            self.assertIn('| B02-A | qwen-8b | Pilot稽核通過；正式待啟動 |', plan.read_text())

    def test_bad_subbatch_progress_row_fails_before_spawning_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / 'plan.md'
            plan.write_text('| B02-B | qwen-8b | waiting | 0/20 | 0/5860 | — | — |\n', encoding='utf-8')
            output = root / 'output'
            args = argparse.Namespace(output=output, workspace=root, plan=plan)
            with patch('run_sequence.subprocess.Popen') as popen:
                with self.assertRaisesRegex(RuntimeError, 'exactly one B02-A row'):
                    run_stage(args, 'B02-A', 'pilot', {}, segment='01')
            popen.assert_not_called()
            self.assertFalse((output / 'B02-A-pilot-console.log').exists())

    def test_owned_process_group_cleanup_stops_runner_and_descendant(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / 'descendant.pid'
            descendant = "import time; time.sleep(30)"
            parent = (
                "import subprocess,sys,time; "
                "p=subprocess.Popen([sys.executable,'-c',sys.argv[2]]); "
                "open(sys.argv[1],'w').write(str(p.pid)); time.sleep(30)"
            )
            child = subprocess.Popen([sys.executable, '-c', parent, str(pid_file), descendant],
                                     start_new_session=True)
            deadline = time.monotonic() + 5
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(pid_file.exists(), 'offline child did not launch its test descendant')
            descendant_pid = int(pid_file.read_text())
            stop_owned_child(child, grace_seconds=2)
            self.assertIsNotNone(child.poll())
            deadline = time.monotonic() + 3
            alive = True
            while alive and time.monotonic() < deadline:
                try:
                    os.kill(descendant_pid, 0)
                except ProcessLookupError:
                    alive = False
                else:
                    time.sleep(0.02)
            self.assertFalse(alive, 'controller-owned test descendant escaped the isolated process group')

    def test_split_batch_progress_updates_the_registered_subbatch_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            plan = p / 'plan.md'
            plan.write_text(
                '| B02-A | qwen-8b | waiting | 0/4 | 0/4592 | — | — |\n'
                '| B02-B | qwen-8b | waiting | 0/20 | 0/5860 | — | — |\n',
                encoding='utf-8')
            output = p / 'output'
            pilot = output / 'B02-A' / 'segment-01' / 'pilot'
            pilot.mkdir(parents=True)
            update_plan(plan, output, 'B02-A', 'pilot', pilot)
            self.assertIn('| B02-A | qwen-8b | Pilot驗證中（正式0） | 0/4 | 0/4592 |', plan.read_text())

            official = output / 'B02-A' / 'segment-01' / 'official'
            official.mkdir(parents=True)
            (official / 'committed.json').write_text(
                json.dumps({'eng-eng--qwen-8b--standalone': {'queries': 1148}}), encoding='utf-8')
            update_plan(plan, output, 'B02-A', 'official', official)
            self.assertIn('| B02-A | qwen-8b | 正式執行中 | 1/4 | 1148/4592 |', plan.read_text())

            (official / 'subbatch-completion-audit.json').write_text(
                json.dumps({'status': 'passed'}), encoding='utf-8')
            (official / 'committed.json').write_text(json.dumps({
                f'eng-eng--qwen-8b--{name}': {'queries': 1148}
                for name in ('standalone', 'bge-reranker', 'qwen-reranker-0.6b', 'qwen-reranker-4b')
            }), encoding='utf-8')
            update_plan(plan, output, 'B02-A', 'complete', official)
            self.assertIn('| B02-A | qwen-8b | 稽核通過（B02子批） | 4/4 | 4592/4592 |', plan.read_text())
            self.assertIn('| B02-B | qwen-8b | waiting | 0/20 | 0/5860 |', plan.read_text())

    def test_progress_updater_ignores_same_batch_in_scope_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / 'plan.md'
            plan.write_text(
                '| 延續段 | Dataset scope | Pilot上限/樣本 | Official scope |\n'
                '|---|---|---|---|\n'
                '| B02-A | MLQA eng-eng | 2h；該分項3題長度分位 | 4 summaries／4,592 rows |\n'
                '| B02-B | 其餘五分項 | 2h | 20 summaries／5,860 rows |\n'
                '\n'
                '| 批次 | Embedder ID | 狀態 | 摘要 | 逐題結果 | 台灣時間起訖 | 結果目錄 |\n'
                '|---|---|---|---|---|---|---|\n'
                '| B02-A | qwen-8b | 等待新pilot | 0/4 | 0/4592 | — | — |\n'
                '| B02-B | qwen-8b | 等待新pilot | 0/20 | 0/5860 | — | — |\n',
                encoding='utf-8')
            output = root / 'output'
            pilot = output / 'B02-A' / 'segment-01' / 'pilot'
            update_plan(plan, output, 'B02-A', 'pilot', pilot)
            lines = plan.read_text(encoding='utf-8').splitlines()
            self.assertIn('| B02-A | MLQA eng-eng | 2h；該分項3題長度分位 | 4 summaries／4,592 rows |', lines)
            self.assertTrue(any(line.startswith(
                '| B02-A | qwen-8b | Pilot驗證中（正式0） | 0/4 | 0/4592 |') for line in lines))
            self.assertIn('| B02-B | qwen-8b | 等待新pilot | 0/20 | 0/5860 | — | — |', lines)

    def test_stage_command_uses_one_fixed_batch_and_explicit_resume(self):
        args = argparse.Namespace(workspace=Path('/workspace'), output=Path('/out'), plan=Path('/plan'))
        pilot = stage_command(args, 'B02-A', 'pilot', False, segment='01')
        official = stage_command(args, 'B02-B', 'official', True, segment='02',
                                 continue_from=Path('/out/B02-B/segment-01/official'),
                                 pilot_evidence=Path('/out/B02-B/segment-01/pilot/pilot-decision.json'))
        self.assertIn('--execute', pilot)
        self.assertIn('--pilot', pilot)
        self.assertNotIn('--resume', pilot)
        self.assertIn('--resume', official)
        self.assertNotIn('--pilot', official)
        self.assertEqual(official[official.index('--batch')+1], 'B02-B')
        self.assertEqual(official[official.index('--segment')+1], '02')
        self.assertEqual(official[official.index('--continue-compatible-from')+1],
                         '/out/B02-B/segment-01/official')
        self.assertEqual(official[official.index('--pilot-evidence')+1],
                         '/out/B02-B/segment-01/pilot/pilot-decision.json')

    def test_sequence_requires_explicit_execute_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'no-run'
            with self.assertRaises(SystemExit) as raised:
                main(['--workspace', str(root), '--output', str(output)])
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(output.exists())

    def test_pilot_never_counts_as_formal_rows_and_passed_row_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            plan = p / 'plan.md'
            plan.write_text('| B03 | embeddinggemma-2 | 待執行 | 0/24 | 0/10452 | — | — |\n')
            output = p / 'output'
            home = output / 'B03' / 'segment-01' / 'pilot'
            home.mkdir(parents=True)
            (home / 'committed.json').write_text(json.dumps({'a':{'queries':3}}))
            update_plan(plan, output, 'B03','pilot',home)
            self.assertIn('| 0/24 | 0/10452 |',plan.read_text())
            plan.write_text('| B03 | embeddinggemma-2 | 稽核通過 | 24/24 | 10452/10452 | done | path |\n')
            original = plan.read_text()
            update_plan(plan, output, 'B03','official',home)
            self.assertEqual(plan.read_text(),original)

    def test_only_actual_committed_rows_enter_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            plan=p/'plan.md'
            plan.write_text('| B03 | embeddinggemma-2 | 待執行 | 0/24 | 0/10452 | — | — |\n')
            output=p/'output'
            home=output/'B03'/'segment-01'/'official'
            home.mkdir(parents=True)
            (home/'committed.json').write_text(json.dumps({'a':{'queries':504},'b':{'queries':200}}))
            update_plan(plan, output, 'B03','official',home)
            self.assertIn('| 2/24 | 704/10452 |',plan.read_text())


if __name__=='__main__':
    unittest.main()
