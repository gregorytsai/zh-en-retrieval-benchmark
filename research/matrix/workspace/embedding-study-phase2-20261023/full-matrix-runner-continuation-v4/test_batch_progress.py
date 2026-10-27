import unittest
from batch_progress import render_completed


class BatchProgressTests(unittest.TestCase):
    def setUp(self):
        self.text = '# Plan\n\n| B01 | qwen-4b | 待執行 | 0/24 | 0/10452 | — | — |\n| B02 | qwen-8b | 待執行 | 0/24 | 0/10452 | — | — |\n'
        self.report = dict(status='passed', batch='B01', embedder='qwen-4b', summaries=24,
                           rows=10452, errors=0, started=1792771200,
                           finished='2026-10-24T08:00:00+00:00', run='/tmp/example/B01/official')

    def test_changes_only_selected_row(self):
        result = render_completed(self.text, self.report)
        self.assertIn('| B01 | qwen-4b | 稽核通過 | 24/24 | 10452/10452 |', result)
        self.assertEqual(self.text.splitlines()[-1], result.splitlines()[-1])

    def test_partial_not_complete(self):
        for field,value in [('rows',10451),('summaries',23),('errors',1),('status','running')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                render_completed(self.text, {**self.report,field:value})

    def test_unknown_or_duplicate_row_rejected(self):
        with self.assertRaises(ValueError):
            render_completed(self.text, {**self.report,'batch':'B09'})
        with self.assertRaises(ValueError):
            render_completed(self.text+self.text,self.report)

    def test_table_path_injection_rejected(self):
        with self.assertRaises(ValueError):
            render_completed(self.text,{**self.report,'run':'bad|path'})


if __name__ == '__main__':
    unittest.main()
