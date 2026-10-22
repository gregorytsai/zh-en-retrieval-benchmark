"""Small deterministic evaluator checks; no model loading or network."""
import unittest
import numpy as np
from run_study import metrics

class MetricsTests(unittest.TestCase):
    def test_perfect(self):
        result = metrics(['a'], ['a'], {'a'})
        self.assertTrue(all(value == 1 for value in result.values()))

    def test_missing(self):
        result = metrics(['x'], ['x'], {'a'})
        self.assertTrue(all(value == 0 for value in result.values()))

    def test_multiple_relevant_not_hit_rate(self):
        result = metrics(['a', 'x'], ['a', 'x'], {'a', 'b'})
        self.assertEqual(result['recall10'], .5)
        self.assertEqual(result['hit10'], 1)
        self.assertEqual(result['candidate_recall50'], .5)
        self.assertLess(result['ndcg10'], 1)

    def test_retrieved_but_outside_top10(self):
        order = [str(i) for i in range(10)] + ['a']
        result = metrics(order, order, {'a'})
        self.assertEqual(result['mrr10'], 0)
        self.assertEqual(result['recall10'], 0)
        self.assertEqual(result['recall50'], 1)

    def test_rerank_cannot_change_candidate_recall(self):
        result = metrics(['x', 'a'], ['a', 'x'], {'a'})
        self.assertEqual(result['top1'], 0)
        self.assertEqual(result['mrr10'], .5)
        self.assertEqual(result['candidate_recall50'], 1)

    def test_missing_qrels_fails(self):
        with self.assertRaises(AssertionError):
            metrics(['a'], ['a'], set())

    def test_exact_search_stable_ties(self):
        corpus = np.asarray([[1., 0.], [1., 0.], [0., 1.]])
        self.assertEqual(np.argsort(-(corpus @ np.array([1., 0.])), kind='stable').tolist(), [0, 1, 2])

if __name__ == '__main__':
    unittest.main()
