"""Verify the published Phase 2 release entirely offline.

Run: python scripts/verify_phase2.py
Requires NumPy (the same PCG64 bootstrap generator as the original run).
Reads released ZIPs in memory; does not extract, download, run inference, or
modify results. MIRACL passage text is intentionally absent: its original full
text SHA-256 is provenance, not independently verified by this public check.
"""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import math
import statistics
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = json.loads((ROOT / 'data/phase2/analysis.json').read_text(encoding='utf-8'))

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def metric_values(order, candidates, judgments, graded):
    relevant = {d for d, g in judgments.items() if g > 0}
    positives = [i + 1 for i, d in enumerate(order[:10]) if d in relevant]
    gain = lambda g: 2 ** g - 1 if graded else int(g > 0)
    discounts = [math.log2(i + 2) for i in range(10)]
    dcg = sum(gain(judgments.get(d, 0)) / discounts[i] for i, d in enumerate(order[:10]))
    ideal = sum(gain(g) / discounts[i] for i, g in enumerate(sorted(judgments.values(), reverse=True)[:10]))
    out = dict(top1=float(order[0] in relevant), mrr10=1 / positives[0] if positives else 0,
               ndcg10=dcg / ideal, recall10=len(set(order[:10]) & relevant) / len(relevant),
               recall50=len(set(order[:50]) & relevant) / len(relevant), hit10=float(bool(positives)),
               candidate_recall50=len(set(candidates[:50]) & relevant) / len(relevant),
               p5=len(set(order[:5]) & relevant) / 5, relevant_count=len(relevant), judged_count=len(judgments))
    for key, ids in [('judged_coverage10', order[:10]), ('judged_coverage50', order[:50]),
                     ('candidate_judged_coverage50', candidates[:50])]:
        out[key] = len(set(ids) & set(judgments)) / len(ids)
    if graded:
        for g in range(1, 5):
            out[f'p5_grade{g}'] = sum(judgments.get(d, 0) >= g for d in order[:5]) / 5
    return out

def audit(batch):
    official = zipfile.Path(ROOT / 'downloads' / f'phase2-{batch}-results.zip')
    materials = zipfile.Path(ROOT / 'downloads/phase2-materials.zip')
    rows = [json.loads(l) for l in (official / 'query-results.jsonl').read_text(encoding='utf-8').splitlines()]
    summaries = read(official / 'summary.json')
    selection = read(official / 'query-selection.json')
    qrels = read(official / 'qrels.json')
    manifest = read(official / 'run-manifest.json')
    data_path = materials / ('batch1/datasets.json' if batch == 'batch1' else 'batch2/selection-without-passage-text.json')
    data = read(data_path)['datasets']
    failures = Counter()
    max_diffs = defaultdict(float)
    def same(category, a, b, tol=1e-10):
        if isinstance(a, (float, int)) and isinstance(b, (float, int)):
            diff = abs(a - b)
            max_diffs[category] = max(max_diffs[category], diff)
            if diff > tol: failures[category] += 1
        elif a != b:
            failures[category] += 1
    keys = [(r['dataset'], r['embedding'], r['reranker'], r['query_id']) for r in rows]
    failures['duplicate_keys'] = len(keys) - len(set(keys))
    baselines = {(r['dataset'], r['embedding'], r['query_id']): r for r in rows if r['reranker'] == 'standalone'}
    corpus_ids = {d: {x['id'] for x in v['corpus']} for d, v in data.items()}
    grouped = defaultdict(list)
    metrics_by_key = {}
    for r, key in zip(rows, keys):
        d, e, rr, q = key
        same('status', r['status'], 'ok')
        same('query_selected', q in selection['datasets'][d]['query_ids'], True)
        same('grade_mode', r['grade_mode'], selection['datasets'][d]['grade_mode'])
        ids = r['final_ids']
        same('duplicate_ranked_documents', len(ids), len(set(ids)))
        same('ranked_document_in_corpus', set(ids) <= corpus_ids[d], True)
        same('ranked_length', len(ids), 100 if rr == 'standalone' else 50)
        base = baselines[(d, e, q)]
        candidates = base['final_ids'][:50]
        if rr != 'standalone':
            same('fixed_candidate_set', set(ids), set(candidates))
            for k in ('embedding_ms', 'retrieval_ms', 'cluster_id', 'cluster_unit', 'previous100'):
                same('baseline_reuse_' + k, r[k], base[k])
            same('rerank_score_count', len(r['scores']), len(ids))
            same('rerank_score_order', r['scores'], sorted(r['scores'], reverse=True))
        same('latency_sum', r['total_ms'], r['embedding_ms'] + r['retrieval_ms'] + (r['rerank_ms'] or 0))
        computed = metric_values(ids, candidates, qrels['datasets'][d]['qrels'][q], r['grade_mode'] == 'graded')
        for k, v in computed.items(): same('row_metrics', r[k], v)
        metrics_by_key[key] = computed
        grouped[(d, e, rr)].append(r)
    rebuilt = []
    for s in summaries:
        g = (s['dataset'], s['embedding'], s['reranker'])
        part = grouped[g]
        same('group_query_count', len(part), s['queries'])
        same('group_query_ids', {r['query_id'] for r in part}, set(selection['datasets'][g[0]]['query_ids']))
        same('summary_error_count', sum(r['status'] != 'ok' for r in part), s['errors'])
        out = dict(dataset=g[0], embedding=g[1], reranker=g[2], queries=len(part), errors=sum(r['status'] != 'ok' for r in part))
        for k in s:
            if k in out: continue
            if k.endswith(('_p50', '_p95')):
                field, percentile = k.rsplit('_', 1)
                values = [r[field] for r in part if r[field] is not None]
                value = float(np.percentile(values, int(percentile[1:]))) if values else None
            else:
                value = statistics.mean(metrics_by_key[(r['dataset'], r['embedding'], r['reranker'], r['query_id'])][k] for r in part)
            same('summary_values', s[k], value)
            out[k] = value
        rebuilt.append(out)
    same('summary_group_count', len(summaries), len(grouped))
    for s in rebuilt:
        matches = [a for a in ANALYSIS['rows'] if all(a[k] == s[k] for k in ('dataset', 'embedding', 'reranker'))]
        same('published_analysis_group_count', len(matches), 1)
        if len(matches) != 1:
            continue
        for k, value in s.items():
            same('published_analysis_summary', matches[0].get(k), value)
        part = grouped[(s['dataset'], s['embedding'], s['reranker'])]
        same('published_analysis_mean_latency', matches[0]['total_ms_mean'], statistics.mean(r['total_ms'] for r in part))
    for field, file in [('selection_sha256', 'query-selection.json'), ('selected_qrels_sha256', 'qrels.json')]:
        canonical = json.dumps(read(official / file), ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
        same('manifest_' + field, manifest[field], hashlib.sha256(canonical).hexdigest())
    if batch == 'batch1':
        same('dataset_hash', manifest['dataset_sha256'], digest(data_path))
    same('dataset_hash_provenance', manifest['dataset_sha256'], ANALYSIS['batches'][batch]['dataset_sha256'])
    for name, expected in ANALYSIS['batches'][batch]['source_files_sha256'].items():
        same('published_file_hash', digest(official / name), expected)
    same('matrix_query_rows', manifest['matrix_full_query_rows'], len(rows))
    same('matrix_summary_count', manifest['matrix_summary_count'], len(summaries))
    # Reproduce the documented seed and cluster-resampling sequence, using independently
    # recalculated metrics and cluster sums/counts (not the runner's metric/bootstrap functions).
    rng = np.random.default_rng(20261023)
    rebuilt_pairs = []
    for pair in read(official / 'paired-confidence.json'):
        d = pair['dataset']
        left_e, left_rr = pair['left'].split('+')
        right_e, right_rr = pair['right'].split('+')
        left = {r['query_id']: r for r in grouped[(d, left_e, left_rr)]}
        right = {r['query_id']: r for r in grouped[(d, right_e, right_rr)]}
        same('paired_query_sets', set(left), set(right))
        clusters = defaultdict(list)
        deltas = []
        for q in sorted(left):
            delta = metrics_by_key[(d, right_e, right_rr, q)][pair['metric']] - metrics_by_key[(d, left_e, left_rr, q)][pair['metric']]
            deltas.append(delta)
            clusters[left[q].get('cluster_id') or q].append(delta)
        cluster_sums = np.array([sum(clusters[c]) for c in sorted(clusters)])
        cluster_sizes = np.array([len(clusters[c]) for c in sorted(clusters)])
        sampled = rng.integers(0, len(clusters), size=(pair['bootstrap_replicates'], len(clusters)))
        estimates = cluster_sums[sampled].sum(axis=1) / cluster_sizes[sampled].sum(axis=1)
        ci = np.quantile(estimates, [.025, .975]).tolist()
        delta = statistics.mean(deltas)
        same('paired_delta', pair['paired_delta_right_minus_left'], delta)
        same('paired_cluster_count', pair['cluster_count'], len(clusters))
        same('paired_query_count', pair['query_count'], len(left))
        for a, b in zip(ci, pair['bootstrap95']): same('paired_bootstrap95', a, b)
        rebuilt_pairs.append(dict(pair, independently_computed_delta=delta, independently_computed_bootstrap95=ci))
    datasets = {}
    for d, sel in selection['datasets'].items():
        meta = {q['id']: q for q in sel['queries']}
        for q in data[d]['queries']:
            for field in ('cluster_id', 'cluster_unit', 'previous100'):
                same('material_query_metadata', q.get(field, q.get('metadata', {}).get(field)), meta[q['id']].get(field))
        published = ANALYSIS['datasets'][d]
        same('published_dataset_queries', published['queries'], len(sel['query_ids']))
        same('published_dataset_corpus', published['corpus'], len(corpus_ids[d]))
        same('published_dataset_grade_mode', published['grade_mode'], sel['grade_mode'])
        same('published_dataset_batch', published['batch'], batch)
        same('published_dataset_metadata', published['metadata'], sel['dataset_metadata'])
        if batch == 'batch2':
            same('miracl_passage_text_absent', all('text' not in c for c in data[d]['corpus']), True)
        same('dataset_query_selection', set(meta), {q['id'] for q in data[d]['queries']})
        same('qrels_from_dataset', qrels['datasets'][d]['qrels'], {q: data[d]['qrels'][q] for q in meta})
        same('corpus_unique_ids', len(corpus_ids[d]), len(data[d]['corpus']))
        same('qrels_docs_in_corpus', all(set(q) <= corpus_ids[d] for q in qrels['datasets'][d]['qrels'].values()), True)
        for r in rows:
            if r['dataset'] != d: continue
            for field in ('cluster_id', 'cluster_unit', 'previous100'):
                same('selection_metadata_' + field, r.get(field), meta[r['query_id']].get(field))
        datasets[d] = dict(queries=len(sel['query_ids']), corpus_documents=len(corpus_ids[d]),
                           clusters=len({q['cluster_id'] for q in sel['queries']}),
                           previous100_count=sum(q.get('previous100', False) for q in sel['queries']),
                           judged_grades=sorted({g for q in qrels['datasets'][d]['qrels'].values() for g in q.values()}))
    return dict(status='passed' if not any(failures.values()) else 'failed', rows=len(rows), groups=len(grouped),
                unique_keys=len(set(keys)), dataset_qualified_queries=sum(x['queries'] for x in datasets.values()),
                datasets=datasets, failures={k: v for k, v in failures.items() if v}, maximum_numeric_differences=dict(max_diffs),
                independently_recomputed_summary=rebuilt, independently_recomputed_pairs=rebuilt_pairs,
                input_hashes={f: digest(official / f) for f in ('query-results.jsonl', 'qrels.json', 'summary.json', 'paired-confidence.json')})

if __name__ == '__main__':
    result = {batch: audit(batch) for batch in ('batch1', 'batch2')}
    all_summary = result['batch1']['independently_recomputed_summary'] + result['batch2']['independently_recomputed_summary']
    macro = {}
    for name, dataset_ids in [('six_datasets_equal_weight', {s['dataset'] for s in all_summary}),
                              ('mlqa_three_directions_equal_weight', {'eng-eng', 'eng-zho', 'zho-zho'})]:
        by_config = defaultdict(list)
        for s in all_summary:
            if s['dataset'] in dataset_ids:
                by_config[(s['embedding'], s['reranker'])].append(s)
        macro[name] = [dict(embedding=e, reranker=rr, datasets=len(part),
                            top1=statistics.mean(s['top1'] for s in part),
                            ndcg10=statistics.mean(s['ndcg10'] for s in part))
                      for (e, rr), part in sorted(by_config.items()) if len(part) == len(dataset_ids)]
    macro['six_dataset_F_minus_E'] = [dict(dataset=s['dataset'],
        top1=s['top1'] - next(e['top1'] for e in all_summary if e['dataset'] == s['dataset'] and e['reranker'] == 'qwen-reranker-0.6b'),
        ndcg10=s['ndcg10'] - next(e['ndcg10'] for e in all_summary if e['dataset'] == s['dataset'] and e['reranker'] == 'qwen-reranker-0.6b'))
        for s in all_summary if s['reranker'] == 'qwen-reranker-4b']
    macro['interpretation'] = 'Equal-weight descriptive macro averages, not pooled leaderboard scores. Six-dataset comparison is restricted to the three configurations tested on all six datasets. Per-dataset confidence intervals cannot be applied to these macro averages.'
    result['macro_averages'] = macro
    result['scope'] = 'Offline release verification; no inference, no performance remeasurement, no network access. MIRACL corpus IDs verified; withheld passage text and original full-text hash cannot be reverified.'
    result['interpretation_limits'] = [
        'Show dataset rankings alongside explicitly equal-weight macro summaries. Corpus, question set, relevance grades and number of configurations differ; no all-dataset pooled winner.',
        'Old v3 had 100 questions per direction and 22 configurations; phase 2 has expanded questions and six configurations (three for MIRACL). Do not interpret score changes as model upgrades.',
        'Only Qwen 4B embeddings were paired with rerankers in phase 2. Do not extrapolate results to untested small-embedder pairs.',
        'ACORD has incomplete judgments, only 57 queries, low judged coverage and low candidate recall; unjudged does not mean irrelevant.',
        'MIRACL is a union of dev judged passages, not the official full corpus benchmark; its absolute scores cannot be compared to full-corpus leaderboards.',
        'Bootstrap intervals describe query/cluster sampling uncertainty in these selected corpora; MLQA clusters are passages rather than articles, and multiple comparisons are unadjusted.',
        'Latency is the sum of separately recorded query embedding, retrieval and reranking times, not measured production end-to-end time or hardware sizing.'
    ]
    expected = read(ROOT / 'data/phase2/independent-audit.json')
    snapshot_failures = []
    for batch in ('batch1', 'batch2'):
        for field in ('rows', 'groups', 'unique_keys', 'dataset_qualified_queries', 'datasets', 'input_hashes', 'independently_recomputed_summary', 'independently_recomputed_pairs'):
            if result[batch][field] != expected[batch][field]:
                snapshot_failures.append(batch + ':' + field)
    if result['macro_averages'] != expected['macro_averages']:
        snapshot_failures.append('macro_averages')
    public_group_keys = [(r['dataset'], r['embedding'], r['reranker']) for r in ANALYSIS['rows']]
    computed_group_keys = [(r['dataset'], r['embedding'], r['reranker']) for r in all_summary]
    if sorted(public_group_keys) != sorted(computed_group_keys):
        snapshot_failures.append('analysis_group_keys')
    ok = all(result[b]['status'] == 'passed' for b in ('batch1', 'batch2')) and not snapshot_failures
    report = {b: {f: result[b][f] for f in ('status', 'rows', 'groups', 'unique_keys', 'dataset_qualified_queries', 'failures')} for b in ('batch1', 'batch2')}
    report.update(status='passed' if ok else 'failed', snapshot_failures=snapshot_failures,
                  macro_averages=result['macro_averages'], scope=result['scope'])
    print(json.dumps(report, indent=2))
    sys.exit(0 if ok else 1)
