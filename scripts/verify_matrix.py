"""Verify the published 28-configuration matrix offline (Python 3.10+, stdlib only).

Usage: python scripts/verify_matrix.py [--root /path/to/site]
Reads downloads/matrix-{B01..B07}-results.zip, matrix-shared.zip,
and data/matrix/{analysis,original-ranking}.json.
No network, inference, extraction, or private source files are used.
Returns exit code 0 on success, 1 on any discrepancy or malformed/missing input.
This validates measured records, not model weight bytes or corpus full text.
"""
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict, Counter
import argparse, hashlib, json, math, statistics, sys

BATCH_EMBEDDERS = ['qwen-4b', 'qwen-8b', 'embeddinggemma-2', 'qwen-0.6b',
                   'bge-m3', 'f2llm-v2-0.6b', 'granite-97m-r2']
RERANKERS = {'standalone', 'bge-reranker', 'qwen-reranker-0.6b', 'qwen-reranker-4b'}

def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()

def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

def metric_values(order, candidates, judgments, graded):
    relevant = {d for d, g in judgments.items() if g > 0}
    positives = [i + 1 for i, d in enumerate(order[:10]) if d in relevant]
    gain = lambda g: 2 ** g - 1 if graded else int(g > 0)
    discounts = [math.log2(i + 2) for i in range(10)]
    dcg = sum((gain(judgments.get(d, 0)) / discounts[i] for i, d in enumerate(order[:10])))
    ideal = sum((gain(g) / discounts[i] for i, g in enumerate(sorted(judgments.values(), reverse=True)[:10])))
    out = dict(top1=float(order[0] in relevant), mrr10=1 / positives[0] if positives else 0, ndcg10=dcg / ideal, recall10=len(set(order[:10]) & relevant) / len(relevant), recall50=len(set(order[:50]) & relevant) / len(relevant), hit10=float(bool(positives)), candidate_recall50=len(set(candidates[:50]) & relevant) / len(relevant), p5=len(set(order[:5]) & relevant) / 5, relevant_count=len(relevant), judged_count=len(judgments))
    for key, ids in [('judged_coverage10', order[:10]), ('judged_coverage50', order[:50]), ('candidate_judged_coverage50', candidates[:50])]:
        out[key] = len(set(ids) & set(judgments)) / len(ids)
    if graded:
        for g in range(1, 5):
            out[f'p5_grade{g}'] = sum((judgments.get(d, 0) >= g for d in order[:5])) / 5
    return out

def verify(root):
    failures = Counter()
    maximum_differences = defaultdict(float)
    checks = Counter()
    def check(label, actual, expected, tol=1e-10):
        checks[label] += 1
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            delta = abs(actual - expected)
            maximum_differences[label] = max(maximum_differences[label], delta)
            if not math.isfinite(delta) or delta > tol:
                failures[label] += 1
        elif actual != expected:
            failures[label] += 1
    downloads = root / 'downloads'
    analysis = json.loads((root / 'data/matrix/analysis.json').read_text(encoding='utf-8'))
    ranking = json.loads((root / 'data/matrix/original-ranking.json').read_text(encoding='utf-8'))
    with ZipFile(downloads / 'matrix-shared.zip') as archive:
        qrels = json.loads(archive.read('qrels.json'))
        selection = json.loads(archive.read('query-selection.json'))
        for name in ('qrels.json', 'query-selection.json'):
            check('website_shared_raw_sha256', hashlib.sha256(archive.read(name)).hexdigest(),
                  analysis['shared_sha256'][name])
    check('shared_dataset_digest', qrels['dataset_sha256'], selection['dataset_sha256'])
    check('shared_datasets', set(qrels['datasets']), set(selection['datasets']))
    check('six_datasets', len(selection['datasets']), 6)
    metadata = {}
    for dataset, selected in selection['datasets'].items():
        ids = selected['query_ids']
        metadata[dataset] = {q['id']: q for q in selected['queries']}
        check('selection_unique', len(set(ids)), len(ids))
        check('selection_metadata_ids', set(metadata[dataset]), set(ids))
        check('qrels_query_ids', set(qrels['datasets'][dataset]['qrels']), set(ids))
        check('grade_mode', selected['grade_mode'], qrels['datasets'][dataset]['grade_mode'])
    allrows, summaries, manifests = [], [], []
    batch_counts = {}
    for index, embedder in enumerate(BATCH_EMBEDDERS, 1):
        batch = f'B{index:02}'
        with ZipFile(downloads / f'matrix-{batch}-results.zip') as archive:
            rows = [json.loads(line) for line in archive.read('query-results.jsonl').splitlines() if line.strip()]
            summary = json.loads(archive.read('summary.json'))
            audit = json.loads(archive.read('batch-completion-audit.json'))
            for name in ('query-results.jsonl', 'summary.json'):
                check('website_batch_raw_sha256', hashlib.sha256(archive.read(name)).hexdigest(),
                      analysis['batches'][batch]['source_files_sha256'][name])
            check('website_batch_rows', analysis['batches'][batch]['rows'], len(rows))
            current_manifests = [json.loads(archive.read(name)) for name in archive.namelist()
                                 if name.endswith('run-manifest.json')]
        batch_counts[batch] = len(rows)
        check('batch_rows', len(rows), 10452)
        check('batch_summaries', len(summary), 24)
        check('batch_embedder', {r['embedding'] for r in rows}, {embedder})
        check('batch_rerankers', {r['reranker'] for r in rows}, RERANKERS)
        check('completion_status', audit['status'], 'passed')
        check('completion_rows', audit['rows'], len(rows))
        check('completion_summaries', audit['summaries'], len(summary))
        check('completion_errors', audit['errors'], 0)
        check('manifest_count', len(current_manifests), 2 if batch == 'B02' else 1)
        covered = []
        for manifest in current_manifests:
            ids = manifest['dataset_ids']
            covered.extend(ids)
            check('manifest_dataset_digest', manifest['dataset_sha256'], selection['dataset_sha256'])
            qr = {**qrels, 'datasets': {d: qrels['datasets'][d] for d in ids}}
            sel = {**selection, 'datasets': {d: selection['datasets'][d] for d in ids}}
            check('qrels_manifest_hash', canonical(qr), manifest['selected_qrels_sha256'])
            check('selection_manifest_hash', canonical(sel), manifest['selection_sha256'])
            check('manifest_configs', set(manifest['effective_configs']), {embedder + '__' + rr for rr in RERANKERS})
            if manifests:
                for field in ('model_sha256', 'binary_sha256'):
                    check('common_' + field, manifest[field], manifests[0][field])
                check('common_calculation_core', manifest['code_sha256']['run_study.py'],
                      manifests[0]['code_sha256']['run_study.py'])
            manifests.append(manifest)
        check('manifest_dataset_coverage', sorted(covered), sorted(selection['datasets']))
        allrows.extend(rows)
        summaries.extend(summary)
    rowkey = lambda r: (r['dataset'], r['embedding'], r['reranker'], r['query_id'])
    check('rows', len(allrows), 73164)
    check('unique_row_keys', len({rowkey(r) for r in allrows}), len(allrows))
    baselines = {(r['dataset'], r['embedding'], r['query_id']): r
                 for r in allrows if r['reranker'] == 'standalone'}
    groups = defaultdict(list)
    metric_count = 0
    for row in allrows:
        dataset, embedder, reranker, query = rowkey(row)
        base = baselines[(dataset, embedder, query)]
        candidates, order = base['final_ids'], row['final_ids']
        check('status', row['status'], 'ok')
        check('top50_length', len(order), 50)
        check('top50_unique', len(set(order)), 50)
        check('fixed_candidate_set', set(order), set(candidates))
        check('row_grade_mode', row['grade_mode'], selection['datasets'][dataset]['grade_mode'])
        for field in ('cluster_id', 'cluster_unit', 'previous100'):
            check('selection_' + field, row.get(field), metadata[dataset][query].get(field))
        for field in ('embedding_ms', 'retrieval_ms'):
            check('shared_' + field, row[field], base[field])
        for field in ('embedding_ms', 'retrieval_ms', 'total_ms'):
            check('finite_nonnegative_time', math.isfinite(row[field]) and row[field] >= 0, True)
        check('time_components_sum', row['total_ms'],
              row['embedding_ms'] + row['retrieval_ms'] + (row['rerank_ms'] or 0))
        if reranker == 'standalone':
            check('standalone_rerank_time', row['rerank_ms'], None)
        else:
            check('rerank_time', math.isfinite(row['rerank_ms']) and row['rerank_ms'] >= 0, True)
            check('candidate_order_hash', row['candidate_ids_sha256'], canonical(candidates))
            check('score_count', len(row['scores']), 50)
            check('score_finite', all(math.isfinite(s) for s in row['scores']), True)
            check('score_order', row['scores'], sorted(row['scores'], reverse=True))
        computed = metric_values(order, candidates, qrels['datasets'][dataset]['qrels'][query],
                                 row['grade_mode'] == 'graded')
        for metric, value in computed.items():
            check('quality_' + metric, row[metric], value)
            metric_count += 1
        groups[(dataset, embedder, reranker)].append(row)
    check('group_count', len(groups), 168)
    check('summary_count', len(summaries), 168)
    check('unique_summary_keys', len({(s['dataset'], s['embedding'], s['reranker']) for s in summaries}), 168)
    for summary in summaries:
        key = (summary['dataset'], summary['embedding'], summary['reranker'])
        rows = groups[key]
        check('complete_query_set', {r['query_id'] for r in rows}, set(selection['datasets'][key[0]]['query_ids']))
        check('summary_queries', summary['queries'], len(rows))
        check('summary_errors', summary['errors'], 0)
        for field, actual in summary.items():
            if field in {'dataset', 'embedding', 'reranker', 'queries', 'errors'}:
                continue
            if field.endswith(('_p50', '_p95')):
                name, pct = field.rsplit('_', 1)
                expected = percentile([r[name] for r in rows if r[name] is not None], int(pct[1:]))
            else:
                expected = statistics.mean(r[field] for r in rows)
            check('summary_' + field, actual, expected)
    # Validate the exact inputs consumed by the public website, not only the ZIPs.
    website_rows = analysis['rows']
    website_map = {(r['dataset'], r['embedding'], r['reranker']): r for r in website_rows}
    check('website_summary_count', len(website_rows), 168)
    check('website_unique_summary_keys', len(website_map), 168)
    check('website_summary_keys', set(website_map), set(groups))
    check('website_batches', set(analysis['batches']), set(batch_counts))
    check('website_models', set(analysis['models']), set(manifests[0]['model_sha256']))
    for model, meta in analysis['models'].items():
        check('website_model_sha256', meta['sha256'], manifests[0]['model_sha256'][model])
        check('website_model_expected_sha256', meta['expected_sha256'], meta['sha256'])
        check('website_model_expected_bytes', meta['expected_bytes'], meta['bytes'])
        check('website_model_weight_gib', meta['weight_gib'], meta['bytes'] / 2**30)
    for dataset, selected in selection['datasets'].items():
        check('website_dataset_queries', analysis['datasets'][dataset]['queries'], len(selected['query_ids']))
        check('website_dataset_grade_mode', analysis['datasets'][dataset]['grade_mode'], selected['grade_mode'])
        check('website_dataset_metadata', analysis['datasets'][dataset]['metadata'], selected['dataset_metadata'])
    for summary in summaries:
        key = (summary['dataset'], summary['embedding'], summary['reranker'])
        published = website_map[key]
        check('website_summary_fields', set(published), set(summary) | {'total_ms_mean', 'weight_gib_sum'})
        for field, expected in summary.items():
            check('website_summary_' + field, published[field], expected)
        check('website_total_ms_mean', published['total_ms_mean'], statistics.mean(r['total_ms'] for r in groups[key]))
        weight = analysis['models'][key[1]]['weight_gib']
        if key[2] != 'standalone':
            weight += analysis['models'][key[2]]['weight_gib']
        check('website_weight_gib_sum', published['weight_gib_sum'], weight)
    configs = {(e, rr) for d, e, rr in groups}
    published_ranking = ranking['configurations']
    check('website_ranking_count', len(published_ranking), 28)
    check('website_ranking_configs', {(r['embedding'], r['reranker']) for r in published_ranking}, configs)
    macro_ndcg = {}
    for embedder, reranker in configs:
        parts = [rows for (d, e, rr), rows in groups.items() if (e, rr) == (embedder, reranker)]
        check('macro_six_datasets', len(parts), 6)
        macro_ndcg[(embedder, reranker)] = statistics.mean(statistics.mean(r['ndcg10'] for r in rows) for rows in parts)
    for published in published_ranking:
        key = (published['embedding'], published['reranker'])
        expected_coverage = {d: len(rows) for (d, e, rr), rows in groups.items() if (e, rr) == key}
        check('website_macro_ndcg10', published['macro_ndcg10_equal_weight'], macro_ndcg[key])
        check('website_ranking_coverage', published['coverage'], expected_coverage)
        check('website_ranking_eligible', published['eligible'], True)
        check('website_ranking_expected', published['dataset_count_expected'], 6)
        check('website_ranking_present', published['dataset_count_present'], 6)
        check('website_ranking_reasons', published['reasons'], [])
    check('website_ranking_order', [r['macro_ndcg10_equal_weight'] for r in published_ranking],
          sorted([r['macro_ndcg10_equal_weight'] for r in published_ranking], reverse=True))
    check('website_reranker_timing_keys', set(analysis['reranker_timing']), RERANKERS - {'standalone'})
    for reranker, published in analysis['reranker_timing'].items():
        times = [r['rerank_ms'] for r in allrows if r['reranker'] == reranker]
        check('website_reranker_timing_rows', published['rows'], len(times))
        for pct in (50, 95):
            check('website_reranker_timing_p' + str(pct), published[f'p{pct}_ms'], percentile(times, pct))
        check('website_reranker_mean_macro_ndcg10', published['mean_macro_ndcg10'],
              statistics.mean(v for (e, rr), v in macro_ndcg.items() if rr == reranker))
    check('website_embedder_timing_keys', set(analysis['embedder_timing']), set(BATCH_EMBEDDERS))
    for embedder, published in analysis['embedder_timing'].items():
        times = [r['embedding_ms'] for r in baselines.values() if r['embedding'] == embedder]
        check('website_embedder_timing_rows', published['rows'], len(times))
        for pct in (50, 95):
            check('website_embedder_timing_p' + str(pct), published[f'p{pct}_ms'], percentile(times, pct))
    return {'status': 'failed' if failures else 'passed', 'failures': dict(failures),
            'rows': len(allrows), 'groups': len(groups), 'configurations': len(groups) // 6,
            'metric_values_recomputed': metric_count, 'batch_counts': batch_counts,
            'checks': dict(checks), 'maximum_differences': dict(maximum_differences),
            'scope': 'Offline published-record validation; no inference or network.',
            'limits': ['No model weight bytes or corpus full text are required or verified.',
                       'Manifest hashes verify recorded selection/qrels and cross-run consistency; they are not signatures.',
                       'Bootstrap confidence intervals and hardware repeatability are outside this verifier.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = verify(args.root)
    except Exception as exc:
        result = {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
