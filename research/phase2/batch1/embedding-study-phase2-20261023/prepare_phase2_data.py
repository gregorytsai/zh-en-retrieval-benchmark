#!/usr/bin/env python3
"""Prepare verified first-batch data without model inference or network access."""
import csv
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    data = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists():
        if path.read_text() != data:
            raise RuntimeError(f'Refusing to overwrite changed frozen dataset: {path}')
        return
    path.write_text(data)


def query_sort(row):
    return hashlib.sha256(('20261022:' + str(row['id'])).encode()).hexdigest(), str(row['id'])


def validate(data, expected_docs, expected_queries):
    docs = {x['id'] for x in data['corpus']}
    queries = {x['id'] for x in data['queries']}
    assert len(docs) == len(data['corpus']) == expected_docs
    assert len(queries) == len(data['queries']) == expected_queries
    assert set(data['qrels']) == queries
    assert all(isinstance(x['text'], str) and x['text'].strip() for x in data['corpus'] + data['queries'])
    for judgments in data['qrels'].values():
        assert set(judgments) <= docs
        assert all(isinstance(v, int) and 0 <= v <= 4 for v in judgments.values())
        assert any(v > 0 for v in judgments.values())
    ordered = sorted(data['queries'], key=lambda x: (len(x['text']), x['id']))
    data['pilot_query_ids'] = [ordered[i]['id'] for i in (0, len(ordered)//2, len(ordered)-1)]


def main():
    raw = WORK / 'qwen-reranker-rebuilt/real-dataset'
    manifest = json.loads((raw/'manifest.json').read_text())
    previous = json.loads((WORK/'embedding-study-20261022/run-20261022-v3-guarded/query-selection-final.json').read_text())
    datasets, evidence = {}, []
    for config in ('eng-zho', 'zho-zho', 'eng-eng'):
        entry = manifest['configs'][config]
        loaded = {}
        for kind, spec in entry['files'].items():
            path = raw/spec['local_file']
            assert path.stat().st_size == spec['bytes'] and sha(path) == spec['sha256']
            evidence.append({'path': str(path), 'sha256': spec['sha256'], 'bytes': spec['bytes']})
            loaded[kind] = pq.read_table(path).to_pylist()
        corpus = sorted([{'id': str(x['id']), 'text': x['text']} for x in loaded['corpus']], key=lambda x: x['id'])
        qrels = defaultdict(dict)
        for x in loaded['qrels']:
            qid, did, grade = str(x['query-id']), str(x['corpus-id']), int(x['score'])
            assert grade in (0, 1) and did not in qrels[qid]
            qrels[qid][did] = grade
        old = set(previous['configs'][config])
        assert len(old) == 100
        queries = []
        for x in loaded['queries']:
            qid = str(x['id'])
            positives = sorted(d for d, g in qrels[qid].items() if g > 0)
            assert len(positives) == 1, 'Cluster identity requires exactly one source passage here'
            queries.append({'id': qid, 'text': x['text'], 'metadata': {
                'cluster_id': config + ':' + positives[0],
                'cluster_unit': 'source_passage_not_article',
                'previous100': qid in old}})
        assert old <= {q['id'] for q in queries}
        data = {'corpus': corpus, 'queries': sorted(queries, key=query_sort), 'qrels': dict(qrels),
                'grade_mode': 'binary', 'metadata': {'source': manifest['dataset'],
                'revision': manifest['source_revision'], 'license': manifest['license'],
                'split': 'validation', 'query_language': entry['query_language'],
                'corpus_language': entry['corpus_language'], 'previous100': 100,
                'cluster_limitation': 'Passage-level only: source article grouping unavailable in this adapter.'}}
        counts = entry['full_validation_counts']
        validate(data, counts['corpus'], counts['queries'])
        datasets[config] = data
    archive = ROOT/'sources/acord/original.zip'
    expected = 'c9e851456a14aed161033b856e9f40f3cc9f46cf7282c2007b143420e93a65c3'
    assert sha(archive) == expected
    evidence.append({'path': str(archive), 'sha256': expected, 'bytes': archive.stat().st_size})
    with zipfile.ZipFile(archive) as z:
        def member(suffix):
            matches = [n for n in z.namelist() if n.endswith(suffix)]
            assert len(matches) == 1
            return z.read(matches[0]).decode('utf-8')
        corpus = sorted([{'id': str(x['_id']), 'text': x['text']} for x in map(json.loads, member('/corpus.jsonl').splitlines())], key=lambda x: x['id'])
        all_queries = [json.loads(x) for x in member('/queries.jsonl').splitlines()]
        qrels = defaultdict(dict)
        for x in csv.DictReader(io.StringIO(member('/qrels/test.tsv')), delimiter='\t'):
            qid, did, grade = x['query-id'], x['corpus-id'], int(x['score'])
            assert did not in qrels[qid]
            qrels[qid][did] = grade
        assert sum(map(len, qrels.values())) == 61988
        queries = []
        for x in all_queries:
            if str(x['_id']) in qrels:
                meta = dict(x.get('metadata', {}))
                assert meta['split'] == 'test'
                meta.update(cluster_id=meta.get('parent_query_id') or str(x['_id']),
                            cluster_unit='parent_query', previous100=False)
                queries.append({'id': str(x['_id']), 'text': x['text'], 'metadata': meta})
        data = {'corpus': corpus, 'queries': sorted(queries, key=query_sort), 'qrels': dict(qrels),
                'grade_mode': 'graded', 'metadata': {'source': 'theatticusproject/acord',
                'revision': '7ba1c6f2e0baaa32e78c78c2b02e9c57b102af4e',
                'license': 'CC-BY-4.0', 'split': 'test', 'query_language': 'eng',
                'corpus_language': 'eng', 'ndcg_gain': '2^grade-1',
                'binary_metrics_relevant_threshold': 1, 'stars': 'grade+1'}}
        validate(data, 3931, 57)
        datasets['acord-eng-eng'] = data
    payload = {'schema_version': 1, 'seed': '20261022', 'datasets': datasets}
    save(ROOT/'datasets.json', payload)
    stats = {k: {'corpus': len(v['corpus']), 'queries': len(v['queries']),
                 'qrels': sum(map(len, v['qrels'].values())), 'grade_mode': v['grade_mode'],
                 'max_document_chars': max(len(d['text']) for d in v['corpus']),
                 'pilot_query_ids': v['pilot_query_ids']} for k, v in datasets.items()}
    report = {'datasets_sha256': sha(ROOT/'datasets.json'), 'source_files': evidence,
              'datasets': stats, 'expected_summaries': 24,
              'expected_rows': sum(v['queries'] for v in stats.values())*6}
    assert report['expected_rows'] == 13278
    save(ROOT/'data-validation.json', report)
    print(json.dumps({k: v for k,v in report.items() if k!='source_files'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
