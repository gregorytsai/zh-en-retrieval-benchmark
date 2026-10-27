#!/usr/bin/env python3
"""Prepare second-batch retrieval inputs; no model execution or remote code."""
import gzip
import hashlib
import json
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from prepare_phase2_data import save, sha, validate

ROOT = Path(__file__).resolve().parent


def key(value):
    return hashlib.sha256(('phase2-20261023:' + value).encode()).hexdigest(), value


def miracl(lang):
    root = ROOT/'sources/miracl'/lang
    expected = {'zh': 'd09a33559280fe5e7b4a6ce9330ff9e261906d1ef4bf991ea8c02d309657f9fa',
                'en': 'f333c75491c8dac69f946b021511a67666318efb799dff3a6d927a841f8148eb'}
    assert sha(root/'passages.jsonl.gz') == expected[lang]
    with gzip.open(root/'passages.jsonl.gz', 'rt') as stream:
        docs = {x['docid']: x for x in map(json.loads, stream)}
    queries = dict(x.split('\t', 1) for x in (root/'topics-dev.tsv').read_text().splitlines())
    qrels = defaultdict(dict)
    for row in (root/'qrels-dev.tsv').read_text().splitlines():
        qid, _, did, relevance = row.split()
        assert did not in qrels[qid]
        qrels[qid][did] = int(relevance)
    assert len(queries) == {'zh':393, 'en':799}[lang]
    all_judged = {did for values in qrels.values() for did in values}
    missing = all_judged - docs.keys()
    recovered = []
    if missing:
        official = root/'official-docs-0.jsonl.gz'
        assert lang == 'zh'
        assert sha(official) == 'b036ba6a927a4598b314de55a1de200d2a619b28b4327f49e97ec4233d324827'
        controls_checked = []
        with gzip.open(official, 'rt') as stream:
            for row in stream:
                item = json.loads(row)
                did = item['docid']
                if did in all_judged and did in docs and len(controls_checked) < 2:
                    assert item == docs[did]
                    controls_checked.append(did)
                if did in missing:
                    docs[did] = item
                    recovered.append(did)
        assert missing <= set(recovered), sorted(missing-set(recovered))
        assert len(controls_checked) == 2, 'Require overlap check between original and author subset'
    assert all_judged <= docs.keys()
    selected = sorted(queries, key=key)[:200]
    # All judged documents from the entire dev set, not only selected positives.
    corpus = [{'id': did, 'text': (docs[did]['title']+'\n'+docs[did]['text']).strip(),
               'metadata': {'source_docid': did, 'article_id': did.split('#')[0]}}
              for did in sorted(all_judged)]
    data = {'grade_mode': 'binary', 'corpus': corpus,
            'queries': [{'id': qid, 'text': queries[qid], 'metadata': {
                'cluster_id': qid, 'cluster_unit': 'query', 'previous100': False}}
                for qid in selected],
            'qrels': {qid:qrels[qid] for qid in selected},
            'metadata': {'source': 'miracl/miracl', 'revision': '5be20db9509754dadad47689368639fcec739c00',
                'split': 'dev', 'license_topics_qrels': 'Apache-2.0',
                'corpus_mode': 'all dev judged passages union; NOT full-corpus MIRACL',
                'text_source': 'miracl/nomiracl corpus plus original corpus for missing IDs',
                'text_subset_revision': 'ecd08778d0426a5ca28ac99763b0c9ddc2c78e68',
                'original_corpus_revision': 'd921ec7e349ce0d28daf30b2da9da5ee698bef0d',
                'language': lang, 'recovered_missing_ids': sorted(recovered)}}
    validate(data, len(all_judged), 200)
    return data


def qasper():
    path = ROOT/'sources/qasper/qasper-test-and-evaluator-v0.3.tgz'
    with tarfile.open(path) as archive:
        raw = json.load(archive.extractfile('qasper-test-v0.3.json'))
    reasons = Counter()
    eligible = []
    paragraphs_by_paper = {}
    # Conservative contextual-reference filter. Remaining questions require semantic review.
    vague = re.compile(r'\b(this|these|their|they|authors?|paper)\b|\bthe (proposed |presented )?(model|method|approach|system|dataset|framework)\b', re.I)
    for paper_id, paper in raw.items():
        paragraphs = [(f'{paper_id}:{section_i}:{para_i}', text)
                      for section_i, section in enumerate(paper['full_text'])
                      for para_i, text in enumerate(section['paragraphs']) if text.strip()]
        paragraphs_by_paper[paper_id] = paragraphs
        lookup = defaultdict(list)
        for did, text in paragraphs:
            lookup[text.strip()].append(did)
        for query in paper['qas']:
            text = query['question'].strip()
            if vague.search(text):
                reasons['contextual_reference_filter'] += 1
                continue
            answers = [x['answer'] for x in query['answers']]
            if not answers or any(x.get('unanswerable') for x in answers):
                reasons['unanswerable_or_disputed'] += 1
                continue
            evidence = {e.strip() for x in answers for e in x.get('evidence', [])}
            if not evidence or any(e.startswith('FLOAT SELECTED') for e in evidence):
                reasons['missing_or_figure_evidence'] += 1
                continue
            if any(len(lookup[e]) != 1 for e in evidence):
                reasons['evidence_not_uniquely_mapped'] += 1
                continue
            eligible.append({'id': query['question_id'], 'text': text,
                             'metadata': {'cluster_id': paper_id, 'cluster_unit': 'paper',
                                          'paper_title': paper['title'], 'previous100': False},
                             'relevant': {lookup[e][0]: 1 for e in evidence}})
    selected, papers = [], set()
    for query in sorted(eligible, key=lambda q: key(q['id'])):
        paper_id = query['metadata']['cluster_id']
        if paper_id not in papers:
            selected.append(query)
            papers.add(paper_id)
        if len(selected) == 100:
            break
    assert len(selected) == 100, (len(selected), reasons)
    corpus = [{'id': did, 'text': text, 'metadata': {'paper_id': paper}}
              for paper in sorted(papers) for did, text in paragraphs_by_paper[paper]]
    data = {'grade_mode': 'binary', 'corpus': corpus,
            'queries': [{k:v for k,v in q.items() if k!='relevant'} for q in selected],
            'qrels': {q['id']:q['relevant'] for q in selected},
            'metadata': {'source': 'allenai/qasper', 'version': '0.3', 'split': 'test',
                'license': 'CC-BY-4.0', 'source_sha256': sha(path),
                'selection': 'fixed hash; one query per paper; full paragraphs from selected papers',
                'counts_raw': {'papers': len(raw), 'queries': sum(len(p['qas']) for p in raw.values())},
                'exclusions': dict(reasons), 'eligible_queries': len(eligible),
                'status': 'awaiting_semantic_query_review_and_token_audit',
                'score_scope': 'derived evidence retrieval, not official QASPER QA score'}}
    validate(data, len(corpus), 100)
    return data


def main():
    # User decision: QASPER is excluded from this round; retain its draft only as history.
    datasets = {'miracl-zh-zh': miracl('zh'), 'miracl-en-en': miracl('en')}
    save(ROOT/'datasets-batch2.json', {'schema_version': 1, 'seed': 'phase2-20261023', 'datasets': datasets})
    checks = {name: {'queries': len(d['queries']), 'corpus': len(d['corpus']),
                     'qrels': sum(map(len, d['qrels'].values())),
                     'max_doc_chars': max(len(x['text']) for x in d['corpus'])}
              for name,d in datasets.items()}
    save(ROOT/'batch2-miracl-validation.json', {'datasets': checks,
        'expected_rows': 1200, 'expected_summaries': 6,
        'status': 'draft_not_ready_for_inference',
        'excluded_by_user': ['QASPER'],
        'pending': ['length audit', 'second-batch runner and guarded pilot'],
        'datasets_sha256': sha(ROOT/'datasets-batch2.json')})
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
