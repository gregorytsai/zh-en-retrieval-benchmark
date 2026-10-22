#!/usr/bin/env python3
"""Isolated, checkpointed MLQA text retrieval benchmark. Never changes router settings."""
import argparse
import hashlib
import json
import math
import platform
import signal
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
DATA = WORK / 'qwen-reranker-rebuilt/real-dataset'
BINARY = WORK / 'llama.cpp-qwen38/build/bin/llama-server'
sys.path.insert(0, str(WORK / 'bilingual-retrieval-benchmark'))
import run_benchmark as base

INSTRUCTION = 'Instruct: Given a web search query, retrieve relevant passages\nQuery: '
EMBEDDERS = {
    'qwen-0.6b': (ROOT/'models/Qwen3-Embedding-0.6B-Q8_0.gguf', 'last', 1024, INSTRUCTION, '', True),
    'qwen-4b': (WORK/'Qwen3-Embedding-4B.Q8_0.gguf', 'last', 2560, INSTRUCTION, '', True),
    'qwen-8b': (ROOT/'models/Qwen3-Embedding-8B-Q8_0.gguf', 'last', 4096, INSTRUCTION, '', True),
    'bge-m3': (ROOT/'models/bge-m3-q8_0.gguf', 'cls', 1024, '', '', True),
    'embeddinggemma-2': (ROOT/'models/embeddinggemma-2-Q8_0.gguf', 'mean', 768,
                        'task: search result | query: ', 'title: none | text: ', True),
    'f2llm-v2-0.6b': (ROOT/'models/F2LLM-v2-0.6B.Q8_0.gguf', 'last', 1024, INSTRUCTION, '', False),
    'granite-97m-r2': (ROOT/'models/granite-embedding-97M-multilingual-r2-Q8_0.gguf', 'cls', 384, '', '', False),
}
RERANKERS = {
    'bge-reranker': WORK/'bge-reranker-v2-m3-q8_0.gguf',
    'qwen-reranker-0.6b': WORK/'qwen-reranker-rebuilt/Qwen3-Reranker-0.6B-Q8_0.gguf',
    'qwen-reranker-4b': WORK/'qwen-reranker-rebuilt/Qwen3-Reranker-4B-Q8_0.gguf',
}
FIXTURES = [
    ('太陽能電池如何發電？', 'Solar cells convert sunlight into electricity.', 'Espresso is made from ground coffee.'),
    ('水的化學式是什麼？', 'The chemical formula of water is H2O.', 'Saturn has rings of ice and rock.'),
    ('哪個行星以紅色聞名？', 'Mars is known as the red planet.', 'Bread is baked from flour and yeast.'),
    ('光合作用需要什麼能量？', 'Photosynthesis uses energy from sunlight.', 'A bicycle usually has two wheels.'),
    ('蝙蝠如何在黑暗中導航？', 'Bats use echolocation to navigate in the dark.', 'Copper is used in electrical wiring.'),
    ('What do bees produce?', 'Bees produce honey.', 'Earthquakes are caused by movements of tectonic plates.'),
    ('How many legs does a spider have?', 'A spider has eight legs.', 'The ocean contains salt water.'),
    ('冰融化後變成什麼？', '冰融化後會變成液態水。', '火車在鐵軌上行駛。'),
    ('植物的根有什麼作用？', '植物的根會吸收土壤中的水和養分。', '電腦鍵盤用來輸入文字。'),
    ('誰寫了哈姆雷特？', '《哈姆雷特》的作者是莎士比亞。', '長頸鹿是脖子很長的動物。'),
]
METRICS = ['top1', 'mrr10', 'ndcg10', 'recall10', 'recall50', 'hit10', 'candidate_recall50']

def stamp():
    return datetime.now(timezone.utc).isoformat()

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)

def append(path, value):
    with path.open('a') as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + '\n')

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

def metrics(order, candidates, relevant):
    assert relevant
    hits = [i for i, doc in enumerate(order[:10], 1) if doc in relevant]
    ideal = sum(1/math.log2(i+1) for i in range(1, min(10, len(relevant))+1))
    return {'top1': float(bool(order) and order[0] in relevant),
            'mrr10': 1/hits[0] if hits else 0,
            'ndcg10': sum(1/math.log2(i+1) for i in hits)/ideal,
            'recall10': len(set(order[:10]) & relevant)/len(relevant),
            'recall50': len(set(order[:50]) & relevant)/len(relevant),
            'hit10': float(bool(hits)),
            'candidate_recall50': len(set(candidates[:50]) & relevant)/len(relevant)}

def prepare(out):
    manifest = json.loads((DATA/'manifest.json').read_text())
    datasets = {}
    selection = {'seed': '20261022', 'revision': manifest['source_revision'], 'configs': {}}
    for config, entry in manifest['configs'].items():
        loaded = {kind: pq.read_table(DATA/item['local_file']).to_pylist() for kind, item in entry['files'].items()}
        corpus = sorted(loaded['corpus'], key=lambda r: str(r['id']))
        queries = sorted(loaded['queries'], key=lambda r: (hashlib.sha256(('20261022:'+str(r['id'])).encode()).hexdigest(), str(r['id'])))[:100]
        relevant = defaultdict(set)
        for row in loaded['qrels']:
            if row['score'] > 0:
                relevant[str(row['query-id'])].add(str(row['corpus-id']))
        ids = {str(r['id']) for r in corpus}
        assert len(ids) == len(corpus)
        for query in queries:
            assert relevant[str(query['id'])] and relevant[str(query['id'])] <= ids
        dataset = {'corpus': corpus, 'queries': queries, 'relevant': {str(q['id']): sorted(relevant[str(q['id'])]) for q in queries}}
        datasets[config] = dataset
        dump(out/'data'/f'{config}.json', dataset)
        selection['configs'][config] = [str(q['id']) for q in queries]
    dump(out/'query-selection.json', selection)
    return datasets

class Server(base.LlamaServer):
    def __init__(self, model, mode, name, pooling='last'):
        super().__init__(model, mode, BINARY, name, ctx_size=8192)
        self.pooling = pooling

    def _command(self, port):
        command = super()._command(port)
        for key, value in [('--batch-size', '4096'), ('--ubatch-size', '4096'), ('--log-verbosity', '3')]:
            command[command.index(key)+1] = value
        if self.mode == 'embedding':
            command[command.index('--pooling')+1] = self.pooling
        return command + ['--n-gpu-layers', 'all']

def request(server, endpoint, body, timeout=180):
    return base.json_request(server.url+endpoint, body, timeout=timeout)

def token_count(server, text):
    response, _ = request(server, '/tokenize', {'content': text, 'add_special': True})
    return len(response['tokens'])

def embed(server, texts, dim):
    response, elapsed = request(server, '/v1/embeddings', {'input': texts, 'encoding_format': 'float'})
    data = sorted(response['data'], key=lambda r: r['index'])
    assert [r['index'] for r in data] == list(range(len(texts)))
    vectors = np.asarray([r['embedding'] for r in data], dtype=np.float32)
    assert vectors.shape == (len(texts), dim) and np.isfinite(vectors).all()
    norms = np.linalg.norm(vectors, axis=1)
    assert np.all(np.abs(norms-1) < .01), norms
    server.sample_rss()
    return vectors/norms[:, None], elapsed

def rerank(server, query, documents):
    response, elapsed = request(server, '/v1/rerank', {'query': query, 'documents': documents, 'top_n': len(documents)})
    result = response['results']
    assert sorted(r['index'] for r in result) == list(range(len(documents)))
    scored = [(r['index'], float(r.get('relevance_score', r.get('score')))) for r in result]
    assert all(math.isfinite(score) for _, score in scored)
    order = [i for i, _ in sorted(scored, key=lambda x: (-x[1], x[0]))]
    server.sample_rss()
    return order, elapsed, dict(scored)

class Study:
    def __init__(self, out, hours):
        self.out = out
        self.started = time.monotonic()
        self.deadline = self.started + hours*3600
        self.data = prepare(out)
        self.n = 100
        base.ROOT = out
        self.progress('prepared')

    def progress(self, phase, **extra):
        value = {'time': stamp(), 'phase': phase, 'elapsed_seconds': time.monotonic()-self.started, **extra}
        dump(self.out/'progress.json', value)
        print(json.dumps(value), flush=True)

    def check_time(self):
        if time.monotonic() > self.deadline:
            raise TimeoutError('Study time budget reached; checkpoints retained')

    @contextmanager
    def server(self, name, model, mode, pooling='last'):
        self.check_time()
        server = Server(model, mode, name, pooling)
        try:
            with server:
                append(self.out/'server-events.jsonl', {'time': stamp(), 'model': name, 'load_ms': server.load_ms,
                       'command': server._command(server.port), 'props': base.get_json(server.url+'/props')})
                yield server
        finally:
            server.stop()
            append(self.out/'server-events.jsonl', {'time': stamp(), 'model': name, 'event': 'stopped',
                   'sampled_peak_rss_mb': server.peak_rss_mb})

    def pilot(self):
        means = {}
        for name, model in RERANKERS.items():
            self.progress('pilot', model=name)
            with self.server('pilot-'+name, model, 'rerank') as server:
                gates = []
                for query, positive, negative in FIXTURES:
                    order, _, scores = rerank(server, query, [positive, negative])
                    gates.append({'query': query, 'order': order, 'scores': scores})
                    assert order[0] == 0 and scores[0] > scores[1], (name, query, scores)
                repeated = [rerank(server, FIXTURES[0][0], list(FIXTURES[0][1:]))[2] for _ in range(3)]
                assert all(abs(r[0]-repeated[0][0]) < 1e-4 for r in repeated)
                dump(self.out/f'gate-{name}.json', {'fixtures': gates, 'repeat': repeated})
                durations = []
                for config, data in self.data.items():
                    # Timing-only pilot: fixed mixed-length candidates with one known positive.
                    byid = {str(d['id']): d['text'] for d in data['corpus']}
                    sampled = sorted(byid, key=lambda i: hashlib.sha256(i.encode()).hexdigest())[:50]
                    max_doc = max(token_count(server, text) for text in byid.values())
                    max_query = max(token_count(server, q['text']) for q in data['queries'])
                    assert max_doc + max_query + 512 < 4096, (name, config, max_doc, max_query)
                    append(self.out/'token-audit.jsonl', {'model': name, 'config': config,
                           'max_doc': max_doc, 'max_query': max_query, 'template_reserve': 512, 'truncated': False})
                    for query in data['queries'][:10]:
                        self.check_time()
                        ids = list(sampled)
                        positive = data['relevant'][str(query['id'])][0]
                        if positive not in ids:
                            ids[-1] = positive
                        _, elapsed, _ = rerank(server, query['text'], [byid[i] for i in ids])
                        durations.append(elapsed/1000)
                        append(self.out/'pilot-timings.jsonl', {'model': name, 'config': config, 'query_id': query['id'], 'seconds': elapsed/1000})
                    self.progress('pilot', model=name, config=config, requests_done=len(durations))
                means[name] = statistics.mean(durations)
        # Reserve 2 h indexing + 2 h reporting/buffer and apply 30% inference margin.
        for n in (100, 75, 50):
            projected = 4*3600 + 1.3*5*3*n*sum(means.values())
            if projected <= 10*3600:
                self.n = n
                break
        else:
            raise RuntimeError('Even N=50 exceeds the pre-registered timing budget')
        dump(self.out/'pilot-decision.json', {'mean_seconds': means, 'n': self.n, 'projected_main_hours': projected/3600,
             'candidate_policy': 'fixed deterministic sample with inserted positive; timing only, not retrieval quality'})
        selection = json.loads((self.out/'query-selection.json').read_text())
        selection['selected_n'] = self.n
        selection['configs'] = {c: ids[:self.n] for c, ids in selection['configs'].items()}
        dump(self.out/'query-selection-final.json', selection)
        self.progress('pilot_complete', n=self.n, projected_main_hours=projected/3600)

    def embeddings(self):
        for name, (model, pooling, dim, qp, dp, core) in EMBEDDERS.items():
            self.progress('embedding', model=name)
            with self.server('embed-'+name, model, 'embedding', pooling) as server:
                gates = []
                for query, positive, negative in FIXTURES:
                    vec, _ = embed(server, [qp+query, dp+positive, dp+negative], dim)
                    good, bad = float(vec[0]@vec[1]), float(vec[0]@vec[2])
                    gates.append({'query': query, 'positive': good, 'negative': bad})
                    assert good > bad, (name, query, good, bad)
                repeat = [embed(server, [qp+FIXTURES[0][0]], dim)[0] for _ in range(3)]
                assert all(np.allclose(repeat[0], v, atol=1e-5) for v in repeat[1:])
                dump(self.out/f'gate-{name}.json', gates)
                for config, data in self.data.items():
                    corpus = data['corpus']
                    queries = data['queries'][:self.n]
                    texts = [dp+d['text'] for d in corpus]
                    qtexts = [qp+q['text'] for q in queries]
                    lengths = [token_count(server, text) for text in texts]
                    qlengths = [token_count(server, text) for text in qtexts]
                    assert max(lengths+qlengths) < 4096, (name, config, max(lengths+qlengths))
                    append(self.out/'token-audit.jsonl', {'model': name, 'config': config,
                           'max_doc': max(lengths), 'max_query': max(qlengths), 'truncated': False})
                    # Keep physical batches below the audited budget; no silent truncation.
                    vectors, corpus_ms = [], 0.0
                    start = 0
                    while start < len(texts):
                        self.check_time()
                        end = start
                        tokens = 0
                        while end < len(texts) and end-start < 8 and tokens+lengths[end] < 3800:
                            tokens += lengths[end]
                            end += 1
                        if end == start:
                            end += 1
                        vec, elapsed = embed(server, texts[start:end], dim)
                        vectors.append(vec)
                        corpus_ms += elapsed
                        start = end
                        if start % 100 < 8:
                            self.progress('embedding', model=name, config=config, documents_done=start, documents_total=len(texts))
                    corpus_vec = np.concatenate(vectors)
                    cache = self.out/'cache'/name
                    cache.mkdir(parents=True, exist_ok=True)
                    np.save(cache/f'{config}-corpus.npy', corpus_vec)
                    # Explicit query warmup, excluded from measured rows.
                    for text in qtexts[:10]:
                        embed(server, [text], dim)
                    query_vec, query_ms = [], []
                    for i, text in enumerate(qtexts):
                        self.check_time()
                        vec, elapsed = embed(server, [text], dim)
                        query_vec.append(vec[0])
                        query_ms.append(elapsed)
                    np.save(cache/f'{config}-queries.npy', np.asarray(query_vec))
                    candidates = []
                    ids = [str(d['id']) for d in corpus]
                    for query, vec, elapsed in zip(queries, query_vec, query_ms):
                        tick = time.perf_counter()
                        order = np.argsort(-(corpus_vec@vec), kind='stable')
                        ranked = [ids[i] for i in order]
                        retrieval_ms = (time.perf_counter()-tick)*1000
                        candidate = {'query_id': str(query['id']), 'ids': ranked[:100], 'embedding_ms': elapsed, 'retrieval_ms': retrieval_ms}
                        candidates.append(candidate)
                        append(self.out/'query-results.jsonl', {'config': config, 'embedding': name, 'reranker': 'standalone',
                               'query_id': str(query['id']), 'k': 0, 'status': 'ok', 'embedding_ms': elapsed,
                               'retrieval_ms': retrieval_ms, 'rerank_ms': None, 'total_ms': elapsed+retrieval_ms,
                               **metrics(ranked, ranked[:50], set(data['relevant'][str(query['id'])]))})
                    dump(cache/f'{config}-candidates.json', candidates)
                    append(self.out/'index-stats.jsonl', {'model': name, 'config': config, 'corpus_docs': len(corpus),
                           'corpus_ms': corpus_ms, 'docs_per_second': len(corpus)/(corpus_ms/1000), 'dimension': dim,
                           'query_prefix': qp, 'document_prefix': dp, 'pooling': pooling, 'normalized': True})
                    self.progress('embedding_config_complete', model=name, config=config)

    def rerank_all(self):
        for name, model in RERANKERS.items():
            self.progress('reranking', model=name)
            with self.server('main-'+name, model, 'rerank') as server:
                for query, positive, negative in FIXTURES:
                    rerank(server, query, [positive, negative])
                for embedding, spec in EMBEDDERS.items():
                    if not spec[-1]:
                        continue
                    for config, data in self.data.items():
                        byid = {str(d['id']): d['text'] for d in data['corpus']}
                        candidates = json.loads((self.out/'cache'/embedding/f'{config}-candidates.json').read_text())
                        for i, (query, candidate) in enumerate(zip(data['queries'][:self.n], candidates)):
                            self.check_time()
                            ids = candidate['ids'][:50]
                            row = {'config': config, 'embedding': embedding, 'reranker': name, 'k': 50,
                                   'query_id': str(query['id']), 'embedding_ms': candidate['embedding_ms'],
                                   'retrieval_ms': candidate['retrieval_ms']}
                            try:
                                order, elapsed, scores = rerank(server, query['text'], [byid[d] for d in ids])
                                final = [ids[j] for j in order]
                                row.update(status='ok', rerank_ms=elapsed,
                                           total_ms=candidate['embedding_ms']+candidate['retrieval_ms']+elapsed,
                                           final_ids=final, scores=[scores[j] for j in order],
                                           **metrics(final, ids, set(data['relevant'][str(query['id'])])))
                            except Exception as exc:
                                row.update(status='error', error=str(exc), rerank_ms=None, total_ms=None,
                                           **{key: 0.0 for key in METRICS})
                                append(self.out/'errors.jsonl', row)
                            append(self.out/'query-results.jsonl', row)
                            if (i+1) % 10 == 0:
                                self.progress('reranking', model=name, embedding=embedding, config=config, queries_done=i+1, queries_total=self.n)
                        self.report()

    def report(self):
        groups = defaultdict(list)
        for row in rows(self.out/'query-results.jsonl'):
            groups[(row['config'], row['embedding'], row['reranker'])].append(row)
        summaries = []
        for (config, embedding, reranker), part in sorted(groups.items()):
            summary = {'config': config, 'embedding': embedding, 'reranker': reranker, 'queries': len(part),
                       'errors': sum(r['status'] != 'ok' for r in part),
                       **{m: statistics.mean(r[m] for r in part) for m in METRICS}}
            for key in ('embedding_ms', 'rerank_ms', 'total_ms'):
                values = [r[key] for r in part if r.get(key) is not None]
                for label, quantile in [('p50', .5), ('p95', .95)]:
                    summary[key+'_'+label] = float(np.quantile(values, quantile)) if values else None
            summaries.append(summary)
        dump(self.out/'summary.json', summaries)
        lines = ['# MLQA bilingual retrieval — local Q8_0 benchmark', '',
                 'Partial until progress.json says complete. Fixed seeded validation subset; not official MTEB scores.',
                 'eng-zho = Chinese query → English corpus. zho-zho = Chinese → Chinese; eng-eng = English → English.',
                 'Full small validation corpora, exact cosine, model-specific prefixes/pooling, Metal GPU, rerank k=50.',
                 'Total latency is the sum of separately measured embedding, exact search and rerank, not production end-to-end latency.',
                 'Errors count as zero quality; latency percentiles only include successful requests. RSS samples exclude unsampled transient peaks.', '',
                 '| direction | embedder | reranker | n | errors | Top1 | MRR10 | nDCG10 | Recall10 | Recall50 | rerank p50/p95 ms |',
                 '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for r in summaries:
            latency = '—' if r['rerank_ms_p50'] is None else f"{r['rerank_ms_p50']:.0f}/{r['rerank_ms_p95']:.0f}"
            lines.append(f"| {r['config']} | {r['embedding']} | {r['reranker']} | {r['queries']} | {r['errors']} | " +
                         ' | '.join(f"{r[m]:.3f}" for m in ['top1','mrr10','ndcg10','recall10','recall50']) + ' | '+latency+' |')
        (self.out/'report.md').write_text('\n'.join(lines)+'\n')
        return summaries

    def confidence(self):
        allrows = rows(self.out/'query-results.jsonl')
        groups = defaultdict(dict)
        for r in allrows:
            groups[(r['config'], r['embedding'], r['reranker'])][r['query_id']] = r
        rng = np.random.default_rng(20261022)
        result = []
        for (config, embedding, reranker), part in groups.items():
            if reranker == 'standalone':
                continue
            baseline = groups[(config, embedding, 'standalone')]
            ids = sorted(part)
            delta = np.asarray([part[i]['top1']-baseline[i]['top1'] for i in ids])
            estimates = delta[rng.integers(0, len(ids), size=(2000, len(ids)))].mean(axis=1)
            result.append({'config': config, 'embedding': embedding, 'reranker': reranker,
                           'top1_delta_vs_standalone': float(delta.mean()),
                           'paired_bootstrap95': np.quantile(estimates, [.025, .975]).tolist(),
                           'interpretation': 'exploratory; no multiple-comparison correction'})
        dump(self.out/'paired-confidence.json', result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=12)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.prepare_only:
        prepare(out)
        print('Prepared data only', out)
        return
    readiness = json.loads((ROOT/'materials-readiness.json').read_text())
    assert readiness['status'] == 'ready'
    version = subprocess.run([str(BINARY), '--version'], capture_output=True, text=True, check=True)
    metadata = {'started_at': stamp(), 'binary': str(BINARY), 'binary_sha256': base.sha256_file(BINARY),
                'binary_version': (version.stdout + version.stderr).strip(),
                'platform': platform.platform(), 'python': sys.version, 'numpy': np.__version__,
                'model_and_data_hashes': readiness, 'runner_sha256': base.sha256_file(Path(__file__)),
                'time_budget_hours': args.hours, 'production_modified': False}
    dump(out/'run-manifest.json', metadata)
    study = Study(out, args.hours)
    try:
        study.pilot()
        study.embeddings()
        study.rerank_all()
        summary = study.report()
        study.confidence()
        assert len(summary) == 66 and all(r['queries'] == study.n for r in summary)
        errors = sum(r['errors'] for r in summary)
        study.progress('complete' if errors == 0 else 'complete_with_errors', summaries=len(summary), query_rows=66*study.n, errors=errors)
    except BaseException as exc:
        append(out/'fatal-errors.jsonl', {'time': stamp(), 'type': type(exc).__name__, 'error': str(exc)})
        study.report()
        study.progress('stopped', reason=str(exc))
        raise

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: sys.exit('Termination requested'))
    main()
