#!/usr/bin/env python3
"""Verify all planned model/data files, without running inference or changing sources."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def verify(path, size, expected, kind, source):
    row = {'path': str(path.relative_to(WORKSPACE)), 'kind': kind,
           'expected_bytes': size, 'expected_sha256': expected, 'source': source}
    if not path.is_file():
        return dict(row, status='missing')
    row.update(bytes=path.stat().st_size, sha256=sha256(path))
    row['status'] = 'verified' if row['bytes'] == size and row['sha256'] == expected else 'mismatch'
    print(row['status'], row['path'], flush=True)
    return row

def main():
    results = []
    downloads = json.loads((ROOT / 'download-manifest.json').read_text())
    for model in downloads['models']:
        results.append(verify(ROOT / 'models' / model['file'], model['bytes'],
                              model['sha256'], 'embedding', model['url']))
    deployment = json.loads((WORKSPACE / 'qwen-reranker-rebuilt/deployment/deployment-manifest.json').read_text())
    for model in deployment['models']:
        results.append(verify(WORKSPACE / model['file'], model['size_bytes'],
                              model['sha256'], model['kind'], model['source_url']))
    results.append(verify(WORKSPACE / 'qwen-reranker-rebuilt/Qwen3-Reranker-4B-Q8_0.gguf',
                          4279678912,
                          '4178dde89ef52ba4c986269466c2f1150bf4d94ed1b7ec91cb78ed87361794d1',
                          'reranker', 'https://huggingface.co/Qwen/Qwen3-Reranker-4B'))
    data_root = WORKSPACE / 'qwen-reranker-rebuilt/real-dataset'
    dataset = json.loads((data_root / 'manifest.json').read_text())
    counts = {}
    for config, entry in dataset['configs'].items():
        counts[config] = entry['full_validation_counts']
        for kind, item in entry['files'].items():
            source = ('https://huggingface.co/datasets/' + dataset['dataset'] +
                      '/resolve/' + dataset['source_revision'] + '/' + item['repo_file'])
            results.append(verify(data_root / item['local_file'], item['bytes'],
                                  item['sha256'], 'dataset-' + kind, source))
    report = {
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'ready' if all(r['status'] == 'verified' for r in results) else 'incomplete',
        'scope': '7 embedders, 3 rerankers, MLQA three-direction validation raw files',
        'dataset_revision': dataset['source_revision'],
        'dataset_license': dataset['license'], 'full_validation_counts': counts,
        'files': results,
        'not_done': ['new 100-query manifests', 'benchmark driver adaptations',
                     'full benchmark', 'production configuration changes'],
    }
    (ROOT / 'materials-readiness.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('Overall:', report['status'], 'Files:', len(results), flush=True)
    return 0 if report['status'] == 'ready' else 1

if __name__ == '__main__':
    raise SystemExit(main())
