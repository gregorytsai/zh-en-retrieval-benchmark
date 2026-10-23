#!/usr/bin/env python3
"""Fetch only missing judged passages via HF's documented filter API."""
import gzip
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent/'sources/miracl/zh'
REVISION = 'd921ec7e349ce0d28daf30b2da9da5ee698bef0d'


def get(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        body = response.read()
        return json.loads(body), dict(response.headers)


def main():
    destination = ROOT/'supplement-from-official-corpus.json'
    if destination.exists():
        raise RuntimeError('Output already exists; inspect rather than overwrite')
    version, _ = get('https://huggingface.co/api/datasets/miracl/miracl-corpus')
    assert version['sha'] == REVISION
    with gzip.open(ROOT/'passages.jsonl.gz', 'rt') as stream:
        docs = {x['docid']: x for x in map(json.loads, stream)}
    wanted = {x.split()[2] for x in (ROOT/'qrels-dev.tsv').read_text().splitlines()}
    missing = sorted(wanted - docs.keys())
    controls = sorted(wanted & docs.keys())[:2]
    requested = missing + controls
    assert len(requested) <= 100
    where = ' OR '.join('"docid"=\'' + x.replace("'", "''") + "'" for x in requested)
    url = 'https://datasets-server.huggingface.co/filter?' + urllib.parse.urlencode({
        'dataset': 'miracl/miracl-corpus', 'config': 'zh', 'split': 'train',
        'where': where, 'offset': 0, 'length': 100})
    payload, headers = get(url)
    assert all(not x.get('truncated_cells') for x in payload['rows'])
    found = {x['row']['docid']: x['row'] for x in payload['rows']}
    assert set(found) == set(requested), (set(requested)-set(found))
    for did in controls:
        assert found[did] == docs[did], 'Upstream subset and original corpus text differ'
    version_after, _ = get('https://huggingface.co/api/datasets/miracl/miracl-corpus')
    assert version_after['sha'] == REVISION
    result = {'url': url, 'upstream_main_revision_checked_before_and_after': REVISION,
              'note': 'Filter endpoint does not accept revision; saved response and controls provide provenance.',
              'headers': headers, 'missing_ids': missing, 'control_ids': controls,
              'response': payload}
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'missing_recovered': len(missing), 'controls_match': len(controls),
                      'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
