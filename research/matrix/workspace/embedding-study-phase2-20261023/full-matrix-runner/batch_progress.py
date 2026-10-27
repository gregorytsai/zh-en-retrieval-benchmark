"""Audit a completed official batch before updating its plan-table row.

This module never starts inference. Incomplete runs cannot be marked complete.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from matrix_spec import BATCHES, DATASET_COUNTS, RERANKERS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit_completed(batch_id: str, home: Path, workspace: Path) -> dict:
    import core
    import guarded_engine as engine
    core.configure(workspace)
    embedder = dict(BATCHES)[batch_id]
    manifest = read(home / 'run-manifest.json')
    require(manifest['run_kind'] == 'official', 'Pilot cannot complete an official batch')
    require(read(home / 'progress.json')['phase'] == 'complete', 'Run is not complete')
    require(manifest['batch_id'] == batch_id, 'Batch ID differs from frozen manifest')
    require(set(manifest['dataset_ids']) == set(DATASET_COUNTS), 'Missing dataset')
    require(core.sha256_file(core.BINARY) == manifest['binary_sha256'], 'Binary changed')
    code_root = Path(__file__).resolve().parent
    code_aliases = {'run_study.py': 'core.py', 'legacy_run_benchmark.py': 'server_helper.py'}
    for name, digest in manifest['code_sha256'].items():
        path = code_root / code_aliases.get(name, name)
        require(path.parent == code_root and core.sha256_file(path) == digest, f'Code changed: {name}')
    deadline = read(home / 'deadline.json')
    require(core.sha256_json(deadline) == manifest['deadline_record_sha256'], 'Deadline record changed')
    require(0 < deadline['wall_hours'] <= 12, 'Invalid wall-clock budget')
    require(deadline['deadline_epoch'] == manifest['deadline_epoch'], 'Deadline extended')
    finished = datetime.fromisoformat(read(home / 'progress.json')['time']).timestamp()
    require(finished <= manifest['deadline_epoch'], 'Run completed after deadline')
    require(core.sha256_json(read(home / 'query-selection.json')) == manifest['selection_sha256'], 'Selection artifact changed')
    require(core.sha256_json(read(home / 'qrels.json')) == manifest['selected_qrels_sha256'], 'Qrels artifact changed')
    for name, digest in manifest['model_sha256'].items():
        path = core.EMBEDDERS[name][0] if name in core.EMBEDDERS else core.RERANKERS[name]
        require(core.sha256_file(path) == digest, f'Model changed: {name}')
    committed = read(home / 'committed.json')
    selection = read(home / 'query-selection.json')['datasets']
    qrels = read(home / 'qrels.json')['datasets']
    summary = read(home / 'summary.json')
    require(len(committed) == len(summary) == 24, 'Expected 24 committed summaries')
    main_rows = core.rows(home / 'query-results.jsonl')
    require(len(main_rows) == 10452, 'Expected 10452 rows')
    key = lambda r: (r['dataset'], r['embedding'], r['reranker'], r['query_id'])
    main = {key(r): r for r in main_rows}
    require(len(main) == 10452, 'Duplicate query keys')
    traffic = core.rows(home / 'traffic-events.jsonl')
    accepted_keys = {e['combo'] for e in traffic if e.get('event') == 'combo_accepted'}
    require(accepted_keys == set(committed), 'Traffic acceptance evidence mismatch')
    observed = set()
    for dataset in DATASET_COUNTS:
        query_ids = selection[dataset]['query_ids']
        require(len(query_ids) == DATASET_COUNTS[dataset]['queries'], 'Query count mismatch')
        cache_dir = home / 'cache' / embedder / dataset
        cache_manifest, cache_hash = engine.verified_cache_manifest(cache_dir, query_ids)
        require(cache_manifest['model_sha256'] == manifest['model_sha256'][embedder], 'Cache model pin differs')
        require(cache_manifest['binary_sha256'] == manifest['binary_sha256'], 'Cache binary pin differs')
        require(cache_manifest['code_sha256'] == manifest['code_sha256'], 'Cache code pin differs')
        candidates = read(cache_dir / 'candidates.json')
        pools = {r['query_id']: r['ids'] for r in candidates}
        for reranker in ('standalone', *RERANKERS):
            commit_key = f'{dataset}--{embedder}--{reranker}'
            attempt_rows = engine.validate_accepted_attempt(home, (dataset, embedder, reranker, commit_key), committed, query_ids)
            attempt = home / committed[commit_key]['attempt']
            require(read(attempt / 'validity.json').get('traffic_epoch') is not None, 'Missing traffic epoch')
            events = [e for e in core.rows(attempt / 'server-events.jsonl') if e.get('command')]
            require(len(events) == 1, 'Expected one measured server')
            command = events[0]['command']
            require(command[command.index('--n-gpu-layers')+1] == 'all', 'GPU not requested')
            log = Path(command[command.index('--log-file')+1]).read_text(errors='replace')
            offloads = re.findall(r'offloaded\s+(\d+)/(\d+) layers to GPU', log)
            require(offloads and int(offloads[-1][0]) > 0, 'Missing actual GPU offload')
            if reranker != 'standalone':
                inp = read(attempt / 'cache-input.json')
                require(inp['cache_manifest_sha256'] == cache_hash, 'Different candidate cache')
                gate = read(attempt / 'gate-reranker.json')
                require(gate['status'] == 'passed' and len(gate['fixtures']) == 10, 'Reranker fixture gate missing')
            for row in attempt_rows:
                require(main.get(key(row)) == row, 'Main row differs from accepted attempt')
                observed.add(key(row))
                pool = pools[row['query_id']]
                require(len(pool) == len(set(pool)) == len(row['final_ids']) == 50, 'Not exactly top50')
                require(set(pool) == set(row['final_ids']), 'Rerank changed candidate set')
                if reranker == 'standalone':
                    require(pool == row['final_ids'], 'Standalone changed candidate order')
                else:
                    require(row['candidate_ids_sha256'] == core.sha256_json(pool), 'Candidate input order hash mismatch')
                values = core.metrics(row['final_ids'], pool, qrels[dataset]['qrels'][row['query_id']], qrels[dataset]['grade_mode'])
                require(all(abs(row[k]-v) < 1e-10 for k,v in values.items()), 'Metric recomputation mismatch')
    require(observed == set(main), 'Unexpected main rows')
    require(core.summarize(main_rows) == summary, 'Summary differs from rows')
    preflight = read(home / 'preflight.json')['models']
    require(set(preflight) == {embedder, *RERANKERS}, 'Preflight model coverage mismatch')
    for name, item in preflight.items():
        require(item['status'] == 'passed', 'Preflight not passed')
        audits = core.rows(home / item['attempt'] / 'token-audit.jsonl')
        require(len(audits) == 6 and {a['dataset'] for a in audits} == set(DATASET_COUNTS), 'Preflight dataset coverage mismatch')
        require(all(a['status'] == 'ok' and not a['truncated'] for a in audits), 'Token failure/truncation')
    return {'status': 'passed', 'batch': batch_id, 'embedder': embedder,
            'summaries': 24, 'rows': 10452, 'errors': 0,
            'accepted_gpu_attempts': 24, 'candidate_hashes_verified': True,
            'started': manifest['original_started_epoch'], 'finished': read(home / 'progress.json')['time'],
            'run': str(home.resolve()), 'checked_at': datetime.now(ZoneInfo('Asia/Taipei')).isoformat()}


def render_completed(plan_text: str, report: dict) -> str:
    require(report.get('status') == 'passed' and report.get('summaries') == 24 and report.get('rows') == 10452 and report.get('errors') == 0, 'Incomplete batch cannot update plan')
    batch = report['batch']
    require(batch in dict(BATCHES), 'Unknown batch')
    lines = plan_text.splitlines()
    matches = [i for i, line in enumerate(lines) if line.startswith(f'| {batch} |')]
    require(len(matches) == 1, 'Plan must contain exactly one row for this batch')
    start = datetime.fromtimestamp(report['started'], ZoneInfo('Asia/Taipei')).strftime('%m/%d %H:%M')
    finish = datetime.fromisoformat(report['finished']).astimezone(ZoneInfo('Asia/Taipei')).strftime('%m/%d %H:%M')
    require('|' not in report['run'] and '\n' not in report['run'], 'Unsafe table path')
    lines[matches[0]] = f"| {batch} | {report['embedder']} | 稽核通過 | 24/24 | 10452/10452 | {start}–{finish} | {report['run']} |"
    return '\n'.join(lines) + '\n'


def update_completed(batch, home, workspace, plan):
    import fcntl
    report = audit_completed(batch, home, workspace)
    # Dedicated documentation lock, not the inference lock already held by runner.
    with plan.with_suffix('.progress.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        original = plan.read_text(encoding='utf-8')
        replacement = render_completed(original, report)
        import core
        core.dump(home / 'batch-completion-audit.json', report)
        from matrix_spec import equal_weight_macro
        summaries = read(home / 'summary.json')
        ranking = []
        for reranker in ('standalone', *RERANKERS):
            rows = [r for r in summaries if r['reranker'] == reranker]
            macro = equal_weight_macro(rows)
            require(macro['eligible'], 'Audited batch has ineligible six-section coverage')
            ranking.append({'batch': batch, 'embedding': report['embedder'], 'reranker': reranker,
                            'completion_audit_passed': True, **macro})
        ranking.sort(key=lambda r: (-r['macro_ndcg10_equal_weight'], r['reranker']))
        core.dump(home / 'ranking.json', {'scope': 'this batch only; not a 28-configuration global ranking', 'configurations': ranking})
        temp = plan.with_suffix('.progress.tmp')
        temp.write_text(replacement, encoding='utf-8')
        temp.replace(plan)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', choices=dict(BATCHES), required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(update_completed(args.batch, args.run_dir, args.workspace, args.plan), ensure_ascii=False, indent=2))
