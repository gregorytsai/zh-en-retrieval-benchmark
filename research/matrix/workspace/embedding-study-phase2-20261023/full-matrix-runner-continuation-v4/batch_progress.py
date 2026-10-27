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

from matrix_spec import BATCHES, RUN_BATCHES, BATCH_DATASETS, DATASET_COUNTS, RERANKERS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _progress_row_indices(lines: list[str], batch: str) -> list[int]:
    """Resolve a batch ID only inside the progress table, never scope tables."""
    header = next((i for i, line in enumerate(lines)
                   if line.strip().startswith("| 批次 | Embedder ID | 狀態 | 摘要 |")), None)
    if header is None:
        candidates = range(len(lines))
    else:
        stop = len(lines)
        for i in range(header + 1, len(lines)):
            line = lines[i].strip()
            if not line or not line.startswith("|"):
                stop = i
                break
        candidates = range(header + 1, stop)
    return [i for i in candidates if lines[i].strip().startswith(f"| {batch} |")]


def latest_audited_segment(run_root: Path, batch: str, audit_name: str = 'batch-completion-audit.json') -> Path:
    homes = sorted((run_root / batch).glob('segment-*/official'), key=lambda path: path.parent.name)
    for home in reversed(homes):
        path = home / audit_name
        if path.is_file() and read(path).get('status') == 'passed':
            return home
    raise ValueError(f'{batch} has no passed audited official segment')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit_completed(batch_id: str, home: Path, workspace: Path) -> dict:
    import core
    import guarded_engine as engine
    core.configure(workspace)
    embedder = dict(RUN_BATCHES).get(batch_id, dict(BATCHES).get(batch_id))
    require(embedder is not None, 'Unknown batch')
    dataset_ids = list(BATCH_DATASETS[batch_id])
    manifest = read(home / 'run-manifest.json')
    require(manifest['run_kind'] == 'official', 'Pilot cannot complete an official batch')
    require(read(home / 'progress.json')['phase'] == 'complete', 'Run is not complete')
    require(manifest['batch_id'] == batch_id, 'Batch ID differs from frozen manifest')
    require(manifest['dataset_ids'] == dataset_ids, 'Run dataset scope differs from the registered batch')
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
    expected_summaries = len(dataset_ids) * 4
    expected_rows = sum(DATASET_COUNTS[name]['queries'] for name in dataset_ids) * 4
    require(len(committed) == len(summary) == expected_summaries, 'Committed summary count differs from batch scope')
    main_rows = core.rows(home / 'query-results.jsonl')
    require(len(main_rows) == expected_rows, f'Expected {expected_rows} rows for batch scope')
    key = lambda r: (r['dataset'], r['embedding'], r['reranker'], r['query_id'])
    main = {key(r): r for r in main_rows}
    require(len(main) == expected_rows, 'Duplicate query keys')
    traffic = core.rows(home / 'traffic-events.jsonl')
    accepted_keys = {e['combo'] for e in traffic if e.get('event') == 'combo_accepted'}
    require(accepted_keys == set(committed), 'Traffic acceptance evidence mismatch')
    observed = set()
    for dataset in dataset_ids:
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
        require(len(audits) == len(dataset_ids) and {a['dataset'] for a in audits} == set(dataset_ids), 'Preflight dataset coverage mismatch')
        require(all(a['status'] == 'ok' and not a['truncated'] for a in audits), 'Token failure/truncation')
    return {'status': 'passed', 'batch': batch_id, 'embedder': embedder,
            'dataset_ids': dataset_ids, 'summaries': expected_summaries, 'rows': expected_rows, 'errors': 0,
            'accepted_gpu_attempts': len(committed), 'candidate_hashes_verified': True,
            'started': manifest['original_started_epoch'], 'finished': read(home / 'progress.json')['time'],
            'run': str(home.resolve()), 'checked_at': datetime.now(ZoneInfo('Asia/Taipei')).isoformat()}


def render_completed(plan_text: str, report: dict) -> str:
    require(report.get('status') == 'passed' and report.get('summaries') == 24 and report.get('rows') == 10452 and report.get('errors') == 0, 'Incomplete batch cannot update plan')
    batch = report['batch']
    require(batch in dict(BATCHES), 'Unknown batch')
    lines = plan_text.splitlines()
    matches = _progress_row_indices(lines, batch)
    require(len(matches) == 1, 'Plan must contain exactly one row for this batch')
    start = datetime.fromtimestamp(report['started'], ZoneInfo('Asia/Taipei')).strftime('%m/%d %H:%M')
    finish = datetime.fromisoformat(report['finished']).astimezone(ZoneInfo('Asia/Taipei')).strftime('%m/%d %H:%M')
    require('|' not in report['run'] and '\n' not in report['run'], 'Unsafe table path')
    lines[matches[0]] = f"| {batch} | {report['embedder']} | 稽核通過 | 24/24 | 10452/10452 | {start}–{finish} | {report['run']} |"
    return '\n'.join(lines) + '\n'


def update_completed(batch, home, workspace, plan):
    import fcntl
    report = audit_completed(batch, home, workspace)
    if batch in ('B02-A', 'B02-B'):
        import core
        core.dump(home / 'subbatch-completion-audit.json', report)
        return report
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


def aggregate_b02_splits(run_root: Path, workspace: Path, plan: Path) -> dict | None:
    """Combine only two fully audited, disjoint B02 subbatches into six sections."""
    import core
    from matrix_spec import equal_weight_macro
    sources = [latest_audited_segment(run_root, batch, 'subbatch-completion-audit.json')
               for batch in ("B02-A", "B02-B")]
    audit_paths = [home / "subbatch-completion-audit.json" for home in sources]
    if not all(path.is_file() for path in audit_paths):
        return None
    audits = [read(path) for path in audit_paths]
    require(all(item.get("status") == "passed" for item in audits), "Both B02 split audits must pass")
    require([item.get("batch") for item in audits] == ["B02-A", "B02-B"], "B02 split audit identities differ")
    require(set(audits[0]["dataset_ids"]).isdisjoint(audits[1]["dataset_ids"]), "B02 split datasets overlap")
    require(set(audits[0]["dataset_ids"]) | set(audits[1]["dataset_ids"]) == set(DATASET_COUNTS),
            "B02 split does not cover all six datasets")
    manifests = [read(home / "run-manifest.json") for home in sources]
    for field in ("dataset_sha256", "binary_sha256", "model_sha256", "code_sha256", "effective_configs"):
        require(manifests[0].get(field) == manifests[1].get(field), f"B02 split {field} pins differ")
    summaries = []
    rows = []
    source_files = []
    for home in sources:
        summaries.extend(read(home / "summary.json"))
        rows.extend(core.rows(home / "query-results.jsonl"))
        source_files.append({
            "run": str(home.resolve()),
            "audit_sha256": core.sha256_file(home / "subbatch-completion-audit.json"),
            "summary_sha256": core.sha256_file(home / "summary.json"),
            "rows_sha256": core.sha256_file(home / "query-results.jsonl"),
        })
    expected_rows = sum(spec["queries"] for spec in DATASET_COUNTS.values()) * 4
    expected_summaries = len(DATASET_COUNTS) * 4
    key = lambda row: (row["dataset"], row["embedding"], row["reranker"], str(row["query_id"]))
    require(len(summaries) == expected_summaries, "B02 combined summary count is incomplete")
    require(len(rows) == expected_rows and len({key(row) for row in rows}) == expected_rows,
            "B02 combined query rows are incomplete or duplicated")
    require(all(row.get("status") == "ok" for row in rows), "B02 combined result has non-ok rows")
    require(core.summarize(rows) == sorted(summaries, key=lambda row: (row["dataset"], row["embedding"], row["reranker"])),
            "B02 combined summaries do not reproduce from accepted rows")
    batch = "B02"
    embedder = dict(BATCHES)[batch]
    require({row.get("embedding") for row in summaries} == {embedder}, "B02 combined embedding ID differs")
    combined = run_root / "B02" / "combined"
    combined.mkdir(parents=True, exist_ok=True)
    core.dump(combined / "summary.json", sorted(summaries, key=lambda row: (row["dataset"], row["embedding"], row["reranker"])))
    rows_path = combined / "query-results.jsonl"
    temp_rows = rows_path.with_suffix(".jsonl.tmp")
    temp_rows.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temp_rows.replace(rows_path)
    started = min(item["started"] for item in audits)
    finished = max(item["finished"] for item in audits)
    report = {
        "status": "passed", "batch": batch, "embedder": embedder,
        "dataset_ids": list(DATASET_COUNTS), "summaries": expected_summaries,
        "rows": expected_rows, "errors": 0, "accepted_gpu_attempts": 24,
        "candidate_hashes_verified": True, "split_sources": source_files,
        "started": started, "finished": finished,
        "run": str(combined.resolve()), "checked_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
    }
    core.dump(combined / "batch-completion-audit.json", report)
    ranking = []
    for reranker in ("standalone", *RERANKERS):
        config_rows = [row for row in summaries if row["reranker"] == reranker]
        macro = equal_weight_macro(config_rows)
        require(macro["eligible"], f"B02 {reranker} is not six-dataset ranking eligible")
        # Keep configuration identity in the persisted record. The ranking
        # sort and downstream audit both key on reranker; omitting it here
        # used to raise KeyError after the combined audit had already passed.
        ranking.append({"batch": batch, "embedding": embedder, "reranker": reranker,
                        "completion_audit_passed": True,
                        "source_subbatches": ["B02-A", "B02-B"], **macro})
    ranking.sort(key=lambda row: (-row["macro_ndcg10_equal_weight"], row["reranker"]))
    core.dump(combined / "ranking.json", {
        "scope": "B02 only; merged from two independently audited disjoint dataset subbatches",
        "configurations": ranking,
    })
    # The two registered B02-A/B rows are completed independently by the
    # controller. There is intentionally no aggregate B02 row in the plan.
    plan_lines = plan.read_text(encoding="utf-8").splitlines()
    for subbatch, expected_count, expected_row_count in (
        ("B02-A", 4, 4592), ("B02-B", 20, 5860),
    ):
        matches = [plan_lines[index] for index in _progress_row_indices(plan_lines, subbatch)]
        require(len(matches) == 1, f"Plan must contain exactly one {subbatch} progress row")
        require("稽核通過" in matches[0] and f"{expected_count}/{expected_count}" in matches[0] and
                f"{expected_row_count}/{expected_row_count}" in matches[0],
                f"Plan progress row for {subbatch} is not marked audited/completed")
    return report


def aggregate_global(run_root: Path, workspace: Path, carryover_manifest: Path) -> dict:
    """Build a global ranking only when all seven embedder scopes are audited."""
    import core
    from matrix_spec import equal_weight_macro
    carry = read(carryover_manifest)
    homes = {"B01": Path(carry["batches"]["B01"]["run"])}
    homes["B02"] = run_root / "B02" / "combined"
    for batch in ("B03", "B04", "B05", "B06", "B07"):
        homes[batch] = latest_audited_segment(run_root, batch)
    summaries = []
    rows = []
    sources = []
    for batch, home in homes.items():
        audit_path = home / "batch-completion-audit.json"
        require(audit_path.is_file() and read(audit_path).get("status") == "passed",
                f"{batch}: completion audit is missing or not passed")
        batch_summary = read(home / "summary.json")
        batch_rows = core.rows(home / "query-results.jsonl")
        require(len(batch_summary) == 24 and len(batch_rows) == 10452, f"{batch}: incomplete full matrix")
        require(all(item.get("errors") == 0 for item in batch_summary), f"{batch}: summary errors are nonzero")
        summaries.extend(batch_summary)
        rows.extend(batch_rows)
        sources.append({
            "batch": batch, "run": str(home.resolve()),
            "audit_sha256": core.sha256_file(audit_path),
            "summary_sha256": core.sha256_file(home / "summary.json"),
            "rows_sha256": core.sha256_file(home / "query-results.jsonl"),
        })
    expected_summaries = 168
    expected_rows = 73164
    row_key = lambda row: (row["dataset"], row["embedding"], row["reranker"], str(row["query_id"]))
    require(len(summaries) == expected_summaries, "Global matrix does not contain 168 summaries")
    require(len(rows) == expected_rows and len({row_key(row) for row in rows}) == expected_rows,
            "Global matrix does not contain 73164 unique rows")
    require(all(row.get("status") == "ok" for row in rows), "Global matrix contains failed rows")
    global_by_config = {}
    for embedder in (value for _, value in BATCHES):
        for reranker in ("standalone", *RERANKERS):
            group = [row for row in summaries if row.get("embedding") == embedder and row.get("reranker") == reranker]
            macro = equal_weight_macro(group)
            require(macro["eligible"], f"Global configuration {embedder}/{reranker} lacks six clean dataset summaries")
            global_by_config[(embedder, reranker)] = {"embedding": embedder, "reranker": reranker, **macro}
    ranking = sorted(global_by_config.values(), key=lambda row: (-row["macro_ndcg10_equal_weight"], row["embedding"], row["reranker"]))
    report = {
        "status": "passed", "summaries": expected_summaries, "rows": expected_rows,
        "unique_query_keys": expected_rows, "errors": 0, "eligible_configurations": 28,
        "sources": sources, "checked_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
    }
    core.dump(run_root / "global-completion-audit.json", report)
    core.dump(run_root / "global-ranking.json", {
        "scope": "full 7-embedder × 4-config × 6-dataset matrix",
        "macro": "equal-weight mean nDCG@10 across all six datasets; incomplete configurations are excluded and block this file",
        "configurations": ranking,
    })
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', choices=dict(BATCHES), required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(update_completed(args.batch, args.run_dir, args.workspace, args.plan), ensure_ascii=False, indent=2))
