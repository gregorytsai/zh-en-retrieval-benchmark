#!/usr/bin/env python3
"""Run each combination transactionally; retry a whole combo after external traffic."""
import argparse
import fcntl
import json
import shutil
import signal
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import run_study as core
from traffic_guard import TrafficGuard, Interference

ALL_EMBEDDERS = dict(core.EMBEDDERS)
ALL_RERANKERS = dict(core.RERANKERS)


class GuardedStudy(core.Study):
    def __init__(self, out, guard, deadline):
        super().__init__(out, max(.001, deadline-time.time())/3600)
        self.guard = guard
        self.wall_deadline = deadline
        self.home = out

    def check_time(self):
        if time.time() >= self.wall_deadline:
            raise TimeoutError('Original 12-hour wall-clock budget reached, including traffic pauses')
        self.guard.check()

    def progress(self, phase, **extra):
        target = getattr(self, 'home', self.out)
        value = {'time': core.stamp(), 'phase': phase, 'elapsed_seconds': time.monotonic()-self.started, **extra}
        core.dump(target/'progress.json', value)
        print(json.dumps(value), flush=True)

    @contextmanager
    def server(self, name, model, mode, pooling='last'):
        self.check_time()
        server = core.Server(model, mode, name, pooling)
        original = server._command
        def command(port):
            args = original(port)
            args[args.index('--log-verbosity')+1] = '4'
            return args
        server._command = command
        self.guard.bind(server)
        try:
            with server:
                self.check_time()
                core.append(self.out/'server-events.jsonl', {'time': core.stamp(), 'model': name,
                            'load_ms': server.load_ms, 'command': server._command(server.port)})
                yield server
        finally:
            server.stop()
            self.guard.unbind()
            core.append(self.out/'server-events.jsonl', {'time': core.stamp(), 'model': name,
                        'event': 'stopped', 'sampled_peak_rss_mb': server.peak_rss_mb})


def consolidate(home, committed):
    target = home/'query-results.jsonl'
    temp = home/'query-results.jsonl.tmp'
    with temp.open('w') as output:
        for item in committed.values():
            output.write((home/item['attempt']/'query-results.jsonl').read_text())
    temp.replace(target)
    obj = object.__new__(core.Study)
    obj.out = home
    summary = obj.report()
    return obj, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', default='2026-10-22T22:08:13+00:00')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    home = args.output.resolve()
    home.mkdir(parents=True, exist_ok=args.resume)
    lockfile = (home/'runner.lock').open('a')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    committed_path = home/'committed.json'
    committed = json.loads(committed_path.read_text()) if committed_path.exists() else {}
    def event(kind, **extra):
        core.append(home/'traffic-events.jsonl', {'time': core.stamp(), 'event': kind, **extra})
    guard = TrafficGuard(event)
    study = GuardedStudy(home, guard, deadline)
    data = dict(study.data)
    source = core.ROOT/'run-20261022-v2'
    decision = json.loads((source/'pilot-decision.json').read_text())
    study.n = decision['n']
    core.dump(home/'pilot-decision.json', dict(decision, source_run=str(source), note='Timing budget estimate only; official timings are remeasured under traffic guard'))
    metadata = json.loads((source/'run-manifest.json').read_text())
    metadata.update(started_at=core.stamp(), runner_sha256=core.base.sha256_file(Path(__file__)),
                    core_runner_sha256=core.base.sha256_file(Path(core.__file__)),
                    guard_sha256=core.base.sha256_file(core.ROOT/'traffic_guard.py'),
                    external_traffic_policy='1-second metrics polling, fail closed, 30-second quiet window, whole-combo retry',
                    wall_deadline=args.deadline, old_run_role='archived unguarded evidence; not merged into official latency')
    if (home/'run-manifest.json').exists():
        core.append(home/'resume-history.jsonl', {'resumed_at': core.stamp(),
                    'previous_manifest': json.loads((home/'run-manifest.json').read_text()),
                    'committed_before_resume': len(committed)})
    core.dump(home/'run-manifest.json', metadata)
    shutil.copy2(source/'query-selection-final.json', home/'query-selection-final.json')
    jobs = [(config, embed, 'standalone') for embed in ALL_EMBEDDERS for config in data]
    jobs += [(config, embed, rerank) for rerank in ALL_RERANKERS
             for embed, spec in ALL_EMBEDDERS.items() if spec[-1] for config in data]
    guard.start()
    try:
        for config, embed, rerank in jobs:
            key = f'{config}--{embed}--{rerank}'
            if key in committed:
                continue
            attempt_number = 0
            while True:
                if time.time() >= deadline:
                    raise TimeoutError('Wall-clock budget reached while waiting for production traffic')
                study.out = home
                study.progress('waiting_for_idle', combo=key, committed=len(committed), required_quiet_seconds=30)
                while not guard.ready():
                    if time.time() >= deadline:
                        raise TimeoutError('Wall-clock budget reached while waiting for production traffic')
                    time.sleep(1)
                try:
                    guard.begin()
                except Interference:
                    continue
                attempt_number += 1
                attempt = home/'attempts'/f'{key}--{time.time_ns()}'
                attempt.mkdir(parents=True)
                study.out = attempt
                study.data = {config: data[config]}
                core.base.ROOT = attempt
                core.EMBEDDERS = {embed: ALL_EMBEDDERS[embed]}
                core.RERANKERS = {rerank: ALL_RERANKERS[rerank]} if rerank != 'standalone' else {}
                event('combo_started', combo=key, attempt=str(attempt.relative_to(home)))
                try:
                    if rerank == 'standalone':
                        study.embeddings()
                    else:
                        (attempt/'cache').symlink_to(home/'cache', target_is_directory=True)
                        study.rerank_all()
                    guard.finish()
                    result = core.rows(attempt/'query-results.jsonl')
                    assert len(result) == study.n and len({r['query_id'] for r in result}) == study.n
                    assert all(r['status'] == 'ok' for r in result), 'Non-traffic benchmark error; inspect attempt'
                    if rerank == 'standalone':
                        dest = home/'cache'/embed
                        dest.mkdir(parents=True, exist_ok=True)
                        for file in (attempt/'cache'/embed).iterdir():
                            shutil.copy2(file, dest/file.name)
                    # Only accepted transactions appear in the authoritative result set.
                    committed[key] = {'attempt': str(attempt.relative_to(home)), 'accepted_at': core.stamp(), 'queries': len(result)}
                    core.dump(committed_path, committed)
                    consolidate(home, committed)
                    core.dump(attempt/'validity.json', {'status': 'accepted', 'traffic_epoch': guard.epoch})
                    event('combo_accepted', combo=key, committed=len(committed))
                    study.progress('combo_complete', combo=key, committed=len(committed), total=66)
                    break
                except BaseException as exc:
                    try:
                        guard.check()
                        contaminated = isinstance(exc, Interference)
                    except Interference:
                        contaminated = True
                    core.dump(attempt/'validity.json', {'status': 'invalidated' if contaminated else 'failed', 'reason': str(exc)})
                    if contaminated and isinstance(exc, Exception):
                        event('combo_invalidated', combo=key, reason=str(exc))
                        study.progress('paused_external_traffic', combo=key, committed=len(committed), reason=str(exc))
                        continue
                    raise
        study.out = home
        report, summary = consolidate(home, committed)
        report.confidence()
        assert len(summary) == 66 and all(r['queries'] == study.n and r['errors'] == 0 for r in summary)
        study.progress('complete', summaries=66, query_rows=66*study.n, errors=0)
    except BaseException as exc:
        study.out = home
        core.append(home/'fatal-errors.jsonl', {'time': core.stamp(), 'error': str(exc)})
        study.progress('stopped', reason=str(exc), committed=len(committed))
        raise
    finally:
        guard.close()


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: sys.exit('Termination requested'))
    main()
