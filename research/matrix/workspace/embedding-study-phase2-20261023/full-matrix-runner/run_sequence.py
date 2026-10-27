#!/usr/bin/env python3
"""Run the fixed B01–B07 schedule serially with guarded per-batch gates."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from matrix_spec import BATCHES

HERE = Path(__file__).resolve().parent
CHILD = None


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def update_plan(plan, batch, state, home, started=None):
    """Publish actual accepted counts; only batch_progress may publish completion."""
    committed = read(home / 'committed.json', {})
    rows = sum(c.get('queries', 0) for c in committed.values())
    label = {'pilot': 'Pilot 驗證中', 'official': '正式執行中', 'blocked': '停止待處理'}.get(state, state)
    time_label = datetime.fromtimestamp(started, ZoneInfo('Asia/Taipei')).strftime('%m/%d %H:%M') if started else '—'
    # Pilot counts belong in status.json, not official completion columns.
    summaries_label, rows_label = (f'{len(committed)}/24', f'{rows}/10452') if state == 'official' else ('0/24', '0/10452')
    if state == 'blocked' and home.name == 'official':
        summaries_label, rows_label = f'{len(committed)}/24', f'{rows}/10452'
    with plan.with_suffix('.progress.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = plan.read_text()
        lines = text.splitlines()
        matches = [i for i, line in enumerate(lines) if line.startswith(f'| {batch} |')]
        if len(matches) != 1:
            raise ValueError('Missing or ambiguous batch progress row')
        if '稽核通過' in lines[matches[0]]:
            return
        lines[matches[0]] = f'| {batch} | {dict(BATCHES)[batch]} | {label} | {summaries_label} | {rows_label} | {time_label} 起 | {home} |'
        tmp = plan.with_suffix('.progress.tmp')
        tmp.write_text('\n'.join(lines) + '\n')
        tmp.replace(plan)


def stage_command(args, batch, kind, resume):
    command = [sys.executable, str(HERE / 'run_full_matrix.py'), '--workspace', str(args.workspace),
               '--output', str(args.output), '--batch', batch, '--execute', '--plan-path', str(args.plan)]
    if kind == 'pilot':
        command.append('--pilot')
    if resume:
        command.append('--resume')
    return command


def stop(signum, frame):
    global CHILD
    if CHILD is not None and CHILD.poll() is None:
        CHILD.terminate()
        try:
            CHILD.wait(timeout=15)
        except subprocess.TimeoutExpired:
            # Do not SIGKILL a runner while it still owns a measurement child.
            # Its signal handler/finally is responsible for child cleanup.
            pass
    raise SystemExit(128 + signum)


def run_stage(args, batch, kind, state):
    global CHILD
    home = args.output / batch / kind
    progress = read(home / 'progress.json', {})
    if kind == 'official' and (home / 'batch-completion-audit.json').is_file():
        from batch_progress import update_completed
        update_completed(batch, home, args.workspace, args.plan)
        return
    if kind == 'pilot' and progress.get('phase') == 'pilot_complete':
        return  # Official CLI independently audits this pilot and its forecast.
    manifest = read(home / 'run-manifest.json')
    resume = manifest is not None
    if resume and time.time() >= manifest['deadline_epoch']:
        raise TimeoutError(f'{batch}/{kind} original deadline expired; cannot reset it')
    if manifest and progress.get('phase') == 'blocked_preflight':
        raise RuntimeError(f'{batch}/{kind} blocked by token preflight; see artifacts')
    command = stage_command(args, batch, kind, resume)
    state.update(batch=batch, embedder=dict(BATCHES)[batch], stage=kind,
                 phase='launching', updated_at=stamp(), run_dir=str(home))
    dump(args.output / 'sequence-status.json', state)
    with (args.output / f'{batch}-{kind}-console.log').open('a') as log:
        CHILD = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
        state.update(phase='running', child_pid=CHILD.pid, updated_at=stamp())
        dump(args.output / 'sequence-status.json', state)
        last_doc_update = 0
        while CHILD.poll() is None:
            progress = read(home / 'progress.json', {})
            manifest = read(home / 'run-manifest.json', {})
            state.update(progress=progress, deadline_epoch=manifest.get('deadline_epoch'), updated_at=stamp())
            dump(args.output / 'sequence-status.json', state)
            if time.monotonic() - last_doc_update >= 30:
                update_plan(args.plan, batch, kind, home, manifest.get('original_started_epoch'))
                last_doc_update = time.monotonic()
            time.sleep(5)
        code = CHILD.returncode
        CHILD = None
    if code:
        update_plan(args.plan, batch, 'blocked', home, manifest.get('original_started_epoch'))
        raise RuntimeError(f'{batch}/{kind} exited {code}; preserve run and inspect logs before resume')
    terminal = read(home / 'progress.json', {}).get('phase')
    if terminal != ('pilot_complete' if kind == 'pilot' else 'complete'):
        raise RuntimeError(f'{batch}/{kind} has no expected completion marker: {terminal}')
    if kind == 'official' and not (home / 'batch-completion-audit.json').is_file():
        raise RuntimeError(f'{batch} has no successful completion audit')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    args.workspace = args.workspace.resolve()
    project = args.workspace / 'embedding-study-phase2-20261023'
    args.output = (args.output or project / 'run-full-matrix').resolve()
    args.plan = (args.plan or project / 'FULL_MATRIX_PLAN_ZH_TW.md').resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (project / '.full-matrix-sequence.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another full-matrix sequence controller is active') from exc
        previous = read(args.output / 'sequence-status.json')
        if previous and not args.resume:
            raise RuntimeError('Sequence state already exists; use --resume without changing deadlines')
        if previous and previous.get('child_pid') and previous.get('phase') == 'running':
            try:
                os.kill(int(previous['child_pid']), 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('The previous runner may still be alive; inspect it before resuming the controller')
        state = previous or {'created_at': stamp(), 'completed_batches': [], 'schedule': list(BATCHES)}
        state.update(controller_pid=os.getpid(), phase='running', updated_at=stamp())
        dump(args.output / 'sequence-status.json', state)
        try:
            for batch, embedder in BATCHES:
                if batch in state['completed_batches']:
                    home = args.output / batch / 'official'
                    from batch_progress import update_completed
                    update_completed(batch, home, args.workspace, args.plan)
                    continue
                run_stage(args, batch, 'pilot', state)
                run_stage(args, batch, 'official', state)
                state['completed_batches'].append(batch)
                state.update(updated_at=stamp(), phase='batch_complete')
                dump(args.output / 'sequence-status.json', state)
            state.update(phase='complete', updated_at=stamp(), expected_summaries=168, expected_rows=73164)
            dump(args.output / 'sequence-status.json', state)
        except BaseException as exc:
            state.update(phase='stopped', error=str(exc), updated_at=stamp())
            dump(args.output / 'sequence-status.json', state)
            raise


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    main()
