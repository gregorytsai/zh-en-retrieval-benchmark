#!/usr/bin/env python3
"""Assemble the authorized NAS delivery locally; never runs inference."""
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

WORK = Path(__file__).resolve().parent.parent
DEST = WORK / 'retrieval-delivery-20261022'
SOURCES = ('embedding-study-20261022', 'bilingual-retrieval-benchmark', 'qwen-reranker-rebuilt')
EXCLUDE_DIRS = {'.venv', '__pycache__', '.git', '.pytest_cache', 'models'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ignore(directory, names):
    return [name for name in names if name in EXCLUDE_DIRS
            or name.endswith(('.gguf', '.pyc', '.safetensors', '.bin'))
            or name == '.DS_Store']


def main():
    workspace = DEST / 'workspace'
    if workspace.exists():
        raise SystemExit('Refusing to overwrite existing delivery workspace')
    workspace.mkdir()
    for source in SOURCES:
        shutil.copytree(WORK / source, workspace / source, ignore=ignore)
        print('Copied ' + source, flush=True)
    study = workspace / 'embedding-study-20261022'
    audit = subprocess.run(['python3', 'audit_v3.py'], cwd=study, capture_output=True, text=True, check=True)
    parsed = json.loads(audit.stdout)
    assert all(parsed['checks'].values()), parsed['checks']
    (DEST / 'audit-results.json').write_text(audit.stdout, encoding='utf-8')
    manifest = json.loads((study / 'run-20261022-v3-guarded/run-manifest.json').read_text())
    commands = {
        'hardware': ['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'],
        'memory_bytes': ['/usr/sbin/sysctl', '-n', 'hw.memsize'],
        'os': ['/usr/bin/sw_vers'],
    }
    observed = {}
    for key, command in commands.items():
        result = subprocess.run(command, capture_output=True, text=True)
        observed[key] = {'stdout': result.stdout.strip(), 'stderr': result.stderr.strip(), 'exit_code': result.returncode}
    environment = {'packaged_at_utc': datetime.now(timezone.utc).isoformat(),
                   'benchmark_manifest': manifest, 'read_only_delivery_time_checks': observed}
    (DEST / 'environment.json').write_text(json.dumps(environment, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    records = []
    for path in sorted(DEST.rglob('*')):
        if path.is_symlink():
            raise RuntimeError('Unexpected symlink: ' + str(path))
        if path.is_file() and path.name not in {'SHA256SUMS', 'package-manifest.json'}:
            records.append({'path': path.relative_to(DEST).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest(path)})
    package = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'scope': list(SOURCES),
               'excluded': ['GGUF/model weights', '.venv', '__pycache__', '.git', 'llama.cpp source and build outputs'],
               'benchmark_status': 'complete and audited', 'files': records,
               'file_count_excluding_manifest_and_checksums': len(records),
               'total_bytes_excluding_manifest_and_checksums': sum(record['bytes'] for record in records)}
    manifest_path = DEST / 'package-manifest.json'
    manifest_path.write_text(json.dumps(package, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    checksum_rows = [(record['sha256'], record['path']) for record in records]
    checksum_rows.append((digest(manifest_path), 'package-manifest.json'))
    (DEST / 'SHA256SUMS').write_text(''.join(f'{sha}  {path}\n' for sha, path in checksum_rows), encoding='utf-8')
    print(json.dumps({'path': str(DEST), 'files': len(checksum_rows) + 1,
                      'bytes': package['total_bytes_excluding_manifest_and_checksums'], 'audit_passed': True}))


if __name__ == '__main__':
    main()
