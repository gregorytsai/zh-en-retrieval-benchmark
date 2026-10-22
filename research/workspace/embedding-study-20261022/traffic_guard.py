"""Fail-closed monitoring of OTHER local llama-server processes, without inference."""
import concurrent.futures
import json
import os
import re
import shlex
import subprocess
import threading
import time
import urllib.request


class Interference(RuntimeError):
    pass


def parse_metrics(text):
    result = {}
    for line in text.splitlines():
        if line.startswith('llamacpp:') and '{' not in line:
            key, value = line.split()[:2]
            result[key.split(':', 1)[1]] = float(value)
    required = {'requests_processing', 'requests_deferred', 'n_decode_total', 'prompt_tokens_total', 'tokens_predicted_total'}
    if not required <= result.keys():
        raise ValueError('Required metrics missing')
    return result


def fetch(url, is_json=True):
    with urllib.request.urlopen(url, timeout=1.5) as response:
        text = response.read().decode()
    return json.loads(text) if is_json else text


def discover():
    output = subprocess.run(['/bin/ps', '-axo', 'pid=,ppid=,command='], check=True,
                            capture_output=True, text=True, timeout=3).stdout
    workers, routers = {}, {}
    for line in output.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        pid, parent, command = fields
        # Only actual server executables, never grep/shell commands mentioning them.
        if not re.match(r'^\S*/?llama-server(?:\s|$)', command):
            continue
        if int(parent) == os.getpid():
            continue
        args = shlex.split(command)
        port = int(args[args.index('--port')+1]) if '--port' in args else 8080
        (routers if '--models-preset' in args or '--models-dir' in args else workers)[int(pid)] = port
    return workers, routers


class TrafficGuard:
    def __init__(self, event, idle_seconds=30, period=1):
        self.event = event
        self.idle_seconds = idle_seconds
        self.period = period
        self.lock = threading.RLock()
        self.poll_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.previous = None
        self.epoch = 0
        self.reason = 'Initializing monitor'
        self.last_activity = time.monotonic()
        self.last_poll = 0.0
        self.server = None
        self.terminating_process = None
        self.terminate_started = None
        self.combo_epoch = None
        self.last_event_reason = None
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=12)
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def worker_snapshot(self, item):
        pid, port = item
        metrics = parse_metrics(fetch(f'http://127.0.0.1:{port}/metrics', False))
        # No /slots: it can wake sleeping workers. /metrics is sleep-safe in this build.
        counters = {k: v for k, v in metrics.items() if k.endswith('_total')}
        return pid, {'port': port, 'counters': counters,
                     'busy': metrics['requests_processing'] > 0 or metrics['requests_deferred'] > 0}

    def snapshot(self):
        workers, routers = discover()
        result = dict(self.pool.map(self.worker_snapshot, workers.items()))
        router_states = {}
        for pid, port in routers.items():
            payload = fetch(f'http://127.0.0.1:{port}/models')
            states = {r['id']: r.get('status', {}).get('value', 'unknown') for r in payload['data']}
            router_states[pid] = states
            if any(v not in ('loaded', 'unloaded', 'sleeping') for v in states.values()):
                raise RuntimeError('Router model transition or unknown state')
        return {'workers': result, 'routers': router_states}

    def poll(self):
        with self.poll_lock:
            reason = None
            try:
                current = self.snapshot()
                if any(v['busy'] for v in current['workers'].values()):
                    reason = 'External requests processing or queued'
                elif self.previous is not None and current != self.previous:
                    reason = 'External inference counter or server topology changed'
                self.previous = current
            except Exception as exc:
                reason = 'Monitoring unavailable: ' + str(exc)
            with self.lock:
                self.last_poll = time.monotonic()
                if (self.terminating_process is not None and
                        self.terminating_process.poll() is None and
                        self.last_poll - self.terminate_started >= 2):
                    try:
                        self.terminating_process.kill()
                    except ProcessLookupError:
                        pass
                if reason:
                    self.last_activity = self.last_poll
                    self.epoch += 1
                    self.reason = reason
                    # This reference is ONLY assigned to a benchmark-owned Popen.
                    if self.server and self.server.process and self.server.process.poll() is None:
                        try:
                            self.server.process.terminate()
                            if self.terminating_process is not self.server.process:
                                self.terminating_process = self.server.process
                                self.terminate_started = self.last_poll
                        except ProcessLookupError:
                            pass
                    if reason != self.last_event_reason:
                        self.event('external_activity', reason=reason)
                    self.last_event_reason = reason
                else:
                    self.reason = None
                    self.last_event_reason = None

    def loop(self):
        while not self.stop_event.is_set():
            self.poll()
            self.stop_event.wait(self.period)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=10)
        self.pool.shutdown(wait=True)

    def ready(self):
        with self.lock:
            return (self.last_poll > 0 and self.reason is None and
                    time.monotonic()-self.last_poll < 5 and
                    time.monotonic()-self.last_activity >= self.idle_seconds)

    def begin(self):
        with self.lock:
            if not self.ready():
                raise Interference('Not yet continuously idle')
            self.combo_epoch = self.epoch

    def check(self):
        with self.lock:
            if (self.combo_epoch != self.epoch or self.reason is not None or
                    time.monotonic()-self.last_poll > 5):
                raise Interference(self.reason or 'External activity occurred during this combination')

    def finish(self):
        # A synchronous end scrape also detects short calls between periodic polls.
        self.poll()
        self.check()

    def bind(self, server):
        with self.lock:
            self.server = server

    def unbind(self):
        with self.lock:
            self.server = None
