#!/usr/bin/env python3
"""Diagnose llama.cpp Metal/offload behavior for a rebuilt Qwen reranker.

The script keeps a separate stdout/stderr log and measures the same warmed
small rerank batch with explicit --gpu-layers 0 and 99.  It is intentionally
stdlib-only so the result does not depend on the benchmark virtualenv.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BINARY = (Path(__file__).resolve().parents[1] / 'llama.cpp-qwen38/build-v0.4.1/bin/llama-server')
MODEL = ROOT / "Qwen3-Reranker-0.6B-Q8_0.gguf"
LOG_DIR = ROOT / "offload-diagnostics"


def request(url: str, payload: dict, timeout: float = 120.0) -> tuple[dict, float]:
    body = json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.loads(response.read().decode())
    return result, (time.perf_counter() - started) * 1000.0


def health(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/health", timeout=2) as response:
            return json.loads(response.read().decode()).get("status") == "ok"
    except Exception:
        return False


def free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run(layers: int) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    port = free_port()
    log_path = LOG_DIR / f"reranker-gpu-layers-{layers}.log"
    command = [
        str(BINARY), "-m", str(MODEL), "--no-webui", "--offline",
        "--ctx-size", "2048", "--parallel", "1", "--threads", "8",
        "--threads-batch", "8", "--batch-size", "512", "--ubatch-size", "512",
        "--host", "127.0.0.1", "--port", str(port), "--rerank",
        "--gpu-layers", str(layers), "--log-verbosity", "5",
        "--log-colors", "off", "--log-timestamps",
    ]
    started = time.perf_counter()
    with log_path.open("wb") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        url = f"http://127.0.0.1:{port}"
        load_ms = None
        deadline = time.time() + 300
        while time.time() < deadline:
            if process.poll() is not None:
                break
            if health(url):
                load_ms = (time.perf_counter() - started) * 1000.0
                break
            time.sleep(0.5)
        batch = [
            "A battery management system monitors cell voltage and temperature.",
            "The espresso machine uses a thermoblock to heat water quickly.",
            "Metal shaders accelerate matrix multiplication on Apple silicon.",
            "A battery charger regulates current during the constant-current phase.",
            "Coffee beans are roasted before they are ground for espresso.",
        ]
        latencies: list[float] = []
        error = None
        if load_ms is not None:
            # Warm-up request, then measure the same batch three times.
            try:
                request(url + "/v1/rerank", {"model": MODEL.name, "query": "How does a battery charger work?", "documents": batch})
                for _ in range(3):
                    _, elapsed = request(url + "/v1/rerank", {"model": MODEL.name, "query": "How does a battery charger work?", "documents": batch})
                    latencies.append(elapsed)
            except Exception as exc:
                error = str(exc)
        else:
            error = f"server exited with code {process.returncode}"
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        return {
            "gpu_layers": layers,
            "command": command,
            "log": str(log_path),
            "load_ms": load_ms,
            "latencies_ms": latencies,
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "error": error,
            "returncode": process.returncode,
        }


def main() -> int:
    if not MODEL.exists():
        raise SystemExit(f"missing model: {MODEL}")
    results = [run(0), run(99)]
    output = {
        "binary": str(BINARY),
        "model": str(MODEL),
        "results": results,
        "note": "Interpret --gpu-layers 99 only as GPU use when the corresponding log shows Metal device allocation/offloaded layers; a Metal-linked binary alone is insufficient.",
    }
    (LOG_DIR / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
