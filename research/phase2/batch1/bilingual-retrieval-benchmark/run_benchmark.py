#!/usr/bin/env python3
"""Run a reproducible local bilingual embedding/reranking benchmark.

The script talks only to llama-server instances that it starts itself.  It
chooses an ephemeral loopback port and terminates each child in a finally
block, so an unrelated server (including a user's server on port 8080) is not
modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DEFAULT_BINARY = WORKSPACE / "llama.cpp-qwen38/build-v0.4.1/bin/llama-server"
EMBED_MODELS = {
    "qwen3-embedding-8b-q8_0": WORKSPACE / "Qwen3-Embedding-8B-Q8_0.gguf",
    "qwen3-embedding-4b-q8_0": WORKSPACE / "Qwen3-Embedding-4B.Q8_0.gguf",
}
RERANK_MODELS = {
    "qwen3-reranker-0.6b-q8_0": WORKSPACE / "Qwen3-Reranker-0.6B.Q8_0.gguf",
    "qwen3-reranker-4b-q5_k_m": WORKSPACE / "Qwen3-Reranker-4B.Q5_K_M.gguf",
    "qwen3-reranker-4b-q8_0": WORKSPACE / "Qwen3-Reranker-4B.Q8_0.gguf",
    "bge-reranker-v2-m3-q8_0": WORKSPACE / "bge-reranker-v2-m3-q8_0.gguf",
}
THREADS = int(os.environ.get("BENCHMARK_THREADS", "8"))
EMBED_BATCH = int(os.environ.get("BENCHMARK_EMBED_BATCH", "8"))
QUERY_INSTRUCTION = "Instruct: Given a web search query, retrieve relevant passages\nQuery: "


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def json_request(url: str, payload: dict[str, Any], timeout: float = 900.0) -> tuple[dict[str, Any], float]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = now_ms()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:1000]}") from exc
    elapsed = now_ms() - started
    try:
        return json.loads(raw.decode("utf-8")), elapsed
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON response: {raw[:300]!r}") from exc


def get_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def reserve_port() -> int:
    # An ephemeral port avoids the existing user's llama-server on 8080.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def rss_mb(pid: int) -> float | None:
    try:
        result = subprocess.run(
            ["/bin/ps", "-o", "rss=", "-p", str(pid)],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        value = result.stdout.strip().split()[0]
        # Darwin's ps reports KiB.
        return float(value) / 1024.0
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


class LlamaServer:
    def __init__(self, model: Path, mode: str, binary: Path, run_tag: str, ctx_size: int = 8192):
        self.model = model
        self.mode = mode
        self.binary = binary
        self.run_tag = run_tag
        self.ctx_size = ctx_size
        self.port: int | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self.log_handle: Any = None
        self.load_ms: float | None = None
        self.peak_rss_mb: float | None = None
        self.start_error: str | None = None

    @property
    def url(self) -> str:
        if self.port is None:
            raise RuntimeError("server is not started")
        return f"http://127.0.0.1:{self.port}"

    def _command(self, port: int) -> list[str]:
        command = [
            str(self.binary),
            "-m", str(self.model),
            "--no-webui",
            "--ctx-size", str(self.ctx_size),
            "--parallel", "1",
            "--threads", str(THREADS),
            "--threads-batch", str(THREADS),
            "--batch-size", "512",
            "--ubatch-size", "512",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--log-verbosity", "1",
            "--log-file", str(ROOT / "logs" / f"{self.run_tag}.log"),
        ]
        if self.mode == "embedding":
            command += ["--embedding", "--pooling", "last", "--embd-normalize", "2"]
        elif self.mode == "rerank":
            command += ["--rerank"]
        else:
            raise ValueError(self.mode)
        return command

    def start(self, timeout_s: float = 1200.0) -> float:
        ROOT.joinpath("logs").mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        last_error = "unknown startup error"
        for attempt in range(1, 4):
            port = reserve_port()
            self.port = port
            stdout_path = ROOT / "logs" / f"{self.run_tag}.stdout"
            self.log_handle = stdout_path.open("wb")
            try:
                self.process = subprocess.Popen(
                    self._command(port),
                    stdout=self.log_handle,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    cwd=str(WORKSPACE),
                )
            except OSError as exc:
                last_error = f"spawn: {exc}"
                self.log_handle.close()
                self.process = None
                continue
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                if self.process.poll() is not None:
                    last_error = f"server exited with code {self.process.returncode}"
                    break
                try:
                    health = get_json(self.url + "/health", timeout=2)
                    if health.get("status") == "ok":
                        self.load_ms = (time.perf_counter() - started) * 1000.0
                        self.sample_rss()
                        return self.load_ms
                except Exception as exc:  # startup connection errors are expected
                    last_error = str(exc)
                self.sample_rss()
                time.sleep(1.0)
            self.stop()
        self.start_error = last_error
        raise RuntimeError(f"failed to start {self.model.name}: {last_error}")

    def sample_rss(self) -> None:
        if self.process is None:
            return
        value = rss_mb(self.process.pid)
        if value is not None:
            self.peak_rss_mb = max(self.peak_rss_mb or 0.0, value)

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.sample_rss()
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()


def embed_texts(server: LlamaServer, texts: list[str], model_name: str, batch_size: int = EMBED_BATCH) -> tuple[list[list[float]], list[float], int]:
    vectors: list[list[float]] = []
    timings: list[float] = []
    tokens = 0
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response, elapsed = json_request(server.url + "/v1/embeddings", {"model": model_name, "input": batch})
        data = response.get("data")
        if not isinstance(data, list) or len(data) != len(batch):
            raise RuntimeError(f"embedding response returned {len(data) if isinstance(data, list) else type(data)} items for {len(batch)} inputs")
        data = sorted(data, key=lambda item: int(item.get("index", 0)))
        vectors.extend(l2_normalize([float(x) for x in item["embedding"]]) for item in data)
        timings.append(elapsed)
        usage = response.get("usage", {})
        tokens += int(usage.get("total_tokens", 0))
        server.sample_rss()
    return vectors, timings, tokens


def rank_embedding(query_vector: list[float], corpus_vectors: list[list[float]]) -> list[int]:
    return sorted(range(len(corpus_vectors)), key=lambda idx: (-dot(query_vector, corpus_vectors[idx]), idx))


def rerank(server: LlamaServer, query: str, documents: list[str], model_name: str) -> tuple[list[int], float, int, float | None]:
    response, elapsed = json_request(
        server.url + "/v1/rerank",
        {"model": model_name, "query": query, "documents": documents},
    )
    raw_results = response.get("results")
    if not isinstance(raw_results, list):
        raise RuntimeError(f"rerank response has no results: {str(response)[:500]}")
    scored = []
    for result in raw_results:
        index = int(result["index"])
        score = float(result.get("relevance_score", result.get("score")))
        scored.append((index, score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    order = [index for index, _ in scored]
    usage = response.get("usage", {})
    tokens = int(usage.get("total_tokens", 0))
    server.sample_rss()
    return order, elapsed, len(raw_results), float(tokens)


def position(order: list[str], target: str) -> int | None:
    try:
        return order.index(target) + 1
    except ValueError:
        return None


def metrics_for(order: list[str], target: str, candidate_order: list[str]) -> dict[str, float | int | None]:
    final_rank = position(order, target)
    candidate_rank = position(candidate_order, target)
    return {
        "final_rank": final_rank,
        "candidate_rank": candidate_rank,
        "top1": int(final_rank == 1),
        "mrr10": (1.0 / final_rank) if final_rank is not None and final_rank <= 10 else 0.0,
        # Exactly one judged relevant document means ideal DCG@10 is 1.
        "ndcg10": (1.0 / math.log2(final_rank + 1)) if final_rank is not None and final_rank <= 10 else 0.0,
        "recall10": int(final_rank is not None and final_rank <= 10),
        "candidate_recall": int(candidate_rank is not None),
    }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def aggregate(query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(field: str) -> float | None:
        values = [float(row[field]) for row in query_rows if row.get(field) is not None]
        return statistics.mean(values) if values else None

    result: dict[str, Any] = {
        "queries": len(query_rows),
        "top1": mean("top1"),
        "mrr10": mean("mrr10"),
        "ndcg10": mean("ndcg10"),
        "recall10": mean("recall10"),
        "candidate_recall": mean("candidate_recall"),
        "query_latency_ms_mean": mean("total_query_latency_ms"),
        "query_latency_ms_p50": percentile([float(r["total_query_latency_ms"]) for r in query_rows], 0.50),
        "query_latency_ms_p95": percentile([float(r["total_query_latency_ms"]) for r in query_rows], 0.95),
    }
    directions = sorted({str(row["direction"]) for row in query_rows})
    for direction in directions:
        subset = [row for row in query_rows if row["direction"] == direction]
        for key in ("top1", "mrr10", "ndcg10", "recall10", "candidate_recall"):
            values = [float(row[key]) for row in subset]
            result[f"{direction}_{key}"] = statistics.mean(values) if values else None
    rerank_values = [float(row["rerank_latency_ms"]) for row in query_rows if row.get("rerank_latency_ms") is not None]
    if rerank_values:
        result["rerank_latency_ms_mean"] = statistics.mean(rerank_values)
        result["rerank_latency_ms_p50"] = percentile(rerank_values, 0.50)
        result["rerank_latency_ms_p95"] = percentile(rerank_values, 0.95)
        result["rerank_docs_per_sec"] = sum(int(row["candidate_count"]) for row in query_rows) / (sum(rerank_values) / 1000.0)
    else:
        result["rerank_latency_ms_mean"] = None
        result["rerank_latency_ms_p50"] = None
        result["rerank_latency_ms_p95"] = None
        result["rerank_docs_per_sec"] = None
    return result


def row_error(base: dict[str, Any], error: str) -> dict[str, Any]:
    row = dict(base)
    row.update({"status": "error", "error": error})
    return row


def base_summary(embedding_name: str, reranker_name: str | None, candidate_k: int | None) -> dict[str, Any]:
    return {
        "status": "ok",
        "embedding": embedding_name,
        "reranker": reranker_name or "standalone",
        "candidate_k_requested": candidate_k,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(path: Path, manifest: dict[str, Any], summaries: list[dict[str, Any]], errors: list[dict[str, Any]], binary_version: str) -> None:
    lines = [
        "# Bilingual retrieval benchmark report",
        "",
        "This report is generated by `run_benchmark.py` from the fixed corpus and query files. Each query has exactly one judged relevant anchor document. The corpus contains 200 documents (100 English and 100 Chinese) across 10 topics; there are 30 queries: 10 zh->en, 10 en->en, and 10 zh->zh.",
        "",
        "## Method",
        "",
        f"- Server binary: `{binary_version}`",
        "- Embedding query input: `Instruct: Given a web search query, retrieve relevant passages\\nQuery: ...`; document input is the document text without an instruction.",
        "- Embeddings use `--pooling last --embd-normalize 2`; retrieval is cosine (dot product after local normalization).",
        "- Reranking sends the complete candidate list in one `/v1/rerank` request; requested candidate sizes are 50 and 100.",
        "- To measure the requested language direction without rewarding a language shortcut, each query is ranked against the 100-document target-language pool: zh->en and en->en use English-only documents; zh->zh uses Chinese-only documents. Standalone rows use embedding ranking over that same pool and do not call a reranker.",
        "- `candidate_recall` is the fraction of queries whose judged document entered the candidate set; `recall10` is final top-10 recall.",
        "- Latencies are wall-clock client request times on one local server slot; they include JSON transfer and inference. Model-load latency is from process spawn until `/health` is ready.",
        "- Hardware is the host reported by `platform.platform()`/`platform.machine()` and the CPU-only llama.cpp build; no GPU device was available to this run.",
        "- Limitation: this is a controlled synthetic corpus with one judged anchor per query, so the scores compare configurations under these labels and are not a substitute for a production relevance set or human judgments.",
        "",
        "## Summary",
        "",
        "| embedding | reranker | k | top-1 | MRR@10 | nDCG@10 | Recall@10 | candidate recall | query ms mean/p95 | rerank ms mean/p95 | load ms | peak RSS MB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row.get('embedding','')} | {row.get('reranker','')} | {row.get('candidate_k_requested') or 'all'} | "
            f"{fmt(row.get('top1'), 3)} | {fmt(row.get('mrr10'), 3)} | {fmt(row.get('ndcg10'), 3)} | "
            f"{fmt(row.get('recall10'), 3)} | {fmt(row.get('candidate_recall'), 3)} | "
            f"{fmt(row.get('query_latency_ms_mean'), 1)}/{fmt(row.get('query_latency_ms_p95'), 1)} | "
            f"{fmt(row.get('rerank_latency_ms_mean'), 1)}/{fmt(row.get('rerank_latency_ms_p95'), 1)} | "
            f"{fmt(row.get('model_load_ms'), 1)} | {fmt(row.get('peak_rss_mb'), 1)} |"
        )
    lines += ["", "## Direction breakdown", "", "| embedding | reranker | k | direction | top-1 | MRR@10 | nDCG@10 | Recall@10 | candidate recall |", "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        for direction in ("zh->en", "en->en", "zh->zh"):
            if f"{direction}_top1" not in row:
                continue
            lines.append(
                f"| {row['embedding']} | {row['reranker']} | {row.get('candidate_k_requested') or 'all'} | {direction} | "
                f"{fmt(row.get(direction + '_top1'), 3)} | {fmt(row.get(direction + '_mrr10'), 3)} | "
                f"{fmt(row.get(direction + '_ndcg10'), 3)} | {fmt(row.get(direction + '_recall10'), 3)} | {fmt(row.get(direction + '_candidate_recall'), 3)} |"
            )
    lines += ["", "## Reproducibility", "", f"- Corpus/query manifest: `{json.dumps(manifest, ensure_ascii=False, sort_keys=True)}`", "- Reranker failures or unavailable combinations are recorded in `errors.jsonl` and the CSV/JSON summaries."]
    if errors:
        lines += ["", "## Errors", ""]
        for error in errors:
            lines.append(f"- `{error.get('embedding')}` + `{error.get('reranker')}` k={error.get('candidate_k_requested')}: {error.get('error')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    ROOT.joinpath("logs").mkdir(parents=True, exist_ok=True)
    if not (ROOT / "corpus.jsonl").exists() or not (ROOT / "queries.jsonl").exists():
        subprocess.run([sys.executable, str(ROOT / "generate_corpus.py")], check=True)
    corpus = load_jsonl(ROOT / "corpus.jsonl")
    queries = load_jsonl(ROOT / "queries.jsonl")
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    corpus_hash = sha256_file(ROOT / "corpus.jsonl")
    query_hash = sha256_file(ROOT / "queries.jsonl")
    binary = Path(args.binary).resolve()
    try:
        version = subprocess.check_output([str(binary), "--version"], text=True, stderr=subprocess.STDOUT, timeout=15).strip().replace("\n", "; ")
    except Exception as exc:
        version = f"version unavailable: {exc}"
    run_meta = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "binary": str(binary),
        "binary_version": version,
        "threads": THREADS,
        "embed_batch": EMBED_BATCH,
        "corpus_sha256": corpus_hash,
        "queries_sha256": query_hash,
        "corpus_documents": len(corpus),
        "queries": len(queries),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "hardware_note": "CPU-only llama.cpp build; --list-devices reported no usable GPU device.",
    }
    (ROOT / "run_metadata.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    corpus_texts = [doc["text"] for doc in corpus]
    corpus_ids = [doc["id"] for doc in corpus]
    query_texts = [QUERY_INSTRUCTION + query["query"] for query in queries]
    all_summary: list[dict[str, Any]] = []
    all_query_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    server_events: list[dict[str, Any]] = []

    def save_progress() -> None:
        write_rows(ROOT / "summary.jsonl", all_summary)
        write_rows(ROOT / "query_results.jsonl", all_query_rows)
        write_rows(ROOT / "errors.jsonl", errors)
        write_rows(ROOT / "server_events.jsonl", server_events)
        write_csv(ROOT / "summary.csv", all_summary)

    for embedding_name, embedding_model in EMBED_MODELS.items():
        print(f"[benchmark] embedding {embedding_name} ({embedding_model.name})", flush=True)
        if not embedding_model.exists():
            error = row_error(base_summary(embedding_name, None, None), f"missing model: {embedding_model}")
            errors.append(error)
            save_progress()
            continue
        embed_server = LlamaServer(embedding_model, "embedding", binary, f"embedding-{embedding_name}", ctx_size=args.ctx_size)
        embed_metrics = {"load_ms": None, "corpus_ms": None, "query_ms": None, "tokens": 0, "dim": None, "peak_rss_mb": None, "status": "ok"}
        try:
            with embed_server:
                event = {"stage": "embedding", "model": embedding_name, "port": embed_server.port, "load_ms": embed_server.load_ms, "status": "ok"}
                server_events.append(event)
                corpus_vectors, corpus_timings, corpus_tokens = embed_texts(embed_server, corpus_texts, embedding_model.name)
                # Use one request per query so query latency is directly
                # measured rather than estimated from a batched request.
                query_vectors, query_timings, query_tokens = embed_texts(embed_server, query_texts, embedding_model.name, batch_size=1)
                embed_server.sample_rss()
                embed_metrics.update({
                    "load_ms": embed_server.load_ms,
                    "corpus_ms": sum(corpus_timings),
                    "query_ms": sum(query_timings),
                    "tokens": corpus_tokens + query_tokens,
                    "dim": len(corpus_vectors[0]),
                    "peak_rss_mb": embed_server.peak_rss_mb,
                    "corpus_docs_per_sec": len(corpus_vectors) / (sum(corpus_timings) / 1000.0),
                    "query_per_sec": len(query_vectors) / (sum(query_timings) / 1000.0),
                })
                print(f"[benchmark] embedded {len(corpus_vectors)} docs + {len(query_vectors)} queries; dim={embed_metrics['dim']} load_ms={embed_metrics['load_ms']:.1f}", flush=True)
        except Exception as exc:
            embed_metrics["status"] = "error"
            embed_metrics["error"] = str(exc)
            errors.append(row_error(base_summary(embedding_name, None, None), f"embedding: {exc}"))
            save_progress()
            continue

        # Standalone embedding-only baseline: rank the whole corpus.
        standalone_rows: list[dict[str, Any]] = []
        for qidx, query in enumerate(queries):
            started = now_ms()
            eligible_idx = [idx for idx, doc in enumerate(corpus) if doc["language"] == query["target_language"]]
            eligible_vectors = [corpus_vectors[idx] for idx in eligible_idx]
            order_local = rank_embedding(query_vectors[qidx], eligible_vectors)
            order = [corpus_ids[eligible_idx[idx]] for idx in order_local]
            query_elapsed = now_ms() - started + (query_timings[qidx] if qidx < len(query_timings) else 0.0)
            metrics = metrics_for(order, query["target_doc_id"], order)
            standalone_rows.append({
                "embedding": embedding_name,
                "reranker": "standalone",
                "candidate_k_requested": None,
                "candidate_count": len(eligible_idx),
                "query_id": query["id"],
                "direction": query["direction"],
                "topic": query["topic"],
                "target_doc_id": query["target_doc_id"],
                "total_query_latency_ms": query_elapsed,
                "rerank_latency_ms": None,
                **metrics,
            })
        summary = {**base_summary(embedding_name, None, None), **aggregate(standalone_rows), **{
            "candidate_k_actual": len(eligible_idx),
            "model_load_ms": embed_metrics["load_ms"],
            "corpus_embedding_ms": embed_metrics["corpus_ms"],
            "query_embedding_ms_total": embed_metrics["query_ms"],
            "embedding_dim": embed_metrics["dim"],
            "embedding_tokens": embed_metrics["tokens"],
            "corpus_docs_per_sec": embed_metrics.get("corpus_docs_per_sec"),
            "query_per_sec": embed_metrics.get("query_per_sec"),
            "peak_rss_mb": embed_metrics["peak_rss_mb"],
        }}
        all_summary.append(summary)
        all_query_rows.extend(standalone_rows)
        save_progress()

        # Every reranker is loaded independently for each embedding so the
        # 2 x 4 x 2 matrix is measured rather than inferred.
        for reranker_name, reranker_model in RERANK_MODELS.items():
            print(f"[benchmark] reranker {embedding_name} + {reranker_name}", flush=True)
            if not reranker_model.exists():
                for requested_k in (50, 100):
                    error = row_error(base_summary(embedding_name, reranker_name, requested_k), f"missing model: {reranker_model}")
                    errors.append(error)
                save_progress()
                continue
            rerank_server = LlamaServer(reranker_model, "rerank", binary, f"rerank-{embedding_name}-{reranker_name}", ctx_size=args.ctx_size)
            try:
                with rerank_server:
                    server_events.append({"stage": "rerank", "embedding": embedding_name, "model": reranker_name, "port": rerank_server.port, "load_ms": rerank_server.load_ms, "status": "ok"})
                    for requested_k in (50, 100):
                        rows: list[dict[str, Any]] = []
                        for qidx, query in enumerate(queries):
                            retrieve_started = now_ms()
                            eligible_idx = [idx for idx, doc in enumerate(corpus) if doc["language"] == query["target_language"]]
                            eligible_vectors = [corpus_vectors[idx] for idx in eligible_idx]
                            retrieval_local = rank_embedding(query_vectors[qidx], eligible_vectors)
                            candidate_idx = [eligible_idx[idx] for idx in retrieval_local[:requested_k]]
                            candidate_ids = [corpus_ids[idx] for idx in candidate_idx]
                            candidate_texts = [corpus[idx]["text"] for idx in candidate_idx]
                            retrieval_ms = now_ms() - retrieve_started
                            try:
                                rerank_order_local, rerank_ms, returned_count, rerank_tokens = rerank(
                                    rerank_server,
                                    query["query"],
                                    candidate_texts,
                                    reranker_model.name,
                                )
                                final_ids = [candidate_ids[idx] for idx in rerank_order_local if 0 <= idx < len(candidate_ids)]
                                metrics = metrics_for(final_ids, query["target_doc_id"], candidate_ids)
                                row = {
                                    "embedding": embedding_name,
                                    "reranker": reranker_name,
                                    "candidate_k_requested": requested_k,
                                    "candidate_count": len(candidate_ids),
                                    "rerank_returned_count": returned_count,
                                    "rerank_tokens": rerank_tokens,
                                    "query_id": query["id"],
                                    "direction": query["direction"],
                                    "topic": query["topic"],
                                    "target_doc_id": query["target_doc_id"],
                                    "retrieval_ms": retrieval_ms,
                                    "rerank_latency_ms": rerank_ms,
                                    "total_query_latency_ms": (query_timings[qidx] if qidx < len(query_timings) else 0.0) + retrieval_ms + rerank_ms,
                                    "status": "ok",
                                    **metrics,
                                }
                                rows.append(row)
                            except Exception as exc:
                                row = {
                                    "embedding": embedding_name,
                                    "reranker": reranker_name,
                                    "candidate_k_requested": requested_k,
                                    "candidate_count": len(candidate_ids),
                                    "query_id": query["id"],
                                    "direction": query["direction"],
                                    "topic": query["topic"],
                                    "target_doc_id": query["target_doc_id"],
                                    "status": "error",
                                    "error": str(exc),
                                }
                                errors.append(row)
                                rows.append(row)
                                # One request can fail due to an incompatible
                                # endpoint. Continue to preserve missing rows.
                        good_rows = [row for row in rows if row.get("status") == "ok"]
                        if good_rows:
                            summary = {**base_summary(embedding_name, reranker_name, requested_k), **aggregate(good_rows), **{
                                "candidate_k_actual": min(requested_k, len(corpus)),
                                "model_load_ms": rerank_server.load_ms,
                                "corpus_embedding_ms": embed_metrics["corpus_ms"],
                                "query_embedding_ms_total": embed_metrics["query_ms"],
                                "embedding_dim": embed_metrics["dim"],
                                "embedding_tokens": embed_metrics["tokens"],
                                "corpus_docs_per_sec": embed_metrics.get("corpus_docs_per_sec"),
                                "query_per_sec": embed_metrics.get("query_per_sec"),
                                "peak_rss_mb": max(embed_metrics["peak_rss_mb"] or 0.0, rerank_server.peak_rss_mb or 0.0),
                                "actual_returned_mean": statistics.mean(float(row["rerank_returned_count"]) for row in good_rows),
                            }}
                            all_summary.append(summary)
                            print(f"[benchmark] finished {embedding_name} + {reranker_name} k={requested_k}: top1={summary.get('top1')} mrr10={summary.get('mrr10')} rerank_ms={summary.get('rerank_latency_ms_mean')}", flush=True)
                        else:
                            errors.append(row_error(base_summary(embedding_name, reranker_name, requested_k), "all rerank requests failed"))
                        all_query_rows.extend(rows)
                        save_progress()
            except Exception as exc:
                for requested_k in (50, 100):
                    errors.append(row_error(base_summary(embedding_name, reranker_name, requested_k), f"server: {exc}"))
                save_progress()

    # Final outputs are written even if some model combinations failed.
    manifest_out = dict(manifest)
    manifest_out.update({"corpus_sha256_observed": corpus_hash, "queries_sha256_observed": query_hash})
    (ROOT / "manifest.observed.json").write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run_meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    run_meta["summary_rows"] = len(all_summary)
    run_meta["error_rows"] = len(errors)
    (ROOT / "run_metadata.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(ROOT / "report.md", manifest_out, all_summary, errors, version)
    save_progress()
    print(json.dumps({"summary_rows": len(all_summary), "errors": len(errors), "report": str(ROOT / "report.md")}, ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=str(DEFAULT_BINARY), help="llama-server binary")
    parser.add_argument("--ctx-size", type=int, default=8192, help="server context size")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except KeyboardInterrupt:
        raise SystemExit(130)
