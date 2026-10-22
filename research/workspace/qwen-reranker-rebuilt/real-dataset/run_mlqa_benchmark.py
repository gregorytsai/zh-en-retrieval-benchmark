#!/usr/bin/env python3
"""Run a small real MLQARetrieval comparison through llama-server/Metal.

The input is the deterministic subset produced by prepare_mlqa_retrieval.py.
Each direction keeps its complete validation corpus and official qrels for 30
queries.  The default run compares one Qwen embedding model as standalone,
with the rebuilt Qwen reranker, and with the existing BGE reranker at k=50 and
k=100.  Use --embeddings both to add the local Qwen 4B embedder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any


BENCHMARK = (Path(__file__).resolve().parents[2] / 'bilingual-retrieval-benchmark')
REAL = (Path(__file__).resolve().parents[2] / 'qwen-reranker-rebuilt/real-dataset')
OUTPUT = REAL / "benchmark-metal"
sys.path.insert(0, str(BENCHMARK))
import run_benchmark as base  # noqa: E402


WORKSPACE = (Path(__file__).resolve().parents[2] / '')
REBUILT = (Path(__file__).resolve().parents[2] / 'qwen-reranker-rebuilt')
base.ROOT = OUTPUT
base.EMBED_MODELS = {
    "qwen3-embedding-8b-q8_0": WORKSPACE / "Qwen3-Embedding-8B-Q8_0.gguf",
    "qwen3-embedding-4b-q8_0": WORKSPACE / "Qwen3-Embedding-4B.Q8_0.gguf",
}
base.RERANK_MODELS = {
    "qwen3-reranker-0.6b-rebuilt-q8_0": REBUILT / "Qwen3-Reranker-0.6B-Q8_0.gguf",
    "bge-reranker-v2-m3-q8_0": WORKSPACE / "bge-reranker-v2-m3-q8_0.gguf",
}
_original_command = base.LlamaServer._command


def _metal_command(self: base.LlamaServer, port: int) -> list[str]:
    command = _original_command(self, port)
    # Real MLQA passages can make a query-document pair exceed the base
    # benchmark's 512-token physical batch.  4096 covers the observed
    # maximum (~1203 tokens) while leaving the 8192-token context unchanged.
    command[command.index("--batch-size") + 1] = "4096"
    command[command.index("--ubatch-size") + 1] = "4096"
    return command + ["--gpu-layers", "99"]


base.LlamaServer._command = _metal_command


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_relevant(order: list[str], relevant: set[str]) -> int | None:
    for rank, doc_id in enumerate(order, 1):
        if doc_id in relevant:
            return rank
    return None


def metrics(final_order: list[str], candidate_order: list[str], relevant: set[str]) -> dict[str, float | int | None]:
    final_rank = first_relevant(final_order, relevant)
    candidate_rank = first_relevant(candidate_order, relevant)
    hits10 = [rank for rank, doc_id in enumerate(final_order[:10], 1) if doc_id in relevant]
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(10, len(relevant)) + 1))
    dcg = sum(1.0 / math.log2(rank + 1) for rank in hits10)
    return {
        "final_rank": final_rank,
        "candidate_rank": candidate_rank,
        "top1": int(final_rank == 1),
        "mrr10": (1.0 / final_rank) if final_rank is not None and final_rank <= 10 else 0.0,
        "ndcg10": (dcg / ideal) if ideal else 0.0,
        "recall10": int(bool(hits10)),
        "candidate_recall": int(candidate_rank is not None),
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(field: str) -> float | None:
        vals = [float(row[field]) for row in rows if row.get(field) is not None]
        return statistics.mean(vals) if vals else None

    result: dict[str, Any] = {
        "queries": len(rows),
        "top1": mean("top1"),
        "mrr10": mean("mrr10"),
        "ndcg10": mean("ndcg10"),
        "recall10": mean("recall10"),
        "candidate_recall": mean("candidate_recall"),
        "query_latency_ms_mean": mean("total_query_latency_ms"),
        "query_latency_ms_p50": percentile([float(row["total_query_latency_ms"]) for row in rows], 0.5),
        "query_latency_ms_p95": percentile([float(row["total_query_latency_ms"]) for row in rows], 0.95),
    }
    for direction in sorted({str(row["direction"]) for row in rows}):
        part = [row for row in rows if row["direction"] == direction]
        for field in ("top1", "mrr10", "ndcg10", "recall10", "candidate_recall"):
            result[f"{direction}_{field}"] = statistics.mean(float(row[field]) for row in part)
    rerank_ms = [float(row["rerank_latency_ms"]) for row in rows if row.get("rerank_latency_ms") is not None]
    result["rerank_latency_ms_mean"] = statistics.mean(rerank_ms) if rerank_ms else None
    result["rerank_latency_ms_p50"] = percentile(rerank_ms, 0.5)
    result["rerank_latency_ms_p95"] = percentile(rerank_ms, 0.95)
    result["rerank_docs_per_sec"] = (sum(int(row["candidate_count"]) for row in rows) / (sum(rerank_ms) / 1000.0)) if rerank_ms else None
    return result


def save(output: Path, summaries: list[dict[str, Any]], query_rows: list[dict[str, Any]], errors: list[dict[str, Any]], events: list[dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in summaries), encoding="utf-8")
    (output / "query_results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in query_rows), encoding="utf-8")
    (output / "errors.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in errors), encoding="utf-8")
    (output / "server_events.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in events), encoding="utf-8")
    if summaries:
        fields: list[str] = []
        for row in summaries:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summaries)


def write_report(path: Path, metadata: dict[str, Any], summaries: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    lines = [
        "# MLQARetrieval real-data benchmark",
        "",
        "This is a real, qrels-backed MTEB MLQARetrieval validation subset, not the synthetic 200-document benchmark. Each direction retains its complete validation corpus and 30 deterministic query IDs selected by sorted ID; qrels are copied from the official source.",
        "",
        "## Method",
        "",
        "- Dataset: `mteb/MLQARetrieval`, license `CC-BY-SA-3.0`, source revision `" + str(metadata["dataset_revision"]) + "`.",
        "- Config orientation is `corpus-language-query-language`: `eng-zho` is Chinese query to English corpus, `zho-zho` Chinese to Chinese, and `eng-eng` English to English.",
        "- Embeddings use Qwen instruction-prefixed queries, `--pooling last --embd-normalize 2`, and cosine/dot ranking over the complete validation corpus for that config.",
        "- Reranking sends each retrieved candidate list in one request; k is 50 or 100. Standalone is the embedding-only baseline.",
        "- Server command explicitly uses `--gpu-layers 99`; v0.4.1 Metal offload was verified separately in `../offload-diagnostics/`.",
        "- Metrics use the official binary qrels (score 1). Because this is a fixed 30-query subset rather than the full MTEB task, results are a reproducible smoke comparison, not official leaderboard scores.",
        "",
        "## Summary",
        "",
        "| config | embedding | reranker | k | queries | top1 | MRR@10 | nDCG@10 | Recall@10 | candidate recall | query ms mean/p95 | rerank ms mean/p95 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def f(value: Any, digits: int = 3) -> str:
        return "n/a" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))
    for row in summaries:
        lines.append(
            f"| {row['config']} | {row['embedding']} | {row['reranker']} | {row.get('candidate_k_requested') or 'all'} | {row.get('queries')} | "
            f"{f(row.get('top1'))} | {f(row.get('mrr10'))} | {f(row.get('ndcg10'))} | {f(row.get('recall10'))} | {f(row.get('candidate_recall'))} | "
            f"{f(row.get('query_latency_ms_mean'), 1)}/{f(row.get('query_latency_ms_p95'), 1)} | {f(row.get('rerank_latency_ms_mean'), 1)}/{f(row.get('rerank_latency_ms_p95'), 1)} |"
        )
    if errors:
        lines += ["", "## Errors", ""]
        lines.extend(f"- {json.dumps(error, ensure_ascii=False, sort_keys=True)}" for error in errors)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    global OUTPUT
    requested_output = Path(args.output_dir)
    OUTPUT = requested_output if requested_output.is_absolute() else REAL / requested_output
    base.ROOT = OUTPUT
    manifest = json.loads((REAL / "manifest.json").read_text(encoding="utf-8"))
    selected_embeddings = list(base.EMBED_MODELS)
    if args.embeddings == "8b":
        selected_embeddings = ["qwen3-embedding-8b-q8_0"]
    elif args.embeddings == "4b":
        selected_embeddings = ["qwen3-embedding-4b-q8_0"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "logs").mkdir(parents=True, exist_ok=True)
    metadata = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": manifest["dataset"],
        "dataset_revision": manifest["source_revision"],
        "license": manifest["license"],
        "binary": str(base.DEFAULT_BINARY),
        "binary_version": "0.4.1 (build 177, commit b29c606)",
        "gpu_layers": 99,
        "device": "MTL0 Apple M3 Ultra (verified in offload diagnostics)",
        "batch_size": 4096,
        "ubatch_size": 4096,
        "offload_diagnostic": {
            "results": str(REBUILT / "offload-diagnostics/results.json"),
            "cpu_log": str(REBUILT / "offload-diagnostics/reranker-gpu-layers-0.log"),
            "metal_log": str(REBUILT / "offload-diagnostics/reranker-gpu-layers-99.log"),
        },
        "configs": list(manifest["configs"]),
        "queries_per_config": args.queries_per_config,
        "embeddings": selected_embeddings,
    }
    (OUTPUT / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summaries: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    for config in manifest["configs"]:
        config_dir = REAL / "subset" / config
        corpus = load_jsonl(config_dir / "corpus.jsonl")
        queries = load_jsonl(config_dir / "queries.jsonl")[: args.queries_per_config]
        qrels = load_jsonl(config_dir / "qrels.jsonl")
        relevant_by_query: dict[str, set[str]] = {}
        for qrel in qrels:
            if int(qrel["score"]) > 0:
                relevant_by_query.setdefault(str(qrel["query-id"]), set()).add(str(qrel["corpus-id"]))
        corpus_ids = [str(item["id"]) for item in corpus]
        corpus_texts = [str(item["text"]) for item in corpus]
        query_texts = [base.QUERY_INSTRUCTION + str(item["text"]) for item in queries]
        for embedding_name in selected_embeddings:
            embedding_model = base.EMBED_MODELS[embedding_name]
            print(f"[real] {config} embedding {embedding_name}", flush=True)
            if not embedding_model.exists():
                errors.append({"config": config, "embedding": embedding_name, "error": f"missing model {embedding_model}"})
                save(OUTPUT, summaries, query_rows, errors, events)
                continue
            embed_server = base.LlamaServer(embedding_model, "embedding", base.DEFAULT_BINARY, f"real-embedding-{config}-{embedding_name}", ctx_size=args.ctx_size)
            try:
                with embed_server:
                    events.append({"stage": "embedding", "config": config, "model": embedding_name, "port": embed_server.port, "load_ms": embed_server.load_ms, "gpu_layers": 99, "device": "MTL0"})
                    corpus_vectors, corpus_timings, corpus_tokens = base.embed_texts(embed_server, corpus_texts, embedding_model.name)
                    query_vectors, query_timings, query_tokens = base.embed_texts(embed_server, query_texts, embedding_model.name, batch_size=1)
                    embedding_metrics = {
                        "load_ms": embed_server.load_ms,
                        "corpus_ms": sum(corpus_timings),
                        "query_ms": sum(query_timings),
                        "tokens": corpus_tokens + query_tokens,
                        "dim": len(corpus_vectors[0]),
                        "peak_rss_mb": embed_server.peak_rss_mb,
                        "corpus_docs_per_sec": len(corpus_vectors) / (sum(corpus_timings) / 1000.0),
                        "query_per_sec": len(query_vectors) / (sum(query_timings) / 1000.0),
                    }
                    standalone: list[dict[str, Any]] = []
                    for qidx, query in enumerate(queries):
                        started = base.now_ms()
                        local_order = base.rank_embedding(query_vectors[qidx], corpus_vectors)
                        final_order = [corpus_ids[idx] for idx in local_order]
                        row_metrics = metrics(final_order, final_order, relevant_by_query.get(str(query["id"]), set()))
                        standalone.append({
                            "config": config, "embedding": embedding_name, "reranker": "standalone", "candidate_k_requested": None,
                            "candidate_count": len(corpus), "query_id": query["id"], "query_language": manifest["configs"][config]["query_language"],
                            "corpus_language": manifest["configs"][config]["corpus_language"], "total_query_latency_ms": base.now_ms() - started + query_timings[qidx],
                            "direction": f"{manifest['configs'][config]['query_language']}->{manifest['configs'][config]['corpus_language']}",
                            "rerank_latency_ms": None, "status": "ok", **row_metrics,
                        })
                    summary = {
                        "config": config, "embedding": embedding_name, "reranker": "standalone", "candidate_k_requested": None,
                        **aggregate(standalone), "candidate_k_actual": len(corpus), "model_load_ms": embedding_metrics["load_ms"],
                        "corpus_embedding_ms": embedding_metrics["corpus_ms"], "query_embedding_ms_total": embedding_metrics["query_ms"],
                        "embedding_dim": embedding_metrics["dim"], "embedding_tokens": embedding_metrics["tokens"], "peak_rss_mb": embedding_metrics["peak_rss_mb"],
                        "corpus_docs_per_sec": embedding_metrics["corpus_docs_per_sec"], "query_per_sec": embedding_metrics["query_per_sec"], "status": "ok",
                    }
                    summaries.append(summary)
                    query_rows.extend(standalone)
                    save(OUTPUT, summaries, query_rows, errors, events)
                    print(f"[real] {config} {embedding_name} standalone top1={summary['top1']}", flush=True)
            except Exception as exc:
                errors.append({"config": config, "embedding": embedding_name, "reranker": "standalone", "error": str(exc)})
                save(OUTPUT, summaries, query_rows, errors, events)
                continue

            for reranker_name, reranker_model in base.RERANK_MODELS.items():
                print(f"[real] {config} {embedding_name} reranker {reranker_name}", flush=True)
                if not reranker_model.exists():
                    errors.extend({"config": config, "embedding": embedding_name, "reranker": reranker_name, "candidate_k_requested": k, "error": f"missing model {reranker_model}"} for k in (50, 100))
                    save(OUTPUT, summaries, query_rows, errors, events)
                    continue
                rerank_server = base.LlamaServer(reranker_model, "rerank", base.DEFAULT_BINARY, f"real-rerank-{config}-{embedding_name}-{reranker_name}", ctx_size=args.ctx_size)
                try:
                    with rerank_server:
                        events.append({"stage": "rerank", "config": config, "embedding": embedding_name, "model": reranker_name, "port": rerank_server.port, "load_ms": rerank_server.load_ms, "gpu_layers": 99, "device": "MTL0"})
                        for requested_k in (50, 100):
                            rows_for_combo: list[dict[str, Any]] = []
                            for qidx, query in enumerate(queries):
                                retrieval_started = base.now_ms()
                                local_order = base.rank_embedding(query_vectors[qidx], corpus_vectors)
                                candidate_idx = local_order[:requested_k]
                                candidate_ids = [corpus_ids[idx] for idx in candidate_idx]
                                candidate_texts = [corpus_texts[idx] for idx in candidate_idx]
                                retrieval_ms = base.now_ms() - retrieval_started
                                try:
                                    rerank_order, rerank_ms, returned_count, rerank_tokens = base.rerank(rerank_server, str(query["text"]), candidate_texts, reranker_model.name)
                                    final_ids = [candidate_ids[idx] for idx in rerank_order if 0 <= idx < len(candidate_ids)]
                                    row_metrics = metrics(final_ids, candidate_ids, relevant_by_query.get(str(query["id"]), set()))
                                    row = {
                                        "config": config, "embedding": embedding_name, "reranker": reranker_name, "candidate_k_requested": requested_k,
                                        "candidate_count": len(candidate_ids), "rerank_returned_count": returned_count, "rerank_tokens": rerank_tokens,
                                        "query_id": query["id"], "query_language": manifest["configs"][config]["query_language"], "corpus_language": manifest["configs"][config]["corpus_language"],
                                        "direction": f"{manifest['configs'][config]['query_language']}->{manifest['configs'][config]['corpus_language']}",
                                        "retrieval_ms": retrieval_ms, "rerank_latency_ms": rerank_ms, "total_query_latency_ms": query_timings[qidx] + retrieval_ms + rerank_ms,
                                        "status": "ok", **row_metrics,
                                    }
                                    rows_for_combo.append(row)
                                except Exception as exc:
                                    row = {"config": config, "embedding": embedding_name, "reranker": reranker_name, "candidate_k_requested": requested_k, "query_id": query["id"], "status": "error", "error": str(exc)}
                                    rows_for_combo.append(row)
                                    errors.append(row)
                            good = [row for row in rows_for_combo if row.get("status") == "ok"]
                            if good:
                                summary = {
                                    "config": config, "embedding": embedding_name, "reranker": reranker_name, "candidate_k_requested": requested_k,
                                    **aggregate(good), "candidate_k_actual": min(requested_k, len(corpus)), "model_load_ms": rerank_server.load_ms,
                                    "corpus_embedding_ms": embedding_metrics["corpus_ms"], "query_embedding_ms_total": embedding_metrics["query_ms"],
                                    "embedding_dim": embedding_metrics["dim"], "embedding_tokens": embedding_metrics["tokens"], "peak_rss_mb": max(embedding_metrics["peak_rss_mb"] or 0.0, rerank_server.peak_rss_mb or 0.0),
                                    "corpus_docs_per_sec": embedding_metrics["corpus_docs_per_sec"], "query_per_sec": embedding_metrics["query_per_sec"],
                                    "actual_returned_mean": statistics.mean(float(row["rerank_returned_count"]) for row in good), "status": "ok",
                                }
                                summaries.append(summary)
                                print(f"[real] finished {config} {embedding_name} + {reranker_name} k={requested_k}: top1={summary['top1']} mrr10={summary['mrr10']}", flush=True)
                            else:
                                errors.append({"config": config, "embedding": embedding_name, "reranker": reranker_name, "candidate_k_requested": requested_k, "error": "all requests failed"})
                            query_rows.extend(rows_for_combo)
                            save(OUTPUT, summaries, query_rows, errors, events)
                except Exception as exc:
                    errors.extend({"config": config, "embedding": embedding_name, "reranker": reranker_name, "candidate_k_requested": k, "error": f"server: {exc}"} for k in (50, 100))
                    save(OUTPUT, summaries, query_rows, errors, events)

    metadata["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    metadata["summary_rows"] = len(summaries)
    metadata["query_rows"] = len(query_rows)
    metadata["error_rows"] = len(errors)
    (OUTPUT / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(OUTPUT / "report.md", metadata, summaries, errors)
    save(OUTPUT, summaries, query_rows, errors, events)
    print(json.dumps({"summary_rows": len(summaries), "query_rows": len(query_rows), "errors": len(errors), "report": str(OUTPUT / "report.md")}, ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", choices=("8b", "4b", "both"), default="8b")
    parser.add_argument("--queries-per-config", type=int, default=30)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--output-dir", default="benchmark-metal", help="output directory, relative to real-dataset unless absolute")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
