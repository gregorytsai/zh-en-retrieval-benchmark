#!/usr/bin/env python3
"""Validate rerank API ordering and preserve raw scores for small known cases."""

from __future__ import annotations

import json
import time
from pathlib import Path

from run_benchmark import DEFAULT_BINARY, LlamaServer, RERANK_MODELS, json_request


ROOT = Path(__file__).resolve().parent
CASES = [
    {
        "id": "battery-en",
        "query": "How can a residential storage pack limit aging from deep cycling, and which metric describes charge-discharge losses?",
        "doc_ids": ["en_01_anchor", "en_01_d01", "en_02_anchor"],
    },
    {
        "id": "espresso-zh",
        "query": "說明濃縮咖啡的研磨、粉量、水溫和接觸時間如何共同決定萃取結果。",
        "doc_ids": ["zh_03_anchor", "zh_03_d02", "zh_07_anchor"],
    },
    {
        "id": "signatures-zh-to-en",
        "query": "公開金鑰系統怎麼驗證訊息沒有被改過並連結寄件者身分？若金鑰外洩該怎麼辦？",
        "doc_ids": ["en_05_anchor", "en_05_d01", "en_03_anchor"],
    },
]


def main() -> None:
    docs = {row["id"]: row["text"] for row in (json.loads(line) for line in (ROOT / "corpus.jsonl").read_text(encoding="utf-8").splitlines())}
    results = []
    tested = dict(RERANK_MODELS)
    patched = ROOT / "Qwen3-Reranker-0.6B.Q8_0.rerank-template.gguf"
    if patched.exists():
        tested["qwen3-reranker-0.6b-q8_0-corrected-template"] = patched
    for name, model in tested.items():
        server = LlamaServer(model, "rerank", DEFAULT_BINARY, f"sanity-{name}", ctx_size=8192)
        started = time.perf_counter()
        try:
            with server:
                for case in CASES:
                    payload = {
                        "model": model.name,
                        "query": case["query"],
                        "documents": [docs[doc_id] for doc_id in case["doc_ids"]],
                    }
                    response, elapsed = json_request(server.url + "/v1/rerank", payload, timeout=180)
                    results.append({
                        "model": name,
                        "model_path": str(model),
                        "case": case["id"],
                        "doc_ids_in_request_order": case["doc_ids"],
                        "elapsed_ms": elapsed,
                        "response": response,
                        "returned_order_ids": [case["doc_ids"][int(item["index"])] for item in response.get("results", [])],
                        "server_load_ms": server.load_ms,
                        "server_peak_rss_mb": server.peak_rss_mb,
                        "status": "ok",
                    })
        except Exception as exc:
            results.append({"model": name, "model_path": str(model), "status": "error", "error": str(exc), "server_load_ms": server.load_ms, "server_peak_rss_mb": server.peak_rss_mb})
        print(f"[sanity] {name} done", flush=True)
    (ROOT / "sanity_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(results), "file": str(ROOT / "sanity_results.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
