#!/usr/bin/env python3
"""Run a small raw-score smoke test for the rebuilt Qwen3 reranker."""

from __future__ import annotations

import json
import time
from pathlib import Path

import sys

BENCHMARK = (Path(__file__).resolve().parents[1] / 'bilingual-retrieval-benchmark')
sys.path.insert(0, str(BENCHMARK))
from run_benchmark import DEFAULT_BINARY, LlamaServer, json_request  # noqa: E402


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "Qwen3-Reranker-0.6B-f16.gguf"
CASES = [
    {
        "id": "battery-en",
        "query": "How can a residential storage pack limit aging from deep cycling, and which metric describes charge-discharge losses?",
        "documents": [
            "Battery storage systems can reduce aging by limiting depth of discharge and avoiding extreme state-of-charge windows. Round-trip efficiency describes the energy lost between charging and discharging.",
            "Espresso extraction depends on grind size, dose, water temperature, and contact time.",
            "Public-key signatures let a verifier check message integrity and associate the signature with a key holder.",
        ],
        "positive_index": 0,
    },
    {
        "id": "espresso-zh",
        "query": "說明濃縮咖啡的研磨、粉量、水溫和接觸時間如何共同決定萃取結果。",
        "documents": [
            "電池儲能系統可透過限制放電深度與避免極端荷電狀態來降低老化；往返效率描述充放電損失。",
            "濃縮咖啡的研磨粗細、粉量、水溫與接觸時間會共同影響流速、萃取率與風味平衡。",
            "公開金鑰簽章可以驗證訊息完整性，並把簽章連結到持有私鑰的人。",
        ],
        "positive_index": 1,
    },
    {
        "id": "signatures-zh-to-en",
        "query": "公開金鑰系統怎麼驗證訊息沒有被改過並連結寄件者身分？",
        "documents": [
            "Battery storage systems can reduce aging by limiting depth of discharge.",
            "Digital signatures let anyone with the sender's public key verify that a message was not altered and that it was signed by the corresponding private key holder.",
            "Espresso extraction depends on grind size and water temperature.",
        ],
        "positive_index": 1,
    },
]


def main() -> None:
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)
    rows = []
    server = LlamaServer(MODEL, "rerank", DEFAULT_BINARY, "smoke-qwen3-reranker-0.6b-rebuilt", ctx_size=8192)
    started = time.perf_counter()
    try:
        with server:
            for case in CASES:
                response, elapsed = json_request(
                    server.url + "/v1/rerank",
                    {"model": MODEL.name, "query": case["query"], "documents": case["documents"]},
                    timeout=180,
                )
                scored = [
                    {"index": int(item["index"]), "score": float(item.get("relevance_score", item.get("score")))}
                    for item in response.get("results", [])
                ]
                scored.sort(key=lambda item: (-item["score"], item["index"]))
                positive = next(item for item in scored if item["index"] == case["positive_index"])
                negatives = [item for item in scored if item["index"] != case["positive_index"]]
                rows.append({
                    "case": case["id"],
                    "elapsed_ms": elapsed,
                    "scores": scored,
                    "positive_index": case["positive_index"],
                    "positive_score": positive["score"],
                    "max_negative_score": max(item["score"] for item in negatives),
                    "positive_above_all_negatives": all(positive["score"] > item["score"] for item in negatives),
                    "response": response,
                })
    finally:
        server.stop()
    output = {
        "model": str(MODEL),
        "binary": str(DEFAULT_BINARY),
        "server_load_ms": server.load_ms,
        "server_peak_rss_mb": server.peak_rss_mb,
        "elapsed_s": time.perf_counter() - started,
        "cases": rows,
        "all_positive_above_all_negatives": all(row["positive_above_all_negatives"] for row in rows),
    }
    (ROOT / "smoke_results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
