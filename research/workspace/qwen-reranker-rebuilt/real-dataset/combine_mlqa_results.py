#!/usr/bin/env python3
"""Combine the completed 4B and 8B MLQA benchmark artifacts."""

from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path


ROOT = (Path(__file__).resolve().parents[2] / 'qwen-reranker-rebuilt/real-dataset')
SOURCES = [ROOT / "benchmark-metal", ROOT / "benchmark-metal-4b"]
OUTPUT = ROOT / "benchmark-metal-combined"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def mean(rows: list[dict], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.mean(values) if values else None


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def num(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def label_embedding(name: str) -> str:
    return "4B" if "4b" in name.lower() else "8B"


def label_reranker(name: str, k: int | None) -> str:
    if name == "standalone":
        return "standalone"
    prefix = "Qwen 0.6B" if "qwen3-reranker" in name else "BGE"
    return f"{prefix} k={k}"


def main() -> None:
    summaries: list[dict] = []
    queries: list[dict] = []
    errors: list[dict] = []
    source_metadata: list[dict] = []
    for source in SOURCES:
        summaries.extend(read_jsonl(source / "summary.jsonl"))
        queries.extend(read_jsonl(source / "query_results.jsonl"))
        errors.extend(read_jsonl(source / "errors.jsonl"))
        source_metadata.append(json.loads((source / "run_metadata.json").read_text(encoding="utf-8")))

    summaries.sort(key=lambda row: (label_embedding(row["embedding"]), row["config"], row["reranker"], row.get("candidate_k_requested") or 0))
    queries.sort(key=lambda row: (label_embedding(row["embedding"]), row["config"], row["reranker"], row.get("candidate_k_requested") or 0, str(row["query_id"])))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT / "summary.jsonl", summaries)
    write_jsonl(OUTPUT / "query_results.jsonl", queries)
    write_jsonl(OUTPUT / "errors.jsonl", errors)

    fields: list[str] = []
    for row in summaries:
        for field in row:
            if field not in fields:
                fields.append(field)
    with (OUTPUT / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset": "mteb/MLQARetrieval",
        "dataset_revision": source_metadata[0]["dataset_revision"],
        "license": source_metadata[0]["license"],
        "source_directories": [str(path) for path in SOURCES],
        "models": sorted({row["embedding"] for row in summaries}),
        "configs": ["eng-zho", "zho-zho", "eng-eng"],
        "queries_per_config_per_embedding": 30,
        "summary_rows": len(summaries),
        "query_rows": len(queries),
        "error_rows": len(errors),
        "ok_query_rows": sum(row.get("status") == "ok" for row in queries),
        "gpu_layers": 99,
        "device": "MTL0 Apple M3 Ultra",
        "batch_size": 4096,
        "ubatch_size": 4096,
        "notes": [
            "This directory combines completed runs; no inference was rerun during combination.",
            "RSS in each summary is the maximum of sequential model runs, not the sum of simultaneously resident services.",
        ],
    }
    (OUTPUT / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    grouped: dict[tuple[str, str, int | None], list[dict]] = {}
    for row in summaries:
        key = (row["embedding"], row["reranker"], row.get("candidate_k_requested"))
        grouped.setdefault(key, []).append(row)

    overall_rows: list[dict] = []
    for (embedding, reranker, k), rows in sorted(grouped.items(), key=lambda item: (label_embedding(item[0][0]), item[0][1], item[0][2] or 0)):
        overall_rows.append({
            "embedding": embedding,
            "reranker": reranker,
            "k": k,
            "top1": mean(rows, "top1"),
            "mrr10": mean(rows, "mrr10"),
            "ndcg10": mean(rows, "ndcg10"),
            "recall10": mean(rows, "recall10"),
            "candidate_recall": mean(rows, "candidate_recall"),
            "query_ms": mean(rows, "query_latency_ms_mean"),
            "query_p50_ms": mean(rows, "query_latency_ms_p50"),
            "query_p95_ms": mean(rows, "query_latency_ms_p95"),
            "rerank_ms": mean(rows, "rerank_latency_ms_mean"),
        })
    write_jsonl(OUTPUT / "overall_summary.jsonl", overall_rows)

    lines = [
        "# Qwen 4B / 8B MLQA 中英檢索完整測試報告",
        "",
        "## 結論摘要",
        "",
        "- **雙語與中文查英文的預設組合：Qwen3 Embedding 4B Q8_0 + Qwen3 Reranker 0.6B Q8_0，rerank k=50。** 在本子集上，它與 8B 的 Qwen rerank 組合有相同的整體 Top-1（91.1%），但 4B embedding 平均 query latency 較低、corpus indexing 較快，RSS 也較小。",
        "- **英文為主且追求最高 Top-1：Qwen3 Embedding 8B + BGE k=50。** 英文→英文 Top-1 為 93.3%；4B+BGE k=50 為 90.0%。",
        "- **極低延遲：4B embedding standalone。** 三方向平均 Top-1 81.1%，平均 query 約 85.7 ms，不需常駐 reranker。",
        "- **k=50 是合理預設。** Qwen k=100 的整體 Top-1 沒有高於 k=50，rerank latency 約翻倍；BGE k=100 甚至在部分方向下降。",
        "",
        "## 測試方法",
        "",
        "- 資料集：`mteb/MLQARetrieval` validation，revision `cf59ddd8f4aaf39ce1869361e09252698c340945`，授權 `CC-BY-SA-3.0`。",
        "- 每個 embedding 在 `eng-zho`（中文 query→英文 corpus）、`zho-zho`、`eng-eng` 各取固定排序的 30 queries，保留該 config 的完整 corpus 與官方 qrels。",
        "- 每個 embedding 測 standalone，以及 Qwen3 Reranker 0.6B Q8_0、BGE reranker v2 m3 Q8_0 的 k=50/100。",
        "- `llama-server` v0.4.1，明確使用 `--gpu-layers 99 --batch-size 4096 --ubatch-size 4096`；既有診斷日誌證實 Qwen 0.6B 為 Metal 29/29 layers offload。",
        "- 共 30 summary rows、900 query rows，全部 `status=ok`，errors=0。這是固定 30-query 子集，不是官方完整 MTEB leaderboard 分數。",
        "",
        "## 三方向 Top-1",
        "",
        "| embedding | 組合 | 中文→英文 | 中文→中文 | 英文→英文 | 三方向平均 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    configs = ["eng-zho", "zho-zho", "eng-eng"]
    for overall in overall_rows:
        matching = [row for row in summaries if row["embedding"] == overall["embedding"] and row["reranker"] == overall["reranker"] and row.get("candidate_k_requested") == overall["k"]]
        by_config = {row["config"]: row for row in matching}
        lines.append(
            f"| {label_embedding(overall['embedding'])} | {label_reranker(overall['reranker'], overall['k'])} | "
            + " | ".join(pct(by_config[c]["top1"]) for c in configs)
            + f" | {pct(overall['top1'])} |"
        )

    lines += [
        "",
        "## 整體準確率與延遲",
        "",
        "三方向各 30 queries，因此以下為等權平均。延遲為每方向 mean 的平均。",
        "",
        "| embedding | 組合 | Top-1 | MRR@10 | nDCG@10 | Recall@10 | query mean ms | rerank mean ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall_rows:
        lines.append(
            f"| {label_embedding(row['embedding'])} | {label_reranker(row['reranker'], row['k'])} | {pct(row['top1'])} | "
            f"{row['mrr10']:.3f} | {row['ndcg10']:.3f} | {pct(row['recall10'])} | {num(row['query_ms'])} | {num(row['rerank_ms'])} |"
        )

    standalone = [row for row in summaries if row["reranker"] == "standalone"]
    lines += [
        "",
        "## Embedding 本身的效率",
        "",
        "| embedding | 方向 | corpus docs/s | query mean/p50/p95 ms | peak RSS MB | embedding dim |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(standalone, key=lambda item: (label_embedding(item["embedding"]), item["config"])):
        lines.append(
            f"| {label_embedding(row['embedding'])} | {row['config']} | {row['corpus_docs_per_sec']:.2f} | "
            f"{row['query_latency_ms_mean']:.1f}/{row['query_latency_ms_p50']:.1f}/{row['query_latency_ms_p95']:.1f} | "
            f"{row['peak_rss_mb']:.0f} | {row['embedding_dim']} |"
        )

    s4 = [row for row in standalone if "4b" in row["embedding"].lower()]
    s8 = [row for row in standalone if "8b" in row["embedding"].lower()]
    throughput4, throughput8 = mean(s4, "corpus_docs_per_sec"), mean(s8, "corpus_docs_per_sec")
    latency4, latency8 = mean(s4, "query_latency_ms_mean"), mean(s8, "query_latency_ms_mean")
    rss4, rss8 = mean(s4, "peak_rss_mb"), mean(s8, "peak_rss_mb")
    lines += [
        "",
        "### 4B 對 8B",
        "",
        f"- 4B corpus indexing 平均 {throughput4:.2f} docs/s，8B 為 {throughput8:.2f} docs/s；4B 約快 {(throughput4 / throughput8 - 1) * 100:.0f}%。",
        f"- 4B standalone query 平均 {latency4:.1f} ms，8B 為 {latency8:.1f} ms；4B 約低 {(1 - latency4 / latency8) * 100:.0f}% latency。",
        f"- 本次大 batch 測試的 embedding peak RSS 平均約為 4B {rss4 / 1024:.2f} GiB、8B {rss8 / 1024:.2f} GiB；4B 約少 {(1 - rss4 / rss8) * 100:.0f}%。",
        "- Standalone 整體 Top-1：4B 81.1%、8B 80.0%；但 8B 的整體 MRR@10 略高。只有 30 queries/方向，這個小差距不應解讀為 4B 普遍比 8B 更準。",
        "- 搭配 Qwen 0.6B k=50 後，4B 與 8B 的整體 Top-1 都是 91.1%，表示這份子集上 reranker 補回了大部分首階段排序差異。",
        "",
        "## 架設建議",
        "",
        "1. **一般中英 RAG / 中文查英文：**常駐 4B embedding 與 Qwen 0.6B reranker，首階段取 50 筆後 rerank；最終只送前 5–10 筆給生成模型。這是本次品質、速度、記憶體最均衡的組合。",
        "2. **英文知識庫且準確率優先：**選 8B embedding + BGE k=50；若記憶體較重要，4B+BGE k=50 只少一個 30-query 樣本的 Top-1 命中。",
        "3. **低流量或極低延遲：**可先只部署 4B embedding；對低信心、長 query 或跨語 query 才呼叫 reranker。",
        "4. **不要預設 k=100：**先監控 production relevance set 的 candidate recall@50；只有它不足時才升到 100。這次 k=100 幾乎沒有 Top-1 收益，延遲約兩倍。",
        "5. **服務配置：**文件 embedding 離線預算並存入向量庫；線上 embedding 與 reranker 各用一個長駐 Metal process，明確指定 `--gpu-layers 99`，不要依賴此版本的 auto offload。先以 `--parallel 1` 建立基線，再依壓測增加 worker/slot。",
        "6. **記憶體規劃：**報告的 peak RSS 是依序測量時的最大值，不是兩服務同時常駐的總和。4B embedding 本次約 6 GiB；加上 0.6B Q8 reranker、Metal/批次 buffer 與安全餘裕，建議至少預留約 8–10 GiB 可用 unified memory。8B 組合建議至少預留約 12–14 GiB。正式上線前仍應做同時常駐壓測。",
        "7. **API 行為：**真實 MLQA passage 曾超過 512 tokens，因此本測試使用 batch/ubatch 4096。正式環境若允許更長 passage，需做截斷或 chunking，並確認 physical batch 足以容納單一 query-passage pair。",
        "",
        "## 限制",
        "",
        "- 每方向只有固定 30 queries；適合本機組合篩選，不等同完整 MLQA/MTEB 評測。",
        "- 每個 query 使用官方 binary qrels；實務知識庫常有多個同樣合理答案，應另建 production judgments。",
        "- 4B 與 8B 是不同時間的順序測試；同機背景負載會影響 latency，模型間效率應以 embedding throughput、p50/p95 與後續並行壓測共同判讀。",
        "- BGE 與 Qwen reranker 的最佳提示、文件截斷及批次參數仍可能依資料領域而變。",
        "",
        "## 可重現檔案",
        "",
        "- `summary.jsonl` / `summary.csv`: 30 組方向級摘要。",
        "- `overall_summary.jsonl`: 10 組跨方向平均。",
        "- `query_results.jsonl`: 900 筆逐 query 結果。",
        "- `run_metadata.json`: 來源、列數與 Metal 設定。",
        "- 原始 8B 與 4B 報告分別保留於 `../benchmark-metal/` 和 `../benchmark-metal-4b/`。",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert len(summaries) == 30, len(summaries)
    assert len(queries) == 900, len(queries)
    assert not errors, errors
    assert all(row.get("status") == "ok" for row in queries)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
