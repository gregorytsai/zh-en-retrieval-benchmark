# 本機重現指引：MLQA retrieval guarded v3

本文件描述 2026-10-22 在目前這台 Mac 上已驗證的基準測試建置。它不是其他主機的移植承諾，也不是 production 部署規格。正式結果在 [`run-20261022-v3-guarded/`](run-20261022-v3-guarded/)；本機完整逐組結果見 [`FINAL_REPORT_ZH_TW.md`](FINAL_REPORT_ZH_TW.md)。本指引只記錄流程，不執行推論或修改服務。

## 已驗證環境

| 元件 | 此次實測值 |
|---|---|
| 硬體／OS | Apple M3 Ultra；macOS 26.7 arm64 |
| llama.cpp server | `<WORKSPACE>/llama.cpp-qwen38/build/bin/llama-server` |
| binary 識別 | 0.7.0-dev，build 883，commit `1fd20a7f4`；SHA-256 `5f7bee49eaf08bf77b3cc523d88d385750d6b7da8479775b32bc9be030464ae5` |
| Python／套件 | Python 3.13.7、NumPy 2.5.4、PyArrow 26.0.0；`requirements.txt` 有固定 NumPy／PyArrow 版本 |
| 加速 | llama-server log 顯示 Metal；66 個 accepted attempt 的 GPU offload 均為完整 N/N layer |

本次 benchmark 使用既有 binary，並沒有在這個工作中重編 llama.cpp。換 Mac、Linux、不同 GPU/backend 或 binary 後，必須重新建置、核對版本並檢查每個模型的 offload log；目前沒有其他主機的驗證結果。

## 工作目錄與素材相依

程式以目前資料夾為 `ROOT`，其上一層 `<WORKSPACE>` 為 `WORK`。單獨複製本資料夾不足以重現；程式還會讀取下列 sibling 資產：

```text
<WORKSPACE>/
├── embedding-study-20261022/
│   ├── run_study.py, run_guarded.py, traffic_guard.py
│   ├── audit_v3.py, materials-readiness.json, requirements.txt, .venv/
│   ├── models/                         # 六款新 GGUF
│   └── run-20261022-v2/                # pilot-decision.json、query-selection-final.json
├── bilingual-retrieval-benchmark/run_benchmark.py
├── llama.cpp-qwen38/build/bin/llama-server
├── Qwen3-Embedding-4B.Q8_0.gguf
├── bge-reranker-v2-m3-q8_0.gguf
└── qwen-reranker-rebuilt/
    ├── Qwen3-Reranker-0.6B-Q8_0.gguf
    ├── Qwen3-Reranker-4B-Q8_0.gguf
    └── real-dataset/
```

測試啟動前可先用 `verify_materials.py` 檢查素材與 SHA-256；模型和資料的完整清單、來源、容量及 hash 在 [`materials-readiness.json`](materials-readiness.json)。完成的 v3 run manifest 也記錄素材、binary、runner、guard hash。不要把原始 validation 資料、外部模型路徑或本機絕對路徑直接當成可公開套件。

### 模型設定

以下檔名為本機檔案；Q8_0 檔案完整 SHA-256 以 `materials-readiness.json` 為準。Core embedder 才會進行 3×3 reranker 矩陣；F2LLM 與 Granite 只有 standalone。

| 模型 | 本機相對路徑 | Pooling／輸出維度 | 輸入格式 |
|---|---|---|---|
| Qwen3 Embedding 0.6B | `embedding-study-20261022/models/Qwen3-Embedding-0.6B-Q8_0.gguf` | last／1024 | query 前綴 `Instruct: Given a web search query, retrieve relevant passages\nQuery: `；document 無前綴 |
| Qwen3 Embedding 4B | `Qwen3-Embedding-4B.Q8_0.gguf` | last／2560 | 同 Qwen query 前綴；document 無前綴 |
| Qwen3 Embedding 8B | `embedding-study-20261022/models/Qwen3-Embedding-8B-Q8_0.gguf` | last／4096 | 同 Qwen query 前綴；document 無前綴 |
| BGE-M3 | `embedding-study-20261022/models/bge-m3-q8_0.gguf` | cls／1024 | query/document 均無前綴 |
| EmbeddingGemma 2 | `embedding-study-20261022/models/embeddinggemma-2-Q8_0.gguf` | mean／768 | query `task: search result \| query: `；document `title: none \| text: ` |
| F2LLM-v2 0.6B | `embedding-study-20261022/models/F2LLM-v2-0.6B.Q8_0.gguf` | last／1024 | Qwen query 前綴；document 無前綴；standalone only |
| Granite Embedding 97M r2 | `embedding-study-20261022/models/granite-embedding-97M-multilingual-r2-Q8_0.gguf` | cls／384 | query/document 均無前綴；standalone only |
| BGE reranker v2 m3 | `bge-reranker-v2-m3-q8_0.gguf` | rerank endpoint | Qwen reranker 之外的比較基線 |
| Qwen3 Reranker 0.6B | `qwen-reranker-rebuilt/Qwen3-Reranker-0.6B-Q8_0.gguf` | rerank endpoint | 測試 top-50 documents |
| Qwen3 Reranker 4B | `qwen-reranker-rebuilt/Qwen3-Reranker-4B-Q8_0.gguf` | rerank endpoint | 測試 top-50 documents |

Embedding response 會驗證向量數量、維度、finite 值及 L2 norm，並做正負例與重複 embedding gate。`run_study.py` 另記錄 corpus/query token 長度；本次沒有截斷。Q8_0、prefix、pooling 與輸出維度不可任意省略或交換。

## Runner 與 server 參數

應以 `run_guarded.py` 為正式入口。它對其他本機 llama-server 做唯讀 metrics/router 健康監控：每秒檢查，連續 30 秒安靜後才開始組合，監控不可用時 fail closed；組合期間發生外部活動則整組 invalidated、待閒置後重跑。只會終止自己啟動的 benchmark child process，不會停 production worker。不要在共用本機服務時改用無 guard 的 `run_study.py` 正式執行。

此次 accepted command 共用設定：loopback host、隨機空閒 port、`--ctx-size 8192`、`--parallel 1`、`--threads 8`、`--threads-batch 8`、`--batch-size 4096`、`--ubatch-size 4096`、`--n-gpu-layers all`、`--log-verbosity 4`。Embedding server 加上 `--embedding --pooling <依模型>` 與 `--embd-normalize 2`；reranker server 使用 `--rerank`。完整逐次實際 argv 和日誌在各 accepted attempt 的 `server-events.jsonl` 與 `logs/`。

資料配置在 `run_study.py`：完整 corpus exact cosine 搜尋；每方向 query 以 `sha256("20261022:" + query_id)` 排序，取 100 題；每個 reranker 處理 embedding top-50。批次 embedding 最多 8 個輸入、token 合計小於 3,800；測試前另檢查本次資料未超過 token gate。`eng-zho` 指中文 query 查英文 corpus。

## 新 run 的執行方式

先確認素材、套件、binary、空間與共用 GPU 使用狀況。此範例是操作模板，不可直接照抄佔位字串；每次請用新的 output 目錄和**新的未來 UTC deadline**：

```bash
cd <WORKSPACE>/embedding-study-20261022
python3 verify_materials.py
.venv/bin/python -u run_guarded.py \
  --output run-NEW-ID \
  --deadline FUTURE-UTC-ISO8601
```

例如 deadline 的格式為 `2030-01-01T00:00:00+00:00`，實際執行時需自行選合理的未來截止時間。程式內原始截止時間是本次 run 的 `2026-10-23 06:08 Asia/Taipei`；它是歷史預算界線，未必適合新的執行，**新 run 請明確覆寫 `--deadline`**。runner 會拒絕覆蓋已存在的 output，會建立排他 lock，並從同層 `run-20261022-v2/pilot-decision.json` 讀取這次已選定的 n=100 規劃；v2 只提供 pilot/預算輸入，不提供正式 v3 效能數字。若換位置，這個 pilot 路徑也需明確調整。

只在同一 run 因可恢復中斷而續跑時，才使用原 output 目錄加 `--resume`；先確認沒有其他 runner 正執行。正式 v3 已 complete，請勿對它 resume 或重跑推論。

## 完成後的檢查與輸出

唯讀核對已保存的 v3 結果，不啟動任何模型服務：

```bash
python3 audit_v3.py
```

成功輸出 JSON，並要求 66 expected/committed/summary 組、6,600 唯一 rows、每組 100 題、所有 accepted attempt 內容對應主表、無 query error、無效 attempt 未混入、以及每個 accepted GPU log 均完整 offload。正式產出位於 run 目錄：

- `query-results.jsonl`：逐題正式資料；`summary.json`、`report.md`：逐組品質與 latency。
- `committed.json`、`attempts/*/validity.json`、`traffic-events.jsonl`：組合接受／流量隔離稽核。
- `server-events.jsonl`、`attempts/*/logs/`：argv、load/RSS 樣本、GPU offload 證據。
- `index-stats.jsonl`、`cache/`、`query-selection-final.json`：corpus indexing、向量／候選快取與固定 query manifest。

`total_ms` 不是 production end-to-end：不含載入模型、corpus indexing、router、網路、queue 與併發。`sampled_peak_rss_mb` 是取樣值，非保證捕捉到的真實瞬時最大值。

## 可攜性與授權注意

雖然 runner 部分素材路徑由 `ROOT.parent` 組出，`materials-readiness.json`、事件 argv、metadata 及 scripts 仍保存 `<WORKSPACE>/...` 絕對路徑，而且 4B embedding、reranker、資料、binary、benchmark helper、pilot 資料夾分布在多個 sibling 位置。複製 run output 或 repository 子資料夾本身不等於可移植；要搬到其他主機，先整理相依目錄與路徑、重建環境，重新驗證模型 hash、dataset revision、binary/backend、GPU offload 和外部流量保護，再開始測試。

模型各自授權不會延伸至 MLQA 資料或 benchmark 程式；dataset revision 與 CC-BY-SA-3.0、模型來源及 hash 以 `materials-readiness.json` 為準。公開程式或結果前仍需另做授權、attribution、個人／本機路徑與資料散佈審查。本次未改 production，亦未上傳 GitHub。
