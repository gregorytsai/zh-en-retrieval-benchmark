# 執行環境與程式碼

## 十月正式測試環境

| 項目 | 設定 |
|---|---|
| 硬體 | Apple M3 Ultra，256 GiB unified memory |
| OS | macOS 26.7，arm64 |
| llama.cpp binary | `llama.cpp-qwen38/build/bin/llama-server` |
| 版本 | 0.7.0-dev，build883，commit `1fd20a7f4` |
| binary SHA-256 | `5f7bee49eaf08bf77b3cc523d88d385750d6b7da8479775b32bc9be030464ae5` |
| 編譯器 | AppleClang21.0.0.21000334，Darwin arm64 |
| Python | 3.13.7 |
| 套件 | NumPy2.5.4、PyArrow26.0.0；requirements固定版本 |
| 模型 | 7 embedder、3 reranker，正式矩陣均Q8_0 |
| 加速 | Metal GPU，66 accepted日誌完整N/N layers offload |
| Server | loopback隨機port、ctx8192、parallel1、threads／threads-batch8、batch／ubatch4096、n-gpu-layers all、verbosity4 |
| Embedding | 對應模型pooling、embd-normalize2；query／document格式依模型 |
| Reranking | `/v1/rerank`，每次50篇 |
| 外部服務 | 本機production llama.cpp持續服務；以guard在外部流量期間暫停與整組重測 |

主測環境的權威原始證據為[run-manifest.json](workspace/embedding-study-20261022/run-20261022-v3-guarded/run-manifest.json)、`resume-history.jsonl`及accepted attempt中的argv／log。`environment.json`另保存交付時唯讀取得的硬體資訊；交付時的查詢不等於重新執行基準測試。

## 程式位置

| 目的 | 路徑，相對workspace |
|---|---|
| 正式入口與checkpoint | `embedding-study-20261022/run_guarded.py` |
| 流量監控、等待與終止自己的server | `embedding-study-20261022/traffic_guard.py` |
| 資料準備、向量、指標、rerank與報表 | `embedding-study-20261022/run_study.py` |
| 完整結果核查 | `embedding-study-20261022/audit_v3.py` |
| 單元檢查 | `embedding-study-20261022/test_study.py`、`test_traffic_guard.py` |
| 新模型下載、素材驗證、smoke | `download_models.py`、`verify_materials.py`、`smoke_embeddings.py`，同上資料夾 |
| 共用server與HTTP實作 | `bilingual-retrieval-benchmark/run_benchmark.py` |
| 九月原始及修復後測試 | `bilingual-retrieval-benchmark/*.py`、`qwen-reranker-rebuilt/*.py` |
| 歷史MLQA取材與4B／8B整合 | `qwen-reranker-rebuilt/real-dataset/*.py` |
| 部署文件、patch與計畫 | `qwen-reranker-rebuilt/deployment/` |

九月與十月的binary、query子集、參數與流量保護不同。九月v0.4.1的指令及重轉方式保存在歷史README與manifest，不套用十月延遲數字作跨版本加速結論。

## 在另一台主機重現

參考[完整建置指引](workspace/embedding-study-20261022/BUILD_GUIDE_ZH_TW.md)，先安裝對應Python套件、備妥指定GGUF及相容llama.cpp binary，再核對SHA-256與GPU offload。`workspace/`保留研究程式所依賴的三個sibling資料夾；權重與runtime需另外放入對應位置。部分metadata保留當時的絕對路徑，換機前需檢查並調整。新測試使用新output與未來deadline。

重跑推論會占用GPU，請使用guarded入口。只檢查本交付包與已完成矩陣則執行`verify_package.py`與`audit_v3.py`，不需要模型server。
