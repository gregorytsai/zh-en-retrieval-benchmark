# MLQA 雙語檢索本機測試：最終結果與建議

**有效正式執行：** `run-20261022-v3-guarded/`，於 2026-10-22 22:23:41（Asia/Taipei）完成。以下結論只採用 guarded v3 的已接受結果；v2 是未受外部流量監控的歷史紀錄，不能與 v3 混算效能。

## 完成與資料完整性

- 矩陣為 7 款 embedding standalone（21 組），再加 5 款主要 embedder × 3 款 reranker（45 組），共 **66 組、每組 100 題、6,600 筆**。
- `progress.json`、`summary.json`、`committed.json` 與主 `query-results.jsonl` 相符：66 組全部提交，每組 100 題，6,600 個 `(方向, embedder, reranker, query_id)` 唯一鍵，正式列與 query-level errors 均為 0。
- 逐一比對每組 committed attempt 的 100 筆資料與主表，內容完全相同；66 個 attempt 的 `validity.json` 全為 `accepted`。
- 18:51 外部流量事件曾使一組未完成 attempt 中 46 筆失效。它保留在 `attempts/`，狀態為 `invalidated`，沒有併入主表；同一組後來在閒置窗口完整重跑。`traffic-events.jsonl` 記錄 67 次開始、66 次接受、1 次外部活動。`fatal-errors.jsonl` 裡的單一 `Termination requested` 是安全停止舊控制程序的歷史紀錄，不是正式 query error；續跑後的主矩陣已正常完成。
- 66 個 accepted attempt 的 server log 均出現完整 `offloaded N/N layers to GPU` 證據；命令列亦逐筆核對到指定 llama-server binary、`--n-gpu-layers all`、verbosity 4、ctx 8192、parallel 1、threads 8、batch/ubatch 4096。Metal log 顯示 Apple M3 Ultra。這是本機 accepted run 的證據，不代表其他主機。

可重現的唯讀檢查：

```bash
python3 audit_v3.py
```

此腳本不連模型服務、不做推論、不改結果；成功時輸出 JSON 並以 exit code 0 結束。它重新核對矩陣數量、唯一鍵、attempt validity、主表來源、錯誤、流量事件與所有 accepted GPU log。

## 評測設計與方向

資料來自固定 revision `cf59ddd8f4aaf39ce1869361e09252698c340945` 的 MLQA retrieval validation，授權 CC-BY-SA-3.0。完整 corpus 為 eng-zho 430 篇、zho-zho 443 篇、eng-eng 978 篇；可用 query 分別為 504、504、1,148 題。本次每方向以 SHA-256（seed `20261022`）固定選 100 題、保留完整 corpus，並使用官方正相關 qrels。

| 標籤 | 本報告採用的方向定義 | 正式題數／完整 corpus |
|---|---|---:|
| `eng-zho` | **中文 query → 英文 corpus** | 100／430 |
| `zho-zho` | 中文 query → 中文 corpus | 100／443 |
| `eng-eng` | 英文 query → 英文 corpus | 100／978 |

因此本次並不是「英文查中文」。每個 reranked query 都先以 embedding 的 exact cosine 取 top-50，再將這 50 篇交給 reranker；沒有測 k=100。

## Standalone embedding 品質與效率

以下為每款模型跨三方向共 300 題的逐 query 平均；`query embed` 是 query embedding 時間，`query total` 為 embedding 加 exact search，不含建立全 corpus 向量、載入模型或 production router。

| Embedder | Top-1 | MRR@10 | nDCG@10 | Recall@10 | Recall@50 | Query embed 平均 (ms) | Query total 平均 (ms) | 三個 corpus 建索引總秒數／加權 docs/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 Embedding 0.6B | 0.7467 | 0.8245 | 0.8577 | 0.9600 | 0.9833 | 12.708 | 12.775 | 34.574／53.54 |
| Qwen3 Embedding 4B | 0.8233 | 0.8766 | 0.9002 | 0.9733 | 0.9933 | 40.569 | 40.648 | 168.279／11.00 |
| Qwen3 Embedding 8B | **0.8567** | **0.8958** | **0.9131** | 0.9667 | 0.9933 | 51.377 | 51.483 | 285.550／6.48 |
| BGE-M3 | 0.7767 | 0.8449 | 0.8738 | 0.9633 | 0.9867 | 8.071 | 8.138 | 31.032／59.65 |
| EmbeddingGemma 2 | 0.7633 | 0.8308 | 0.8609 | 0.9533 | 0.9800 | 11.242 | 11.309 | 27.177／68.11 |
| F2LLM-v2 0.6B | 0.7800 | 0.8428 | 0.8697 | 0.9533 | 0.9833 | 12.824 | 12.890 | 34.758／53.25 |
| Granite Embedding 97M r2 | 0.6567 | 0.7482 | 0.7879 | 0.9100 | 0.9867 | **4.061** | **4.125** | **9.199／201.22** |

Qwen4B 較 Qwen0.6B standalone 提升 Top-1 7.67 個百分點、MRR 5.21 點、nDCG 4.25 點；Qwen8B 再較 Qwen4B 提升 Top-1 3.33 點、MRR 1.92 點、nDCG 1.29 點，但 query embedding 平均再慢約 10.8 ms，三份 corpus 索引約由 168 秒增至 286 秒。8B 是 standalone 排名品質最佳者；4B／8B 的 Recall@50 同為 0.9933，8B 的 Recall@10 本次反而略低於 4B（0.9667 vs. 0.9733）。

Granite 是本輪最快的純 embedding／exact-search 選項，但 Top-1 與 MRR 明顯較低；BGE-M3、EmbeddingGemma 2 則提供較小 query 延遲及較平衡的 standalone 品質。這些時間只代表本機 Q8_0、完整但規模較小的這三份 corpus。

## Reranker 整體品質與實測時間

下表彙整每個 reranker 的 15 個組合（5 個主要 embedder × 3 方向），每款共 1,500 個 query。品質欄是逐題平均；`rerank` 的平均、p50、p95 直接由 1,500 筆 query-level 時間計算。`total` 是 query embedding + exact search + rerank 的總和，不含服務啟動、向量建索引、佇列、路由、併發與網路延遲。

| Reranker | Top-1 | MRR@10 | nDCG@10 | Recall@10 | Recall@50 | 對同 embedder standalone：ΔTop-1／ΔMRR／ΔnDCG | Rerank 平均／p50／p95 (ms) | Total 平均 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BGE reranker v2 m3 | 0.8747 | 0.9121 | 0.9280 | 0.9767 | 0.9873 | +8.13／+5.76／+4.68 pp | **875**／864／1,085 | **900** |
| Qwen3 reranker 0.6B | 0.8920 | 0.9258 | 0.9391 | 0.9793 | 0.9873 | +9.87／+7.13／+5.80 pp | 1,163／1,146／1,416 | 1,188 |
| Qwen3 reranker 4B | **0.9367** | **0.9551** | **0.9626** | **0.9853** | 0.9873 | **+14.33／+10.06／+8.14 pp** | 6,291／6,176／8,080 | 6,316 |

三種 reranker 使用同一組 embedding top-50 候選，所以 aggregate candidate/Recall@50 都是 0.9873；更大的 reranker 可以改善排序，但不能找回 top-50 以外漏掉的相關文件。Qwen4B reranker 相較 0.6B 的跨矩陣平均增益為 Top-1 **+4.47 pp**、MRR **+2.93 pp**、nDCG **+2.34 pp**、Recall@10 **+0.60 pp**；代價是平均 rerank 約 **5.4 倍**（多約 5.13 秒／query），p95 約由 1.42 秒升至 8.08 秒。

### 各方向的主要矩陣平均

Standalone 基準只計入有進入 rerank 矩陣的五款 core embedder（每方向 500 題）；每個 reranker 也是每方向 500 題。表內時間是 rerank 平均 ms。

| 方向 | 設定 | Top-1 | MRR@10 | nDCG@10 | Recall@10 | Recall@50 | Rerank 平均 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| eng-zho（中文→英文） | core standalone | 0.7680 | 0.8406 | 0.8728 | 0.9720 | 1.0000 | — |
| eng-zho（中文→英文） | BGE reranker | 0.8660 | 0.9061 | 0.9263 | 0.9900 | 1.0000 | 932 |
| eng-zho（中文→英文） | Qwen reranker 0.6B | 0.8500 | 0.9099 | 0.9307 | 0.9920 | 1.0000 | 1,223 |
| eng-zho（中文→英文） | Qwen reranker 4B | **0.9520** | **0.9686** | **0.9763** | **1.0000** | 1.0000 | 6,681 |
| zho-zho | core standalone | 0.8260 | 0.8800 | 0.9017 | 0.9680 | 0.9820 | — |
| zho-zho | BGE reranker | 0.8920 | 0.9240 | 0.9360 | 0.9720 | 0.9820 | **763** |
| zho-zho | Qwen reranker 0.6B | 0.9100 | 0.9303 | 0.9390 | 0.9660 | 0.9820 | 1,045 |
| zho-zho | Qwen reranker 4B | **0.9380** | **0.9540** | **0.9595** | **0.9760** | 0.9820 | 5,534 |
| eng-eng | core standalone | 0.7860 | 0.8429 | 0.8689 | 0.9500 | 0.9800 | — |
| eng-eng | BGE reranker | 0.8660 | 0.9061 | 0.9215 | 0.9680 | 0.9800 | **931** |
| eng-eng | Qwen reranker 0.6B | 0.9160 | 0.9372 | 0.9476 | 0.9800 | 0.9800 | 1,220 |
| eng-eng | Qwen reranker 4B | **0.9200** | **0.9426** | **0.9518** | **0.9800** | 0.9800 | 6,660 |

這些是方向 macro 結果，個別 5 embedder × 3 reranker × 3 方向的 66 組細項（Top-1、MRR、nDCG、Recall@10/@50、rerank p50/p95、錯誤數）保留於[完整逐組結果表](run-20261022-v3-guarded/report.md)與[原始 summary](run-20261022-v3-guarded/summary.json)。Standalone 七款的詳細方向列也在該完整表內。

### Qwen embedding 尺寸與 reranker 的實測搭配

每列為同一 embedding/reranker 跨三方向 300 題平均；`embed`、`rerank` 是各階段 query-level 平均時間。此表顯示使用大 reranker 時，embedding 換成 8B 沒有帶來本測試中可見的 Top-1 增益，且排序主要由 reranker 支配。

| Embedder | 排序方式 | Top-1 | MRR@10 | nDCG@10 | Embed 平均 (ms) | Rerank 平均 (ms) |
|---|---|---:|---:|---:|---:|---:|
| Qwen 0.6B | standalone | 0.7467 | 0.8245 | 0.8577 | 12.708 | — |
| Qwen 0.6B | + BGE | 0.8867 | 0.9191 | 0.9325 | 12.708 | **847** |
| Qwen 0.6B | + Qwen 0.6B | 0.8933 | 0.9271 | 0.9396 | 12.708 | 1,127 |
| Qwen 0.6B | + Qwen 4B | 0.9367 | 0.9539 | 0.9612 | 12.708 | 5,951 |
| Qwen 4B | standalone | 0.8233 | 0.8766 | 0.9002 | 40.569 | — |
| Qwen 4B | + BGE | 0.8767 | 0.9173 | 0.9330 | 40.569 | **855** |
| Qwen 4B | + Qwen 0.6B | 0.8933 | 0.9280 | 0.9418 | 40.569 | 1,140 |
| Qwen 4B | + Qwen 4B | 0.9367 | 0.9566 | 0.9649 | 40.569 | 6,027 |
| Qwen 8B | standalone | **0.8567** | **0.8958** | **0.9131** | 51.377 | — |
| Qwen 8B | + BGE | 0.8833 | 0.9186 | 0.9338 | 51.377 | **858** |
| Qwen 8B | + Qwen 0.6B | 0.8933 | 0.9273 | 0.9412 | 51.377 | 1,143 |
| Qwen 8B | + Qwen 4B | 0.9367 | 0.9567 | 0.9650 | 51.377 | 6,046 |

## 解讀、CI 與建置建議

**大 reranker 是否值得：取決於可接受延遲。** 在本機本組件，4B 的 4.47 pp Top-1 增益可能適合離線重排、批次處理，或品質優先且可容忍數秒的情境；若是互動服務，約 6.29 秒平均、8.08 秒 rerank p95 的成本通常很難抵銷這段增益。Qwen 0.6B reranker 是中間點；BGE 平均最快（約 0.88 秒 rerank、1.08 秒 rerank p95；total p95 約 1.11 秒），而 Top-1 比 Qwen 0.6B 低約 1.73 pp。這不是 production SLA：尚未測併發、queue、模型常駐、不同硬體與端到端 router 延遲。

若不加 reranker，Qwen8B 是本輪品質最佳 standalone，但成本是約 51 ms/query embed、三 corpus 索引約 286 秒。若採 rerank，建議先以 BGE 或 Qwen0.6B reranker 做工作負載 A/B，再依品質門檻決定是否升 4B；既然 reranker 已主導排序，Qwen8B + Qwen4B reranker 在此資料上沒有高於 Qwen4B embedder + Qwen4B reranker 的 Top-1，僅有極小 MRR/nDCG 差異。

paired bootstrap 檔包含 45 個逐組 Top-1 差值區間（每組 n=100、2,000 次 bootstrap、固定 seed）。下界高於 0 的組數：BGE 8/15、Qwen0.6B 11/15、Qwen4B 14/15；其餘區間跨 0。這些是**探索性、未做多重比較校正**的逐組區間，不能當成整體泛化或顯著性保證。

主要限制：每方向只抽 100/504、100/504、100/1,148 題；雖保留完整 corpus，但資料仍是三個規模較小的 MLQA validation 子集，不能代表長尾領域、真實使用者分布或其他語言／語料。Q8_0 與本地 Metal backend 也會影響精度及速度。所有延遲都是分階段 warmed query 測量；不含 corpus 索引、模型載入、router/HTTP 外部路徑、併發、cache/DB、使用者網路或 production 排隊。尚未測 k=100。

更多重現命令、模型前綴／pooling／維度與本機路徑限制見[建置指引](BUILD_GUIDE_ZH_TW.md)。本輪沒有變更 production 設定，也沒有把任何結果上傳 GitHub。
