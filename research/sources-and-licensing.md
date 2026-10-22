# 官方來源、授權與可強調發現（查核日期：2026-10-22）

供網站編輯使用；以下區分官方已知內容、本機結果與尚未驗證事項。未發布到任何服務。

## 素材授權結論

**可隨專案提供 MLQA / MLQARetrieval 測試素材、節錄與重整版本，但資料部分須保留 CC BY-SA 3.0 與來源署名。**

- [MLQA 原始官方專案](https://github.com/facebookresearch/MLQA#license)明確寫明：資料來自 Wikipedia 段落，採 CC BY-SA 3.0；程式的 LICENSE 另計。不可把原始 repo 程式的授權當作資料授權。
- [MTEB MLQARetrieval 官方資料卡](https://huggingface.co/datasets/mteb/MLQARetrieval)標示 cc-by-sa-3.0，並引用 MLQA。
- [CC BY-SA 3.0 官方摘要](https://creativecommons.org/licenses/by-sa/3.0/)允許複製、分享、改作及商用；需適當署名、附授權連結，衍生作應說明修改、相同授權分享，不另加限制。
- 原始作者：Patrick Lewis、Barlas Oguz、Ruty Rinott、Sebastian Riedel、Holger Schwenk。[ACL 2020 正式論文與引用](https://aclanthology.org/2020.acl-main.653/)，DOI 10.18653/v1/2020.acl-main.653。
- 保留 Wikipedia contributors 的來源說明；若本地上游檔提供文章 title、URL 或識別資料，應完整保留。不要猜造不存在的逐篇 Wikipedia attribution；MTEB corpus 的 title 可能為空，宜透明記載沿用上游所提供的來源層級與 ID。
- 本地 materials-readiness.json 記載 revision `cf59ddd8f4aaf39ce1869361e09252698c340945`。本次已在線核實現行資料卡與原始官方授權；工具無法取得此 revision 的 README / tree / API（shell 亦 DNS 失敗），因此**未聲稱在線重新驗證此 commit 的 metadata**。維持本地 revision / SHA256 證據，網站可提供固定版本下載 URL。
- 專案可將自有程式碼、圖表、第三方資料分層授權；模型權重不必隨專案重傳，提供官方模型卡與本次 GGUF 來源及 SHA256 即可。MIT / Apache 模型授權不會自動成為資料或自有程式的授權。

建議資料 NOTICE：

> 本資料為 MLQA / MTEB MLQARetrieval 的研究用子集與格式轉換，來源為 Patrick Lewis、Barlas Oguz、Ruty Rinott、Sebastian Riedel、Holger Schwenk（ACL 2020）、MTEB 專案及 Wikipedia contributors。MLQA 原文及本資料衍生部分以 CC BY-SA 3.0 分享： https://creativecommons.org/licenses/by-sa/3.0/ 。本專案以固定 revision cf59ddd8f4aaf39ce1869361e09252698c340945 取得資料，依方向抽樣、轉換格式並附加檢索排名結果；並非上游背書。原文若有截短、繁體化或額外翻譯，應逐項標示。

## 不應把檢索結果稱為翻譯結果

MLQA 原始論文是跨語抽取式問答基準，含簡體中文；Wikipedia 文章段落先跨語對齊，再建立／人工翻譯問題及標註答案。中英段落不保證是逐字互譯。

本次 embedding 與 reranker 輸出的是向量、相似度與排名，**沒有模型生成的翻譯**。可提供「中文問題找到英文原文的範例」「找錯文章／漏掉答案的範例」，若為讀者另譯中文解說，應明確標示「網站編輯譯文，非受測模型輸出」，且作為資料衍生部分沿用 CC BY-SA 3.0。不能把未命中的檢索結果描述成翻錯，也不能聲稱這些結果證明翻譯品質。

## 官方模型資料（名稱均指原始模型，本次量化配置另列）

| 模型 | 官方規格與已知內容 | 授權 / 來源 |
|---|---|---|
| Qwen3-Embedding-0.6B / 4B / 8B | 0.6B / 4B / 8B；32K；1024 / 2560 / 4096 維；多語、MRL、自訂任務提示。官方建議 query 加任務說明，跨語情境使用英文提示。 | Apache-2.0；[官方卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)、[論文](https://arxiv.org/abs/2506.05176) |
| Qwen3-Reranker-0.6B / 4B | 0.6B / 4B；32K；100+ 語言；query-document 配對評分。官方格式與 yes/no 評分需正確，不能當一般文字生成 API。 | Apache-2.0；[0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)、[4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) |
| BGE-M3 | 約 0.6B；1024 維；8192 tokens；100+ 語言；支援 dense、sparse、多向量。本測若只有 dense，不代表完整混合檢索能力。官方不要求 query 加 instruction。 | MIT；[官方卡](https://huggingface.co/BAAI/bge-m3) |
| BGE-reranker-v2-m3 | 約 0.6B；以 BGE-M3 為基底，多語 cross-encoder；直接對問題與段落評分，官方定位較省資源。 | Apache-2.0；[官方卡](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| EmbeddingGemma 2 | **全文字部分 270M，完整多模態 740M**；768 維，支援 512 / 256 / 128 維截短；8192 tokens；100+ 語言。文字測試不要稱 740M 文字模型，也不要誤植第一代 308M。官方建議 BF16 / FP32，明確警告 FP16 可 NaN 或降質；該警告不能直接證明本次 Q8 GGUF 有問題。 | Apache-2.0；[Google 官方卡](https://huggingface.co/google/embeddinggemma-2) |
| F2LLM-v2-0.6B | 0.6B；1024 維；Qwen3 架構；多語指令 embedding；query 需 prompt，documents 不加；MRL 截短後重新 normalize。全系列涵蓋 80M–14B，官方訓練強調中低資源語言。 | Apache-2.0；[官方卡](https://huggingface.co/codefuse-ai/F2LLM-v2-0.6B)、[論文](https://arxiv.org/abs/2603.19223) |
| Granite-embedding-97m-multilingual-r2 | 97M；384 維；32768 tokens；ModernBERT 架構；52 種語言明確強化訓練，200+ 來自預訓練支援。小型、經剪枝與蒸餾；官方跨任務表不能直接拿來對比這份 MLQA Top-1。 | Apache-2.0；[IBM 官方卡](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2)、[論文](https://arxiv.org/abs/2605.13521) |

官方最大 context 是模型能力上限，不等於本次 llama.cpp server / batch 的實際可用長度。記憶體須區分 GGUF 檔案大小、量到的 RSS、實際裝置可用空間，不能由參數數直接保證 8GB / 16GB 裝置效能。

## 已知共識與值得突出的本地觀察

**已知，不能包裝成首創：** 多語 embedding 可直接做跨語搜尋；先召回再 rerank 為常見管線；較大模型與耗時／記憶體取捨；模型提示、pooling、量化與轉換正確性會影響結果；小型 encoder 可提供較低計算成本。Qwen 與 BGE 官方卡已支持前幾項。

**適合標示「這次測試觀察到」的結果（以本地 RESULTS_OVERVIEW_ZH_TW.md 與最新 v3 report 為證）：**

1. Qwen 4B reranker 相對 0.6B 的平均中查英 Top-1 多 10.2 個百分點（95.2% vs 85.0%），英文查英文僅多 0.4 點（92.0% vs 91.6%）。多出的計算預算在本資料的跨語方向較有價值。不得稱全球首次、普遍規則或顯著差異，除非另核對成對統計。
2. 固定相同 Qwen 4B reranker 時，Qwen 4B / 8B embedding 的三方向平均 Top-1 同為 93.67%；standalone 則 82.33% vs 85.67%。這是在小候選庫、top-50 幾乎找全相關文件時，增大 embedder 的第一名優勢縮小；不是證明所有資料庫 4B 等同 8B。
3. Granite 97M r2 standalone 約 4.13 ms 平均 query+exact search、Top-1 65.67%；BGE-M3 約 8.14 ms、77.67%；F2LLM 78.00%、EmbeddingGemma 2 76.33%。可呈現品質與等待時間取捨；一兩百分點不要宣稱穩定勝負。
4. BGE / Qwen 0.6B / Qwen 4B reranker 的一次 50 篇平均排序時間約 0.875 / 1.163 / 6.291 秒，而 Top-1 平均 87.47% / 89.20% / 93.67%。這是實測 Apple M3 Ultra 的局部觀察，不是一般筆電的預期速度。

新意在同一套本機 GGUF / Metal 環境、相同中英方向抽樣與候選設定下，讓讀者看到可重現的配對與成本差異；本次網路查核沒有完成窮盡的文獻新穎性審查。避免「首次發現」「打敗所有模型」。

## 網站編輯注意

- 三方向各 100 題，6,600 筆為不同組合反覆評估，**不是 6,600 個獨立問題**。同一 reranker 跨五個 embedder 的 1,500 次也非 1,500 個獨立題目。
- F2LLM 與 Granite 只有 standalone；其 reranker 組合只能標「尚未測試」，不可因尺寸小就推薦成已實測最佳組合。
- 不提供英文查中文方向結論，本次未測此方向。
- Top-1 指找對相關文件，不是翻譯正確率，也不是最終 RAG 回答正確率。
- 不將整體平均當作方向相同；首頁同時列出三方向、品質、時間、測試數與硬體。
- 原始 README 還有「正式測試尚未開始」的早期文字；最新整體報告與 v3 completion 應作首頁依據，避免直接公開舊 README 為最新說明。
