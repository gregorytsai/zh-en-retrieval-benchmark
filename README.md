# 中英檢索實驗室

7 款 embedder、3 款 reranker 的本機中英檢索實測。只有最新完成的 **2026-10-22 guarded v3** 納入排行：22 種組合、3 方向、66 組、6,600 筆測量，重複使用 300 個方向／問題任務。

[繁體中文網站](https://gregorytsai.github.io/zh-en-retrieval-benchmark/) · [English website](https://gregorytsai.github.io/zh-en-retrieval-benchmark/en/) · [完整方法](https://gregorytsai.github.io/zh-en-retrieval-benchmark/methods.html) · [下載與授權](https://gregorytsai.github.io/zh-en-retrieval-benchmark/resources.html)

The complete website is available in Traditional Chinese and English, including all 10 model pages, interactive reranking comparisons, sortable results, methodology and licensing. Original research documents and test materials retain their original language. Contact: [gregorytsai@gmail.com](mailto:gregorytsai@gmail.com).

這是相關文章檢索與排序，沒有生成翻譯；不代表最終 RAG 回答正確率。中文查英文、中文查中文、英文查英文；未測英文查中文。

## 重點

- 五款主要 embedder 配 Qwen 4B reranker 都為 281/300（93.67%），但逐題和完整排序不相同。
- Qwen 4B reranker 相較 0.6B，中查英平均增益 10.2 百分點，英查英 0.4 點；重排平均約從 1.16 秒增至 6.29 秒。
- 單用 Qwen 8B 品質最高（85.67%）；BGE-M3 77.67%／8.14 ms，F2LLM 78%／12.89 ms，只差一題。
- 硬體：M3 Ultra 256 GiB，Metal，Q8_0。容量是模型檔案大小，未測其他硬體的 RAM 下限或延遲。

## 本機檢查與建置

```sh
python scripts/verify_release.py
python scripts/build_site.py
python scripts/package_release.py
python scripts/verify_release.py
python -m http.server 8765 --bind 127.0.0.1
```

建置、封裝與驗證只用 Python 標準函式庫，不連模型、不重跑推論。封裝會統一文字換行並更新公開檔案雜湊。瀏覽 http://127.0.0.1:8765/ 。GitHub Pages 可直接由 main 分支根目錄發佈，已含 `.nojekyll`。

## 資料與程式

- `downloads/results-v3.zip`：完整6600逐題結果（含rerank final_ids/scores）、2100 standalone top100排序、摘要、選題、paired CI。
- `downloads/mlqa-materials.zip`：原始9份Parquet、選用JSON（300題與完整文章庫）、qrels、來源manifest與NOTICE。
- `downloads/benchmark-code.zip` 與 `research/workspace/`：正式與歷史研究程式、保留相依目錄結構。只使用v3結果，歷史程式不是新排行依據。
- `data/all-22-combinations.csv`、`data/all-66-groups.csv`、`data/analysis.json`：所有彙總。
- `research/PUBLICATION_CHANGES.md`：公開版路徑整理、驗證界線。

重新推論需要權重、相容llama.cpp與macOS/Unix程序監控環境；原runner並非Windows一鍵工具。請先閱讀研究建置說明，選新的output與未來deadline，勿resume已完成v3。原始audit_v3.py依賴完整研究封存的attempts/logs；公開成品使用scripts/verify_release.py。

## 授權

自有程式MIT。MLQA素材、節錄與衍生導讀CC BY-SA 3.0，作者／來源／修改聲明見 `data/NOTICE.md`。模型及runtime各自授權，權重不隨本專案散佈。原始資料hash與固定revision保留於manifest。
