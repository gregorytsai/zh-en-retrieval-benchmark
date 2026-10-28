# ACORD judged-top50-v1

協定版本：`judged-top50-v1`。本文件定義施測方法，不代表正式執行已完成；實際狀態、run ID、施測日期與驗收結果以 run manifest、逐題檔案及 validation report 為準。

## 範圍與來源

本輪只重測 `acord-eng-eng`，不修改 MLQA、MIRACL、歷史 run、舊向量快取或網站。ACORD 維持獨立法律專項，不加入首頁五分項或六分項平均。

沿用 ACORD revision `7ba1c6f2e0baaa32e78c78c2b02e9c57b102af4e` 的既有凍結 test 子集：57 個問題、3,931 篇條款、61,988 個明確標註的 query-document 配對。每題已標註條款數範圍421–1,342。保留 query/corpus 順序、原始 ID、文字與 qrels；官方 category、parent_query_id、type 等資料保存在 `metadata`，研究自行衍生欄位留在 `study_metadata`，不可混充官方來源群聚。

資料 qrels 已把1–5星轉成0–4分，不能再次減1。明確 grade=0 是已標註負例；qrels 中缺少的配對是未知，不能當成0。資料 hash、順序與來源證據必須通過才可施測；數量不符須定位差異，不改題或抽樣補數。

來源：[ACORD dataset](https://huggingface.co/datasets/theatticusproject/acord)，CC BY 4.0。理論依據：[ACORD paper](https://aclanthology.org/2025.acl-long.1206.pdf)，C.5與F.1。衍生變更是候選及評分協定，沒有改寫來源問題、條款或標籤。

## 模型與凍結條件

| 批次 | Embedder |
|---|---|
| A01 | qwen-4b |
| A02 | qwen-8b |
| A03 | embeddinggemma-2（精確 registry 版本） |
| A04 | qwen-0.6b |
| A05 | bge-m3 |
| A06 | f2llm-v2-0.6b |
| A07 | granite-97m-r2 |

每批四種設定：`standalone`、`bge-reranker`、`qwen-reranker-0.6b`、`qwen-reranker-4b`。每組57題，共28組1,596筆，包含399筆 standalone、1,197筆 reranker 結果。

十個模型沿用原 Q8_0 檔案、SHA256、pooling、向量維度、prefix、instruction、reranker template、context及執行參數。模型名稱不足以證明相容性，須查 `models-manifest.json` 及原始 registry 的交叉核對。實機須記錄CPU/GPU、記憶體、作業系統、llama.cpp build與binary及依賴雜湊，載入後另查實際GPU offload。

不得自動下載替代權重或升級runtime。需要改設定時另建版本與run，保留舊證據，不能把不同條件的時間混合比較。歷史manifest只提供launcher等既有pins、未完整保存動態函式庫hash；本輪另建立函式庫inventory，不倒推宣稱所有歷史library位元組身份均已核實。

## 候選與排序

對每個 Embedder e、問題 q：

1. 以沿用的正規化 float32 向量對全部3,931篇條款計算 cosine（正規化後 dot product）。保存全部文件的原始rank、ID、有限相似度與是否有qrel。
2. 相似度降冪排列；數值相同時依原始 `doc_id` 字典序升冪，不能用grade或輸入位置打平。
3. 依文件ID是否存在該題qrels移除未知配對，保留明確grade=0；再取前50篇形成有序 `C(e,q)`。不可先取50才篩未知，也不可補入正解、按星等挑選或改成正例池。
4. 同一 Embedder 的三種 Reranker 必須收到完全相同、順序相同的50篇原始條款。以 ordered candidate SHA256驗證；不同 Embedder 的候選可以不同。
5. Reranker只接收原始query/document文字和固定模板，不接收grade、qrels、標籤或參考答案；必須返回50個有限分數，保留原candidate index。最終依分數降冪、doc ID升冪排序，final ID集合必須等於candidate集合。
6. standalone用完整 judged-only 排名評估；reranker用新候選的重排結果評估。兩者IDCG均使用該題完整qrels。

有序候選SHA256採UTF-8 canonical JSON：`ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False`，對有序ID陣列hash。完整排名及其他JSON產物採相同canonical規則，檔案則另做原始bytes SHA256。

**評估限制：**此流程利用了「哪些配對有人工標註」的資訊。真實RAG通常沒有此資訊；本輪分數不等於一般部署命中率、答案正確率或完整端到端搜尋表現。主協定的judged coverage必然100%，不能視為模型進步。未篩選原始Top50的coverage僅可另列診斷，不混入主排行榜；一般部署unfiltered Top50須另立protocol ID。

## 品質指標

令 `g(q,d)` 為已存在qrels中的grade（0–4）。所有品質平均以query等權；下載保存完整精度，文章nDCG顯示三位小數、百分比兩位。每個指標同時報有效題數。

### 線性 nDCG@5／@10

`DCG@k = Σ(i=1..k) g(q,d_i) / log2(i+1)`。

`IDCG@k`：對該題**完整qrels**的grade由高到低排序後套用相同公式。`nDCG@k = DCG@k / IDCG@k`。禁止指數gain `2^grade−1`，禁止只以候選池正規化。IDCG為0時保存null並從該指標平均排除。

### 二元指標

本站門檻為原始星等≥3，即grade≥2；不是grade>0。

- Top-1：第一篇是否達門檻。
- MRR@10：前10篇首個達門檻者的倒數rank，沒有則0。
- Hit@10：前10篇是否至少一篇達門檻。
- Recall@10：前10篇達門檻的唯一文件數／完整qrels中達門檻的文件總數。
- candidate Recall@50：候選50篇達門檻的文件數／完整qrels中達門檻的文件總數。

Recall分母為0則null並排除平均。重排不改候選，因此同一Embedder的三個reranker candidate Recall@50必須相同。

### 3／4／5星 normalized P@5

三個grade門檻分別為2、3、4。分子為前5篇達門檻的數量；分母為 `min(5,完整qrels中達門檻的文件數)`。零分母保存null、排除該指標平均並報有效題數，不補0。本凍結資料5星可評題數為29。

名稱必須保留normalized，不混稱一般固定分母為5的Precision@5。不同門檻有效題集合可能不同，配對比較也必須按共同有效query對齊。

## 快取與真正推論

可重用驗證通過的完整corpus/query向量做品質準備；須核模型、資料、prefix、pooling、dimension、normalization、ID順序、cache-manifest及向量檔SHA。原始向量不得放入公開套件。

重用前須由完整向量重現399份舊raw Top50，並重現七個公開修正standalone結果。公開資料只是回歸證據，不可倒推調整新排名。已知公開ZIP的69份judged尾端有相鄰排名差異；所有差異在Top50之外，本機原公式、候選及品質回歸相符。ZIP缺少原始逐文件cosine，不能確定其差異來源；完整診斷保存於cache regression report。

三種reranker都必須對本輪新候選真正推論，不拼接歷史分數。中斷後只可重用本輪已接受且輸入、模型、資料與程式條件相容的checkpoint，保存來源attempt與恢復標記。只重跑失敗或缺失單位，不依分數高低挑測。

## 正式時間邊界

每組57題至少一次正式測量；暖機另記且不計入正式n。每次正式請求均實際執行新的query embedding、全語料搜尋、judged篩選及指定reranker。語料向量可預載；不能把query快取讀取或歷史推論時間當新測量。

| 欄位 | 邊界 |
|---|---|
| query_embedding_ms | 本次query embedding推論及該階段輸入/輸出處理 |
| retrieval_ms | 對完整語料的相似度計算與完整排序 |
| evaluation_filter_ms | 依qrels membership篩選及建立有序50篇候選 |
| rerank_ms | 本次完整50篇推論與該階段結果處理；standalone為0 |
| total_ms | 上述四階段加總 |
| wall_total_ms | 以單調時鐘包住整個正式query流程，另列與階段加總的差距 |

JSON內 `times_ms` 使用上述名稱去掉 `_ms` 後綴。各階段計時採同一邊界，單獨記錄完整HTTP request trace；loopback HTTP傳輸包含在對應模型階段，無法可靠拆分時不得填0或聲稱已扣除。模型載入、語料準備、預熱及等待分開記錄，不偷加／偷扣。記錄guard與證據生成造成的wall差距。

各階段均報mean、p50、p95、樣本數；分位數採Hyndman–Fan type7（線性插值）。額外計時重複用repeat_id，不增加獨立query樣本數。權重GiB是檔案容量，不是RAM；無可靠peak-memory測量就留null。

## 執行、容錯與驗收

先完成獨立評分與邊界測試，再對十模型做完整token/prefix/template稽核與固定ID smoke。Qwen reranker使用GGUF中的精確rerank template、相同placeholder替換順序及special-token設定；BGE驗證fallback三個special tokens。對本輪所有實際候選配對檢查完整模板長度，禁止靜默截斷。smoke不得充正式結果，不按其分數挑設定。

單一controller依A01–A07序列執行並持有全域鎖。Embedder與reranker可同時載入，但實際推論序列化。guard每秒檢查其他llama-server及router；外部活動或監控失效只停止本run明確擁有的測試server。受干擾單位作廢，30秒連續quiet後按相同條件重跑；不得停止production、改router或繞過guard。

每批有限wall deadline含等待，不延長原deadline。已完成正式資料與失敗紀錄保留；需接續則依相容性建立新的有界run，不因粗估超過10小時放棄整體，也不無限重試。

正式驗收至少包含：28唯一設定、每組57唯一query、1,596品質rows、399完整排名、399個50篇有序候選，以及59,850個reranker pair scores。每次正式embedding/search/rerank須有可關聯的request與retrieval evidence；不能用自報provenance代替實際執行查核。獨立驗證器從原始排名、qrels與配對分數重新計分，不匯入runner的評分函式。

空排名、重複ID、漏文件、非有限值、截斷、hash mismatch及錯誤必須記錄，不視為0分。最終完整性報告區分未解決錯誤與已重跑的歷史失敗/重試。僅數值驗證通過仍不足以寫complete，還須查模型/程式/資料freeze、token/smoke、GPU、guard、accepted checkpoint與交付checksum。未通過保持incomplete。

## 分析與論文差異

比較每個Embedder加reranker後的提升或下降、候選Recall與候選內排序差距，以及4/5星條款在前5名的表現。候選oracle只做事後分析：在固定候選內按grade理想排序，但IDCG仍為full qrels；絕不把oracle或grade供應模型。

指定比較Qwen4B＋Qwen0.6B與BGE-M3＋Qwen0.6B，按同query報勝／負／平手與各指標差值。官方category/type可做描述性分項，n<5只列題數；較大分項也不宣稱穩定優勢。原始57題的parent_query_id為空，未確認可靠來源群聚前不做bootstrap。不得將28×57當作1,596個獨立樣本；不自行編造群聚或法律領域標籤。本輪不做多重比較校正或統計顯著宣稱。

論文C.5/F.1支持unknown處理、線性DCG與normalized星等P@5。**judged-top50-v1是本站本輪明定設計**；論文模型、資料切分與兩階段流程（BM25 Top100）不同，不聲稱論文採相同GGUF模型、50候選或本站重排流程，不宣稱完整複製官方leaderboard。

最後產生完整離線包與獨立可發布allowlist包。公開包只含必要程式、來源/授權資訊、排名、分數、結果與分析；排除憑證、私有主機資訊、個人路徑、完整runtime logs、模型權重與向量快取。文章不新增真實題目案例。NAS必須用新的唯一資料夾、同步後核對SHA256；本任務不直接發布GitHub網站。
