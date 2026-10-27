"""Render the verified complete 7 x 4 x 6 matrix in both languages."""
import json,statistics,re,csv
import build_latest as old
ROOT=old.ROOT
ORDER=old.ORDER
LABELS=old.LABELS
tab=old.tab
tr=old.tr
A=json.loads((ROOT/'data/matrix/analysis.json').read_text('utf-8'))

C=json.loads((ROOT/'data/matrix/acord-corrected.json').read_text('utf-8'))
FIVE=[d for d in ORDER if d!='acord-eng-eng']
def group(scope):
 if scope=='acord-eng-eng':return C['rows']
 ids=FIVE if scope=='all' else ORDER[:3] if scope=='mlqa' else [scope]
 groups={}
 for r in A['rows']:
  if r['dataset'] in ids:groups.setdefault((r['embedding'],r['reranker']),[]).append(r)
 out=[]
 for (e,rr),rows in groups.items():
  assert len(rows)==len(ids)
  item={'embedding':e,'reranker':rr,'queries':sum(r['queries'] for r in rows),'datasets':len(rows),'weight_gib_sum':rows[0]['weight_gib_sum']}
  for k in ['top1','ndcg10','mrr10','recall50','judged_coverage50','total_ms_mean']:item[k]=statistics.mean(r[k] for r in rows)
  for k in ['total_ms_p50','total_ms_p95']:item[k]=rows[0][k] if len(rows)==1 else None
  out.append(item)
 return sorted(out,key=lambda r:-r['ndcg10'])
def name(k,lang):return old.name(k,lang)
def combo(r,lang):return old.combo(r,lang)
def pick(e,rr):return next(r for r in group('all') if r['embedding']==e and r['reranker']==rr)
def page(path,title,body,lang):
 s=old.page(path,title,body,lang)
 return s.replace('六分項等權平均','五分項等權平均，ACORD獨立重算').replace('six-task macro averages','five-task macro averages and separately corrected ACORD').replace('2026-10-24 01:47','2026-10-26 05:31').replace('24 Oct 2026, 01:47','26 Oct 2026, 05:31').replace('14,478','73,164')
def findings(lang):
 t=lambda z,e:tr(z,e,lang)
 items=[
 (t('記憶體有限，先考慮小模型加小重排器。','On a tight memory budget, consider a small embedder and small reranker.'),t('Qwen 0.6B 加上 0.6B 重排器後，五分項平均 nDCG 從 0.82 升至 0.89，Top-1 從 70.34% 升至 81.57%，超過單用 Qwen 8B 的 0.87／77.24%。權重合計約 1.19 GiB，但平均查詢時間約 1.08 秒，比單用 8B 的約 52 ms 更久。','Adding a 0.6B reranker to Qwen 0.6B raises five-task mean nDCG from 0.82 to 0.89 and Top-1 from 70.34% to 81.57%, above Qwen 8B alone at 0.87 / 77.24%. Combined weights are about 1.19 GiB, but mean query time is about 1.08 s versus 52 ms for 8B alone.')),
 (t('中文提問，也能直接找到英文資料。','Chinese questions can retrieve English sources directly.'),t('中文查英文是本次雙語實測的重點。Qwen 0.6B＋0.6B 的第一名命中率由單用小模型的 72.82% 升至 86.31%；Qwen 8B＋4B 達 92.26%，nDCG 為 0.95。這是跨語言檢索能力，不是翻譯或生成回答的正確率；本批未測英文查中文。','Chinese-to-English retrieval is a focus of this benchmark. Qwen 0.6B + 0.6B improves first-result hits from the small embedder’s 72.82% to 86.31%. Qwen 8B + 4B reaches 92.26% and nDCG 0.95. This measures cross-language retrieval, not translation or generated-answer accuracy. English-to-Chinese was not tested in this matrix.')),
 (t('法律條款較難，需看條款品質與評分規則。','Legal clauses need closer attention to quality and scoring.'),t('ACORD 七個純 Embedder 修正後 nDCG 約 0.52–0.70，數值低於本次多數百科分項，但題目與標註標準不同，不能當成同難度考試。公開研究也指出高品質法律條款難以排到前面；原先更低的分數另受未標註資料計分影響，不能全歸因於模型。','Corrected ACORD nDCG is about 0.52–0.70 for the seven standalone embedders, numerically below most encyclopedia conditions here. Different queries and labels prevent a controlled difficulty comparison. Published research also finds high-quality clauses hard to rank. The earlier much lower scores were additionally affected by unjudged-document handling.')),
 (t('最高品質與最快回應，是不同選擇。','Best ranking quality and fastest responses favor different setups.'),t('Qwen 8B＋4B 的五分項平均最高，但與 Qwen 4B＋4B 僅差約 0.001。七個 Embedder 都以 4B 重排器的平均 nDCG 最高；其重排階段中位數約 6.21 秒，0.6B 約 1.13 秒。小差距不代表顯著勝出，速度與容量也應一起考量。','Qwen 8B + 4B has the highest five-task mean, only about 0.001 ahead of Qwen 4B + 4B. All seven embedders score highest with 4B reranking, whose median reranking stage takes about 6.21 s versus 1.13 s for 0.6B. Small differences do not establish significant wins; consider latency and size too.'))]
 return '<section id="findings"><p class="eyebrow">KEY FINDINGS</p><h2>'+t('這次實測的整體結論','Overall conclusions')+'</h2><div class="findings">'+''.join('<article class="finding"><h3>'+h+'</h3><p>'+b+'</p></article>' for h,b in items)+'</div><p class="small">'+t('本機固定資料下的描述性結果；未計算全域信賴區間。','Descriptive results on fixed local inputs; no global confidence intervals were computed.')+' <a href="latest-methods.html#published">'+t('公開研究對照','Published research')+'</a> · <a href="acord.html">'+t('法律檢索完整說明','Legal retrieval details')+'</a></p></section>'

def recommendations(lang):
 t=lambda z,e:tr(z,e,lang)
 choices=[('qwen-reranker-0.6b',t('成效優先','Quality first'),t('同樣使用Qwen 4B Embedder，搭配Qwen 0.6B重排器，五分項平均nDCG與Top-1較高；適合願意多等一些時間、優先考慮搜尋品質的使用者。','With the same Qwen 4B embedder, the Qwen 0.6B reranker gives a higher five-task mean nDCG and Top-1. Choose it when retrieval quality matters more than the additional wait.')),
 ('bge-reranker',t('速度優先','Speed first'),t('保留Qwen 4B Embedder，改搭BGE重排器。這兩組相比，平均查詢總時間較短，排序品質略低；適合更在意回應速度的使用者。','Keep Qwen 4B embeddings and use the BGE reranker. Of these two setups, it has a shorter mean total query time with slightly lower ranking quality. Choose it when responsiveness matters more.'))]
 b='<section id="choose"><h2>'+t('兩組實用推薦：成效或速度','Two practical choices: quality or speed')+'</h2><p>'+t('以相同Qwen 4B Embedder搭配不同重排器，依搜尋品質與等待時間取捨。兩組權重皆約4.58 GiB；還需預留向量索引、執行緩衝與系統記憶體。','Pair the same Qwen 4B embedder with different rerankers to choose between quality and wait time. Both use about 4.58 GiB of weights; allow extra memory for the index, runtime buffers and operating system.')+'</p><div class="recommend-grid">'
 for rr,h,why in choices:
  r=pick('qwen-4b',rr)
  b+='<article class="scorecard"><h3>'+h+'</h3><p><b>'+combo(r,lang)+'</b></p><p>'+f'nDCG {r["ndcg10"]:.4f} · Top-1 {r["top1"]*100:.2f}%'+'</p><p>'+t('平均查詢總時間：','Mean total query time: ')+f'{r["total_ms_mean"]/1000:.2f} '+t('秒','s')+'</p><p>'+why+'</p><a href="models/'+rr+'.html">'+t('查看重排器詳細結果','View reranker details')+' →</a></article>'
 return b+'</div><p class="small">'+t('以上品質與時間採相同五分項等權平均，時間包含問題向量化、向量檢索及重排。速度測於M3 Ultra／256 GiB；容量為權重檔案大小，非最低RAM／VRAM需求。','Quality and timing use the same five-task equal-weight mean. Time includes query embedding, vector search and reranking, measured on M3 Ultra / 256 GiB. Size means weight files, not minimum RAM/VRAM.')+'</p></section>'

def home(lang):
 t=lambda z,e:tr(z,e,lang);pre='../' if lang else ''
 featured=[('qwen-4b','qwen-reranker-0.6b',t('成效優先','Quality first')),('qwen-4b','bge-reranker',t('速度優先','Speed first'))]
 cards=''
 for e,rr,tag in featured:
  r=pick(e,rr);cards+=f'<article class="scorecard"><span>{tag}</span><div class="score-number"><strong>{r["ndcg10"]:.2f}</strong><span> / 1.00</span></div><div class="score-meter" role="meter" aria-label="nDCG@10" aria-valuemin="0" aria-valuemax="1" aria-valuenow="{r["ndcg10"]:.4f}" aria-valuetext="{r["ndcg10"]:.2f} / 1.00"><span style="width:{r["ndcg10"]*100:.3f}%"></span></div><div class="score-scale"><span>0</span><span>{t("滿分 1.00","Maximum 1.00")}</span></div><small>{t("五分項平均 nDCG@10","Five-task mean nDCG@10")}</small><b>{combo(r,lang)}</b><small>Top-1 {r["top1"]*100:.2f}% · {r["weight_gib_sum"]:.3f} GiB</small><small>{t("平均查詢總時間","Mean total query time")} {r["total_ms_mean"]/1000:.2f} {t("秒","s")}</small></article>'
 b=f'<section class="phase-hero"><p class="eyebrow">2026-10-26 · COMPLETE 28-SETUP MATRIX</p><h1>{t("用同一套題目完整比較<br><span class=hero-combinations>28種Embedder X Reranker<span class=keep-phrase>搜尋組合</span></span>","One consistent set of questions.<br>28 Embedder X Reranker setups")}</h1><p class="lead">{t("七個 Embedder，搭配不重排或三種重排器。先看五分項綜合平均，再按語言、品質與<span class=keep-phrase>模型容量挑選。</span>","Seven embedders, each alone or with three rerankers. Start with the five-task mean, then choose by language, quality and model size.")}</p><div class="headline-scores two-recommendations">{cards}</div><p class="recommend-context">{t("兩組皆使用 Qwen 4B Embedder：Qwen 0.6B 重排的成效較高、等待較久；BGE 重排則較快。","Both use Qwen 4B embeddings: Qwen 0.6B reranking scores higher with a longer wait; BGE reranking is faster.")}</p><p class="macro-note">{t("nDCG 衡量前十名排序品質，越接近 1 越好，不是命中百分比。Top-1 是第一筆命中的比例。五分項各占 1/5；ACORD 獨立呈現，不納入平均。分數依本測試相關性標準計算，不代表所有領域的普遍排名。","nDCG measures top-ten ranking quality: closer to 1 is better, and it is not a hit percentage. Top-1 is the first-result hit rate. Each of five conditions weighs 1/5; ACORD is reported separately. Scores use this study’s relevance definitions and are not universal rankings.")} <a href="latest-methods.html#average">{t("平均怎麼算","How averages work")} →</a></p></section>'
 b+='<div class="evidence-strip">'+''.join('<div><strong>'+v+'</strong><span>'+l+'</span></div>' for v,l in [('28',t('完整受測組合','complete setups')),('5 + 1',t('五項平均＋法律專項','mean tasks + legal task')),('73,164',t('逐題測量／168組摘要','measurements / 168 summaries')),('2,613',t('資料集內唯一題目','dataset-qualified queries'))])+'</div><p class="context-line">Q8_0 · M3 Ultra / Metal · 256 GiB · '+t('同一題在28種組合重測，不能視為73,164個獨立樣本。','Each query is measured across 28 setups; these are not 73,164 independent samples.')+'</p>'
 b+=findings(lang)
 b+='<section id="results"><h2>'+t('完整比較：切換分項、排序指標與重排器','Compare every setup: task, metric and reranker')+'</h2><p>'+t('綠色是相同 Embedder 單獨使用的分數；橘色是重排後提升，紅色斜線是下降。圖表預設顯示前7名，可展開全部。表格列出所選條件的全部組合。','Teal shows the same embedder alone, orange the reranking gain, and red hatching a loss. The chart initially shows the top seven; expand it to see all. The table includes every setup matching your selection.')+'</p><div class="controls">'
 scopes=[('all',t('綜合平均｜五分項等權','Overall | five equal-weight conditions')),('mlqa',t('MLQA 平均｜三方向等權','MLQA | three equal-weight directions'))]+[(d,LABELS[d][lang]+(t('｜修正純Embedder',' | corrected standalone') if d=='acord-eng-eng' else '')) for d in ORDER]
 metrics=[('ndcg10',t('nDCG@10｜前十名排序品質','nDCG@10 | top-ten ranking quality')),('top1',t('Top-1｜第一名命中率','Top-1 | first-result hit rate')),('mrr10','MRR@10'),('recall50','Recall@50')]
 rr=[('all',t('全部28組','All 28 setups')),('standalone',t('不加重排器','No reranker'))]+[(r,name(r,lang)) for r in A['reranker_timing']]
 for key,label,options in [('scope',t('測試範圍','Test scope'),scopes),('metric',t('品質指標','Quality metric'),metrics),('reranker',t('顯示重排器','Show reranker'),rr)]:b+='<label>'+label+f'<select id="{key}">'+''.join(f'<option value="{k}">{v}</option>' for k,v in options)+'</select></label>'
 b+='</div><p id="scope-note" class="note" aria-live="polite"></p><div class="phase-legend"><span class="base-key">'+t('單獨檢索','Embedding alone')+'</span><span class="gain-key">'+t('提升','Gain')+'</span><span class="loss-key">'+t('下降','Loss')+'</span></div><div id="phase-bars" aria-live="polite"></div><button type="button" id="toggle-bars" class="chart-toggle"></button><p class="small">'+t('nDCG／MRR 的色條以分數×100呈現；差值以原始分數呈現。Top-1／Recall 差值使用百分點（pp）。','nDCG/MRR bars use score ×100; deltas use raw score units. Top-1/Recall deltas use percentage points (pp).')+'</p><h3>'+t('28 種組合完整比較｜點欄位排序','All 28 setups | click a column to sort')+'</h3><div id="phase-table" class="table-scroll" tabindex="0"></div><p class="small">'+t('查詢總時間＝問題向量化＋向量檢索＋所選重排器時間；不含生成回答。綜合時間是五分項逐題平均的等權平均。p50／p95只在單一分項顯示，不直接平均百分位數。ACORD 重算未重新量時，時間顯示「—」。','Total query time = query embedding + vector search + the selected reranker; answer generation is excluded. Aggregate time equally weights five task means. p50/p95 appear only for individual conditions. Corrected ACORD latency was not remeasured and is shown as —.')+'</p></section>'
 b+=recommendations(lang)
 winners=[]
 for d in ORDER:
  rows=group(d);best=rows[0];top=max(r['top1'] for r in rows);meta=A['datasets'][d]
  winners.append([LABELS[d][lang],meta['queries'],f'{meta["corpus"]:,}',combo(best,lang),f'{best["ndcg10"]:.4f}',f'{top*100:.2f}%'])
 b+='<section><h2>'+t('五分項與法律專項，各自比較','Five conditions and a separate legal task')+'</h2>'+tab([t('分項','Condition'),t('題數','Queries'),t('文章數','Documents'),t('nDCG最高組合','Highest-nDCG setup'),'nDCG@10',t('分項最高Top-1¹','Best task Top-1¹')],winners)+'<p class="small">'+t('¹最高Top-1可能來自另一組合，請用上方互動表按Top-1排序。ACORD 僅比較七個純 Embedder 的修正結果，Top-1 以3星以上判定；不含重排器。MIRACL 是有限語料子集。','¹Best Top-1 may belong to a different setup; sort the interactive table by Top-1. ACORD compares only seven corrected standalone results, with Top-1 requiring at least 3 stars; rerankers are excluded. MIRACL uses a bounded corpus subset.')+'</p></section>'
 b+='<section><h2>'+t('想看完整方法、資料或模型？','Explore methods, data and models')+'</h2><p><a href="latest-methods.html">'+t('完整說明與下載','Methods and downloads')+'</a> · <a href="models.html">'+t('每個模型的詳細頁面','Detailed model pages')+'</a> · <a href="'+pre+'data/matrix/summary.csv">168 '+t('分項結果CSV','task results: CSV')+'</a> · <a href="phase2.html">'+t('10/24前一輪結果','Previous Oct 24 results')+'</a> · <a href="baseline.html">'+t('10/22歷史22組','Historical Oct 22 matrix')+'</a></p></section>'
 payload={'corrected':True,'lang':lang,'names':{k:name(k,lang) for k in A['names']},'labels':{k:v[lang] for k,v in LABELS.items()},'datasets':A['datasets'],'scopes':{k:group(k) for k in ['all','mlqa']+ORDER}}
 b+='<script id="phase-data" type="application/json">'+json.dumps(payload,ensure_ascii=False).replace('</','<\\/')+'</script><script src="'+pre+'assets/phase2.js" defer></script>'
 return page('index.html',t('地端RAG中英檢索模型實測｜28種組合','28-setup bilingual retrieval benchmark'),b,lang)

def methods(lang):
 t=lambda z,e:tr(z,e,lang);pre='../' if lang else ''
 b='<section class="phase-hero"><p class="eyebrow">METHODS / REPRODUCIBILITY</p><h1>'+t('測試怎麼做，結果能說到哪裡。','How the test works, and what it can establish.')+'</h1><p class="lead">'+t('28 組以相同五分項比較，法律條款獨立呈現。完整原始結果、修正結果與程式都保留。','Compare 28 setups on the same five conditions, with legal clauses reported separately. Original results, corrections and code remain available.')+'</p></section>'
 sections=[
 ('average',t('首頁平均如何計算？','How is the homepage mean calculated?'),t('先在各分項內平均全部題目，再對 MLQA 三方向與 MIRACL 中英文共五項各給 1/5 權重。MLQA 合占60%，MIRACL 占40%；不是資料家族等權，也不是實際使用流量。28組皆有完整五項，缺項不補零。ACORD 因評分規則不同且重排待補測，暫不納入。首頁顯示兩位小數與固定0–1量尺；完整表保留四位小數。nDCG 是排序品質，不是回答正確率。','Average every query within each condition, then give the three MLQA directions and two MIRACL languages weight 1/5 each. MLQA contributes 60% and MIRACL 40%; this is not family weighting or production traffic. All 28 setups cover all five tasks. ACORD is excluded while corrected reranking is pending. Cards use two decimals on a fixed 0–1 scale; tables keep four decimals. nDCG is ranking quality, not answer accuracy.')),
 ('fairness',t('相同題目、語料與候選數，具體是什麼？','What does consistent testing mean?'),t('同一分項內固定題目、語料、相關性標註與模型版本。每個Embedder取前50篇，三個重排器共用該Embedder的同一份有序候選。不同Embedder可以找到不同文章。重排只能重新排序，不能找回候選外的文章。73,164筆原始測量是2,613個資料集內題目在28配置重複測量；不是73,164個獨立樣本。','Each condition fixes queries, corpus, judgments and model versions. An embedder supplies the same ordered Top50 candidates to all three rerankers. Different embedders may retrieve different documents. Reranking cannot recover documents outside that pool. The 73,164 original measurements repeat 2,613 dataset-qualified queries across 28 setups; they are not independent samples.')),
 ('scope',t('資料範圍與雙語方向','Data scope and bilingual direction'),t('MLQA validation 三方向為中文查英文504題、中文查中文504題、英文查英文1,148題，包含前輪題目。資料檔 eng-zho 指英文context／中文question，網頁依查詢→文章顯示中文查英文，不代表英文查中文。MIRACL每語言200道dev題，語料是全部dev已標註段落聯集：中文3,786、英文7,921篇，不是完整百萬級語料。ACORD為57道test題及3,931條款。MLQA／MIRACL採二元相關性；ACORD修正定義見法律專頁。','MLQA validation includes Chinese → English (504), Chinese → Chinese (504), and English → English (1,148), including earlier queries. The source key eng-zho means English context / Chinese question: the UI correctly labels it Chinese → English, not the reverse. MIRACL uses 200 dev questions per language and the union of all dev-judged passages: 3,786 Chinese and 7,921 English, not the full million-scale corpus. ACORD has 57 test questions and 3,931 clauses. MLQA/MIRACL labels are binary; see the legal page for corrected ACORD definitions.')),
 ('uncertainty',t('第一名一定最好嗎？','Is first place a reliable winner?'),t('這是固定資料的點估計，沒有全域28組bootstrap區間或多重比較校正。前兩名五項平均僅差約0.001。MIRACL候選Recall@50接近滿分，子集對Embedder的區辨力有限；不能推論大型資料庫裡Embedder不重要。ACORD僅57題，也不能由一項測試推論整體法律能力。','These are point estimates on fixed data, without global 28-setup bootstrap intervals or multiplicity correction. The top two five-task means differ by about 0.001. Near-saturated MIRACL candidate Recall@50 limits embedder discrimination in this subset; it does not show that embedders are unimportant in large corpora. ACORD has only 57 questions and does not establish general legal capability.')),
 ('timing',t('秒數包含什麼？','What does the latency include?'),t('逐題查詢總時間＝embedding_ms＋retrieval_ms＋rerank_ms；不加重排器時最後一項為0。已核對全部73,164筆，總時間與三階段加總完全一致。選了重排器的表格時間包含問題向量化、exact vector search及該重排器；不是只計重排。先逐題相加，再算平均或p50/p95，不把階段中位數直接相加。五項平均採各分項平均時間等權；單一分項才顯示p50/p95。ACORD重算沒有新的效能測量，故時間留白。','Per-query total_ms = embedding_ms + retrieval_ms + rerank_ms; the final term is zero without reranking. All 73,164 rows exactly match this sum. Selecting a reranker includes query embedding, exact vector search and that reranker, not reranking alone. Mean and percentiles are computed after summing per query, not by adding stage medians. Aggregate time equally weights task means; p50/p95 are shown only for single tasks. Corrected ACORD has no new performance measurement, so its timing is blank.')),
 ('hardware',t('硬體與速度的限制','Hardware and timing limits'),t('Apple M3 Ultra、256 GiB統一記憶體、Q8_0、Metal全層offload。單一parallel slot，模型分批執行，並非併發服務測試。時間不含模型冷載、建索引、網路、排隊或生成回答；同Embedder的候選快取重用其向量化與檢索時間。GiB是權重檔容量，不是實測RAM／VRAM或最低需求；尚未測一般筆電與其他GPU。','Apple M3 Ultra, 256 GiB unified memory, Q8_0, full Metal offload and one parallel slot. Models run in separate batches, not concurrent serving. Timings exclude cold loading, indexing, network, queues and generated answers. Reranked rows reuse their embedder’s measured embedding/search time. GiB is weight-file size, not measured RAM/VRAM or a minimum requirement. Laptops and other GPUs were not benchmarked.')),
 ('published',t('與公開研究一致之處，以及本次特色','Published findings and this benchmark’s contribution'),t('Qwen官方在MTEB-R、CMTEB-R與MMTEB-R中，4B重排器均高於0.6B；本次七個Embedder的五項平均也呈相同方向。官方使用Qwen0.6B的Top100，本次是各Embedder的Top50，不能直接比絕對分數。本次特色是地端Q8_0完整28組搭配、中文查英文與同語檢索並列，以及容量／等待時間的實際取捨。中文查英文中，Qwen0.6B＋0.6B的Top-1為86.31%，高於8B單用的81.94%；但未測反方向，也不宣稱首次發現。','Qwen reports higher 4B than 0.6B reranking scores on MTEB-R, CMTEB-R and MMTEB-R; the five-task mean for each local embedder shows the same direction. Qwen uses Top100 from Qwen 0.6B; this study uses each embedder’s Top50, preventing direct absolute-score comparison. The contribution is a full local Q8_0 28-setup matrix, Chinese-to-English alongside same-language retrieval, and measured size/latency tradeoffs. Cross-language Top-1 is 86.31% for Qwen 0.6B + 0.6B versus 81.94% for 8B alone. The reverse direction was not tested; no first-discovery claim is made.'))]
 for ident,h,text in sections:b+=f'<section id="{ident}"><h2>{h}</h2><p>{text}</p></section>'
 b+='<p><a href="https://qwenlm.github.io/blog/qwen3-embedding/">Qwen official evaluation</a> · <a href="https://github.com/project-miracl/miracl">MIRACL</a> · <a href="acord.html">'+t('法律檢索與公開研究對照','Legal retrieval and published research')+'</a></p>'
 b+='<section id="correction"><h2>'+t('ACORD修正與原始證據','ACORD correction and original evidence')+'</h2><p>'+t('先前逐題算術核對通過，但未處理ACORD要求忽略未標註配對的特殊規則。現以原始向量還原完整排名，忽略未標註配對並採線性gain，重算七個純Embedder；399份原始Top50排名全部吻合。重排候選不足，需補測。舊六項平均與舊ACORD數字僅作歷史證據，不再作最新推薦依據。原始結果檔不覆寫；新結果獨立保存。','Earlier arithmetic checks passed but missed ACORD’s rule to ignore unjudged pairs. Original vectors now reconstruct full rankings, ignoring unjudged pairs and using linear gain for seven standalone embedders. All 399 original Top50 rankings match. Rerankers need new candidate scoring. Old six-task averages and ACORD scores remain historical evidence, not current recommendations. Raw result files are preserved; corrections are separate.')+'</p><p><a href="acord.html">'+t('修正成績、診斷與驗證方式','Corrected scores, diagnostics and verification')+'</a></p></section>'
 b+='<section><h2>'+t('完整五分項數字','Complete five-task results')+'</h2>'
 for d in FIVE:
  b+='<details><summary>'+LABELS[d][lang]+' · 28 '+t('組合','setups')+'</summary>'+tab([t('組合','Setup'),'Top-1','nDCG@10','MRR@10','Recall@50',t('查詢總時間p50','Total query p50'),t('查詢總時間p95','Total query p95')],[[combo(r,lang),f'{r["top1"]*100:.2f}%',f'{r["ndcg10"]:.4f}',f'{r["mrr10"]:.4f}',f'{r["recall50"]*100:.2f}%',f'{r["total_ms_p50"]/1000:.3f} s',f'{r["total_ms_p95"]/1000:.3f} s'] for r in group(d)])+'</details>'
 b+='</section>'
 b+='<section id="downloads"><h2>'+t('完整結果、素材與程式','Complete results, materials and code')+'</h2><div class="downloads">'
 for batch,e in zip(['B01','B02','B03','B04','B05','B06','B07'],['qwen-4b','qwen-8b','embeddinggemma-2','qwen-0.6b','bge-m3','f2llm-v2-0.6b','granite-97m-r2']):b+=f'<a class="download" href="{pre}downloads/matrix-{batch}-results.zip"><strong>{batch} · {name(e,lang)} · 10,452 {t("筆","rows")}</strong><span>ZIP ↓</span></a>'
 for file,label in [('matrix-shared.zip',t('六分項共同選題與qrels','Shared query selection and qrels')),('matrix-code.zip',t('完整矩陣程式與離線驗證器','Complete matrix code and offline verifier')),('phase2-materials.zip',t('相同凍結測試素材','The same frozen test materials'))]:b+=f'<a class="download" href="{pre}downloads/{file}"><strong>{label}</strong><span>ZIP ↓</span></a>'
 b+='</div><p>'+t('每批結果包含完整逐題排名與分數、時間、摘要及公開metadata。逐題結果／摘要保留原始位元；manifest移除本機私人路徑，原始SHA仍保留於provenance。B02由A/B兩個正式子批合併。素材包的MLQA／ACORD全文與前輪相同；MIRACL提供問題、qrels、corpus IDs及固定來源重建工具，不重新散布文章全文。','Batch archives include every query ranking, score, timing, summary and public metadata. Row/summary bytes are unchanged; manifests omit private local paths, with original hashes retained in provenance. B02 combines two accepted subbatches. MLQA/ACORD full inputs are unchanged from the previous release; MIRACL includes queries, qrels, corpus IDs and a pinned-source reconstruction tool, without repackaged passage text.')+'</p><p><a href="'+pre+'research/matrix/README.md">'+t('重現指引','Reproduction guide')+'</a> · <a href="'+pre+'data/matrix/summary.csv">CSV</a> · <a href="'+pre+'data/matrix/analysis.json">JSON</a> · <a href="'+pre+'data/matrix/independent-audit.json">'+t('獨立驗證報告','Independent verification')+'</a> · <a href="'+pre+'data/matrix/provenance.json">Provenance</a></p><p><code>python scripts/verify_matrix.py</code> '+t('可離線重算正式矩陣，不啟動模型。公開包不包含內部GPU／流量日誌與失敗attempt；其凍結稽核由NAS保留。','recomputes the accepted matrix offline without loading models. Internal GPU/traffic logs and failed attempts remain in the NAS archive, outside the public package.')+'</p></section>'
 b+='<section id="license"><h2>'+t('授權、改動與來源','Licenses, adaptations and sources')+'</h2><p>'+t('MLQA採CC BY-SA 3.0；ACORD採CC BY 4.0；MIRACL topics/qrels採Apache 2.0，Wikipedia段落另保留上游來源與適用條款。改動包括檢索格式、固定選題與群聚metadata，ACORD星等轉0–4。MIT程式授權不取代資料或各模型授權，模型權重不隨附。','MLQA uses CC BY-SA 3.0; ACORD CC BY 4.0; MIRACL topics/qrels Apache 2.0, with Wikipedia passages retaining upstream attribution and applicable terms. Adaptations include retrieval formatting, frozen selection and cluster metadata, and ACORD stars mapped to grades 0–4. The code’s MIT license does not replace dataset or model licenses; model weights are not included.')+'</p><p><a href="https://github.com/facebookresearch/MLQA">MLQA / Facebook Research</a> · <a href="https://huggingface.co/datasets/theatticusproject/acord">ACORD / The Atticus Project</a> · <a href="https://huggingface.co/datasets/miracl/miracl">MIRACL topics/qrels</a> · <a href="https://huggingface.co/datasets/miracl/miracl-corpus">MIRACL corpus</a></p></section>'
 b+='<p><a href="'+pre+'data/matrix/current-comparison.json">'+t('最新五項比較JSON','Current five-task comparison JSON')+'</a> · <a href="'+pre+'data/matrix/current-summary.csv">'+t('修正版147列CSV','Corrected 147-row CSV')+'</a> · <a href="'+pre+'data/matrix/timing-audit.json">'+t('逐題時間加總驗證','Per-query timing audit')+'</a></p>'
 return page('latest-methods.html',t('地端RAG：方法、限制與完整資料','Local RAG: methods, limitations and data'),b,lang)

def acord(lang):
 from build_acord import render
 return render(ROOT,A,C,page,tab,lang,lambda k:name(k,lang))

def build():
 # The earlier generator retains the v3 baseline and Oct 24 release.
 old.build()
 for lang in [0,1]:
  folder=ROOT/('en' if lang else '')
  for src,dst in [('index.html','phase2.html'),('latest-methods.html','phase2-methods.html')]:
   s=(folder/src).read_text('utf-8')
   s=s.replace('index.html','phase2.html').replace('latest-methods.html','phase2-methods.html')
   s=s.replace('<main id="main">','<main id="main"><p class="note">'+tr('歷史版本｜10/24；','Historical Oct 24 release; ',lang)+'<a href="index.html">'+tr('查看最新28組','View the latest 28 setups',lang)+'</a></p>')
   (folder/dst).write_text(s,encoding='utf-8')
 old.A=A
 for lang in [0,1]:
  folder=ROOT/('en' if lang else '');t=lambda z,e:tr(z,e,lang)
  for p in [folder/'models.html',folder/'methods.html',folder/'resources.html',*sorted((folder/'models').glob('*.html'))]:
   s=p.read_text('utf-8');prefix='../' if p.parent.name=='models' else ''
   notice='<aside class="note"><strong>'+t('2026-10-26 完整28組矩陣','Complete 28-setup matrix · 26 Oct 2026')+'</strong> · <a href="'+prefix+'index.html">'+t('最新綜合比較','Latest comparison')+'</a> · <a href="'+prefix+'latest-methods.html">'+t('方法與完整下載','Methods and all downloads')+'</a></aside>'
   if p.parent.name=='models':
    key=p.stem;rows=[r for r in A['rows'] if r['dataset'] in FIVE and key in [r['embedding'],r['reranker']]]
    notice+='<section><h2>'+t('最新完整矩陣實測','Latest complete-matrix results')+'</h2><p>'+t('下表列出此模型的五分項已測配對。時間為已暖機的問題向量化＋檢索＋所選重排器；容量為權重檔案大小。ACORD另列修正結果。','All measured pairings in five conditions. Timing includes warmed query embedding + search + the selected reranker; size is weight-file size. Corrected ACORD is listed separately.')+'</p>'
    notice+=tab([t('分項','Condition'),t('組合','Setup'),'Top-1','nDCG@10','Recall@50','p50 ms','p95 ms'],[[LABELS[r['dataset']][lang],combo(r,lang),f'{r["top1"]*100:.2f}%',f'{r["ndcg10"]:.4f}',f'{r["recall50"]*100:.2f}%',f'{r["total_ms_p50"]:.2f}',f'{r["total_ms_p95"]:.2f}'] for r in sorted(rows,key=lambda r:(ORDER.index(r['dataset']),-r['ndcg10']))])+'</section>'
    legal=[r for r in C['rows'] if r['embedding']==key]
    notice+='<section><h2>'+t('ACORD 法律條款','ACORD legal clauses')+'</h2>'
    if legal:
     r=legal[0];notice+='<p>'+t('純Embedder修正結果：','Corrected standalone result: ')+f'nDCG@10 {r["ndcg10"]:.4f} · Top-1 {r["top1"]*100:.2f}%'+t('（3星以上）。搭配重排器的修正成績待補測。',' (at least 3 stars). Corrected reranker combinations await measurement.')+'</p>'
    else:notice+='<p>'+t('此重排器的ACORD修正成績待補測，舊分數不列入最新比較。','Corrected ACORD scores for this reranker await measurement; old scores are excluded from current comparisons.')+'</p>'
    notice+='<a href="../acord.html">'+t('法律檢索完整說明','Legal retrieval details')+'</a></section>'
    marker=t('<h2>模型介紹與10/22歷史結果</h2>','<h2>Model background and Oct 22 historical results</h2>')
    start=s.index('<main id="main">')+len('<main id="main">');end=s.index(marker,start)
    s=s[:start]+notice+s[end:]
   else:
    s=re.sub(r'<aside class="note">.*?</aside>',notice,s,count=1,flags=re.S)
   p.write_text(s,encoding='utf-8')
  (folder/'index.html').write_text(home(lang),encoding='utf-8')
  (folder/'latest-methods.html').write_text(methods(lang),encoding='utf-8')
  (folder/'acord.html').write_text(acord(lang),encoding='utf-8')
 for p in ROOT.rglob('*.html'):
  s=p.read_text('utf-8')
  s=s.replace('中英檢索實驗室','地端RAG中英檢索模型實測').replace('Bilingual Retrieval Lab','Local RAG Retrieval Benchmark')
  if p.name in ['phase2.html','phase2-methods.html','baseline.html']:
   s=s.replace('<main id="main">','<main id="main"><p class="note">'+tr('歷史證據；舊ACORD與六項平均不適用於最新推薦。','Historical evidence: old ACORD and six-task means are not current recommendations.',int('en' in p.relative_to(ROOT).parts))+'</p>')
  for asset in ['assets/style.css','assets/phase2.css','assets/phase2.js']:
   s=s.replace(asset+'"',asset+'?v=20261027-recommendations"')
  p.write_text(s,encoding='utf-8')
 urls=sorted(str(p.relative_to(ROOT)).replace('\\','/') for p in ROOT.rglob('*.html'))
 (ROOT/'sitemap.xml').write_text('<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'+''.join('<url><loc>'+old.SITE+p+'</loc><lastmod>2026-10-27</lastmod></url>' for p in urls)+'</urlset>',encoding='utf-8')
 current={'version':'2026-10-27-correction','aggregation':'Five task means, equal weight 1/5; ACORD excluded','scopes':{k:group(k) for k in ['all','mlqa']+ORDER}}
 (ROOT/'data/matrix/current-comparison.json').write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 fields=['dataset','embedding','reranker','queries','top1','ndcg10','mrr10','recall50','total_ms_mean','total_ms_p50','total_ms_p95','weight_gib_sum','protocol']
 with (ROOT/'data/matrix/current-summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader()
  for d in ORDER:
   for r in group(d):w.writerow(dict(r,dataset=d,protocol='judged-only-linear-stars3-standalone' if d=='acord-eng-eng' else 'original-binary-top50'))
 print('Built bilingual full matrix;',len(urls),'pages')
