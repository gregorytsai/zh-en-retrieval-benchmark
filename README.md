# 地端RAG中英檢索模型實測 / Local RAG Retrieval Benchmark

[繁體中文](https://gregorytsai.github.io/zh-en-retrieval-benchmark/) · [English](https://gregorytsai.github.io/zh-en-retrieval-benchmark/en/)

**27 October 2026 correction.** Compare 28 Embedder × Reranker setups on the same five conditions: three MLQA directions and two MIRACL language subsets, each weighted 1/5. ACORD is reported separately with seven corrected standalone results. Corrected ACORD reranker results await measurement.

| Five-task equal-weight mean | nDCG@10 | Top-1 | Q8_0 weights |
|---|---:|---:|---:|
| Qwen 8B + Qwen 4B reranker | 0.919 | 85.42% | 11.480 GiB |
| Qwen 4B + Qwen 4B reranker | 0.918 | 85.30% | 7.972 GiB |
| Qwen 4B + Qwen 0.6B reranker | 0.897 | 81.79% | 4.581 GiB |
| BGE-M3 + Qwen 0.6B reranker | 0.896 | 81.82% | 1.186 GiB |
| Qwen 0.6B + Qwen 0.6B reranker | 0.894 | 81.57% | 1.191 GiB |
| Qwen 8B alone | 0.867 | 77.24% | 7.494 GiB |

A small embedder with a small reranker exceeds a large embedder alone on this dataset, with a latency tradeoff: Qwen 0.6B + 0.6B averages about 1.08 s per query, versus about 52 ms for Qwen 8B alone. File size is not measured RAM/VRAM. This is local Q8_0 / Metal performance on M3 Ultra, 256 GiB.

Chinese-to-English retrieval is a focus: Qwen 0.6B + 0.6B reaches 86.31% Top-1, versus 81.94% for Qwen 8B alone; Qwen 8B + 4B reaches 92.26%. The source key `eng-zho` means English context / Chinese question. English-to-Chinese was not tested. Retrieval scores are not translation or generated-answer accuracy.

## ACORD correction

The previous code assigned unjudged pairs zero gain and retained their rank positions. ACORD explicitly requires ignoring unjudged pairs (paper C.5). It also uses linear nDCG gain; our old code used exponential gain. Earlier arithmetic checks missed this protocol mismatch.

Seven original vector caches passed their stored hashes. Reconstructed full rankings reproduce all 399 original Top50 lists exactly. Corrected evaluation ignores unjudged pairs before cutoffs, uses linear nDCG and requires original stars >=3 for binary metrics. Corrected nDCG@10 ranges from 0.5169 to 0.7027. Star P@5 uses the paper's normalized denominator and excludes queries without qualifying clauses for that metric. Corrected latency is not remeasured and is blank.

Rerankers scored only the old Top50 pool. Filtering that pool is a truncated diagnostic, not a complete corrected run. New candidate scoring is required. Old six-task means and ACORD scores remain historical evidence, not current recommendations.

[ACORD results and research context](https://gregorytsai.github.io/zh-en-retrieval-benchmark/en/acord.html) explains why high-quality legal clauses remain challenging without claiming all published legal scores are poor.

## Inspect and reproduce

- [Qwen 4B vs BGE-M3 with the same Qwen 0.6B reranker](https://gregorytsai.github.io/zh-en-retrieval-benchmark/en/pair-comparison.html): all 2,556 paired queries, task-level gains and losses, and exploratory passage-cluster bootstrap intervals. Qwen gains five net first-result hits on 1,148 English queries, but has no Chinese Top-1 advantage. The additional 3.395 GiB of weights is a conditional upgrade, not a universal recommendation. Recompute with `python scripts/compare_qwen_bge.py` (NumPy required, no inference).
- [Methods and downloads](https://gregorytsai.github.io/zh-en-retrieval-benchmark/en/latest-methods.html)
- [Current comparison JSON](data/matrix/current-comparison.json), [147-row corrected CSV](data/matrix/current-summary.csv): 140 five-task rows and seven standalone ACORD rows
- [Corrected ACORD summary/provenance](data/matrix/acord-corrected.json), [all 399 judged rankings](downloads/acord-corrected-rankings.zip), [truncated-pool diagnostics](data/matrix/acord-diagnostic.json)
- [Timing audit](data/matrix/timing-audit.json): all 73,164 original totals exactly equal query embedding + vector search + selected reranker. Cold loading, indexing, network, queuing and answer generation are excluded.
- Preserved original [168-row summary](data/matrix/summary.csv), [analysis JSON](data/matrix/analysis.json), [arithmetic audit](data/matrix/independent-audit.json), seven `downloads/matrix-B01-results.zip` through `matrix-B07-results.zip` archives and [shared qrels](downloads/matrix-shared.zip)
- [Full code](downloads/matrix-code.zip), [frozen materials](downloads/phase2-materials.zip), [release hashes](data/SHA256SUMS.json)

Run `python scripts/verify_correction.py` for current metrics and payloads, and `python scripts/verify_matrix.py` for preserved raw scores. Both use the standard library, with no inference or network. `python scripts/recompute_acord.py CACHE_ROOT` replays original vector caches with NumPy; vectors are not redistributed. Full judged rank evidence supports independent arithmetic verification. Build with `python scripts/build_site.py`; package with `python scripts/package_release.py`.

The original 2,613 dataset-qualified questions are reused across 28 setups, yielding 73,164 measurements, not independent subjects. The five-task mean uses 2,556 questions per setup. No global confidence intervals or multiplicity adjustment were computed. MIRACL uses a bounded dev-judged corpus union, not the full corpus.

MLQA: CC BY-SA 3.0. ACORD: CC BY 4.0. MIRACL topics/qrels: Apache 2.0. MIRACL passage text is reconstructed from pinned upstream sources rather than rehosted. Code: MIT. Model weights and private runtime logs are excluded.

聯絡 / Contact: [gregorytsai@gmail.com](mailto:gregorytsai@gmail.com)
