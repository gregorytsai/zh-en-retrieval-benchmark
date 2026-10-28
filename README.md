# 地端RAG中英檢索模型實測 / Local RAG Retrieval Benchmark

[繁體中文](https://gregorytsai.github.io/zh-en-retrieval-benchmark/) · [English](https://gregorytsai.github.io/zh-en-retrieval-benchmark/en/)

**28 October 2026 update:** ACORD now has 28 complete retested setups. The homepage shows six equal-weight task means, retaining the original-five and individual-task views. Six-task score = (original-five mean × 5 + ACORD) / 6. MLQA contributes 50%, MIRACL 33.3%, legal 16.7%; these weights are not production traffic.

| Setup | Six-task nDCG | Original five | ACORD | Q8_0 weights |
|---|---:|---:|---:|---:|
| Qwen 8B + Qwen 4B reranker | 0.887 | 0.919 | 0.726 | 11.480 GiB |
| Qwen 4B + Qwen 4B reranker | 0.886 | 0.918 | 0.728 | 7.972 GiB |
| Qwen 4B + Qwen 0.6B reranker | 0.862 | 0.897 | 0.684 | 4.581 GiB |
| BGE-M3 + Qwen 0.6B reranker | 0.860 | 0.896 | 0.682 | 1.186 GiB |

Current evidence comprises 71,568 original five-task rows plus 1,596 fresh ACORD rows. The 2,613 dataset-qualified queries recur across 28 setups, not 73,164 independent subjects. Original archives remain unchanged, including superseded ACORD scores.

## What the legal retest changes

ACORD judged-top50-v1 removes unknown qrel pairs before selecting 50 candidates, retaining explicit zero grades. Three rerankers share each embedder's ordered candidates. Linear nDCG uses full-qrels ideal rankings; binary relevance requires original stars >=3. Normalized star P@5 divides by min(5, qualifying clauses), excluding queries with none; 5-star has 29 eligible queries.

Production RAG lacks judged-membership information. The aggregate is a controlled benchmark index, not a production hit rate, answer accuracy or full reproduction of the paper's pipeline. Latency is not pooled across all six tasks.

Reranking can hurt: Qwen 4B alone scores 0.703 on ACORD versus 0.684 with Qwen 0.6B reranking and 0.675 with BGE. F2LLM 0.6B + Qwen 4B has the highest estimate at 0.729; nearby scores do not establish a reliable winner. With the same Qwen 0.6B reranker, Qwen 4B and BGE-M3 achieve candidate recall of 95.26% versus 85.30%, but identical Top-1 on all 57 queries. [Paired analysis](https://gregorytsai.github.io/zh-en-retrieval-benchmark/pair-comparison.html).

## Inspect and reproduce

- [Methods and downloads](https://gregorytsai.github.io/zh-en-retrieval-benchmark/en/latest-methods.html), [ACORD results](https://gregorytsai.github.io/zh-en-retrieval-benchmark/en/acord.html).
- [ACORD retest ZIP](downloads/acord-retest-20261027.zip): 103 allowlisted files including code, licensed frozen materials, full rankings, 59,850 pair scores, 1,596 query results and timing evidence.
- [Publication validation](data/acord/publication-validation.json), [ACORD CSV](data/acord/summary.csv), [normalized site data](data/acord/retest.json).
- [Current comparison JSON](data/matrix/current-comparison.json), [168 current task rows](data/matrix/current-summary.csv).
- [Original matrix code](downloads/matrix-code.zip), seven matrix-B01-results.zip through matrix-B07-results.zip archives in downloads/, [shared qrels](downloads/matrix-shared.zip), [frozen materials](downloads/phase2-materials.zip), [release hashes](data/SHA256SUMS.json).

Run `python scripts/verify_current.py` for independent legal scoring and six-/five-task payload checks. Extract the ACORD ZIP and run `python -m src.independent_validate .` for additional candidate-generation, source-evidence and request-trace checks. Neither command runs models. Build with `python scripts/build_site.py`, package with `python scripts/package_release.py`, then run `python scripts/verify_release.py`.

Publication checks verified 102 payload hashes, identical frozen queries/corpus/qrels and ten model-weight pins. Independent validation found no missing, duplicate, nonfinite or accepted-error rows. The source reports 107 tests passing on macOS. Windows rechecking passed 87 tests; seven test modules could not import Unix fcntl dependencies. Full controller tests require macOS/Unix. Historical runtime libraries do not all have comparable hashes. There are 69 documented historical judged-tail differences, all beyond Top50, including four above the diagnostic epsilon; their cause is unresolved.

Original-five bootstrap intervals remain restricted to those tasks. ACORD source-query clustering was not established, so no legal bootstrap or global significance claim is made. This is not a production stress test or fixed 30-query × 3-repeat latency test. Weight-file GiB is not measured RAM/VRAM. The archived report treats ACORD separately; this website adds the verified six-task index without rewriting the archive.

MLQA: CC BY-SA 3.0. ACORD: CC BY 4.0, revision 7ba1c6f2e0baaa32e78c78c2b02e9c57b102af4e. MIRACL topics/qrels: Apache 2.0; passage text retains upstream terms. Code: MIT. Weights, vectors, credentials and full private runtime logs are excluded.

聯絡 / Contact: [gregorytsai@gmail.com](mailto:gregorytsai@gmail.com)
