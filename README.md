# Bilingual Retrieval Lab / 中英檢索實驗室

**Updated 24 October 2026.** [繁體中文](https://gregorytsai.github.io/zh-en-retrieval-benchmark/) · [English](https://gregorytsai.github.io/zh-en-retrieval-benchmark/en/)

Latest accepted runs: 14,478 measurements, 30 dataset/setup groups, 2,613 dataset-qualified queries. MLQA full validation in three directions, ACORD test, and two MIRACL judged-passage subsets. This evaluates retrieval ranking, not translation or generated-answer quality.

| Six-condition equal-weight average | Top-1 | nDCG@10 |
|---|---:|---:|
| Qwen3 Embedding 4B alone | 65.35% | 0.73535 |
| + Qwen3 Reranker 0.6B | 71.37% | 0.77127 |
| + Qwen3 Reranker 4B | 74.01% | 0.79258 |

Only three setups were tested in all six conditions. Each condition has weight 1/6 (MLQA family 50%, MIRACL 33.33%, ACORD 16.67%). The interactive site also offers MLQA three-direction averages and each individual condition, metric switching, reranker filtering, sortable tables and model pages. No missing setup is imputed. Historical 22-setup results remain under [baseline.html](https://gregorytsai.github.io/zh-en-retrieval-benchmark/baseline.html).

Important limits: ACORD Top-1 uses grade≥1 (2+ stars), nDCG uses exponential gain, and P@5 uses a fixed denominator; these differ from the original ACORD paper. MIRACL uses a small union of dev judged passages, not the official full corpus. Qwen's published measurements support a larger-reranker quality trend, but scores from different protocols are not directly comparable. Our local observations concern task-specific quality/latency tradeoffs, not first-ever findings.

## Inspect and reproduce

- [Methods and source comparison](https://gregorytsai.github.io/zh-en-retrieval-benchmark/en/latest-methods.html)
- [Latest CSV](data/phase2/summary.csv), [analysis](data/phase2/analysis.json), [independent audit](data/phase2/independent-audit.json)
- [Batch 1 complete results](downloads/phase2-batch1-results.zip), [Batch 2 complete results](downloads/phase2-batch2-results.zip)
- [Runner source](research/phase2/README.md), [code ZIP](downloads/phase2-code.zip), [materials](downloads/phase2-materials.zip)
- [Official-source research](data/phase2/web-comparison.json), [release hashes](data/SHA256SUMS.json)

With Python and NumPy, run `python scripts/verify_phase2.py` for the new measurements and `python scripts/verify_release.py` for the original measurements plus release hashes and links. These are offline arithmetic/integrity checks, not model inference. Build both languages with `python scripts/build_site.py`, then refresh archives/hashes with `python scripts/package_release.py`.

MLQA materials retain CC BY-SA 3.0; ACORD retains CC BY 4.0; MIRACL topics/qrels retain Apache 2.0. MIRACL passage text is retrieved from pinned upstream sources using the supplied helper, preserving Wikipedia attribution/terms, rather than rehosted here. Code: MIT. Model weights are not included. See the materials NOTICE and reproduction guide for revisions, hashes and transformations.

聯絡 / Contact: [gregorytsai@gmail.com](mailto:gregorytsai@gmail.com)
