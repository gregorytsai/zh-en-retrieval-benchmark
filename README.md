# Bilingual Retrieval Lab / 中英檢索實驗室

**Updated 27 October 2026; measurements completed 26 October.** [繁體中文](https://gregorytsai.github.io/zh-en-retrieval-benchmark/) · [English](https://gregorytsai.github.io/zh-en-retrieval-benchmark/en/)

Complete matrix: **7 embedders × 4 retrieval/reranking settings × 6 conditions = 28 complete setups, 168 summaries, 73,164 accepted query measurements**. The same 2,613 dataset-qualified queries are repeated across setups, not independent subjects. All setups use identical input corpora and query selections; each embedder supplies the same ordered Top50 candidates to its three rerankers.

| Six-condition equal-weight average | Top-1 | nDCG@10 | Q8_0 weights |
|---|---:|---:|---:|
| Qwen 4B + Qwen 4B reranker | 74.01% | 0.7926 | 7.972 GiB |
| Qwen 8B + Qwen 4B reranker | 74.10% | 0.7917 | 11.480 GiB |
| BGE-M3 + Qwen 4B reranker | 74.05% | 0.7907 | 4.577 GiB |
| Qwen 0.6B + Qwen 4B reranker | 74.09% | 0.7893 | 4.581 GiB |
| BGE-M3 + Qwen 0.6B reranker | 71.69% | 0.7700 | 1.186 GiB |

Each condition weighs 1/6: MLQA contributes 50%, MIRACL 33.33%, ACORD 16.67%. Top-1 and nDCG leaders differ. Small score differences are descriptive point estimates, not significant wins: no global 28-setup bootstrap intervals or multiplicity corrections were computed. This measures retrieval ranking, not translation or generated-answer quality.

The complete matrix makes small-embedder/large-reranker tradeoffs visible. Larger rerankers give higher six-task mean nDCG for all seven embedders, consistent in direction with [Qwen's published evaluation](https://qwenlm.github.io/blog/qwen3-embedding/), but Qwen's Top100 protocol differs. Local results are not first-ever findings. Pooled median reranking stage time is 0.849 s for BGE, 1.127 s for Qwen 0.6B, and 6.217 s for Qwen 4B on the measured M3 Ultra. File size is not a minimum RAM/VRAM requirement.

MIRACL uses a small dev-judged passage union, not the full corpus; near-saturated candidate recall limits embedder discrimination. ACORD has 57 queries; Top-1 uses 2+ stars, nDCG exponential gain and P@5 a fixed denominator, unlike the original paper. Known source-report erratum: standalone Top100 is stale prose; all actual matrix rankings contain Top50.

## Inspect and reproduce

- [Methods, limitations and all downloads](https://gregorytsai.github.io/zh-en-retrieval-benchmark/en/latest-methods.html)
- [168-row summary CSV](data/matrix/summary.csv), [analysis JSON](data/matrix/analysis.json), [independent audit](data/matrix/independent-audit.json)
- Seven complete result archives in `downloads/matrix-B01-results.zip` through `matrix-B07-results.zip`; [shared queries/qrels](downloads/matrix-shared.zip)
- [Reproduction guide](research/matrix/README.md), [complete matrix code](downloads/matrix-code.zip), [frozen materials](downloads/phase2-materials.zip)
- [Publication provenance](data/matrix/provenance.json), [source changes/hashes](research/matrix/PUBLICATION_MANIFEST.json), [all release hashes](data/SHA256SUMS.json)
- Historical [Oct 24 phase 2](https://gregorytsai.github.io/zh-en-retrieval-benchmark/en/phase2.html) and [Oct 22 baseline](https://gregorytsai.github.io/zh-en-retrieval-benchmark/en/baseline.html)

Run `python scripts/verify_matrix.py` (standard library only) for the latest raw scores and website calculations. The check passed for all 73,164 rows and 957,516 quality values. This is offline arithmetic/integrity verification, not new model inference. `verify_phase2.py` checks the preceding release; `verify_release.py` checks the original baseline plus all release hashes and HTML links. Build with `python scripts/build_site.py`, package with `python scripts/package_release.py`.

MLQA retains CC BY-SA 3.0; ACORD CC BY 4.0; MIRACL topics/qrels Apache 2.0. MIRACL passage text is reconstructed from pinned upstream sources with attribution, not rehosted here. Research code: MIT. Weights and private GPU/traffic logs are not included.

聯絡 / Contact: [gregorytsai@gmail.com](mailto:gregorytsai@gmail.com)
