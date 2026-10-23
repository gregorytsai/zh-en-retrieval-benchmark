# Phase 2 reproduction / 第二階段重現

The two batches completed on 23–24 October 2026 (Taipei). The public release contains only accepted measurements. It does not include pilots, internal server logs, failed attempts or model weights. Original numeric result files and frozen manifests are byte-identical to the source archives; source hashes are in `data/phase2/analysis.json` at the repository root.

## Verify the publication without models

From the repository root, with Python and NumPy:

```sh
python scripts/verify_phase2.py
python scripts/verify_release.py
```

The first command recomputes the new 14,478 rows, 30 summaries and 32 paired intervals. The second checks the original v3 results and all public file hashes / HTML links. Neither starts an inference server. The original MIRACL full-input hash cannot be checked from the ID-only public selection; reconstruct it below first.

## Inputs and licenses

Unzip `downloads/phase2-materials.zip`. `batch1/datasets.json` is the complete frozen MLQA/ACORD input. MLQA is CC BY-SA 3.0 (Facebook Research and mteb adapter); ACORD is CC BY 4.0 (The Atticus Project and ACORD authors). Keep the package NOTICE with any redistribution. Adaptations are retrieval formatting, selected query IDs, cluster metadata and ACORD stars minus one as grades 0–4.

MIRACL queries/qrels are Apache 2.0 as declared by MIRACL authors. The public selection includes all corpus IDs but omits Wikipedia passage text. Retrieve the pinned upstream files and reconstruct the exact original input:

```sh
python research/phase2/fetch_miracl_materials.py ./miracl-inputs
```

This downloads about 115 MB, verifies three upstream SHA256 values and the final `fdf802cc...ba3d` dataset hash. Preserve MIRACL/Wikipedia attribution and applicable text terms. It performs no inference and executes no downloaded code.

Sources: https://github.com/facebookresearch/MLQA ; https://huggingface.co/datasets/mteb/MLQARetrieval ; https://huggingface.co/datasets/theatticusproject/acord ; https://github.com/project-miracl/miracl ; https://huggingface.co/datasets/miracl/nomiracl ; https://huggingface.co/datasets/miracl/miracl-corpus .

## Run the original benchmark

`batch1/` and `batch2/` preserve separate source layouts. Each includes `bilingual-retrieval-benchmark/run_benchmark.py` plus `embedding-study-phase2-20261023/`; batch 2 has its runner in that folder's `batch2-runner/`. Consult `run_study.py` model registry and the published batch manifest for exact GGUF names and SHA256. Supply the matching models and `llama.cpp-qwen38/build/bin/llama-server` relative to the batch workspace. Binary: build 883, commit `1fd20a7f4`, SHA256 `5f7bee49eaf08bf77b3cc523d88d385750d6b7da8479775b32bc9be030464ae5`. Original execution used macOS/Metal; Windows/CPU runs are different performance conditions.

Place the batch-1 input as `batch1/embedding-study-phase2-20261023/datasets.json`; place reconstructed batch-2 input as `batch2/embedding-study-phase2-20261023/batch2-runner/datasets.json`. From the respective runner folder:

```sh
python run_guarded.py --validate-data-only --datasets-file datasets.json
python -m unittest test_runner test_study test_traffic_guard
# Explicitly starts an inference experiment; requires compatible runtime and model files:
python run_guarded.py --datasets-file datasets.json --output ./new-run --wall-hours 12
```

Use a fresh output directory. The runner monitors other local inference activity and fails closed if it cannot verify quiet conditions. Do not disable its checks to obtain comparable measurements. The original code is included for inspection; no new hardware performance is claimed by this publication.

## Read the results correctly

The homepage macro average is equal across six conditions, restricted to three common setups. MLQA average weights three directions equally. Missing setups are not zero-filled. Per-task confidence intervals cannot be reused for these macro averages. ACORD uses exponential nDCG gain, a fixed denominator for P@5 and grade≥1 (2-star) Top1; these differ from the original ACORD paper. MIRACL is a judged-passage union, not its full-corpus leaderboard. No generation, translation, production-concurrency or independent repeated-latency test was performed.

The original source files may mention excluded datasets in preparation functions or historical comments. Only the official manifests and the accepted result ZIPs define the published matrix.

本頁提供離線驗證、固定版本素材重建及原始runner路徑。完整方法與授權的中英文說明，請看網站 `latest-methods.html`。測量程式及驗證程式沿用專案MIT；第三方資料與模型保留其原授權。
