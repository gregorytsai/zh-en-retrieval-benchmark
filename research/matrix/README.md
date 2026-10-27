# 完整28組矩陣 / Complete 28-setup matrix

測量於2026-10-26完成，2026-10-27獨立重算並發布。7 embedders × 4 retrieval/reranking choices × 6 fixed conditions = 168 summaries and 73,164 accepted rows. 本套程式測檢索排序，不生成譯文。

## 先核對結果 / Verify the released results

Use Python 3 with the repository's `downloads/matrix-B01-results.zip` through `matrix-B07-results.zip`, `matrix-shared.zip`, and `data/matrix/`:

```sh
python scripts/verify_matrix.py
```

Only Python's standard library is needed. The verifier recomputes quality from final ranking IDs and qrels, summaries, timing, candidate order hashes, complete coverage, source hashes, website values and six-condition macro rankings. It performs no inference, network access or writes to the measured data. This verification passed on the public package; model weights, original hardware and private GPU/traffic logs were not re-executed by the publication check.

## Prepare the same inputs

Use the full repository, or extract `matrix-code.zip` into a new folder and add the downloaded result archives, `matrix-shared.zip`, `phase2-materials.zip`, and the `data/matrix/` directory in their repository-relative locations. Extract `batch1/datasets.json` from `phase2-materials.zip` to `research/matrix/workspace/embedding-study-phase2-20261023/datasets.json`. Preserve its bytes.

Rebuild MIRACL with the pinned-source helper (about 115 MB inbound download):

```sh
python research/matrix/fetch_miracl_materials.py /path/to/new-miracl-materials
```

The helper verifies three upstream archives and the reconstructed final file. Copy the resulting `datasets-batch2.json` into the same project directory as `datasets.json`.

| Input | SHA-256 |
|---|---|
| datasets.json | `bc0657a77090f86b081b378e3f369653dc08f6049da78487b7dc0c134e7adad1` |
| datasets-batch2.json | `fdf802cc03e9059783e5a2cba0785437253f07841ec6956246d5884461e7ba3d` |

The runner's combined `dataset_sha256` is the canonical JSON hash of `{first: first_file_hash, second: second_file_hash}`, not the hash of a third merged file. Both fixed inputs are identical to the October 24 release. The supplied validation sidecars retain historical dataset evidence; stale preparation status text is not a claim about the completed matrix. The independent verifier checks actual records.

MLQA and ACORD contain the full adapted licensed inputs. MIRACL full passage text is not repackaged: its selection, corpus IDs, qrels and reconstruction tool preserve pinned upstream attribution. Model/source links and model licenses are listed on each model page and in `data/models.json`. The registry in each `matrix_spec.py` gives exact GGUF relative paths, sizes and hashes. Obtain matching assets separately; no weights or llama-server binary are bundled.

## Runtime and new measurements

原測量是 macOS／Apple Silicon／Metal。這是完整原始測量程式，並非Windows即裝即用套件。Code uses Unix process control and Metal evidence gates. The measured environment was Python 3.12.14, NumPy 2.3.5, macOS 26.7, M3 Ultra with 256 GiB, Q8_0, server context 8192, batch/ubatch 4096, eight threads, one parallel slot and full GPU offload. Binary SHA-256: `5f7bee49eaf08bf77b3cc523d88d385750d6b7da8479775b32bc9be030464ae5` (llama.cpp build 883 / commit `1fd20a7f4`). A locally rebuilt binary may differ even at the same commit; report that as a new experiment variant, not byte-identical reproduction.

Embedding inputs must be below 4096 tokens including prefixes; reranking pair bounds are at most 4096. The test does not establish 8192-token document support for every model. Preflight stops on excessive length; no silent truncation. BGE uses its audited three-special-token fallback pair overhead; Qwen uses a conservative template bound.

Run from a separate workspace with all fixed assets. Do not resume or overwrite the archived measured run. The original runner handles B01; v4 handles B02-A/B and B03–B07. Each batch needs its own pilot gate before official execution. The historical `run_sequence.py` and `prepare_v4_continuation.py` expect private archived carryover evidence and are not the fresh-run entry points.

```bash
set -euo pipefail
WORKSPACE=/absolute/path/to/research/matrix/workspace
PY=/absolute/path/to/python3
PROJECT="$WORKSPACE/embedding-study-phase2-20261023"
V1="$PROJECT/full-matrix-runner"
V4="$PROJECT/full-matrix-runner-continuation-v4"
NEW_OUTPUT="$WORKSPACE/new-full-matrix-run"
test ! -e "$NEW_OUTPUT"
mkdir "$NEW_OUTPUT"
cp "$PROJECT/FULL_MATRIX_PLAN_ZH_TW.md" "$NEW_OUTPUT/WORKING_PLAN.md"
PLAN="$NEW_OUTPUT/WORKING_PLAN.md"

"$PY" "$V4/run_full_matrix.py" --workspace "$WORKSPACE" --validate --verify-assets > "$NEW_OUTPUT/assets-check.json"
"$PY" - "$NEW_OUTPUT/assets-check.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1], encoding='utf-8'))['assets']
assert len(a['models']) == 10 and all(x.get('pin_match') is True for x in a['models'])
assert a['binary'].get('sha256') == '5f7bee49eaf08bf77b3cc523d88d385750d6b7da8479775b32bc9be030464ae5'
PY

# The following commands start models. Do not skip failed pilot/hash/guard gates.
"$PY" "$V1/run_full_matrix.py" --workspace "$WORKSPACE" --output "$NEW_OUTPUT" --plan-path "$PLAN" --batch B01 --execute --pilot
"$PY" "$V1/run_full_matrix.py" --workspace "$WORKSPACE" --output "$NEW_OUTPUT" --plan-path "$PLAN" --batch B01 --execute
for BATCH in B02-A B02-B B03 B04 B05 B06 B07; do
  "$PY" "$V4/run_full_matrix.py" --workspace "$WORKSPACE" --output "$NEW_OUTPUT" --plan-path "$PLAN" --batch "$BATCH" --segment 01 --execute --pilot
  "$PY" "$V4/run_full_matrix.py" --workspace "$WORKSPACE" --output "$NEW_OUTPUT" --plan-path "$PLAN" --batch "$BATCH" --segment 01 --execute
done
```

`--validate --verify-assets` can exit zero while a model reports `pin_match=false`, so the explicit JSON checks above are necessary. Execution also checks the pinned binary. These commands document fresh per-batch execution, not the historical controller's carryover-specific global report format. Slower hardware may fail the original 2-hour pilot / 12-hour official limits. Guard polling of local services remains enabled. No fresh inference was performed during publication.

## Source and publication notes

- All Python sources in the three runner directories are preserved. `PUBLICATION_MANIFEST.json` records original and public hashes. v4's calculation core is identical to the original matrix core; controller changes repaired reporting and continuation state.
- **Known report-text erratum:** `core.write_report` still says standalone top100; actual code, committed records and every measured `final_ids` list use **top50**. We preserve original source for traceability and correct the website/methods here.
- B02-A/B were merged only after accepted subbatch audits. Pilots, interrupted and invalidated attempts do not appear in public result rows.
- Private host/runtime paths are omitted from public manifests and validation sidecars. Source hashes identify original metadata separately; sanitized metadata is not represented as byte-identical.
- Full historical GPU/traffic/attempt evidence remains in the NAS archive. Archive-only carryover/integration tests need that evidence; public arithmetic verification does not claim to replay those tests.
- `prepare_phase2_data.py` and `prepare_batch2_data.py` preserve the original raw-to-input transformations. They require upstream raw files and previous100 selection metadata; for the exact measured inputs use the distributed frozen materials and MIRACL helper above.

## Licenses

Code: repository MIT license. MLQA: Facebook Research / mteb adapter, [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), source [MLQA](https://github.com/facebookresearch/MLQA). ACORD: The Atticus Project and ACORD authors, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), source [dataset](https://huggingface.co/datasets/theatticusproject/acord). MIRACL topics/qrels: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), [source](https://github.com/project-miracl/miracl); Wikipedia passage text retains upstream attribution and applicable text terms. Adaptations: fixed retrieval format, query selection and cluster metadata; ACORD stars minus one maps to grades 0–4. Dataset licenses are not replaced by the code license.

Contact: gregorytsai@gmail.com
