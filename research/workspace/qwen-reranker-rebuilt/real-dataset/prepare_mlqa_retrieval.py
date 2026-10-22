#!/usr/bin/env python3
"""Download and prepare a small, qrels-backed MLQARetrieval subset.

The MTEB repository stores corpus, query, and qrels as separate Parquet
configs.  This script downloads only validation files for the requested
directions, keeps the official qrels, and writes a deterministic query subset
against the complete (small) validation corpus.  No negative labels are
invented: documents without a qrel simply remain unjudged/non-relevant for
the benchmark evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq


REPO = "mteb/MLQARetrieval"
DEFAULT_CONFIGS = ("eng-zho", "zho-zho", "eng-eng")
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
SUBSET = ROOT / "subset"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS), choices=[
        "eng-zho", "zho-zho", "eng-eng", "zho-eng",
    ])
    parser.add_argument("--queries-per-config", type=int, default=30)
    parser.add_argument("--revision", default="main", help="HF revision; main resolves to the current commit")
    args = parser.parse_args()
    if args.queries_per_config < 1:
        raise SystemExit("--queries-per-config must be positive")

    api = HfApi()
    info = api.dataset_info(REPO, revision=args.revision, files_metadata=True)
    revision = info.sha or args.revision
    sibling_by_name = {item.rfilename: item for item in info.siblings}
    RAW.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "dataset": REPO,
        "source_url": "https://huggingface.co/datasets/mteb/MLQARetrieval",
        "source_revision": revision,
        "requested_revision": args.revision,
        "license": "cc-by-sa-3.0",
        "split": "validation",
        "orientation": "config name is corpus-language minus query-language; eng-zho is Chinese query to English corpus",
        "configs": {},
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }

    for config in args.configs:
        files: dict[str, Any] = {}
        loaded: dict[str, list[dict[str, Any]]] = {}
        for kind in ("corpus", "queries", "qrels"):
            filename = f"{config}-{kind}/validation-00000-of-00001.parquet"
            expected = sibling_by_name.get(filename)
            if expected is None:
                raise SystemExit(f"official file not found at {revision}: {filename}")
            local = Path(hf_hub_download(REPO, filename=filename, repo_type="dataset", revision=revision, local_dir=str(RAW)))
            loaded[kind] = rows(local)
            lfs = getattr(expected, "lfs", None)
            files[kind] = {
                "repo_file": filename,
                "local_file": str(local.relative_to(ROOT)),
                "bytes": local.stat().st_size,
                "sha256": sha256(local),
                "official_size": getattr(expected, "size", None),
                "official_lfs_sha256": getattr(lfs, "sha256", None),
            }

        # Deterministic first-N query IDs, while retaining the complete official
        # validation corpus and only the selected query/qrel rows.
        corpus = sorted(loaded["corpus"], key=lambda item: str(item["id"]))
        query_all = sorted(loaded["queries"], key=lambda item: str(item["id"]))
        selected_queries = query_all[: args.queries_per_config]
        selected_ids = {str(item["id"]) for item in selected_queries}
        selected_qrels = sorted(
            [item for item in loaded["qrels"] if str(item["query-id"]) in selected_ids],
            key=lambda item: (str(item["query-id"]), str(item["corpus-id"])),
        )
        # A qrel must point at a corpus row; fail closed if source files do not
        # join rather than silently dropping a judged document.
        corpus_ids = {str(item["id"]) for item in corpus}
        missing = sorted({str(item["corpus-id"]) for item in selected_qrels} - corpus_ids)
        if missing:
            raise SystemExit(f"{config}: qrels reference missing corpus IDs: {missing[:5]}")
        config_dir = SUBSET / config
        write_jsonl(config_dir / "corpus.jsonl", corpus)
        write_jsonl(config_dir / "queries.jsonl", selected_queries)
        write_jsonl(config_dir / "qrels.jsonl", selected_qrels)
        manifest["configs"][config] = {
            "corpus_language": config.split("-")[0],
            "query_language": config.split("-")[1],
            "files": files,
            "full_validation_counts": {
                "corpus": len(corpus),
                "queries": len(query_all),
                "qrels": len(loaded["qrels"]),
            },
            "subset_counts": {
                "corpus": len(corpus),
                "queries": len(selected_queries),
                "qrels": len(selected_qrels),
            },
            "subset_files": {
                kind: {
                    "path": str((config_dir / f"{kind}.jsonl").relative_to(ROOT)),
                    "bytes": (config_dir / f"{kind}.jsonl").stat().st_size,
                    "sha256": sha256(config_dir / f"{kind}.jsonl"),
                }
                for kind in ("corpus", "queries", "qrels")
            },
        }

    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
