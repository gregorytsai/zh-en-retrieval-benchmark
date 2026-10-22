#!/usr/bin/env python3
"""Download only the files needed to convert a Qwen3 reranker to GGUF."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parent
REVISION = "main"
SMALL_FILES = [
    "README.md",
    "chat_template.jinja",
    "config.json",
    "config_sentence_transformers.json",
    "generation_config.json",
    "merges.txt",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()
    repo = args.repo
    api = HfApi()
    info = api.model_info(repo, revision=REVISION, files_metadata=True)
    allow = [item.rfilename for item in info.siblings if (
        item.rfilename in SMALL_FILES or
        item.rfilename == "model.safetensors.index.json" or
        item.rfilename == "model.safetensors" or
        (item.rfilename.startswith("model-") and item.rfilename.endswith(".safetensors"))
    )]
    model_name = args.output_name or repo.rsplit("/", 1)[-1]
    model_dir = ROOT / "hf" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    local = Path(snapshot_download(
        repo_id=repo,
        revision=info.sha,
        local_dir=model_dir,
        allow_patterns=allow,
        local_dir_use_symlinks=False,
    ))
    expected = {item.rfilename: item for item in info.siblings if item.rfilename in allow}
    files = []
    for name, item in sorted(expected.items()):
        path = local / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({
            "path": name,
            "size": path.stat().st_size,
            "sha256": sha256(path),
            "hub_size": getattr(item, "size", None) or (getattr(item.lfs, "size", None) if item.lfs else None),
            "hub_lfs_sha256": item.lfs.sha256 if item.lfs else None,
        })
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = {
        "repo": repo,
        "revision": REVISION,
        "status": "downloaded",
        "resolved_revision": info.sha,
        "files": files,
        "local_dir": str(local),
    }
    existing = next((item for item in manifest["models"] if item.get("repo") == repo), None)
    if existing is None:
        manifest["models"].append(entry)
    else:
        existing.update(entry)
    manifest["downloads"].append({"repo": repo, "revision": info.sha, "allow_patterns": allow})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"repo": repo, "revision": info.sha, "local_dir": str(local), "files": files}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        raise
