#!/usr/bin/env python3
"""Fetch pinned public GGUFs; resume partial files and verify upstream SHA-256."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import argparse

ROOT = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def download(row):
    target = ROOT / "models" / row["file"]
    partial = target.with_suffix(".gguf.part")
    if target.exists():
        if target.stat().st_size == row["bytes"] and digest(target) == row["sha256"]:
            print(row["file"] + ": already verified", flush=True)
            return
        raise RuntimeError("Existing target differs; refusing overwrite: " + str(target))
    print(row["file"] + ": downloading", flush=True)
    subprocess.run(["curl", "-fL", "--silent", "--show-error", "--retry", "3",
                    "--connect-timeout", "30", "--max-time", "7200", "-C", "-",
                    "-o", str(partial), row["url"]], check=True)
    if partial.stat().st_size != row["bytes"] or digest(partial) != row["sha256"]:
        raise RuntimeError("Size/SHA-256 mismatch: " + str(partial))
    with partial.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise RuntimeError("Not a GGUF: " + str(partial))
    partial.rename(target)
    print(row["file"] + ": SHA-256 OK", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Download a single manifest filename")
    args = parser.parse_args()
    models = json.loads((ROOT / "download-manifest.json").read_text())["models"]
    if args.only:
        models = [row for row in models if row["file"] == args.only]
        if not models:
            parser.error("filename not in manifest")
    (ROOT / "models").mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, models))
    print("All requested GGUF downloads verified.", flush=True)
