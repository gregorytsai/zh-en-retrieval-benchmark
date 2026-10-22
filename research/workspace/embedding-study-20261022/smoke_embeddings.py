#!/usr/bin/env python3
"""Sequential isolated compatibility checks, not a retrieval benchmark."""
import argparse
import json
import math
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
BINARY = ROOT.parent / "llama.cpp-qwen38/build/bin/llama-server"
CASES = [
    ("granite-embedding-97M-multilingual-r2-Q8_0.gguf", "cls", 384,
     "太陽能電池如何發電？",
     "Solar cells convert sunlight into electrical energy.",
     "Espresso is prepared by forcing hot water through ground coffee."),
    ("embeddinggemma-2-Q8_0.gguf", "mean", 768,
     "task: search result | query: 太陽能電池如何發電？",
     "title: none | text: Solar cells convert sunlight into electrical energy.",
     "title: none | text: Espresso is prepared by forcing hot water through ground coffee."),
    ("Qwen3-Embedding-0.6B-Q8_0.gguf", "last", 1024,
     "Instruct: Given a web search query, retrieve relevant passages\nQuery: 太陽能電池如何發電？",
     "Solar cells convert sunlight into electrical energy.",
     "Espresso is prepared by forcing hot water through ground coffee."),
    ("bge-m3-q8_0.gguf", "cls", 1024,
     "太陽能電池如何發電？",
     "Solar cells convert sunlight into electrical energy.",
     "Espresso is prepared by forcing hot water through ground coffee."),
    ("F2LLM-v2-0.6B.Q8_0.gguf", "last", 1024,
     "Instruct: Given a web search query, retrieve relevant passages\nQuery: 太陽能電池如何發電？",
     "Solar cells convert sunlight into electrical energy.",
     "Espresso is prepared by forcing hot water through ground coffee."),
]

def request(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)

def run(case):
    filename, pooling, dim, query, positive, negative = case
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    command = [str(BINARY), "-m", str(ROOT / "models" / filename),
               "--host", "127.0.0.1", "--port", str(port), "--embedding",
               "--pooling", pooling, "--embd-normalize", "2",
               "--n-gpu-layers", "all", "--ctx-size", "8192",
               "--batch-size", "2048", "--ubatch-size", "2048", "--parallel", "1", "--log-verbosity", "5"]
    result = {"model":filename, "command":command, "expected_dimension":dim,
              "purpose":"compatibility only; not benchmark accuracy"}
    with (ROOT / "evidence" / (filename + ".log")).open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited {process.returncode}")
                try:
                    if request(url + "/health").get("status") == "ok":
                        break
                except Exception:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("server startup exceeded 120s")
                time.sleep(0.2)
            start = time.monotonic()
            response = request(url + "/v1/embeddings", {"input":[query, positive, negative], "encoding_format":"float"})
            vectors = [x["embedding"] for x in sorted(response["data"], key=lambda x:x["index"])]
            result.update(elapsed_seconds=time.monotonic()-start,
                          dimensions=[len(v) for v in vectors],
                          finite=all(math.isfinite(x) for v in vectors for x in v),
                          norms=[math.sqrt(sum(x*x for x in v)) for v in vectors])
            assert len(vectors) == 3 and all(len(v) == dim for v in vectors)
            assert result["finite"] and all(abs(n-1)<0.01 for n in result["norms"])
            result["positive_cosine"] = sum(a*b for a,b in zip(vectors[0],vectors[1]))
            result["negative_cosine"] = sum(a*b for a,b in zip(vectors[0],vectors[2]))
            result["positive_above_negative"] = result["positive_cosine"] > result["negative_cosine"]
            assert result["positive_above_negative"], "Relevant document did not outrank negative"
            result["status"] = "ok"
        except Exception as exc:
            result.update(status="failed", error=str(exc))
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only")
    args = parser.parse_args()
    (ROOT / "evidence").mkdir(exist_ok=True)
    selected = [c for c in CASES if not args.only or c[0] == args.only]
    if not selected:
        parser.error("unknown model")
    failed = False
    for case in selected:
        result = run(case)
        (ROOT / "evidence" / (case[0] + ".smoke.json")).write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        failed = failed or result["status"] != "ok"
    raise SystemExit(1 if failed else 0)
