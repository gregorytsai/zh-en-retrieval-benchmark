# Public release changes

Original evidence remains on the source host. Only the accepted v3 matrix is ranked. Public metadata replaces the local user directory with `<WORKSPACE>`. Historical source paths were converted to script-relative paths in:

- qwen-reranker-rebuilt/diagnose_offload.py
- qwen-reranker-rebuilt/run_repaired_benchmark.py
- qwen-reranker-rebuilt/run_repaired_benchmark_metal.py
- qwen-reranker-rebuilt/smoke_rebuilt.py
- qwen-reranker-rebuilt/real-dataset/combine_mlqa_results.py
- qwen-reranker-rebuilt/real-dataset/run_mlqa_benchmark.py

The original audit_v3.py needs full attempts and GPU logs from the private research delivery and is archived as source, not the public validation command. Use scripts/verify_release.py to validate the public release. No inference was rerun for publication. No model weights, private service snapshots, or unfiltered operational logs are distributed.
