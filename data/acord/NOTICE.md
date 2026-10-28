# Notices and attribution

Original benchmark code: Gregory Tsai, 2026, MIT License. See
LICENSES/MIT-project.txt, copied unchanged from the pinned reference project
`gregorytsai/zh-en-retrieval-benchmark` at commit
`2a0c391e4baa9d57168edc61cd57278670962b69`.
The ACORD retest code and modifications are distributed under that project license.

ACORD data and adaptations retain CC BY 4.0 attribution. See
LICENSES/ACORD-CC-BY-4.0.md. This code license does not replace the data license.
MLQA, MIRACL and their results are outside this retest package's experiment scope.

Third-party models, model weights, NumPy, Python and llama.cpp retain their own
licenses. No weights, vector caches or runtime binaries are included in the public
allowlist package. Model identifiers, source information and pinned checksums are
recorded for reproducibility, not relicensed. The frozen runtime and model manifests
must be checked before inference; installing newer software changes the experiment.

This retest is a controlled judged-pair clause retrieval evaluation. It does not
claim to reproduce the original ACORD leaderboard or to measure ordinary RAG
answer correctness. Ranking evidence uses knowledge of which pairs were judged.
