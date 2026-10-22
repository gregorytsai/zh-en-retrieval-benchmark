# MLQA / MLQARetrieval — CC BY-SA 3.0

Source authors: Patrick Lewis, Barlas Oguz, Ruty Rinott, Sebastian Riedel, Holger Schwenk; MTEB contributors; Wikipedia contributors.
MLQA: https://github.com/facebookresearch/MLQA
Paper: https://aclanthology.org/2020.acl-main.653/
Retrieval adaptation: https://huggingface.co/datasets/mteb/MLQARetrieval
Dataset revision: cf59ddd8f4aaf39ce1869361e09252698c340945
License: https://creativecommons.org/licenses/by-sa/3.0/
Legal code: https://creativecommons.org/licenses/by-sa/3.0/legalcode

Changes: 100 queries per direction selected by SHA-256 seed 20261022; full validation corpus retained; JSON format conversion, retrieval annotations and editorial Chinese explanations added. Source question/passage text is preserved; excerpts are labelled. No tested model generated a translation. Our editorial adaptations of dataset content are shared under CC BY-SA 3.0. No endorsement implied.

The upstream retrieval corpus provides document IDs but often empty titles and no per-article URL. Those fields are retained without inventing Wikipedia attribution. Original Parquet files are included unchanged; source and SHA-256 are in dataset-manifest.json. Data is separate from code licensing and model-weight licensing.
