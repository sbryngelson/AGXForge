# Third-party notices

This project's own code and writing are licensed under the MIT License ([LICENSE](LICENSE)). Third-party material
retains its applicable terms: the model metadata below keeps its upstream Apache License 2.0. Files produced by Apple's
tools, or derived from Apple's decoder metadata, are identified with their origin in [PROVENANCE.md](PROVENANCE.md).

These files are copied or extracted from public model repositories at the revisions pinned in this project's own
records (`model.json` beside them). They keep their upstream licence, Apache License 2.0, whose text is in
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt). The project's MIT licence ([LICENSE](LICENSE)) does not apply to
them. No model weights or tokenizer files are included.

| Files | Upstream repository | Revision | Licence (as the repository's metadata declares it) |
|---|---|---|---|
| `evidence/g17-inference-models-v1/qwen/config.json`, `qwen/safetensors-header.json` | Qwen/Qwen2.5-0.5B-Instruct (Hugging Face) | `7ae557604adf67be50417f59c2c2f167def9a775` | Apache-2.0 |
| `evidence/g17-inference-models-v1/minilm/config.json`, `minilm/modules.json`, `minilm/pooling.json`, `minilm/sentence-config.json`, `minilm/safetensors-header.json` | sentence-transformers/all-MiniLM-L6-v2 (Hugging Face) | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | Apache-2.0 |

`safetensors-header.json` is the JSON header of the upstream checkpoint file (tensor names, shapes, dtypes and
offsets), recorded so a fetched checkpoint can be checked before use; the other files are upstream configuration files.
The licence column reproduces each repository's declared licence, recorded in `model.json` (`license: apache-2.0`).
The upstream repository file inventories were checked at the pinned revisions on 2026-10-05 (the Hugging Face model
API for each repository at its revision). Neither contains a NOTICE file.
