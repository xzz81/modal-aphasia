# AGENTS.md

## Repository Portability

- Use the repository root as the project root.
- Do not write absolute paths in code, configs, scripts, or notebooks. All paths must be relative to the repository root.
- Datasets must be accessed through `data/`.
- Models, checkpoints, tokenizers, and adapters must be accessed through `model/`.
- If required data or models already exist on the server, use local symlinks under `data/` or `model/` instead of downloading duplicates.
- After downloading any dataset or model, create or update a reusable download script that recreates it under `data/` or `model/`; this script must be tracked by git, while the downloaded files themselves must not be committed.
- Do not commit server-specific paths, symlinks, datasets, model weights, checkpoints, logs, or outputs.
- A fresh Git clone should be deployable on another server by recreating the required `data/` and `model/` links, without changing source code.

## Benchmarks

- Benchmark datasets are not the same asset type as benchmark code.
- Do not mix benchmark datasets with benchmark or baseline code repositories.
