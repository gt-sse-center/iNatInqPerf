**Project:**
[![License](https://img.shields.io/github/license/gt-sse-center/iNatInqPerf?color=dark-green)](https://github.com/gt-sse-center/iNatInqPerf/blob/main/LICENSE)

**Package:**
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/inatinqperf?color=dark-green)](https://pypi.org/project/inatinqperf/)
[![PyPI - Version](https://img.shields.io/pypi/v/inatinqperf?color=dark-green)](https://pypi.org/project/inatinqperf/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/inatinqperf)](https://pypistats.org/packages/inatinqperf)

**Development:**
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![CI](https://github.com/gt-sse-center/iNatInqPerf/actions/workflows/CICD.yml/badge.svg)](https://github.com/gt-sse-center/iNatInqPerf/actions/workflows/CICD.yml)
[![Code Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ketanbj/521b537b3503957227f91dfb3db59065/raw/iNatInqPerf_code_coverage.json)](https://github.com/gt-sse-center/iNatInqPerf/actions)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/y/gt-sse-center/iNatInqPerf?color=dark-green)](https://github.com/gt-sse-center/iNatInqPerf/commits/main/)

<!-- Content above this delimiter will be copied to the generated README.md file. DO NOT REMOVE THIS COMMENT, as it will cause regeneration to fail. -->

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Development](#development)
- [Additional Information](#additional-information)
- [License](#license)

## Overview

This project provides a **modular benchmark pipeline** for experimenting with different vector databases (FAISS, Qdrant, …).  
It runs end-to-end:

1. **Download** → Hugging Face dataset (optionally export images + manifest)  
2. **Embed** → Generate CLIP embeddings for images  
3. **Build** → Construct indexes with multiple VectorDBs  
4. **Search** → Profile queries (latency + Recall@K vs exact baseline)  
5. **Update** → Test insertions & deletions (index maintenance)

All steps are run with **uv** as the package manager.

### How to use `iNatInqPerf`

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup environment
uv venv .venv && source .venv/bin/activate
uv sync

# Run a small end-to-end benchmark (FAISS IVF+PQ vectordb)
uv run python src/inatinqperf/benchmark/benchmark.py run-all --size small --vectordb faiss.ivfpq
```

---

## Step 1: Download Dataset

Fetch a dataset from Hugging Face, slice by size, and optionally export JPEGs with a manifest.

```bash
# Small slice (200 samples, exports JPEGs)
uv run python src/inatinqperf/benchmark/benchmark.py download --size small --out_dir data/raw --export-images

# Large (full train split only)
uv run python src/inatinqperf/benchmark/benchmark.py download --size large --out_dir data/raw

# XL (train+val+test)
uv run python src/inatinqperf/benchmark/benchmark.py download --size xlarge --out_dir data/raw

# XXL (all data)
uv run python src/inatinqperf/benchmark/benchmark.py download --size xxlarge --out_dir data/raw
```

### Options

- `--size` : `small`, `large`, `xlarge`, `xxlarge`
- `--out_dir` : output folder (default: `data/raw`)
- `--export-images` : save JPEGs + `manifest.csv`
- `--no-export-images` : keep HF Arrow dataset only

**Output structure:**

```sh
data/raw/
  dataset_info.json
  state.json
  data-00000-of-00001.arrow
  images/
    00000000.jpg
    00000001.jpg
    ...
  images/manifest.csv   # [index,filename,label]
```

---

## Step 2: Embed Images

Generate CLIP embeddings and save them into a Hugging Face dataset.

```bash
# Default model + batch size
uv run python src/inatinqperf/benchmark/benchmark.py embed --raw_dir data/raw --emb_dir data/emb

# Override CLIP model & batch size
uv run python src/inatinqperf/benchmark/benchmark.py embed --raw_dir data/raw --emb_dir data/emb --model_id openai/clip-vit-large-patch14 --batch-size 32
```

**Outputs:**

- `data/emb/` — temporary embeddings
- `data/emb_hf/` — HF dataset with `{id, label, embedding}`

---

## Step 3: Build Index

Construct an index on a chosen vectordb.

```bash
# FAISS Flat (exact baseline)
uv run python src/inatinqperf/benchmark/benchmark.py build --vectordb faiss.flat --hf_dir data/emb_hf

# FAISS IVF+PQ (ANN)
uv run python src/inatinqperf/benchmark/benchmark.py build --vectordb faiss.ivfpq --hf_dir data/emb_hf
```

### Supported Vector Databases

- `faiss.flat` (exact)
- `faiss.ivfpq` (IVF + OPQ + PQ)

---

## Step 4: Search Profiling

Profile query latency and compute Recall@K vs FAISS Flat baseline.

```bash
uv run python src/inatinqperf/benchmark/benchmark.py search --vectordb faiss.ivfpq --hf_dir data/emb_hf --topk 10 --queries bench/queries.txt
```

**Outputs:**

- Latency statistics (avg, p50, p95)
- Recall@K vs baseline
- JSON metrics in `.results/`

---

## Step 5: Index Update

Simulate real-time usage: insert (upsert) and delete vectors.

```bash
uv python src/inatinqperf/benchmark/benchmark.py update --vectordb faiss.ivfpq --hf_dir data/emb_hf
```

Configurable counts via `configs/benchmark.yaml`:

```yaml
update:
  add_count: 50
  delete_count: 30
```

---

## Profiling with py-spy

Use `py-spy` to record flamegraphs during any step:

```bash
bash scripts/pyspy_run.sh search-faiss -- python src/inatinqperf/benchmark/benchmark.py search --vectordb faiss.ivfpq --hf_dir data/emb_hf --topk 10 --queries src/inatinqperf/benchmark/queries.txt
```

Outputs:

- `.results/search-faiss.svg` (flamegraph)
- `.results/search-faiss.speedscope.json`

---

<!-- Content below this delimiter will be copied to the generated README.md file. DO NOT REMOVE THIS COMMENT, as it will cause regeneration to fail. -->

## Installation

| Installation Method | Command |
| --- | --- |
| Via [uv](https://github.com/astral-sh/uv) | `uv add inatinqperf` |
| Via [pip](https://pip.pypa.io/en/stable/) | `pip install inatinqperf` |

## Development

Please visit [Contributing](https://github.com/gt-sse-center/iNatInqPerf/blob/main/CONTRIBUTING.md) and [Development](https://github.com/gt-sse-center/iNatInqPerf/blob/main/DEVELOPMENT.md) for information on contributing to this project.

## Additional Information

Additional information can be found at these locations.

| Title | Document | Description |
| --- | --- | --- |
| Code of Conduct | [CODE_OF_CONDUCT.md](https://github.com/gt-sse-center/iNatInqPerf/blob/main/CODE_OF_CONDUCT.md) | Information about the norms, rules, and responsibilities we adhere to when participating in this open source community. |
| Contributing | [CONTRIBUTING.md](https://github.com/gt-sse-center/iNatInqPerf/blob/main/CONTRIBUTING.md) | Information about contributing to this project. |
| Development | [DEVELOPMENT.md](https://github.com/gt-sse-center/iNatInqPerf/blob/main/DEVELOPMENT.md) | Information about development activities involved in making changes to this project. |
| Governance | [GOVERNANCE.md](https://github.com/gt-sse-center/iNatInqPerf/blob/main/GOVERNANCE.md) | Information about how this project is governed. |
| Maintainers | [MAINTAINERS.md](https://github.com/gt-sse-center/iNatInqPerf/blob/main/MAINTAINERS.md) | Information about individuals who maintain this project. |
| Security | [SECURITY.md](https://github.com/gt-sse-center/iNatInqPerf/blob/main/SECURITY.md) | Information about how to privately report security issues associated with this project. |

## License

`iNatInqPerf` is licensed under the <a href="https://choosealicense.com/licenses/MIT/" target="_blank">MIT</a> license.
