#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/pyspy_run.sh <output-stem> -- <python command...>
#
# Example:
#   scripts/pyspy_run.sh search-faiss -- python benchmark/benchmark.py search --vectordb faiss.ivfpq --hf_dir data/emb_hf --topk 10 --queries benchmark/queries.txt

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <output-stem> -- <python command...>" >&2
  exit 2
fi

OUT_STEM="$1"
shift

if [[ "${1:-}" != "--" ]]; then
  echo "ERROR: expected '--' after output-stem" >&2
  exit 2
fi
shift

mkdir -p .results
SVG=".results/${OUT_STEM}.svg"
SCOPE=".results/${OUT_STEM}.speedscope.json"

# Record flamegraph SVG
py-spy record \
  --rate 250 \
  --output "$SVG" \
  --format flamegraph \
  -- "$@"

# Record speedscope JSON
py-spy record \
  --rate 250 \
  --output "$SCOPE" \
  --format speedscope \
  -- "$@"

echo "py-spy artifacts:"
echo "  $SVG"
echo "  $SCOPE"