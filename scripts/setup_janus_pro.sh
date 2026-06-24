#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: scripts/setup_janus_pro.sh [--link-only] [--verify-only]

Install the Janus-Pro-7B base model asset under model/Janus-Pro-7B.

Environment variables:
  JANUS_PRO_MODEL_ID      Hugging Face repo id. Default: deepseek-ai/Janus-Pro-7B
  JANUS_PRO_MODEL_DIR     Target path. Default: <repo>/model/Janus-Pro-7B
  PYTHON                  Python executable for verification/download fallback. Default: python3
  JANUS_PRO_SOURCE_DIR    Existing local model directory to symlink/copy from.
  JANUS_PRO_COPY          If set to 1 with JANUS_PRO_SOURCE_DIR, copy instead of symlink.
  HF_ENDPOINT             Optional Hugging Face mirror endpoint.

Examples:
  JANUS_PRO_SOURCE_DIR=/cache/ummu/model/Janus-Pro-7B scripts/setup_janus_pro.sh --link-only
  PYTHON=/opt/venv/bin/python HF_ENDPOINT=https://hf-mirror.com scripts/setup_janus_pro.sh
USAGE
}

LINK_ONLY=0
VERIFY_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --link-only)
      LINK_ONLY=1
      shift
      ;;
    --verify-only)
      VERIFY_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
MODEL_ID=${JANUS_PRO_MODEL_ID:-deepseek-ai/Janus-Pro-7B}
MODEL_DIR=${JANUS_PRO_MODEL_DIR:-$PROJECT_ROOT/model/Janus-Pro-7B}
SOURCE_DIR=${JANUS_PRO_SOURCE_DIR:-}
PYTHON=${PYTHON:-python3}

has_required_files() {
  local path=$1
  [[ -f "$path/config.json" ]] && \
    [[ -f "$path/tokenizer.json" || -f "$path/tokenizer.model" ]] && \
    { [[ -f "$path/pytorch_model.bin.index.json" ]] || compgen -G "$path/model-*.safetensors" >/dev/null || compgen -G "$path/pytorch_model-*.bin" >/dev/null; }
}

resolve_path() {
  "$PYTHON" - "$1" <<PY
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
}

verify_model_dir() {
  local path=$1
  if ! has_required_files "$path"; then
    echo "Janus-Pro model directory is incomplete: $path" >&2
    echo "Expected config.json, tokenizer files, and model shards." >&2
    exit 1
  fi

  "$PYTHON" - "$PROJECT_ROOT" "$path" <<PY
import sys
from pathlib import Path

project_root = Path(sys.argv[1])
model_dir = Path(sys.argv[2])
sys.path.insert(0, str(project_root))

from modal_aphasia.janus import modeling_vlm

resolved = modeling_vlm._resolve_model_path("deepseek-ai/Janus-Pro-7B")
expected = project_root / "model" / "Janus-Pro-7B"
if Path(resolved).resolve() != expected.resolve():
    raise SystemExit(f"unexpected Janus-Pro resolution: {resolved} != {expected}")

cfg = modeling_vlm.MultiModalityConfig.from_pretrained(model_dir)
print(f"verified Janus-Pro at {model_dir}")
print(f"model_type={cfg.model_type}")
print(f"language_hidden_size={cfg.language_config.hidden_size}")
PY
}

mkdir -p "$(dirname "$MODEL_DIR")"

if [[ "$VERIFY_ONLY" -eq 1 ]]; then
  verify_model_dir "$MODEL_DIR"
  exit 0
fi

if [[ -n "$SOURCE_DIR" ]]; then
  SOURCE_DIR=$(resolve_path "$SOURCE_DIR")
  if ! has_required_files "$SOURCE_DIR"; then
    echo "JANUS_PRO_SOURCE_DIR is not a complete Janus-Pro model directory: $SOURCE_DIR" >&2
    exit 1
  fi

  if [[ -e "$MODEL_DIR" || -L "$MODEL_DIR" ]]; then
    if has_required_files "$MODEL_DIR"; then
      echo "Janus-Pro already installed at $MODEL_DIR"
      verify_model_dir "$MODEL_DIR"
      exit 0
    fi
    echo "Target exists but is incomplete: $MODEL_DIR" >&2
    exit 1
  fi

  if [[ "${JANUS_PRO_COPY:-0}" == "1" ]]; then
    cp -a "$SOURCE_DIR" "$MODEL_DIR"
    echo "Copied Janus-Pro from $SOURCE_DIR to $MODEL_DIR"
  else
    ln -s "$SOURCE_DIR" "$MODEL_DIR"
    echo "Linked Janus-Pro from $SOURCE_DIR to $MODEL_DIR"
  fi
  verify_model_dir "$MODEL_DIR"
  exit 0
fi

if has_required_files "$MODEL_DIR"; then
  echo "Janus-Pro already installed at $MODEL_DIR"
  verify_model_dir "$MODEL_DIR"
  exit 0
fi

if [[ "$LINK_ONLY" -eq 1 ]]; then
  echo "--link-only was requested but JANUS_PRO_SOURCE_DIR was not set." >&2
  exit 1
fi

if command -v hf >/dev/null 2>&1; then
  hf download "$MODEL_ID" --local-dir "$MODEL_DIR" --local-dir-use-symlinks False
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_DIR" --local-dir-use-symlinks False
else
  "$PYTHON" -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2], local_dir_use_symlinks=False)" "$MODEL_ID" "$MODEL_DIR"
fi

verify_model_dir "$MODEL_DIR"
