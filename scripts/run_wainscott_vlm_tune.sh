#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-preflight}"

case "$MODE" in
  preflight)
    TRIALS="${NAVIRL_TRIALS:-1}"
    OUT_DIR="${NAVIRL_OUT_DIR:-out/tune_wainscott_preflight}"
    MAX_FRAMES="${NAVIRL_MAX_FRAMES:-8}"
    AEGIS_TOP_K="${NAVIRL_AEGIS_TOP_K:-1}"
    ;;
  full)
    TRIALS="${NAVIRL_TRIALS:-48}"
    OUT_DIR="${NAVIRL_OUT_DIR:-out/tune_wainscott_vlm}"
    MAX_FRAMES="${NAVIRL_MAX_FRAMES:-14}"
    AEGIS_TOP_K="${NAVIRL_AEGIS_TOP_K:-8}"
    ;;
  *)
    echo "Usage: $0 [preflight|full]" >&2
    exit 2
    ;;
esac

SEED="${NAVIRL_SEED:-73}"
SCENARIO="${NAVIRL_SCENARIO:-navirl/scenarios/library/wainscott_main_demo.yaml}"
SEARCH_SPACE="${NAVIRL_SEARCH_SPACE:-}"

if [[ -x "$ROOT_DIR/.venv311/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv311/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "ERROR: Could not find .venv311/bin/python or python3." >&2
  exit 1
fi

# Keep this literal on one line so shell line wrapping can't inject a newline.
export NAVIRL_CODEX_CMD='/bin/zsh -lc "codex exec - --output-schema {schema_file} --output-last-message {output_file} {image_flags} < {prompt_file}"'

if [[ "$NAVIRL_CODEX_CMD" == *$'\n'* ]]; then
  echo "ERROR: NAVIRL_CODEX_CMD contains a newline. Aborting." >&2
  exit 1
fi

echo "[run_wainscott_vlm_tune] mode=$MODE trials=$TRIALS seed=$SEED out=$OUT_DIR" >&2
echo "[run_wainscott_vlm_tune] python=$PYTHON_BIN" >&2

CMD=("$PYTHON_BIN" -m navirl tune \
  --scenarios "$SCENARIO" \
  --trials "$TRIALS" \
  --seed "$SEED" \
  --judge-mode vlm \
  --judge-provider codex \
  --no-judge-allow-fallback \
  --aegis-rerank \
  --aegis-top-k "$AEGIS_TOP_K" \
  --max-frames "$MAX_FRAMES" \
  --out "$OUT_DIR")

if [[ -n "$SEARCH_SPACE" ]]; then
  CMD+=(--search-space "$SEARCH_SPACE")
fi

"${CMD[@]}"
