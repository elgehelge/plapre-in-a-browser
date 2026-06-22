#!/usr/bin/env bash
# Fetch the converted Plapre ONNX artifacts into web/public/models for local dev.
#
# The runtime artifacts live in the Hugging Face model repo
# elgehelge/plapre-onnx-web — the same base URL the deployed demo fetches from.
# That repo is the single source of truth: the shared Kanade
# decoder/vocoder/clone-encoder sit at the root, and each model variant's LM-side
# files (LM graph + meta, tokenizer, speaker tables) live under its own sub-dir
# (pico/, nano/). The npm package ships code only; these weights are git-ignored
# and only needed to run the demo and tests locally.
#
# Golden/test fixtures (phase0_golden.json, clone_golden.json,
# phase1_toy_golden.json, lm_toy/) are NOT distributed here — they are reference
# outputs produced by the conversion toolchain. See conversion/README.md.
#
# Usage:
#   scripts/models.sh fetch [pico|nano|all]    # default: pico
#
# Requires the Hugging Face CLI `hf` (pip install -U huggingface_hub).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT/web/public/models"
HF_REPO="elgehelge/plapre-onnx-web"

die() { echo "error: $*" >&2; exit 1; }

cmd_fetch() {
  command -v hf >/dev/null || die "the Hugging Face CLI 'hf' is required (pip install -U huggingface_hub)"
  local variant="${1:-pico}"
  case "$variant" in
    pico|nano|all) ;;
    *) die "unknown variant '$variant' (use pico|nano|all)" ;;
  esac

  # The shared decoder/vocoder/clone-encoder are always needed; each variant's
  # LM-side files live under its own sub-dir. Only pull what's asked for, and
  # leave the repo's own README/.gitattributes behind so they can't clobber the
  # local models/ doc.
  local -a patterns=(
    clone_encoder.onnx
    hift_vocoder.onnx hift_vocoder.onnx.data
    kanade_decoder.onnx kanade_decoder.onnx.data
  )
  case "$variant" in
    pico) patterns+=("pico/*") ;;
    nano) patterns+=("nano/*") ;;
    all)  patterns+=("pico/*" "nano/*") ;;
  esac
  local -a include=()
  local p
  for p in "${patterns[@]}"; do include+=(--include "$p"); done

  mkdir -p "$MODELS_DIR"
  echo "Fetching '$variant' artifacts from https://huggingface.co/$HF_REPO -> $MODELS_DIR"
  hf download "$HF_REPO" --repo-type model --local-dir "$MODELS_DIR" "${include[@]}"
  echo "Done. Artifacts in $MODELS_DIR"
}

case "${1:-}" in
  fetch) shift; cmd_fetch "$@" ;;
  *) echo "usage: $0 fetch [pico|nano|all]" >&2; exit 2 ;;
esac
