#!/usr/bin/env bash
# Pack / upload / fetch the converted model artifacts as a GitHub Release bundle.
#
# The artifacts are large and git-ignored; GitHub Releases host the binary
# bundle so apps and CI can fetch a known-good set without re-running the torch
# exports. Two tiers:
#
#   public  Kanade decoder + HiFT vocoder + clone encoder + their goldens, plus
#           the synthetic toy LM. Safe to attach to a public release.
#   full    additionally the Plapre LM, tokenizer, and speaker embeddings, which
#           are derived from the license-gated syvai/plapre-* weights. Only use
#           --full for private/self-hosted distribution; do not attach to a
#           public release unless you are entitled to redistribute those weights.
#
# Usage:
#   scripts/models.sh pack   [--full] [OUTFILE]      # -> plapre-models[-full].tar.gz
#   scripts/models.sh upload <tag> [--full]          # pack + gh release upload
#   scripts/models.sh fetch  [tag]                   # download + extract into web/public/models
#
# Requires: bash, tar, and (for upload/fetch) the GitHub CLI `gh`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT/web/public/models"

PUBLIC_PATHS=(
  kanade_decoder.onnx kanade_decoder.onnx.data
  hift_vocoder.onnx hift_vocoder.onnx.data
  clone_encoder.onnx
  phase0_golden.json clone_golden.json phase1_toy_golden.json
  lm_toy
)
GATED_PATHS=(
  lm tokenizer.json config.json speakers.json speaker_proj.json
)

die() { echo "error: $*" >&2; exit 1; }

# Collect the existing members for the requested tier into the global array PRESENT.
collect() {
  local tier="$1"; shift
  local -a wanted=("${PUBLIC_PATHS[@]}")
  [ "$tier" = full ] && wanted+=("${GATED_PATHS[@]}")
  PRESENT=()
  local p
  for p in "${wanted[@]}"; do
    if [ -e "$MODELS_DIR/$p" ]; then PRESENT+=("$p"); else echo "  (skip, missing) $p" >&2; fi
  done
  [ "${#PRESENT[@]}" -gt 0 ] || die "no artifacts found in $MODELS_DIR — produce them first (see conversion/)"
}

cmd_pack() {
  local tier=public out=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --full) tier=full ;;
      *) out="$1" ;;
    esac; shift
  done
  [ -n "$out" ] || out="$ROOT/plapre-models$([ "$tier" = full ] && echo -full).tar.gz"
  collect "$tier"
  echo "Packing ${#PRESENT[@]} member(s) ($tier) -> $out"
  tar -czf "$out" -C "$MODELS_DIR" "${PRESENT[@]}"
  echo "Wrote $out ($(du -h "$out" | cut -f1))"
  PACKED_FILE="$out"
}

cmd_upload() {
  command -v gh >/dev/null || die "gh (GitHub CLI) is required"
  local tag="${1:-}"; shift || true
  [ -n "$tag" ] || die "usage: models.sh upload <tag> [--full]"
  cmd_pack "$@"
  if ! gh release view "$tag" >/dev/null 2>&1; then
    echo "Creating release $tag"
    gh release create "$tag" --title "$tag" --notes "Model artifact bundle for $tag"
  fi
  gh release upload "$tag" "$PACKED_FILE" --clobber
  echo "Uploaded $(basename "$PACKED_FILE") to release $tag"
}

cmd_fetch() {
  command -v gh >/dev/null || die "gh (GitHub CLI) is required"
  local tag="${1:-}"
  mkdir -p "$MODELS_DIR"
  local tmp; tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  echo "Downloading model bundle from release ${tag:-<latest>}"
  if [ -n "$tag" ]; then
    gh release download "$tag" --pattern 'plapre-models*.tar.gz' --dir "$tmp"
  else
    gh release download --pattern 'plapre-models*.tar.gz' --dir "$tmp"
  fi
  local f
  for f in "$tmp"/*.tar.gz; do
    echo "Extracting $(basename "$f") -> $MODELS_DIR"
    tar -xzf "$f" -C "$MODELS_DIR"
  done
  echo "Done. Artifacts in $MODELS_DIR"
}

case "${1:-}" in
  pack)   shift; cmd_pack "$@" ;;
  upload) shift; cmd_upload "$@" ;;
  fetch)  shift; cmd_fetch "$@" ;;
  *) echo "usage: $0 {pack|upload|fetch} ..." >&2; exit 2 ;;
esac
