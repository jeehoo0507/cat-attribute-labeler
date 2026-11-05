#!/usr/bin/env bash
# Sourced by entry points. Explicit paths avoid accidental use of shared server caches.
: "${DATA_ROOT:?Set DATA_ROOT to an external-cat data directory}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to this annotation run directory}"
: "${HF_HOME:?Set HF_HOME to a server model-cache directory}"
: "${GPUS:?Set GPUS to exactly one physical GPU id, e.g. 0}"
[[ "$GPUS" =~ ^[0-9]+$ ]] || { echo 'GPUS must be one integer GPU id' >&2; exit 1; }
export DATA_ROOT OUTPUT_ROOT HF_HOME
export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
CONFIG="${CONFIG:-configs/pilot.json}"
export CONFIG
