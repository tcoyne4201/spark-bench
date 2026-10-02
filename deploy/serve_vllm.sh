#!/usr/bin/env bash
# Starts vLLM in the foreground. Used directly, or inside a container by setup.sh.
# MODEL can be a HF repo id or a local dir. If vLLM rejects a flag, edit it here and note the change.
set -euo pipefail
exec vllm serve "${MODEL:-Qwen/Qwen3.8-27B-FP8}" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port "${PORT:-8000}" \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 256 \
  --no-enable-prefix-caching \
  --reasoning-parser qwen3
