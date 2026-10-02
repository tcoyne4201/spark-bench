#!/usr/bin/env bash
# Optional comparison run. Usage: ./deploy/serve_llamacpp.sh /path/to/model.gguf
# --kv-unified is needed, otherwise the context is split across slots and long prompts fail.
set -euo pipefail
exec llama-server --model "${1:?usage: serve_llamacpp.sh /path/to/model.gguf}" \
  --alias qwen3.8-27b --host 0.0.0.0 --port 8080 \
  --ctx-size 262144 --parallel 128 --kv-unified \
  --n-gpu-layers 999 --metrics --jinja --reasoning-format deepseek
