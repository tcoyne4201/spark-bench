#!/usr/bin/env bash
# Installs uv, downloads the model, starts vLLM in the background, waits until it answers.
# Env: MODEL (HF repo), HF_TOKEN (only if gated), HF_HOME (cache dir), PORT,
#      VLLM_IMAGE (docker image, only used if `vllm` is not installed on the host).
set -euo pipefail
cd "$(dirname "$0")/.."
export MODEL="${MODEL:-Qwen/Qwen3.8-27B-FP8}"
export PORT="${PORT:-8000}"
export HF_HOME="${HF_HOME:-/workspace/hf}"
mkdir -p results "$HF_HOME"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo ">> downloading $MODEL"
uvx --from huggingface_hub hf download "$MODEL"

# Only reuse a server WE started (pid file). Any other vllm (e.g. the image's own default server)
# has unknown flags and holds GPU memory, so refuse, or kill it with KILL_EXISTING=1.
our_pid=$(cat results/server.pid 2>/dev/null || true)
others=$(pgrep -f 'vllm serve' | grep -vx "${our_pid:-0}" || true)
if [ -n "$others" ]; then
  if [ "${KILL_EXISTING:-0}" = 1 ]; then
    echo ">> killing other vllm processes: $others"
    pkill -f 'vllm' || true
    sleep 10
  else
    echo "ERROR: another vllm server is running (pid $others), likely the image's default." >&2
    echo "Kill it (pkill -f vllm) or re-run with KILL_EXISTING=1. It would compete for memory." >&2
    exit 1
  fi
fi

if [ -n "$our_pid" ] && kill -0 "$our_pid" 2>/dev/null; then
  echo ">> our server already running (pid $our_pid)"
elif command -v vllm >/dev/null; then
  echo ">> starting vllm on host (log: results/server.log)"
  nohup bash ./deploy/serve_vllm.sh > results/server.log 2>&1 &
  echo $! > results/server.pid
else
  : "${VLLM_IMAGE:?vllm not installed: set VLLM_IMAGE to a vLLM docker image that supports the Spark (GB10)}"
  echo ">> starting vllm in docker ($VLLM_IMAGE)"
  docker run -d --name vllm --gpus all --ipc=host --network host \
    -e MODEL -e PORT -e HF_TOKEN -e HF_HOME=/hf -v "$HF_HOME:/hf" \
    -v "$PWD/deploy:/deploy:ro" --entrypoint bash "$VLLM_IMAGE" /deploy/serve_vllm.sh
  (docker logs -f vllm > results/server.log 2>&1 &)
fi

echo ">> waiting for server (up to 30 min); tail -f results/server.log to watch"
for _ in $(seq 360); do
  curl -sf "localhost:$PORT/v1/models" >/dev/null && { echo ">> server ready"; exit 0; }
  sleep 5
done
echo "server did not come up, see results/server.log" >&2
exit 1
