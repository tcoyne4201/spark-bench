#!/usr/bin/env bash
# One command: setup + benchmark + print the scp command. Run inside tmux.
# Extra args go to bench.py, e.g. ./deploy/run_all.sh --quick
set -euo pipefail
cd "$(dirname "$0")/.."
./deploy/setup.sh
export PATH="$HOME/.local/bin:$PATH"
uv run bench.py --stack vllm --base-url "http://localhost:${PORT:-8000}" --model qwen3.8-27b "$@"
tarball=$(ls -t results/bench-results-*.tar.gz | head -1)
echo
echo "Copy results to your machine (run THERE), then destroy the instance:"
echo "  scp -P ${VAST_TCP_PORT_22:-<ssh-port>} root@${PUBLIC_IPADDR:-<ip>}:$PWD/$tarball ."
echo "(also send: FP8 or BF16 used, any flags changed in serve_vllm.sh, results/server.log)"
