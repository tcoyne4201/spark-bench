# Spark benchmark (Qwen3.8-27B FP8 on vLLM)

Files: `bench.py` (runs on the Spark), `report.py` (runs on your laptop), `deploy/` (server setup).
Scenarios: `sweep` (concurrency 1→256), `long_context` (8k→245k + 4×64k), `burst` (256 at once ×3), `thinking`.
Request errors/timeouts at high load are expected: they are recorded as data.

## On the Vast.ai Spark
```bash
ssh -p <port> root@<ip>
tmux new -s bench                      # so a dropped SSH doesn't kill the run
git clone https://github.com/tcoyne4201/spark-bench.git && cd spark-bench
export HF_TOKEN=...                    # only if the model is gated
./deploy/run_all.sh --quick            # ~5 min shakedown first
./deploy/run_all.sh                    # full run, ~30 min after model load
```
`run_all.sh` = `deploy/setup.sh` (install uv, download model, start vLLM, wait) + `bench.py`.
Env overrides: `MODEL` (default `Qwen/Qwen3.8-27B-FP8`; BF16 = `Qwen/Qwen3.8-27B`), `HF_HOME` (default `/workspace/hf`),
`PORT`, `VLLM_IMAGE` (docker image; only needed if `vllm` isn't installed on the host. The image must support the Spark's GB10/ARM64).
Server log: `results/server.log`. If vLLM rejects a flag, edit `deploy/serve_vllm.sh` and note the change.

## Get the data off (before destroying the instance)
At the end the script prints the exact `scp` command. Or, from this repo on your laptop:
```bash
make fetch HOST=<ip> PORT=<ssh-port>        # rsyncs results/*.tar.gz + server.log into fetched/
```
(`REMOTE_DIR` defaults to `/root/spark-bench`; override if you cloned elsewhere.)

## Report (local)
```bash
make report                                  # -> report/report.html, open in a browser
# or: uv run report.py fetched/bench-results-vllm-*.tar.gz
```
Interactive Plotly charts (hover/zoom/toggle), per-scenario tables and a telemetry timeline, in one offline file.
Pass several tarballs (e.g. a llama.cpp run via `deploy/serve_llamacpp.sh` and `--stack llamacpp --base-url http://localhost:8080`) to compare stacks.

## Notes
- Prompt lengths are approximate (random words, calibrated against the server's token usage); the real counts are in the JSONL.
- Non-thinking scenarios use `ignore_eos` for fixed output length and disable thinking; the `thinking` scenario enables it.
- Other servers (Ollama, LM Studio): `uv run bench.py --stack ollama --base-url http://localhost:11434 --model <name>`; make sure concurrency and context limits are raised.
