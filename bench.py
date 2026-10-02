# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "numpy", "psutil"]
# ///
"""Benchmark an OpenAI-compatible server (vLLM, llama.cpp, ...).

    uv run bench.py --stack vllm --base-url http://localhost:8000 --model qwen3.8-27b
    uv run bench.py --quick     # short shakedown run

Writes results/<stack>/{env.json,telemetry.csv,<scenario>.jsonl,summary.json}
and a results/bench-results-<stack>-<timestamp>.tar.gz to copy off the machine.
"""
import argparse
import asyncio
import csv
import json
import platform
import random
import subprocess
import tarfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import psutil

PROFILES = {
    "full": {
        "warmup": 5,
        "sweep": {"levels": [1, 4, 16, 32, 64, 128, 256], "seconds": 30, "prompt": 1024, "output": 256},
        "long_context": {"lengths": [8192, 32768, 131072, 245000], "concurrent_len": 65536,
                         "concurrent_n": 4, "output": 256},
        "burst": {"size": 256, "cycles": 3, "idle": 15, "prompt": 512, "output": 128},
        "thinking": {"single_n": 3, "concurrent_n": 32, "max_tokens": 4096},
    },
    "quick": {
        "warmup": 2,
        "sweep": {"levels": [1, 8, 32], "seconds": 15, "prompt": 1024, "output": 128},
        "long_context": {"lengths": [8192, 32768], "concurrent_len": 16384,
                         "concurrent_n": 2, "output": 64},
        "burst": {"size": 32, "cycles": 1, "idle": 5, "prompt": 512, "output": 64},
        "thinking": {"single_n": 1, "concurrent_n": 4, "max_tokens": 1024},
    },
}

THINKING_PROMPTS = [
    "A train leaves city A at 60 km/h. Two hours later a second train leaves A at 90 km/h on the same track. When does the second catch up, and how far from A?",
    "Write a Python function that returns the longest palindromic substring of a string, and explain its time complexity.",
    "Three boxes are labelled 'apples', 'oranges', 'mixed', and all labels are wrong. You may draw one fruit from one box. How do you fix all labels?",
    "How many ways can 8 non-attacking rooks be placed on a chessboard so that none is on the main diagonal? Show your reasoning.",
    "Explain why the sum of the first n odd numbers is n squared, with two different proofs.",
    "A bat and ball cost $1.10 together; the bat costs $1 more than the ball. Then generalise: what if the total is T and the difference is D?",
]

WORDS = ("the of and to in is that for it as with was on be by at this have from or one had not but what all "
         "were when we there can an your which their said if do will each about how up out them then she many "
         "some so these would other into has more her two like him see time could no make than first been its "
         "who now people my made over did down only way find use may water long little very after").split()


@dataclass
class Ctx:
    client: httpx.AsyncClient
    args: argparse.Namespace
    base: str
    out_dir: Path
    t0: float
    tok_per_word: float = 1.0
    summary: list = None


# ---------- requests ----------

def make_prompt(ctx, n_tokens):
    """Random words (unique each call, so no prefix-cache hits) of roughly n_tokens."""
    n_words = int(n_tokens / ctx.tok_per_word)
    words = random.Random().choices(WORDS, k=n_words)
    return "Summarize the following text in detail.\n\n" + " ".join(words)


def parse_chunk(line):
    """One SSE line -> (usage, kind) where kind is 'think', 'answer' or None."""
    if not line.startswith("data: ") or line.strip() == "data: [DONE]":
        return None, None
    obj = json.loads(line[6:])
    usage = obj.get("usage")
    choices = obj.get("choices") or [{}]
    delta = choices[0].get("delta") or {}
    if delta.get("reasoning_content") or delta.get("reasoning"):
        return usage, "think"
    if delta.get("content"):
        return usage, "answer"
    return usage, None


async def one_request(ctx, scenario, step, prompt, max_tokens, thinking=False):
    body = {
        "model": ctx.args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if not thinking:
        body["ignore_eos"] = True  # fixed output length for comparable throughput
    start = time.perf_counter()
    rec = {"scenario": scenario, "step": step, "t_submit": start - ctx.t0, "max_tokens": max_tokens,
           "status": "ok", "error": "", "chunks": 0}
    t_first = t_answer = usage = None
    try:
        async with ctx.client.stream("POST", ctx.base + "/v1/chat/completions", json=body) as r:
            if r.status_code != 200:
                rec.update(status="error", error=f"HTTP {r.status_code}: {(await r.aread()).decode()[:300]}")
            async for line in r.aiter_lines() if r.status_code == 200 else []:
                u, kind = parse_chunk(line)
                usage = u or usage
                if kind is None:
                    continue
                now = time.perf_counter()
                t_first = t_first or now
                if kind == "answer":
                    t_answer = t_answer or now
                rec["chunks"] += 1
    except httpx.HTTPError as e:
        rec.update(status="error", error=f"{type(e).__name__}: {e}"[:300])
    end = time.perf_counter()
    out = (usage or {}).get("completion_tokens") or rec["chunks"]
    rec.update(
        prompt_tokens=(usage or {}).get("prompt_tokens"),
        output_tokens=out,
        e2e=end - start,
        ttft=t_first - start if t_first else None,
        time_to_answer=t_answer - start if t_answer else None,
    )
    if t_first and out > 1:
        rec["tpot"] = (end - t_first) / (out - 1)
    if rec["status"] == "ok" and not t_first:
        rec.update(status="error", error="no tokens received")
    return rec


async def closed_loop(ctx, scenario, step, n_workers, seconds, prompt_tokens, out_tokens):
    """n_workers each send requests back to back for `seconds`."""
    end = time.perf_counter() + seconds
    recs = []

    async def worker():
        while time.perf_counter() < end:
            recs.append(await one_request(ctx, scenario, step, make_prompt(ctx, prompt_tokens), out_tokens))

    await asyncio.gather(*[worker() for _ in range(n_workers)])
    return recs


async def all_at_once(ctx, scenario, step, n, prompt_tokens, out_tokens):
    return await asyncio.gather(*[
        one_request(ctx, scenario, step, make_prompt(ctx, prompt_tokens), out_tokens) for _ in range(n)])


# ---------- results ----------

def pct(values, q):
    values = [v for v in values if v is not None]
    return round(float(np.percentile(values, q)), 4) if values else None


def summarize(recs, wall, **extra):
    ok = [r for r in recs if r["status"] == "ok"]
    row = dict(extra, n=len(recs), ok=len(ok), errors=len(recs) - len(ok), wall_s=round(wall, 2),
               req_s=round(len(ok) / wall, 3), out_tok_s=round(sum(r["output_tokens"] for r in ok) / wall, 2))
    for key in ("ttft", "tpot", "e2e", "time_to_answer"):
        vals = [r.get(key) for r in ok]
        row.update({f"{key}_p50": pct(vals, 50), f"{key}_p95": pct(vals, 95), f"{key}_p99": pct(vals, 99)})
    prefill = [r["prompt_tokens"] / r["ttft"] for r in ok if r.get("prompt_tokens") and r.get("ttft")]
    row["prefill_tok_s"] = round(float(np.median(prefill)), 1) if prefill else None
    row["hit_cap_rate"] = round(sum(r["output_tokens"] >= r["max_tokens"] for r in ok) / max(len(ok), 1), 3)
    return row


def record_step(ctx, scenario, recs, wall, **extra):
    """Append raw records to <scenario>.jsonl, store + print the summary row."""
    with open(ctx.out_dir / f"{scenario}.jsonl", "a") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    row = summarize(recs, wall, scenario=scenario, **extra)
    ctx.summary.append(row)
    (ctx.out_dir / "summary.json").write_text(json.dumps(ctx.summary, indent=1))
    errs = {r["error"][:80] for r in recs if r["status"] != "ok"}
    print(f"[{scenario}] {extra} ok={row['ok']}/{row['n']} out_tok/s={row['out_tok_s']} "
          f"ttft_p50={row['ttft_p50']} tpot_p50={row['tpot_p50']} wall={row['wall_s']}s"
          + (f" errors={sorted(errs)[:2]}" if errs else ""), flush=True)


# ---------- scenarios ----------

async def calibrate(ctx):
    """Learn tokens-per-word from the server's usage report so prompt sizes are about right."""
    rec = await one_request(ctx, "calibrate", 0, make_prompt(ctx, 2000), 1)
    if rec["status"] == "ok" and rec["prompt_tokens"]:
        ctx.tok_per_word = max((rec["prompt_tokens"] - 30) / 2000, 0.3)
    print(f"[calibrate] tokens/word={ctx.tok_per_word:.3f} (status={rec['status']} {rec['error']})", flush=True)


async def warmup(ctx, p):
    await all_at_once(ctx, "warmup", 0, p["warmup"], 256, 64)


async def sweep(ctx, p):
    cfg = p["sweep"]
    for n in cfg["levels"]:
        t = time.perf_counter()
        recs = await closed_loop(ctx, "sweep", n, n, cfg["seconds"], cfg["prompt"], cfg["output"])
        record_step(ctx, "sweep", recs, time.perf_counter() - t, concurrency=n, prompt_target=cfg["prompt"])


async def long_context(ctx, p):
    cfg = p["long_context"]
    for length in cfg["lengths"]:
        t = time.perf_counter()
        recs = await all_at_once(ctx, "long_context", length, 1, length, cfg["output"])
        record_step(ctx, "long_context", recs, time.perf_counter() - t, concurrency=1, context=length)
    n, length = cfg["concurrent_n"], cfg["concurrent_len"]
    t = time.perf_counter()
    recs = await all_at_once(ctx, "long_context", length, n, length, cfg["output"])
    record_step(ctx, "long_context", recs, time.perf_counter() - t, concurrency=n, context=length)


async def burst(ctx, p):
    cfg = p["burst"]
    for cycle in range(cfg["cycles"]):
        t = time.perf_counter()
        recs = await all_at_once(ctx, "burst", cycle, cfg["size"], cfg["prompt"], cfg["output"])
        record_step(ctx, "burst", recs, time.perf_counter() - t, concurrency=cfg["size"], cycle=cycle)
        await asyncio.sleep(cfg["idle"])


async def thinking(ctx, p):
    cfg = p["thinking"]

    async def run(n, label):
        prompts = [THINKING_PROMPTS[i % len(THINKING_PROMPTS)] for i in range(n)]
        t = time.perf_counter()
        recs = await asyncio.gather(*[
            one_request(ctx, "thinking", label, q, cfg["max_tokens"], thinking=True) for q in prompts])
        record_step(ctx, "thinking", recs, time.perf_counter() - t, concurrency=n)

    for _ in range(cfg["single_n"]):
        await run(1, "single")
    await run(cfg["concurrent_n"], "concurrent")


SCENARIOS = {"sweep": sweep, "long_context": long_context, "burst": burst, "thinking": thinking}


# ---------- environment + telemetry ----------

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except subprocess.TimeoutExpired:
        return ""


def write_env(args, base, out_dir):
    try:
        server_version = httpx.get(base + "/version", timeout=5).text
    except httpx.HTTPError:
        server_version = ""
    env = {
        "args": vars(args),
        "time": time.strftime("%F %T"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "gpu": sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"),
        "nvidia_smi": sh("nvidia-smi"),
        "cpu_count": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "server_version": server_version,
        "models": sh(f"curl -s {base}/v1/models"),
    }
    (out_dir / "env.json").write_text(json.dumps(env, indent=1))


METRIC_KEYS = {  # column -> substrings of the Prometheus metric names (vLLM, llama.cpp)
    "running": ("num_requests_running", "requests_processing"),
    "waiting": ("num_requests_waiting", "requests_deferred"),
    "kv_cache": ("kv_cache_usage_perc", "gpu_cache_usage_perc", "kv_cache_usage_ratio"),
}


def scrape_metrics(base):
    try:
        text = httpx.get(base + "/metrics", timeout=2).text
    except httpx.HTTPError:
        return {}
    values = {}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, value = line.rsplit(" ", 1)
        for col, keys in METRIC_KEYS.items():
            if any(k in name for k in keys):
                values.setdefault(col, value)
    return values


def telemetry_loop(stop, path, base, t0):
    query = "utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem"
    cols = ["t", "gpu_util", "gpu_mem_mib", "power_w", "temp_c", "sm_mhz", "mem_mhz",
            "ram_used_gb", "cpu_pct", *METRIC_KEYS]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        while not stop.is_set():
            gpu = sh(f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits").split("\n")[0]
            gpu = [x.strip() for x in gpu.split(",")] if gpu else [""] * 6
            m = scrape_metrics(base)
            mem = psutil.virtual_memory()
            w.writerow([round(time.time() - t0, 1), *gpu, round((mem.total - mem.available) / 1e9, 1),
                        psutil.cpu_percent(), *[m.get(k, "") for k in METRIC_KEYS]])
            f.flush()
            stop.wait(1.0)


# ---------- main ----------

async def run_all(ctx, profile, names):
    await calibrate(ctx)
    await warmup(ctx, profile)
    for name in names:
        print(f"=== {name} ===", flush=True)
        await SCENARIOS[name](ctx, profile)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", default="vllm", help="label for this run (vllm, llamacpp, ...)")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="qwen3.8-27b", help="served model name")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS), help=f"comma list of {list(SCENARIOS)}")
    ap.add_argument("--quick", action="store_true", help="short shakedown run (~5 min)")
    ap.add_argument("--timeout", type=float, default=900, help="per-request timeout, seconds")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    base = args.base_url.rstrip("/").removesuffix("/v1")
    out_dir = Path(args.out) / args.stack
    out_dir.mkdir(parents=True, exist_ok=True)
    names = args.scenarios.split(",")
    profile = PROFILES["quick" if args.quick else "full"]

    t0 = time.time()
    write_env(args, base, out_dir)
    stop = threading.Event()
    thread = threading.Thread(target=telemetry_loop, args=(stop, out_dir / "telemetry.csv", base, t0))
    thread.start()

    async def go():
        limits = httpx.Limits(max_connections=1000, max_keepalive_connections=1000)
        async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
            ctx = Ctx(client, args, base, out_dir, time.perf_counter(), summary=[])
            await run_all(ctx, profile, names)

    try:
        asyncio.run(go())
    finally:
        stop.set()
        thread.join()

    tar_path = Path(args.out) / f"bench-results-{args.stack}-{time.strftime('%Y%m%d-%H%M%S')}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=args.stack)
    print(f"\nDone. Results: {tar_path.resolve()}")


if __name__ == "__main__":
    main()
