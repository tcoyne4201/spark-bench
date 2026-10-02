# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "matplotlib"]
# ///
"""Turn bench results into report.md + PNG charts.

    uv run report.py results/                      # dir containing <stack>/ subdirs
    uv run report.py bench-results-*.tar.gz        # one or more tarballs (compares stacks)
"""
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def find_stacks(paths, tmp):
    """Return {stack_name: dir containing summary.json}."""
    stacks = {}
    for p in map(Path, paths):
        if p.suffixes[-2:] == [".tar", ".gz"]:
            with tarfile.open(p) as tar:
                tar.extractall(tmp, filter="data")
            p = Path(tmp)
        for summary in p.rglob("summary.json"):
            stacks[summary.parent.name] = summary.parent
    return stacks


def md_table(df):
    df = df.fillna("")
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def line_chart(data, scenario, x, ys, title, ylabel, path, logx=False):
    """One line per (stack, y column) from summary rows of one scenario."""
    fig, ax = plt.subplots(figsize=(7, 4))
    drawn = False
    for stack, df in data.items():
        d = df[df.scenario == scenario].sort_values(x)
        for y in ys:
            if y in d and d[y].notna().any():
                ax.plot(d[x], d[y], marker="o", label=f"{stack} {y}")
                drawn = True
    if drawn:
        ax.set(title=title, xlabel=x, ylabel=ylabel)
        if logx:
            ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return drawn


def telemetry_chart(stack, d, path):
    f = d / "telemetry.csv"
    if not f.exists():
        return False
    t = pd.read_csv(f).apply(pd.to_numeric, errors="coerce")
    cols = [c for c in ("gpu_util", "power_w", "temp_c", "running", "waiting", "kv_cache") if t[c].notna().any()]
    if not cols:
        return False
    fig, axes = plt.subplots(len(cols), 1, figsize=(8, 1.6 * len(cols)), sharex=True)
    for ax, c in zip([axes] if len(cols) == 1 else axes, cols):
        ax.plot(t["t"], t[c])
        ax.set_ylabel(c)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("seconds since start") if len(cols) > 1 else None
    fig.suptitle(f"telemetry: {stack}")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return True


def env_line(d):
    env = json.loads((d / "env.json").read_text()) if (d / "env.json").exists() else {}
    return f"GPU: {env.get('gpu', '?')} | machine: {env.get('machine', '?')} | RAM {env.get('ram_gb', '?')} GB | run at {env.get('time', '?')}"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    tmp = tempfile.mkdtemp()
    stacks = find_stacks(sys.argv[1:], tmp)
    if not stacks:
        sys.exit("no summary.json found")
    out = Path("report")
    out.mkdir(exist_ok=True)
    data = {s: pd.DataFrame(json.loads((d / "summary.json").read_text())) for s, d in stacks.items()}

    charts = [
        ("sweep", "concurrency", ["out_tok_s"], "Throughput vs concurrency", "output tokens/s", "throughput.png", True),
        ("sweep", "concurrency", ["ttft_p50", "ttft_p95", "ttft_p99"], "TTFT vs concurrency", "seconds", "ttft.png", True),
        ("sweep", "concurrency", ["tpot_p50", "tpot_p95"], "Per-token latency vs concurrency", "seconds/token", "tpot.png", True),
        ("long_context", "context", ["ttft_p50"], "TTFT vs context length (single stream)", "seconds", "ctx_ttft.png", True),
        ("long_context", "context", ["prefill_tok_s"], "Prefill speed vs context length", "prompt tokens/s", "ctx_prefill.png", True),
    ]
    md = ["# DGX Spark benchmark report", ""]
    for s, d in stacks.items():
        md.append(f"- **{s}**: {env_line(d)}")
    for scenario, x, ys, title, ylabel, fname, logx in charts:
        if line_chart(data, scenario, x, ys, title, ylabel, out / fname, logx):
            md += ["", f"![{title}]({fname})"]
    for s, d in stacks.items():
        if telemetry_chart(s, d, out / f"telemetry_{s}.png"):
            md += ["", f"![telemetry {s}](telemetry_{s}.png)"]

    show = ["concurrency", "context", "n", "ok", "errors", "wall_s", "out_tok_s", "ttft_p50", "ttft_p95",
            "tpot_p50", "tpot_p95", "e2e_p95", "prefill_tok_s", "time_to_answer_p50", "hit_cap_rate"]
    for s, df in data.items():
        for scenario, g in df.groupby("scenario", sort=False):
            cols = [c for c in show if c in g and g[c].notna().any()]
            md += ["", f"## {s}: {scenario}", "", md_table(g[cols])]
    (out / "report.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
