# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "plotly"]
# ///
"""Turn bench results into one self-contained, interactive report/report.html.

    uv run report.py fetched/*.tar.gz              # one or more tarballs (several stacks are compared)
    uv run report.py results/                      # or a dir containing <stack>/ subdirs
"""
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CHARTS = [  # scenario, x, y columns, title, y label
    ("sweep", "concurrency", ["out_tok_s"], "Throughput vs concurrency", "output tokens/s"),
    ("sweep", "concurrency", ["ttft_p50", "ttft_p95", "ttft_p99"], "Time to first token vs concurrency", "seconds"),
    ("sweep", "concurrency", ["tpot_p50", "tpot_p95"], "Per-token latency vs concurrency", "seconds/token"),
    ("sweep", "concurrency", ["errors"], "Errors vs concurrency", "failed requests"),
    ("long_context", "context", ["ttft_p50"], "TTFT vs context length (single stream)", "seconds"),
    ("long_context", "context", ["prefill_tok_s"], "Prefill speed vs context length", "prompt tokens/s"),
    ("burst", "cycle", ["wall_s", "ttft_p95", "e2e_p95"], "Burst: drain time and latency per cycle", "seconds"),
    ("thinking", "concurrency", ["time_to_answer_p50", "e2e_p50", "hit_cap_rate"], "Thinking mode", "seconds / rate"),
]
TELEMETRY = ["gpu_util", "power_w", "temp_c", "sm_mhz", "gpu_mem_mib", "ram_used_gb", "running", "waiting", "kv_cache"]
TABLE_COLS = ["concurrency", "context", "cycle", "n", "ok", "errors", "wall_s", "out_tok_s", "ttft_p50", "ttft_p95",
              "tpot_p50", "tpot_p95", "e2e_p95", "prefill_tok_s", "time_to_answer_p50", "hit_cap_rate"]


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


def line_chart(data, scenario, x, ys, title, ylabel):
    """One line per (stack, column) from the summary rows of one scenario; None if no data."""
    fig = go.Figure()
    for stack, df in data.items():
        d = df[df.scenario == scenario]
        if x not in d:
            continue
        d = d.sort_values(x)
        for y in ys:
            if y in d and d[y].notna().any():
                fig.add_trace(go.Scatter(x=d[x], y=d[y], mode="lines+markers", name=f"{stack} {y}"))
    if not fig.data:
        return None
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=ylabel, height=380, margin=dict(t=50, b=40))
    if x in ("concurrency", "context"):
        fig.update_xaxes(type="log", dtick=0.30103)  # log2 steps
    return fig


def telemetry_chart(stack, d):
    f = d / "telemetry.csv"
    if not f.exists():
        return None
    t = pd.read_csv(f).apply(pd.to_numeric, errors="coerce")
    cols = [c for c in TELEMETRY if c in t and t[c].notna().any()]
    if not cols:
        return None
    fig = make_subplots(rows=len(cols), cols=1, shared_xaxes=True, subplot_titles=cols, vertical_spacing=0.03)
    for i, c in enumerate(cols, start=1):
        fig.add_trace(go.Scatter(x=t["t"], y=t[c], name=c, showlegend=False), row=i, col=1)
    fig.update_xaxes(title_text="seconds since start", row=len(cols), col=1)
    fig.update_layout(title=f"Telemetry: {stack}", height=170 * len(cols) + 80, margin=dict(t=60, b=40))
    return fig


def env_line(d):
    f = d / "env.json"
    env = json.loads(f.read_text()) if f.exists() else {}
    return (f"GPU: {env.get('gpu') or '?'} | {env.get('machine', '?')} | RAM {env.get('ram_gb', '?')} GB | "
            f"run at {env.get('time', '?')} | server: {env.get('server_version') or '?'}")


def table_html(g):
    cols = [c for c in TABLE_COLS if c in g and g[c].notna().any()]
    return g[cols].to_html(index=False, na_rep="", border=0, classes="t")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    stacks = find_stacks(sys.argv[1:], tempfile.mkdtemp())
    if not stacks:
        sys.exit("no summary.json found")
    data = {s: pd.DataFrame(json.loads((d / "summary.json").read_text())) for s, d in stacks.items()}

    figs = [line_chart(data, *c) for c in CHARTS]
    figs += [telemetry_chart(s, d) for s, d in stacks.items()]
    figs = [f for f in figs if f is not None]

    html = ["<html><head><meta charset='utf-8'><title>Spark benchmark</title><style>"
            "body{font-family:sans-serif;max-width:1000px;margin:2em auto}"
            ".t{border-collapse:collapse;font-size:12px}.t td,.t th{padding:3px 8px;border-bottom:1px solid #ddd;text-align:right}"
            "</style></head><body><h1>DGX Spark benchmark report</h1><ul>"]
    html += [f"<li><b>{s}</b>: {env_line(d)}</li>" for s, d in stacks.items()]
    html.append("</ul>")
    for i, fig in enumerate(figs):  # plotly.js embedded once so the file works offline
        html.append(fig.to_html(full_html=False, include_plotlyjs=True if i == 0 else False))
    for s, df in data.items():
        for scenario, g in df.groupby("scenario", sort=False):
            html.append(f"<h2>{s}: {scenario}</h2>{table_html(g)}")
    html.append("</body></html>")

    out = Path("report")
    out.mkdir(exist_ok=True)
    (out / "report.html").write_text("\n".join(html))
    print(f"wrote {out / 'report.html'}  (open it in a browser)")


if __name__ == "__main__":
    main()
