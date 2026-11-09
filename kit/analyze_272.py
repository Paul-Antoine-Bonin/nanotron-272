"""Read what probe_272 dumped and tell where the half grads really get released.

  python analyze_272.py runs/dp2 --rank 0

For every backward, each half grad is matched in the allocator trace by address, then we look at:
  * free_requested: the tensor was dropped (refcount hit zero). Expected inside _accumulate_grad.
  * free_completed: the allocator can reuse the block. If this lands much later (e.g. in the next
    forward) while free_requested is on time, memory is held by pending stream events, not by a reference.
Writes summary.md and, if matplotlib is installed, memory_rank{r}.png into the run folder.
"""

import argparse
import bisect
import json
import os
import pickle
from collections import defaultdict

MB = 2**20


def load(run_dir, rank):
    with open(os.path.join(run_dir, f"probe_rank{rank}.json")) as f:
        probe = json.load(f)
    with open(os.path.join(run_dir, f"snapshot_rank{rank}.pickle"), "rb") as f:
        snap = pickle.load(f)
    traces = [t for t in snap["device_traces"] if t]
    if len(traces) != 1:
        raise SystemExit(f"expected one device with a trace, got {len(traces)}")
    return probe, traces[0]


def find_markers(trace, probe):
    size_to_kind = {v: k for k, v in probe["marker_bytes"].items()}
    found = [(i, size_to_kind[e["size"]]) for i, e in enumerate(trace) if e["action"] == "alloc" and e["size"] in size_to_kind]
    events = probe["events"]
    # the trace ring buffer may have dropped the oldest entries: align on the tail
    if len(found) < len(events):
        events = events[len(events) - len(found):]
    elif len(found) > len(events):
        found = found[len(found) - len(events):]
    bad = sum(k != e["kind"] for (_, k), e in zip(found, events))
    if bad:
        raise SystemExit(f"{bad} markers don't line up with the probe log (a real tensor with a marker size?)")
    for (i, _), e in zip(found, events):
        e["trace_idx"] = i
    return events


def index_by_addr(trace):
    idx = defaultdict(list)
    for i, e in enumerate(trace):
        if e["action"] in ("alloc", "free_requested", "free_completed"):
            idx[e["addr"]].append(i)
    return idx


def phase_of(i, events):
    """Name of the phase a trace index falls in, e.g. 'step1 mb0 forward'."""
    pos = bisect.bisect_right([e["trace_idx"] for e in events], i) - 1
    if pos < 0:
        return "before recording"
    e = events[pos]
    where = {"fwd_start": "forward", "fwd_end": "backward", "bwd_end": "after backward", "step_end": "between steps"}[e["kind"]]
    return f"step{e['step']} mb{e['mb']} {where}"


def top_frames(entry, n=3):
    frames = [f for f in entry.get("frames", []) if "probe_272" not in f.get("filename", "")]
    return " <- ".join(f"{os.path.basename(f['filename'])}:{f['line']} {f['name']}" for f in frames[:n]) or "(no python frame)"


def analyze(run_dir, rank):
    probe, trace = load(run_dir, rank)
    events = find_markers(trace, probe)
    by_addr = index_by_addr(trace)
    fwd_starts = [e["trace_idx"] for e in events if e["kind"] == "fwd_start"]

    lines = [f"# {run_dir} rank {rank}", "", f"trace entries: {len(trace)}", ""]
    lines.append("| backward | grads | MB | freed in _accumulate_grad | reusable before next forward | late MB | where late ones become reusable |")
    lines.append("|---|---|---|---|---|---|---|")

    late_examples = []
    for e in events:
        if e["kind"] != "bwd_end" or not e.get("grads"):
            continue
        m = e["trace_idx"]
        nxt = next((i for i in fwd_starts if i > m), len(trace))
        n = len(e["grads"])
        total = sum(g["nbytes"] for g in e["grads"])
        on_time_req = on_time_done = 0
        late_bytes = 0
        late_where = defaultdict(int)
        for g in e["grads"]:
            hits = by_addr.get(g["ptr"], [])
            # last alloc at this address before the end of this backward
            allocs = [i for i in hits if i < m and trace[i]["action"] == "alloc"]
            if not allocs:
                late_where["not found in trace"] += g["nbytes"]
                continue
            a = allocs[-1]
            req = next((i for i in hits if i > a and trace[i]["action"] == "free_requested"), None)
            done = next((i for i in hits if i > a and trace[i]["action"] == "free_completed"), None)
            if req is not None and req < m:
                on_time_req += 1
            if done is not None and done < nxt:
                on_time_done += 1
            else:
                late_bytes += g["nbytes"]
                late_where[phase_of(done, events) if done is not None else "never (still held at dump)"] += g["nbytes"]
                if len(late_examples) < 5 and done is not None:
                    late_examples.append((g["name"], phase_of(req, events) if req else "?", top_frames(trace[req]) if req else "", phase_of(done, events), top_frames(trace[done])))
        where = ", ".join(f"{k}: {v / MB:.0f} MB" for k, v in sorted(late_where.items())) or "-"
        lines.append(
            f"| step{e['step']} mb{e['mb']} | {n} | {total / MB:.0f} | {on_time_req}/{n} | {on_time_done}/{n} | {late_bytes / MB:.0f} | {where} |"
        )

    lines += ["", "## Python side at each forward start", "", "| forward | prev grads still alive | MB | allocated MB | active MB | reserved MB |", "|---|---|---|---|---|---|"]
    for e in events:
        if e["kind"] == "fwd_start":
            lines.append(
                f"| step{e['step']} mb{e['mb']} | {e.get('prev_grads_alive', 0)} | {e.get('prev_grads_alive_bytes', 0) / MB:.0f} | "
                f"{e['allocated'] / MB:.0f} | {e['active'] / MB:.0f} | {e['reserved'] / MB:.0f} |"
            )
            if e.get("referrer_types"):
                lines.append(f"|  | referrers: {', '.join(e['referrer_types'])} | | | | |")

    if late_examples:
        lines += ["", "## A few late grads", ""]
        for name, req_ph, req_fr, done_ph, done_fr in late_examples:
            lines.append(f"- `{name}`: dropped at {req_ph} ({req_fr}), reusable at {done_ph} ({done_fr})")

    out = "\n".join(lines) + "\n"
    with open(os.path.join(run_dir, f"summary_rank{rank}.md"), "w") as f:
        f.write(out)
    print(out)
    plot(run_dir, rank, trace, events)


def plot(run_dir, rank, trace, events):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    # curves relative to the moment recording started, shifted by what was allocated then
    base_alloc, base_active = events[0]["allocated"], events[0]["active"]
    alloc, active = base_alloc, base_active
    xs, ya, yb = [], [], []
    for i, e in enumerate(trace):
        a = e["action"]
        if a == "alloc":
            alloc += e["size"]
            active += e["size"]
        elif a == "free_requested":
            alloc -= e["size"]
        elif a == "free_completed":
            active -= e["size"]
        else:
            continue
        xs.append(i)
        ya.append(alloc / 2**30)
        yb.append(active / 2**30)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(xs, yb, label="active (block not reusable yet)", lw=1)
    ax.plot(xs, ya, label="allocated (tensor still referenced)", lw=1)
    colors = {"fwd_start": "tab:green", "bwd_end": "tab:red", "step_end": "black"}
    for e in events:
        if e["kind"] in colors:
            ax.axvline(e["trace_idx"], color=colors[e["kind"]], lw=0.6, alpha=0.6)
    ax.set_xlabel("allocator events (green: forward start, red: backward end, black: step end)")
    ax.set_ylabel("GiB")
    ax.legend(loc="upper left")
    ax.set_title(f"{os.path.basename(run_dir.rstrip('/'))} rank {rank}")
    path = os.path.join(run_dir, f"memory_rank{rank}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--rank", type=int, default=0)
    a = p.parse_args()
    analyze(a.run_dir, a.rank)
