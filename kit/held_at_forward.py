"""Which blocks are freed (free_requested) before a forward starts but only reusable (free_completed) after it?"""
import json, os, pickle, sys
from collections import defaultdict

run = sys.argv[1]
probe = json.load(open(os.path.join(run, "probe_rank0.json")))
snap = pickle.load(open(os.path.join(run, "snapshot_rank0.pickle"), "rb"))
trace = [t for t in snap["device_traces"] if t][0]
size_to_kind = {v: k for k, v in probe["marker_bytes"].items()}
marks = [(i, size_to_kind[e["size"]]) for i, e in enumerate(trace) if e["action"] == "alloc" and e["size"] in size_to_kind]
events = probe["events"][-len(marks):]
for (i, _), e in zip(marks, events):
    e["idx"] = i
streams = defaultdict(int)
for e in trace:
    streams[e.get("stream")] += 1
print("streams in trace:", dict(streams))

def frames(e, n=4):
    fr = [f for f in e.get("frames", []) if "probe_272" not in f["filename"] and "torch/" not in f["filename"]]
    return " <- ".join(f"{os.path.basename(f['filename'])}:{f['line']}:{f['name']}" for f in fr[:n])

last_alloc, req = {}, {}
held = defaultdict(lambda: [0, 0])  # site -> [count, bytes] over all forward starts
per_fwd = []
fwd = [e for e in events if e["kind"] == "fwd_start"]
fi = 0
pending = {}
for i, e in enumerate(trace):
    while fi < len(fwd) and fwd[fi]["idx"] == i:
        tot = sum(sz for (sz, _) in pending.values())
        per_fwd.append((f"step{fwd[fi]['step']} mb{fwd[fi]['mb']}", len(pending), tot))
        for a, (sz, site) in pending.items():
            held[site][0] += 1; held[site][1] += sz
        fi += 1
    a = e.get("addr")
    if e["action"] == "alloc":
        last_alloc[a] = e
    elif e["action"] == "free_requested":
        al = last_alloc.get(a)
        pending[a] = (e["size"], frames(al) if al else "?")
    elif e["action"] == "free_completed":
        pending.pop(a, None)

print("\nfreed but not yet reusable at each forward start:")
for name, n, tot in per_fwd:
    print(f"  {name}: {n} blocks, {tot/2**20:.0f} MB")
print("\nallocation sites of those blocks (summed over forward starts):")
for site, (c, b) in sorted(held.items(), key=lambda kv: -kv[1][1])[:8]:
    print(f"  {b/2**20:7.0f} MB  x{c:3d}  {site}")
