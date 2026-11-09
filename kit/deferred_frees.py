import json, os, pickle, sys, bisect
from collections import Counter
run = sys.argv[1]
probe = json.load(open(os.path.join(run, "probe_rank0.json")))
trace = [t for t in pickle.load(open(os.path.join(run, "snapshot_rank0.pickle"), "rb"))["device_traces"] if t][0]
s2k = {v: k for k, v in probe["marker_bytes"].items()}
marks = [i for i, e in enumerate(trace) if e["action"] == "alloc" and e["size"] in s2k]
events = probe["events"][-len(marks):]
idx = [m for m in marks]
def phase(i):
    p = bisect.bisect_right(idx, i) - 1
    e = events[p]
    return {"fwd_start": "forward", "fwd_end": "backward", "bwd_end": "after backward", "step_end": "between steps"}[e["kind"]] + f" (step{e['step']} mb{e['mb']})"
req = {}
lag = Counter(); where_req = Counter(); where_done = Counter(); gap_events = []
for i, e in enumerate(trace):
    if e["action"] == "free_requested":
        req[e["addr"]] = i
    elif e["action"] == "free_completed" and e["addr"] in req:
        r = req.pop(e["addr"])
        if i - r > 1:
            gap_events.append(i - r)
            where_req[phase(r).split(" (")[0]] += e["size"]
            where_done[phase(i).split(" (")[0]] += e["size"]
mb = 2**20
print(f"{run}: deferred frees = {len(gap_events)}, total {sum(where_req.values())/mb:.0f} MB")
if gap_events:
    print("  freed during :", {k: f"{v/mb:.0f} MB" for k, v in where_req.items()})
    print("  reusable during:", {k: f"{v/mb:.0f} MB" for k, v in where_done.items()})
    gap_events.sort(); print("  delay in allocator events: median", gap_events[len(gap_events)//2], "max", gap_events[-1])
