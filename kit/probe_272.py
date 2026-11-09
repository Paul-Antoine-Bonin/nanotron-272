"""Memory probe for huggingface/nanotron#272.

Goal: find out why bf16 grads set to None in FP32GradientAccumulator._accumulate_grad
are not released before the next forward.

What it records, per rank:
  * a CUDA allocator trace (torch.cuda.memory._record_memory_history)
  * for every backward: the address/size of each half grad right before it is set to None
  * at the start of every forward: how many of those grads are still alive on the Python side
    (weakrefs), plus allocated / active / reserved memory
  * small "marker" allocations of fixed sizes so the trace can be cut into phases

allocated drops when a tensor is freed (free_requested in the trace).
active drops when the allocator can actually reuse the block (free_completed).
If the gap between the two covers the half grads, the memory is held by the allocator
(cross-stream use / pending events), not by a live reference.
"""

import gc
import json
import os
import weakref

import torch

# Sizes in bytes, multiples of 512 so the allocator does not round them,
# below 1MB so they go to the small pool. Odd multipliers to avoid clashing with real tensors.
MARKER_BYTES = {
    "fwd_start": 512 * 1901,
    "fwd_end": 512 * 1907,
    "bwd_end": 512 * 1913,
    "step_end": 512 * 1931,
}


class _State:
    def __init__(self):
        self.active = False
        self.rank = 0
        self.out_dir = "."
        self.step = 0
        self.mb = 0
        self.record_steps = 3
        self.pending = []  # grads seen during the current backward
        self.last_refs = []  # weakrefs to the grads of the previous backward
        self.events = []
        self.referrers_logged = False


S = _State()


def _mem():
    stats = torch.cuda.memory_stats()
    return {
        "allocated": torch.cuda.memory_allocated(),
        "active": stats.get("active_bytes.all.current", 0),
        "reserved": torch.cuda.memory_reserved(),
    }


def _marker(kind):
    t = torch.empty(MARKER_BYTES[kind], dtype=torch.uint8, device="cuda")
    del t


def _log(kind, **extra):
    _marker(kind)
    S.events.append({"kind": kind, "step": S.step, "mb": S.mb, **_mem(), **extra})


def _patch_grad_accumulator():
    from nanotron.optim.gradient_accumulator import FP32GradientAccumulator

    orig = FP32GradientAccumulator._accumulate_grad

    def _accumulate_grad(self, name, half_param):
        if S.active and half_param.grad is not None:
            g = half_param.grad
            S.pending.append(
                {
                    "name": name,
                    "ptr": g.data_ptr(),
                    "nbytes": g.numel() * g.element_size(),
                    "ref": weakref.ref(g),
                }
            )
            del g
        return orig(self, name=name, half_param=half_param)

    FP32GradientAccumulator._accumulate_grad = _accumulate_grad


def _patch_engine():
    from nanotron.parallel.pipeline_parallel.engine import PipelineEngine

    orig_forward = PipelineEngine.forward
    orig_backward = PipelineEngine.backward

    def forward(self, context, state, micro_batch, model):
        if S.active:
            alive = [r() for r in S.last_refs]
            alive = [t for t in alive if t is not None]
            extra = {
                "prev_grads_alive": len(alive),
                "prev_grads_alive_bytes": sum(t.numel() * t.element_size() for t in alive),
            }
            if alive and not S.referrers_logged:
                # Only Python-side holders show up here. Holders living in C++ (autograd graph,
                # DDP reducer) won't, which is itself useful to know.
                extra["referrer_types"] = sorted({type(o).__name__ for o in gc.get_referrers(alive[0])})
                S.referrers_logged = True
            del alive
            _log("fwd_start", **extra)
        out = orig_forward(self, context=context, state=state, micro_batch=micro_batch, model=model)
        if S.active:
            _log("fwd_end")
        return out

    def backward(self, context, state, grad_accumulator):
        out = orig_backward(self, context=context, state=state, grad_accumulator=grad_accumulator)
        if S.active:
            grads = [{k: v for k, v in g.items() if k != "ref"} for g in S.pending]
            S.last_refs = [g["ref"] for g in S.pending]
            S.pending = []
            _log("bwd_end", grads=grads)
            S.mb += 1
        return out

    PipelineEngine.forward = forward
    PipelineEngine.backward = backward


def _patch_trainer():
    from nanotron.trainer import DistributedTrainer

    orig = DistributedTrainer.training_step

    def training_step(self, dataloader):
        out = orig(self, dataloader)
        if S.active:
            _log("step_end")
            S.step += 1
            S.mb = 0
            if S.step >= S.record_steps:
                dump()
        return out

    DistributedTrainer.training_step = training_step


def install(out_dir, record_steps=3, max_entries=2_000_000):
    """Call after the trainer is built (model already on GPU), before trainer.train()."""
    S.rank = int(os.environ.get("RANK", "0"))
    S.out_dir = out_dir
    S.record_steps = record_steps
    os.makedirs(out_dir, exist_ok=True)

    _patch_grad_accumulator()
    _patch_engine()
    _patch_trainer()

    torch.cuda.synchronize()
    torch.cuda.memory._record_memory_history(max_entries=max_entries, stacks="python")
    S.active = True
    _log("step_end", note="recording starts")


def dump():
    if not S.active:
        return
    S.active = False
    torch.cuda.synchronize()
    snap_path = os.path.join(S.out_dir, f"snapshot_rank{S.rank}.pickle")
    torch.cuda.memory._dump_snapshot(snap_path)
    torch.cuda.memory._record_memory_history(enabled=None)
    with open(os.path.join(S.out_dir, f"probe_rank{S.rank}.json"), "w") as f:
        json.dump({"marker_bytes": MARKER_BYTES, "events": S.events}, f)
    print(f"[probe_272] rank {S.rank}: wrote {snap_path}", flush=True)
