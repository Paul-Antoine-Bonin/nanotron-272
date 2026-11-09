"""Tiny NCCL check before any nanotron run: one all_reduce, one all_gather, one reduce_scatter.

  torchrun --nproc_per_node=2 nccl_test.py
If this hangs, nanotron will hang too, and the cause is the NCCL transport (P2P/SHM), not nanotron.
"""

import datetime
import os
import time

import torch
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=60), device_id=torch.device("cuda", rank))
world = dist.get_world_size()

t0 = time.time()
x = torch.ones(1, device="cuda")
dist.all_reduce(x)
torch.cuda.synchronize()
assert x.item() == world, x

big = torch.full((64 * 2**20,), float(rank), device="cuda")  # 256 MB
dist.all_reduce(big)
torch.cuda.synchronize()

out = torch.empty(world * 2**20, device="cuda")
dist.all_gather_into_tensor(out, torch.full((2**20,), float(rank), device="cuda"))
red = torch.empty(2**20, device="cuda")
dist.reduce_scatter_tensor(red, torch.ones(world * 2**20, device="cuda"))
torch.cuda.synchronize()

if rank == 0:
    print(f"nccl ok: world={world}, {time.time() - t0:.2f}s", flush=True)
dist.destroy_process_group()
