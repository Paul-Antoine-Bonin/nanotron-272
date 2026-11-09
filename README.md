# Where do the bf16 grads get released? (nanotron #272)

Scripts and results for [huggingface/nanotron#272](https://github.com/huggingface/nanotron/issues/272), "half grads don't necessarily get released before the next forward".

**Short version.** The bf16 grads are released on time: they die inside `_accumulate_grad` and their memory is reusable before the next forward, in every setup I tried. The pattern described in the issue does show up with async tensor parallelism, on other tensors: buffers used by the async all-gather and reduce-scatter in `_ColumnLinearAsyncCommunication` stay held by the CUDA allocator until NCCL's stream events complete, some of them until the next forward. `TORCH_NCCL_AVOID_RECORD_STREAMS=1` removes it completely at the same throughput.

## Setup

- nanotron main at `a1d47c7e`, with two one-line patches needed to train a Llama at all: the parametrizator config (same change as nanotron#421) and the tp=1 column linear backward (nanotron#438). `setup_pod.sh` applies both.
- torch 2.5.1 (cu124), flash-attn 2.7.4.post1, 2 x A40 on Runpod.
- `examples/config_tiny_llama.yaml` cut down to 4 layers, hidden 2048, vocab 128k with tied embeddings (571M params), micro batch 2, sequence 1024, 3 accumulation steps, dummy data. The "big" runs use 8 layers and sequence 4096.

The probe (`kit/probe_272.py`) patches `FP32GradientAccumulator._accumulate_grad`, the pipeline engine forward/backward and `training_step`. It records the CUDA allocator trace (`torch.cuda.memory._record_memory_history`), the address of every bf16 grad at the moment it is set to `None`, weakrefs to those grads checked at the next forward, and small marker allocations of fixed sizes that cut the trace into phases.

The allocator frees a block in two steps. `free_requested` happens when the tensor dies, and `memory_allocated()` drops. `free_completed` happens when the block can be reused. If the tensor was used on another stream (NCCL calls `record_stream` on the inputs and outputs of async collectives), the second step waits for a GPU event, and the block counts as `active` memory until then.

## Results

### The bf16 grads

9 backwards per scenario (3 steps x 3 micro batches), rank 0. Details in `results/run1/<scenario>/summary_rank0.md`.

| scenario | bf16 grads freed in `_accumulate_grad` | reusable before next forward | still alive at next forward | deferred frees, any tensor | tokens/s |
|---|---|---|---|---|---|
| single (1 GPU) | 26/26 | 26/26 | 0 | 0 MB | 15.4K |
| dp2 | 26/26 | 26/26 | 0 | 0 MB | 21.3K |
| dp2, `TORCH_NCCL_AVOID_RECORD_STREAMS=1` | 26/26 | 26/26 | 0 | 0 MB | 21.3K |
| dp2, `expandable_segments:True` | 26/26 | 26/26 | 0 | 0 MB | 21.1K |
| tp2 (async TP) | 26/26 | 26/26 | 0 | 1064 MB | 25.6K |
| single, AFAB engine | 26/26 | 26/26 | 0 | 0 MB | 15.4K |

With DDP the bf16 grads never reach NCCL: the all-reduce runs on the fp32 accumulation buffers.

### Async TP buffers and `record_stream`

Second run, on another machine. Details in `results/run2_record_stream.md`.

| tp=2 | deferred frees | freed but not reusable at forward start (mb1 / mb2) | reusable only during the next forward | tokens/s |
|---|---|---|---|---|
| 4 layers, seq 1024 | 1064 MB | 152 / 152 to 172 MB | 124 MB | 22.8K |
| same, `TORCH_NCCL_AVOID_RECORD_STREAMS=1` | 0 | 0 / 0 | 0 | 22.8K |
| 8 layers, seq 4096 | 8961 MB | 1120 / 1584 MB | 928 MB | 17.4K |
| same, `TORCH_NCCL_AVOID_RECORD_STREAMS=1` | 0 | 0 / 0 | 0 | 17.5K |

The held blocks come from `_ColumnLinearAsyncCommunication.backward` (`grad_tensor`, the input of the async reduce-scatter, and `sub_grad_tensor`, its output) and from the layer norm outputs that feed the async all-gather. Micro batch 0 never shows any, because the optimizer step synchronizes in between. In a profiler that tracks `active` memory, these blocks look like they are released during the next forward, around `column_linear`.

`_ColumnLinearAsyncCommunication` already waits on every async handle, so the buffers are safe to free once the collective is done, which is what `TORCH_NCCL_AVOID_RECORD_STREAMS=1` does.

`results/run1/tp2/memory_rank0.png` shows the gap between `allocated` and `active` memory. The allocator snapshots (`snapshot_rank0.pickle`) open in https://pytorch.org/memory_viz.

## Limits

- Peak memory (`max_memory_allocated`, `max_memory_reserved`) with and without `TORCH_NCCL_AVOID_RECORD_STREAMS=1` is not measured yet.
- Only tp=2 and small models. tp=4 and a real model size are not tested.
- I could not reproduce a delay on the bf16 grads themselves. The original report may come from a different config or torch version.

## Running it

On a fresh GPU box (Ampere or newer):

```bash
bash kit/go.sh
```

This clones nanotron, builds the environment (`kit/setup_pod.sh`), runs the scenarios (`kit/run_matrix.sh`) and analyzes each one (`kit/analyze_272.py`). One scenario at a time, from the nanotron repo root with the venv active:

```bash
bash /path/to/kit/run_matrix.sh tp2 tp2_norecord
```

```bash
python /path/to/kit/deferred_frees.py runs/tp2
```

`kit/check_272.sh` runs only the four tp2 runs above. `kit/run_in_container.sh` does the same as a container entrypoint, with the kit passed in an environment variable and the results printed to stdout, for machines you can't SSH into.

Notes:

- nanotron main needs `grouped_gemm` even for dense models, and current `transformers`, `datasets` and `pyarrow` are too new for it; the pins are in `setup_pod.sh`.
- On machines where both GPUs sit behind one PCIe switch (`PIX` or `PXB` in `nvidia-smi topo -m`), NCCL P2P can hang; `kit/nccl_test.py` checks it and `NCCL_P2P_DISABLE=1` works around it.
- nanotron's `tests/pytest.ini` sets `-n 35`; on a 2 GPU machine use `-n 0`.
