# tp2 record_stream check (9 Nov 2026)

Pod `enz7u5qu78wmhc`, 2 x A40, CA-MTL-1 (GPUs behind one PCIe switch, so `NCCL_P2P_DISABLE=1`).
nanotron main a1d47c7e + the two local patches, torch 2.5.1. Results read from the pod logs (SSH blocked on the local network), pod terminated afterwards.
Model: tiny Llama, hidden 2048, vocab 128k, mbs 2, accum 3. "big" = 8 layers and seq 4096, otherwise 4 layers and seq 1024.

bf16 grads: freed in `_accumulate_grad` and reusable before the next forward in every backward of every run (26/26, or 50/50 for big).

| run | deferred frees (total over 9 backwards) | freed but not reusable at forward start, mb1 / mb2 | reusable only during the next forward | tokens/s |
|---|---|---|---|---|
| tp2 | 1064 MB (230 blocks) | 152 / 152 to 172 MB | 124 MB | 22.8K |
| tp2, TORCH_NCCL_AVOID_RECORD_STREAMS=1 | 0 | 0 / 0 | 0 | 22.8K |
| tp2_big | 8961 MB (451 blocks) | 1120 / 1584 MB | 928 MB | 17.4K |
| tp2_big, TORCH_NCCL_AVOID_RECORD_STREAMS=1 | 0 | 0 / 0 | 0 | 17.5K |

Allocation sites of the held blocks (tp2_big, summed over forward starts):
- 3168 MB `functional.py:412` `_ColumnLinearAsyncCommunication.backward` (`grad_tensor`, input of the async reduce-scatter)
- 1584 MB `layer_norm.py:300` `_layer_norm_fwd` (layer norm output, input of the async all-gather)
- 1584 MB `functional.py:430` `_ColumnLinearAsyncCommunication.backward` (`sub_grad_tensor`, output of the reduce-scatter)
- 80 MB `layer_norm.py:621` `_layer_norm_bwd`

Microbatch 0 never shows held blocks: the optimizer step between steps synchronizes.
Peak allocated/reserved were not recorded in this run.
