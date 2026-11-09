#!/usr/bin/env bash
# Runs the #272 scenarios one after the other, then the analysis.
# From the nanotron repo root, with the venv active:  bash /path/to/nanotron-272/run_matrix.sh [scenario...]
set -uo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
export CUDA_DEVICE_MAX_CONNECTIONS=1

run() {  # name nproc [env...] -- args...
  local name=$1 nproc=$2; shift 2
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  if [ "$nproc" -gt "$NGPU" ]; then echo "skip $name (needs $nproc GPUs)"; return; fi
  echo "=== $name"
  # a stuck collective otherwise waits for NCCL's 30 min watchdog; each run takes well under a minute
  env "${envs[@]}" timeout "${RUN_TIMEOUT:-600}" torchrun --nproc_per_node="$nproc" "$KIT/run_272.py" --out "runs/$name" "$@" \
    > "runs/$name.log" 2>&1 || { echo "  failed, see runs/$name.log"; rm -rf "runs/$name/ckpt"; return; }
  # nanotron writes a full checkpoint (5 to 8 GB) at the end of every run; we never use it and 40 GB fills up fast
  rm -rf "runs/$name/ckpt"
  python "$KIT/analyze_272.py" "runs/$name" --rank 0 > /dev/null && echo "  ok: runs/$name/summary_rank0.md"
}

mkdir -p runs
want() { [ $# -eq 0 ] || [[ " $* " == *" $name "* ]]; }
SEL=("$@")

for name in single dp2 tp2 dp2_norecord dp2_expandable single_afab tp2_norecord tp2_big tp2_big_norecord; do
  want "${SEL[@]}" || continue
  case $name in
    single)          run single 1 X=1 -- --dp 1 --tp 1 --accum 3 ;;
    single_afab)     run single_afab 1 X=1 -- --dp 1 --tp 1 --accum 3 --pp-engine afab ;;
    dp2)             run dp2 2 X=1 -- --dp 2 --tp 1 --accum 3 ;;
    tp2)             run tp2 2 X=1 -- --dp 1 --tp 2 --accum 3 ;;
    # does the delay go away if NCCL stops calling record_stream on its inputs?
    dp2_norecord)    run dp2_norecord 2 TORCH_NCCL_AVOID_RECORD_STREAMS=1 -- --dp 2 --tp 1 --accum 3 ;;
    # cheap allocator-side mitigation, to compare with empty_cache
    dp2_expandable)  run dp2_expandable 2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -- --dp 2 --tp 1 --accum 3 ;;
    # tp2 holds async TP buffers until NCCL stream events complete: does it go away without record_stream?
    tp2_norecord)    run tp2_norecord 2 TORCH_NCCL_AVOID_RECORD_STREAMS=1 -- --dp 1 --tp 2 --accum 3 ;;
    # same pair at a larger activation size, to see how the held memory scales
    tp2_big)         run tp2_big 2 X=1 -- --dp 1 --tp 2 --accum 3 --layers 8 --seq 4096 ;;
    tp2_big_norecord) run tp2_big_norecord 2 TORCH_NCCL_AVOID_RECORD_STREAMS=1 -- --dp 1 --tp 2 --accum 3 --layers 8 --seq 4096 ;;
  esac
done
