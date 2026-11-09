#!/usr/bin/env bash
# Decisive check for #272 on tp2: async TP buffers held by the allocator, with and without NCCL record_stream.
# Usage (on the pod):  nohup bash ~/nanotron-272/check_272.sh > ~/check.log 2>&1 &   ends with "### check done"
set -uo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "### setup"
bash "$KIT/setup_pod.sh" > ~/setup.log 2>&1 || { echo "setup failed"; tail -20 ~/setup.log; echo "### check done"; exit 1; }
source "$HOME/nanotron/.venv/bin/activate"
cd "$HOME/nanotron" && mkdir -p runs
nvidia-smi topo -m > runs/topo.txt 2>&1

# Some Runpod hosts put both GPUs behind one PCIe switch, where NCCL P2P hangs.
if ! timeout 60 torchrun --nproc_per_node=2 "$KIT/nccl_test.py" > runs/nccl_default.log 2>&1; then
  echo "NCCL default transport hangs here, using NCCL_P2P_DISABLE=1"
  export NCCL_P2P_DISABLE=1
fi
echo "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-unset}"

echo "### scenarios"
bash "$KIT/run_matrix.sh" tp2 tp2_norecord tp2_big tp2_big_norecord

{ nvidia-smi; echo "nanotron commit: $(cat ~/nanotron_commit.txt)"; echo "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-unset}"; git diff; } > runs/env.txt 2>&1
tar czf ~/check_272.tgz runs
ls -lh ~/check_272.tgz
echo "### check done"
