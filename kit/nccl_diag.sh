#!/usr/bin/env bash
# Setup, then find which NCCL transport works on this box. Writes ~/nccl_diag.log, ends with "### diag done".
# Usage (on the pod):  nohup bash ~/nanotron-272/nccl_diag.sh > ~/nccl_diag.log 2>&1 &
set -uo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "### setup"
bash "$KIT/setup_pod.sh" || { echo "setup failed"; echo "### diag done"; exit 1; }
source "$HOME/nanotron/.venv/bin/activate"
cd "$HOME/nanotron" && mkdir -p runs
nvidia-smi topo -m > runs/topo.txt 2>&1

try() {  # label env...
  local label=$1; shift
  echo "=== nccl $label"
  env "$@" NCCL_DEBUG=INFO timeout 120 torchrun --nproc_per_node=2 "$KIT/nccl_test.py" > "runs/nccl_$label.log" 2>&1
  echo "  exit $?: $(grep -m1 'nccl ok' "runs/nccl_$label.log" || echo 'no ok line')"
  grep -m3 -E "via P2P|via SHM|NET/Socket|Channel 00/0 :" "runs/nccl_$label.log" | sed 's/^/    /'
}

try default X=1
try p2p_off NCCL_P2P_DISABLE=1
try p2p_shm_off NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
echo "### diag done"
