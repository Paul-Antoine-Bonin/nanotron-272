#!/usr/bin/env bash
# Everything in one go: setup, all scenarios, analysis, small archive to send back.
# Usage (on the pod):  nohup bash ~/nanotron-272/go.sh > ~/go.log 2>&1 &
#                      tail -f ~/go.log
set -uo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HOME"
START=$(date +%s)

echo "### setup"
bash "$KIT/setup_pod.sh" || { echo "setup failed"; exit 1; }
source "$HOME/nanotron/.venv/bin/activate"
cd "$HOME/nanotron"

echo "### scenarios"
# single first: if it breaks, the others will break the same way, and we see it early in the log
bash "$KIT/run_matrix.sh" single
bash "$KIT/run_matrix.sh" dp2 tp2 dp2_norecord dp2_expandable single_afab

echo "### collect"
{
  nvidia-smi
  echo
  python -c "import torch, flash_attn; print('torch', torch.__version__, torch.version.cuda, '| flash-attn', flash_attn.__version__)"
  echo "nanotron commit: $(cat "$HOME/nanotron_commit.txt")"
  echo "total time: $(( ($(date +%s) - START) / 60 )) min"
} > runs/env.txt 2>&1
cp "$HOME/go.log" runs/ 2>/dev/null
# snapshots are big, they stay on the pod; everything else fits in a small archive
tar czf "$HOME/results_272.tgz" --exclude='*.pickle' --exclude='ckpt' runs
ls -lh "$HOME/results_272.tgz"
# /workspace is the pod volume: it survives a Stop and JupyterLab can download from it
if [ -d /workspace ]; then
  cp "$HOME/results_272.tgz" /workspace/
  cp -r runs /workspace/runs_272
fi
echo "### done"
