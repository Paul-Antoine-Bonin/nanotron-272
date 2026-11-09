#!/usr/bin/env bash
# Container entrypoint for networks where SSH is blocked: the kit arrives base64 in $KIT_B64,
# results are printed to stdout and read back from the pod logs.
mkdir -p ~/nanotron-272 && cd ~/nanotron-272 && echo "$KIT_B64" | base64 -d | tar xz
bash ~/nanotron-272/check_272.sh 2>&1 | grep -vE "^\s*(Downloading|Downloaded|Building|Built|Prepared|Installed|Resolved|Uninstalled|Audited|Updating|Updated|\+|-) "
source ~/nanotron/.venv/bin/activate
cd ~/nanotron/runs
for s in tp2 tp2_norecord tp2_big tp2_big_norecord; do
  echo "##### $s"
  [ -f $s/summary_rank0.md ] || { echo "no summary"; grep -E "Error|error:" $s.log | tail -3 | cut -c1-200; continue; }
  sed -n '5,16p' $s/summary_rank0.md
  python ~/nanotron-272/deferred_frees.py $s
  python ~/nanotron-272/held_at_forward.py $s | grep -vE "^streams"
  grep -oE "tokens_per_sec: [0-9.K]+" $s.log | tail -1
done
echo "### ALL DONE"
sleep infinity
