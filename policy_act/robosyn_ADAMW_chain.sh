#!/usr/bin/env bash
# robosyn_ADAMW_chain.sh — ablation twin of robosyn_B_chain.sh: stock optimizer, 10 tasks, 40k, ckpt 10k.
# Run: cd ~/robosyn/policy/act && nohup bash ~/memo/robosyn_ADAMW_chain.sh > ~/memo/robosyn_ADAMW.log 2>&1 &
# 완료 판정: grep -c TASK-DONE ~/memo/robosyn_ADAMW.log  → 10
set -x
cd $HOME/robosyn/policy/act

TASKS="click_bell mixer_operating table_rearrangement drawer_open_place water_pouring handle_basket manipulate_pipette item_assembly sample_loading items_handover"

for T in $TASKS; do
  DSROOT=$(python - <<PYEOF
from huggingface_hub import snapshot_download
print(snapshot_download("RoboSynChallenge/cobotmagic_Sim_${T}", repo_type="dataset", max_workers=2))
PYEOF
)
  if [ ! -d "$DSROOT" ]; then echo "TASK-FAIL ${T} (no dataset root)"; continue; fi
  python $HOME/memo/robosyn_train_stock.py \
    --dataset-root "$DSROOT" \
    --output-dir $HOME/memo/robosyn_runs/ADAMW_${T} \
    --steps 40000 --save-freq 10000 --seed 42 \
    --chunk-size 50 --n-action-steps 50 --overwrite \
    && echo "TASK-DONE ${T}" || echo "TASK-FAIL ${T}"
done
echo "ADAMW-CHAIN-DONE"
