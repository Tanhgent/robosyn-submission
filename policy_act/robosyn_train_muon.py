#!/usr/bin/env python3
"""robosyn_train_muon.py — organizer train.py with MuonAdamW injected.

주최 스크립트(policy/act/scripts/train.py)를 무수정으로 실행하되, lerobot의
make_optimizer_and_scheduler를 우리 MuonAdamW로 바꿔치기한다 (주최가 make_dataset을
바꿔치기하는 것과 같은 수법). lr은 MUON_LR 환경변수로 (스케줄러는 None = 무스케줄).

Run (conda robosyn, ~/robosyn/policy/act 에서):
  MUON_LR=1e-3 python ~/memo/robosyn_train_muon.py --dataset-root <..> --output-dir <..> \
      --steps 10000 --seed 42 --chunk-size 50 --n-action-steps 50 --overwrite
배너 "[muon_wrap] lr=... injected"가 떠야 유효런.
"""
import os, sys, runpy

MUON_LR = float(os.environ.get("MUON_LR", "1e-3"))

sys.path.insert(0, os.path.expanduser("~/memo"))          # muon_lite
sys.path.insert(0, os.path.expanduser("~/robosyn/policy/act/scripts"))

import lerobot.optim.factory as OF
import lerobot.scripts.train as LT
from muon_lite import MuonAdamW

def make_muon(cfg, policy):
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = MuonAdamW(params, lr=MUON_LR, weight_decay=1e-4)
    print(f"[muon_wrap] lr={MUON_LR} injected | trainable groups={len(params)}", flush=True)
    return opt, None                                       # (optimizer, lr_scheduler=None)

OF.make_optimizer_and_scheduler = make_muon
LT.make_optimizer_and_scheduler = make_muon                # train.py가 이름을 직접 임포트했으므로 양쪽 패치

if __name__ == "__main__":
    runpy.run_path(os.path.expanduser("~/robosyn/policy/act/scripts/train.py"),
                   run_name="__main__")
