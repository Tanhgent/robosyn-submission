#!/usr/bin/env python3
"""robosyn_train_stock.py — organizer train.py UNMODIFIED (stock AdamW+scheduler).

robosyn_train_muon.py의 쌍둥이: sys.path 세팅과 runpy 실행은 동일하고,
옵티마이저 몽키패치만 없다. 유효런 판정 = 로그에 [stock_wrap] 배너가 있고
[muon_wrap] 배너가 없어야 함.
"""
import os, sys, runpy

sys.path.insert(0, os.path.expanduser("~/memo"))
sys.path.insert(0, os.path.expanduser("~/robosyn/policy/act/scripts"))

print("[stock_wrap] organizer default optimizer/scheduler (no patch)", flush=True)

if __name__ == "__main__":
    runpy.run_path(os.path.expanduser("~/robosyn/policy/act/scripts/train.py"),
                   run_name="__main__")
