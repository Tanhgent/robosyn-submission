# RoboSynChallenge Submission — Team Tanhgent

**NeurIPS 2026 RoboSynChallenge** · Solo participant: Il Seon Huh (Tanhgent, independent researcher) · Contact: tanhgent.ai@gmail.com

## Summary

Per-task checkpoints for all 10 tasks. The core contribution is **optimizer-level**: every ACT policy is trained with a Muon-family optimizer (`MuonAdamW`, lr 3e-4, no scheduler) injected into the *unmodified* organizer training script. Under identical data, architecture, steps, and seed, this single change moves the 10-task macro success rate from **37.5% (stock AdamW) to 50.4% (+12.9pp)** in our evaluation harness. One task (`drawer_open_place`) uses a SmolVLA policy instead of ACT, raising the lineup macro to **~55.3%**.

| Configuration | Macro (our harness, 100 ep/task) |
|---|---|
| **Submitted lineup** — ACT+Muon ×9 + SmolVLA drawer | **55.3%** |
| ACT+Muon ×10 (fallback, single policy type) | 50.4% |
| ACT + stock AdamW ×10 (ablation baseline, 20 ep) | 37.5% |

> **Submission configuration note.** Whether mixed policy types are allowed per task is pending an organizer answer ([#76](https://github.com/EDEM-AI/RoboSynChallenge/issues/76)). This repo ships both configurations; `MODELS.md` marks which checkpoint is active per task for each configuration.

## Per-task results (our harness)

All numbers: success rate over **100 episodes**, `random` layout, organizer `eval_policy.py` (patched per `PATCHES.md`), RunPod RTX 4090. The stock-AdamW ablation column is 20 episodes (±10pp noise band at n=20).

| Task | ACT + Muon | ACT + stock AdamW (20 ep) | Lineup policy | Lineup score |
|---|---|---|---|---|
| table_rearrangement | 99 | 85 | ACT+Muon | 99 |
| mixer_operating | 76 | 25 | ACT+Muon | 76 |
| click_bell | 75 | 85 | ACT+Muon | 75 |
| water_pouring | 69 | 30 | ACT+Muon | 69 |
| handle_basket | 66 | 55 | ACT+Muon | 66 |
| manipulate_pipette | 45 | 10 | ACT+Muon | 45 |
| item_assembly | 34 | 35 | ACT+Muon | 34 |
| items_handover | 32 | 35 | ACT+Muon | 32 |
| sample_loading | 7 | 0 | ACT+Muon | 7 |
| drawer_open_place | 1 | 15 | **SmolVLA (ours)** | **50** |
| **Macro** | **50.4** | **37.5** | | **55.3** |

### Honesty footnote: cross-harness calibration

Local scores and leaderboard scores come from **different harnesses and are not directly comparable**. We measured the translation empirically by running organizer-released checkpoints in our harness:

- Official ACT `click_bell`: published **37** ↔ our harness **75** (+38pp — our protocol filters physically infeasible seeds via the expert planner).
- Official SmolVLA `drawer_open_place`: published **62** ↔ our harness **60** (20 ep screen) and **49–50/100** at 100 ep (ours: 50/100; independently 49/100 by another participant, [#53](https://github.com/EDEM-AI/RoboSynChallenge/issues/53)) — i.e. the leaderboard figure is not reproducible with the public adapter.

The translation factor varies per task and model, so **every decision in this work (optimizer choice, per-task policy choice) was made from same-harness comparisons only.** Local macro values should not be read as leaderboard predictions.

## Method

**Training (ACT).** The organizer's `policy/act/scripts/train.py` is executed **unmodified** via a thin wrapper (`policy_act/robosyn_train_muon.py`) that monkey-patches `make_optimizer_and_scheduler` to return `MuonAdamW` (`policy_act/muon_lite.py`, lr 3e-4, weight_decay 1e-4, no LR scheduler). Everything else — dataset pipeline, model config, 40k steps (pipette: 20k), `--seed 42`, `--chunk-size 50 --n-action-steps 50` — is the organizer default. The ablation twin (`policy_act/robosyn_train_stock.py`) is the same wrapper with the patch removed. Chain scripts for all 10 tasks are included.

**Training (SmolVLA, drawer only).** `lerobot/smolvla_base` finetuned with the organizer SmolVLA tutorial defaults (50k steps) on `cobotmagic_Sim_drawer_open_place`. No recipe changes.

**Ablation protocol.** Pre-registered before measurement: judgment at macro level only (per-task n=20 is inside noise); |Δmacro| < 8pp would trigger a 50-ep rerun (not needed: Δ = 12.9pp); a per-task lineup change would require a 15pp+ AdamW win plus a 100-ep confirmation (no task qualified). The drawer policy swap required a pre-registered ≥45% gate at 100 ep (met: 50/100).

**Hardware.** Training: NVIDIA DGX Spark (GB10, aarch64). Evaluation: RunPod RTX 4090 (x86_64).

## Reproduction

1. **Environment** — see `eval/RUNBOOK.md` for the full pod recipe (base image, pinned EmbodiChain v0.2.4, frozen pip requirements, known-good mirrors). A prebuilt conda env tarball is also referenced there for a ~10-minute restore.
2. **Patches** — the organizer evaluation path requires four fixes for SmolVLA (stdout buffering, tokenizer path, uint8 images, `eval()` arity). All are documented with diffs in `PATCHES.md` and reported upstream ([#52](https://github.com/EDEM-AI/RoboSynChallenge/issues/52), [#53](https://github.com/EDEM-AI/RoboSynChallenge/issues/53), PR [#57](https://github.com/EDEM-AI/RoboSynChallenge/pull/57), [#76](https://github.com/EDEM-AI/RoboSynChallenge/issues/76)). ACT evaluation runs unpatched.
3. **Checkpoints** — Hugging Face links and per-task mapping in `MODELS.md` (public as of submission).
4. **Evaluate** — organizer command, e.g. ACT: `bash policy/act/eval.sh <task> random <ckpt> 0 --max_episodes 100 --headless true`; SmolVLA: see `eval/RUNBOOK.md` (worker env + exports).
5. **Train from scratch** — `policy_act/robosyn_B_chain.sh` (Muon lineup) and `policy_act/robosyn_ADAMW_chain.sh` (ablation); dataset download is embedded in the chains.

## Repository structure

```
README.md             - this file
PATCHES.md            - organizer-code fixes with diffs and issue links
MODELS.md             - checkpoint registry (HF links, per-task mapping, both configurations)
policy_act/           - Muon wrapper, muon_lite.py, stock-AdamW twin, 10-task chain scripts
policy_smolvla/       - drawer training command record + eval patches (worker dtype, arity)
eval/RUNBOOK.md       - pod environment recipe (freeze + official-stack variants), eval commands
```

## Methodology notes

All experiments in this project follow a measurement-first protocol: predictions and judgment thresholds are pre-registered in a ledger before measurement; screening results (n=20) are never promoted without confirmation (n=100); negative results are kept (the stock-AdamW lineup, six rejected improvement attempts during tuning); and cross-harness numbers are never compared directly. The ablation in this README is the headline product of that protocol.

---
*Tanhgent — optimizer-first robot learning. MEMO (Memory-Efficient Momentum Optimizer) research line.*
