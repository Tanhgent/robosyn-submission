# MODELS — checkpoint registry

All checkpoints live in one Hugging Face model repo: **`Beeny0814/robosyn-act-v1`** (private during development; **public as of submission**). Each checkpoint folder contains `config.json`, `model.safetensors`, `train_config.json` (SmolVLA additionally ships its processor files).

> **Loading note.** `snapshot_download` materializes files as symlinks into the HF cache. If you copy a checkpoint elsewhere before evaluating, use `cp -rL` (dereference), or the copies are dangling links.

## Submitted lineup (mixed configuration — macro 55.3 in our harness)

| Task | Policy | Checkpoint path (in `Beeny0814/robosyn-act-v1`) | Training |
|---|---|---|---|
| table_rearrangement | ACT + Muon | `table_rearrangement/` | 40k steps, seed 42 |
| mixer_operating | ACT + Muon | `mixer_operating/` | 40k, seed 42 |
| click_bell | ACT + Muon | `click_bell/` | 40k, seed 42 |
| water_pouring | ACT + Muon | `water_pouring/` | 40k, seed 42 |
| handle_basket | ACT + Muon | `handle_basket/` | 40k, seed 42 |
| manipulate_pipette | ACT + Muon | `manipulate_pipette_20k/` | **20k**, seed 42 |
| item_assembly | ACT + Muon | `item_assembly/` | 40k, seed 42 |
| items_handover | ACT + Muon | `items_handover/` | 40k, seed 42 |
| sample_loading | ACT + Muon | `sample_loading/` | 40k, seed 42 |
| drawer_open_place | **SmolVLA (ours)** | `svla_drawer_050000/` | 50k, organizer finetune defaults |

All ACT training used the organizer recipe with only the optimizer swapped (`MuonAdamW`, lr 3e-4, no scheduler); `manipulate_pipette` uses its 20k checkpoint because 20k outperformed 40k for that task in our harness (45 vs 43 at 100 ep). The SmolVLA checkpoint requires the Bug 2 tokenizer-path fix from `PATCHES.md`.

## Fallback configuration (single policy type — ACT only, macro 50.4)

Identical to the table above except:

| Task | Policy | Checkpoint path |
|---|---|---|
| drawer_open_place | ACT + Muon | `drawer_open_place/` |

This configuration applies if per-task policy mixing is not permitted (pending [#76](https://github.com/EDEM-AI/RoboSynChallenge/issues/76)).

## Ablation baseline (not part of the submission)

`adamw_<task>/` × 10 — identical recipe with the organizer's stock optimizer/scheduler (no patch), 40k steps, seed 42. These exist to make the README's optimizer ablation reproducible.

## Everything else in the HF repo

Folders not listed above (`*_ck0?0000`, `*_memo10k`, `*_mix_v1`, `*_ext0?k`, `*_c25`, `drawer_open_place_*` variants, and the 40k `manipulate_pipette/`) are tuning-history artifacts from the development campaign. They are **not** part of the submission and can be ignored.
