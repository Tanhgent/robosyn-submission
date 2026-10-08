# policy/ — deployment folders (organizer harness format)

- `act/` — the organizer's ACT deployment code, **verbatim** (Apache-2.0, see `LICENSE-UPSTREAM`; upstream: EDEM-AI/RoboSynChallenge). Our ACT checkpoints run on unmodified stock code — that is the point.
- `smolvla/` — organizer's SmolVLA deployment code with the two fixes from `../PATCHES.md` applied (`smolvla_worker.py`: Bug 3 uint8→float; `deploy_policy.py`: Bug 4 return arity). Environment prerequisites: `../eval/RUNBOOK.md`, Recipe B (separate worker env via `SMOLVLA_PYTHON`; if a `lerobot/` source folder ships inside this directory, move it aside as described there).

Checkpoint ↔ task mapping: `../MODELS.md`.
