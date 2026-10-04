# PATCHES — fixes required to run the organizer evaluation path

ACT evaluation runs on the **unmodified** organizer code. The notes below concern only the **SmolVLA** evaluation path (`policy/smolvla/`). Upstream status: Bugs 3–4 were first reported in [#53](https://github.com/EDEM-AI/RoboSynChallenge/issues/53) (fix PR [#57](https://github.com/EDEM-AI/RoboSynChallenge/pull/57)); items 1–2 were reported by us in [#76](https://github.com/EDEM-AI/RoboSynChallenge/issues/76), root-caused jointly with @liyifreddy in that thread, and addressed by the organizers in PR [#77](https://github.com/EDEM-AI/RoboSynChallenge/pull/77). The un-pinned lerobot install that underlies the version gap is [#52](https://github.com/EDEM-AI/RoboSynChallenge/issues/52).

Without item 1's workaround on affected checkouts, failures are silent (exit 0, no traceback) and none of the other errors ever surface.

---

## Item 1 — silent failures: buffered stdout discarded by simulator shutdown

**Symptom.** `eval.sh` exits 0 after model load; no traceback, no metrics file.

**Root cause (corrected in #76 — not a worker flush bug).** When the worker dies during load (e.g. from item 2), the parent raises, `eval_policy.py` closes the env, and `SimulationManager.destroy()` calls `os._exit(0)` ([#54](https://github.com/EDEM-AI/RoboSynChallenge/issues/54)) — discarding whatever the main process still has buffered on stdout, including the real traceback. `policy/smolvla/eval.sh` sets `EMBODICHAIN_SIM_EXIT_PROCESS=0` since [#56](https://github.com/EDEM-AI/RoboSynChallenge/pull/56) (commit `8eb3b5f`, Sep 28); our silent-death observations were all on checkouts cloned before that.

**Workaround (still useful).** On pre-#56 checkouts, or when invoking `scripts/eval_policy.py` directly:

```bash
export PYTHONUNBUFFERED=1
```

This drains the parent's buffer before `os._exit`, so errors surface. PR #77 additionally sets `PYTHONUNBUFFERED=1` in the worker subprocess env.

## Item 2 — tokenizer relative path (lerobot version gap)

**Symptom.** `RepositoryNotFoundError: 'tokenizer'` at load time for checkpoints trained on recent lerobot, which store `"tokenizer_name": "tokenizer"` in `policy_preprocessor.json`.

**Root cause (refined in #76).** A lerobot **version gap**, not a checkpoint defect: lerobot main (`713a409f`) resolves processor artifact paths against the checkpoint directory (`_resolve_artifact_paths` in `processor/pipeline.py`); the released `lerobot==0.6.1` — what `pip install "lerobot[smolvla]"` resolves to — does not yet. Pinning lerobot for SmolVLA (#52) covers it; PR #77 instead changes CWD to the checkpoint dir around loading.

**Workaround for released 0.6.x (what we use).** Rewrite the value to an absolute path inside the checkpoint copy:

```python
import json
p = f"{CKPT}/policy_preprocessor.json"
s = json.load(open(p))
def fix(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "tokenizer_name" and v == "tokenizer":
                o[k] = f"{CKPT}/tokenizer"
            else:
                fix(v)
    elif isinstance(o, list):
        for x in o:
            fix(x)
fix(s)
json.dump(s, open(p, "w"), indent=2)
```

(Needed for our own `svla_drawer_050000` under 0.6.1; the organizer-released checkpoints load without it.)

## Bug 3 — uint8 images reach the SmolVLA preprocessor (crash on first inference)

**Symptom.** `RuntimeError: NotImplementedError: "upsample_bilinear2d_out_frame" not implemented for 'Byte'` on the first `infer`. `smolvla_worker.py` passes observations straight through `torch.from_numpy`, so images arrive as uint8 while lerobot's SmolVLA expects float32 in [0, 1] (its training loader uses `return_uint8: false`).

**Patch** (`policy/smolvla/smolvla_worker.py`, the obs-decoding line, L106 at commit `29342ba`):

```diff
- obs = {key: torch.from_numpy(value) for key, value in obs.items()}
+ obs = {key: (torch.from_numpy(value).float().div_(255.0) if value.dtype == np.uint8
+              else torch.from_numpy(value).float() if value.dtype == np.float64
+              else torch.from_numpy(value)) for key, value in obs.items()}
```

Equivalent to the image handling in PR #57.

## Bug 4 — `eval()` return arity mismatch

**Symptom.** After the first episode: `ValueError: not enough values to unpack (expected 4, got 3)` at `scripts/eval_policy.py:802`. The ACT adapter returns four values (incl. `inference_times_s`); the SmolVLA adapter returns three.

**Patch** (`policy/smolvla/deploy_policy.py`, L361 at commit `29342ba`):

```diff
- return final_obs, info, truncated
+ return final_obs, info, truncated, []
```

The empty list keeps `inference_times.extend(...)` a no-op; as in PR #57, this means `average_inference_time_seconds` is not measured for SmolVLA (the adapter has no timing hooks, and timing `model.infer()` from the parent would include the worker round-trip).

---

## Environment prerequisite (not a code patch)

The documented single-env install cannot run the SmolVLA worker: `installation.md` specifies Python 3.11 (resolves lerobot ≤0.4.4), but `smolvla_worker.py` imports `policy_action_to_transition`, which exists only in lerobot ≥0.6.x (Python ≥3.12). We therefore run a **separate worker env** (py3.12, `lerobot[smolvla]==0.6.1`) selected via `SMOLVLA_PYTHON`, alongside the py3.11 sim env. Full recipe: `eval/RUNBOOK.md`.
