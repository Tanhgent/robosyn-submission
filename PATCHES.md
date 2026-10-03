# PATCHES — fixes required to run the organizer evaluation path

ACT evaluation runs on the **unmodified** organizer code. The four patches below are needed only for the **SmolVLA** evaluation path (`policy/smolvla/`). All have been reported upstream: Bugs 3–4 were first reported in [#53](https://github.com/EDEM-AI/RoboSynChallenge/issues/53) (fix PR [#57](https://github.com/EDEM-AI/RoboSynChallenge/pull/57)); Bugs 1–2 are ours, reported with independent confirmation of #53 in [#76](https://github.com/EDEM-AI/RoboSynChallenge/issues/76). The un-pinned lerobot install that underlies the version mismatch is [#52](https://github.com/EDEM-AI/RoboSynChallenge/issues/52).

Apply order matters: without Bug 1's fix, the process dies silently (exit 0, no traceback) before any of the other errors can even surface.

---

## Bug 1 — worker stdout pipe buffering (silent death)

**Symptom.** `eval.sh` exits 0 after model load; no traceback, no metrics file. The parent (`deploy_policy.py`) blocks forever on `readline` because the worker's JSON replies sit in the stdout pipe buffer.

**Fix (workaround used here).** Export before launching:

```bash
export PYTHONUNBUFFERED=1
```

**Proper fix.** Flush after each JSON reply in `smolvla_worker.py`, or set `PYTHONUNBUFFERED=1` in the `worker_env` dict in `deploy_policy.py`.

## Bug 2 — tokenizer relative path (checkpoints from recent lerobot)

**Symptom.** `RepositoryNotFoundError: 'tokenizer'` at load time. Checkpoints trained on recent lerobot store `"tokenizer_name": "tokenizer"` in `policy_preprocessor.json`; lerobot 0.6.1 resolves this against the **CWD**, not the checkpoint directory.

**Fix.** Rewrite the value to an absolute path inside the checkpoint copy, e.g.:

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

(Needed for our own `svla_drawer_050000`; the organizer-released checkpoints load without it.)

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
