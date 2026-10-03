# RUNBOOK — evaluation environment recipes

Two validated environments on RunPod (RTX 4090, 50 GB disk, TCP 22 exposed). **Recipe A** evaluates the ACT policies; **Recipe B** evaluates SmolVLA. Both start from the organizer Docker base image `dexforce/embodichain:ubuntu22.04-cuda12.8` (start command `bash -c "sleep infinity"`).

Common first step on any fresh pod:

```bash
apt-get update && apt-get install -y libusb-1.0-0 ffmpeg git   # without apt-get update, installs 404
```

> **Mirror note.** Direct `files.pythonhosted.org` access is blocked from some pods. Use `--index-url https://mirrors.aliyun.com/pypi/simple` plus `--timeout 60 --retries 3` on every long `pip install` (we observed indefinite hangs on the tuna mirror), and run long installs under `nohup` — web-terminal disconnects kill foreground jobs.

---

## Recipe A — ACT evaluation (frozen environment)

1. `conda create -n robosyn python=3.11 -y && conda activate robosyn`
2. **Fast path (recommended, ~10 min):** restore the prebuilt env tarball instead of pip:
   ```bash
   # robosyn_env_x86.tar.gz (4.8 GB) from HF dataset Beeny0814/robosyn-eval-artifacts
   cd /root/miniconda3/envs && rm -rf robosyn && tar -xzf /path/to/robosyn_env_x86.tar.gz
   ```
3. **From-scratch path:** install the frozen requirements `frozen_robosyn_pod_2026-09-22.txt` (same HF dataset; 271 pinned packages — replace the local cudnn wheel line with `nvidia-cudnn-cu12==9.5.1.17`), then:
   ```bash
   cd /workspace && git clone --depth 1 --branch v0.2.4 https://github.com/DexForce/EmbodiChain.git
   git clone --depth 1 https://github.com/EDEM-AI/RoboSynChallenge.git
   cd EmbodiChain && pip install -e . --no-deps && cd ../RoboSynChallenge && pip install -e . --no-deps
   pip install --no-deps "lerobot @ git+https://github.com/huggingface/lerobot@b883328e6c95681ca90a18b102e4ae5e1f91e2bf"
   ```
4. Verify: `python -c "import lerobot, torch, embodichain; print(lerobot.__version__, torch.__version__)"` → `0.3.3 2.7.1+cu126`
5. Evaluate (per task):
   ```bash
   cd /workspace/RoboSynChallenge/policy/act
   bash eval.sh <task> random <ckpt_dir> 0 --max_episodes 100 --headless true
   ```
   Checkpoints: download per `MODELS.md`; copy out of the HF cache with `cp -rL` (symlinks).

## Recipe B — SmolVLA evaluation (official-docs stack + separate worker env)

1. Sim env per the organizer `installation.md`, verbatim: `conda create -n robosyn python=3.11` → clone EmbodiChain, `git checkout tags/v0.2.4`, `pip install -e . --extra-index-url http://pyp.open3dv.site:2345/simple/ --trusted-host pyp.open3dv.site` → `pip install "numpy<2.0"` → clone RoboSynChallenge, `pip install -e .`
   Verify: `embodichain 0.2.4 / dexsim 0.4.3`.
2. **Separate worker env** (the documented single env cannot run the worker — see `PATCHES.md`, environment prerequisite):
   ```bash
   conda create -n svla312 python=3.12 -y
   /root/miniconda3/envs/svla312/bin/pip install "lerobot[smolvla]==0.6.1"
   # if the pod driver is below CUDA 13: reinstall torch as the cu126 build
   ```
3. Required exports and layout:
   ```bash
   export PYTHONUNBUFFERED=1                                   # PATCHES.md Bug 1
   export ROBOSYN_VENV_DIR=/root/miniconda3/envs/robosyn
   export SMOLVLA_PYTHON=/root/miniconda3/envs/svla312/bin/python
   mv policy/smolvla/lerobot policy/smolvla/lerobot_unused     # auto-detect trap: if present, the repo's main-branch lerobot source shadows the installed 0.6.1
   ```
4. Apply the two code patches from `PATCHES.md` (Bug 3: worker uint8→float; Bug 4: eval arity), and the Bug 2 tokenizer-path fix inside our checkpoint copy.
5. Evaluate:
   ```bash
   bash policy/smolvla/eval.sh drawer_open_place random <ckpt_dir> 0 \
     --pytorch_device cuda --headless true --renderer auto \
     --max_episodes 100 --eval_video_log true --smolvla_rescale_gripper true
   ```

## Validation trail

- Recipe A reproduced the ACT lineup numbers in `README.md` (100 ep/task) and ran the stock-AdamW ablation (2026-10-03).
- Recipe B reproduced the organizer-released drawer checkpoint at 12/20 (screen) and our `svla_drawer_050000` at 50/100 (2026-09-28), consistent with the independent 49/100 in issue #53.
- Known-good timing: pod setup ~40 min from scratch on a healthy connection (observed up to 2–4 h on degraded mirrors — hence the tarball fast path), ACT eval ~20 min/20 ep, SmolVLA eval ~2 min/ep.
