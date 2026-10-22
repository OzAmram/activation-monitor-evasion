# NOTES

## 2026-10-22 — Environment setup (M0, partial)

**Goal:** Set up a dev environment on the new RunPod server; read SPEC.md and raise questions.

**What was done:**
- Hardware: 1x NVIDIA RTX PRO 4500 Blackwell (32 GB, compute capability 12.0 → needs CUDA ≥ 12.8 torch builds), 128 CPUs, 755 GB RAM.
  Root disk is 30 GB and ephemeral; `/workspace` is a persistent network filesystem (slow for many small files).
- Created `pyproject.toml` (uv-managed). Virtual env at `/workspace/activation_monitors/.venv` (persistent, on network FS).
  torch 2.8.0+cu128, transformers 5.20.2. Install command:
  `UV_CACHE_DIR=/workspace/.cache/uv UV_LINK_MODE=copy uv sync` (~5 min on the network FS).
- `HF_HOME=/workspace/.cache/huggingface/` already set by the pod. `HF_HUB_ENABLE_HF_TRANSFER` is set, so `hf_transfer` was added as a dependency.
- Created skeleton dirs `src/multihop`, `tests`, `scripts`, `configs`, `results` and `.gitignore`. Not yet a git repo.

**Results (verified by running):** Smoke test with ungated `Qwen/Qwen2.5-0.5B-Instruct`, `device_map="auto"`, bf16:
plain HuggingFace forward hooks on decoder layers return `[batch, seq, hidden]` and match `output_hidden_states[k+1]` exactly.
Script: `/tmp/claude-0/.../scratchpad/smoke.py` (temporary; to be turned into a real test in `tests/`).

**Open questions / known problems:**
- No HuggingFace token on the pod; Llama-3.1-8B-Instruct is gated → cannot download yet.
- Multi-GPU (`device_map="auto"` across GPUs) untested: only 1 GPU on this pod.

**Next steps:** Get answers to the spec questions, HF token, then M0 (hook shape tests on Llama) and M1 (datasets).

### Update (same day): answers to spec questions
- Recorded in README "Decisions" table: probes both per-entity and per-type; answer position = last prompt token (the pass that generates the letter);
  instruction inserted before "Answer immediately..."; keep items without good distractors.
- Target models go up to GLM 5.3 (near-frontier open weights) → backend must scale beyond `device_map="auto"` eventually.
- Added `UV_CACHE_DIR` / `UV_LINK_MODE` to `/workspace/configs/.bashrc` (that configs repo change is not committed).
- Git repo initialised, remote `git@github.com:OzAmram/activation-monitor-evasion.git` (push works via forwarded SSH agent).
