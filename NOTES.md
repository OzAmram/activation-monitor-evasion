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

## 2026-10-22 — M1 datasets (part 1: item pool)

**Goal:** Load TwoHopFact + Hopping Too Late, build three-hop letter questions, apply model-independent filters.

**What was done:**
- Sources (pinned): `soheeyang/TwoHopFact` @ 59a84cd (HF dataset, 45,595 items); `edenbiran/HoppingTooLate` @ 6c4b43e (GitHub, `datasets/two_hop.csv`, 82,020 rows). Raw copies in `data/raw/` (gitignored).
- `src/multihop/data.py` (loading, question construction, `first_letter`, `grade`), `src/multihop/filtering.py` (filters as boolean columns + `keep`, `drop_reason`), `scripts/build_dataset.py`, `tests/test_data.py`.
- Command: `.venv/bin/python scripts/build_dataset.py --version v1` → `data/three_hop_pool_v1.parquet` (all rows + filter columns) and `data/three_hop_pool_v1.json` (counts).
- Tests: `.venv/bin/python -m pytest tests/test_data.py` → 8 passed (includes the Oregon/Salem shortcut-collision case and a check over all kept real questions that none names e2/e3 or their aliases).

**Results:** 127,615 rows → **28,259 kept** (TwoHopFact 16,370; Hopping Too Late 11,889). First failing filter: aliases 55,672; unique bridge/answer 32,970; collision 5,921; letter undefined 2,954; duplicate 1,302; names e2/e3 432; degenerate 105.
Kept items: 8,544 distinct e2; 533 e2 appear in ≥10 items (e.g. France 421, Stephen King 149). Letters fairly spread (top: C 9.1%, M 8.8%, J 7.1%).
Manual inspection of samples: questions read correctly; alias drops are a mix of genuinely ambiguous (Lewis Carroll / Charles Dodgson) and over-strict (titles like "Sir Jonathan Pryce").

**Open questions / known problems:** see README Decisions (alias strictness, shortcut coverage, wrong-e2-candidate check). Final evaluation subset size not decided (28k × all conditions × paraphrases is too much to record at all layers).

**Next steps:** decide evaluation subset + sampling; chance-baseline null prompts (needs same-type replacement e1); e2/e3 type phrases for instructions.
