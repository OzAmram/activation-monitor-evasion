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

## 2026-10-22 — M0 model infrastructure (background agent) + M1 evaluation subset

**What was done:**
- Model layer (`src/multihop/model.py`, `configs/olmo3_7b.yaml`, `tests/test_model.py`, `scripts/smoke_model.py`), written by a sub-agent, reviewed and committed (ec2226f).
  Re-ran myself: `.venv/bin/python -m pytest tests/` → 19 passed (Qwen 0.5B for model tests).
  Agent-reported on Olmo 3 7B (not re-run by me): 32 layers, hidden 4096, 13.6 GiB GPU, ~98 items/s recording all layers at batch 16-64, 2.0 MiB/item (8 positions, all layers, bf16).
  Olmo 3 inserts a default system prompt ("You are a helpful function-calling AI assistant...") — kept for now.
  Last 8 prompt tokens are `the letter . <|im_end|> \n <|im_start|> assistant \n` (5 template tokens).
  Olmo answered 3/3 example questions with a single letter, all wrong — each time the first letter of the hop-1 entity or a prompt word (France→F, California→C). Supports the e1/e2 letter-collision filter.
  Note: the agent raised the bf16 batched-vs-single tolerance from 2e-2 to 5e-2 after a failure at 0.0201; it first measured bf16-vs-fp32 error of a single prompt at the same size (0.021) and added a strict fp32 test (1e-4) as the correctness check.
- `sample_eval_set` in `data.py` + `scripts/sample_eval_set.py`: 3,000-item evaluation subset (user approved 3k).
  `.venv/bin/python scripts/sample_eval_set.py --pool data/three_hop_pool_v1.parquet --out data/three_hop_eval_v1.parquet` (defaults: 40 probe entities × 10 items, ≥3 distinct e3 per entity, ≤10 entities per e2 type, ≤40 random items per relation pair, seed 0).

**Results:** 3,000 items (1,300 random + ~200 probe-entity items per source), 191 relation pairs, 1,753 distinct e2.
Finding: per-entity variety is limited by the data — an e2 has only 2-6 r2 relations, so at most ~3-6 distinct answers. Without the ≥3-distinct-e3 requirement most probe entities had a single answer for all items (e2 probe confounded with e3).

**Open questions / known problems:**
- Source noise: Hopping Too Late inherits bad Wikidata facts (e.g. "Ralph Lauren Corporation founded by Latvia"). A type-frequency heuristic flags ~400/11,889 kept HTL items but mostly real facts, so no filter added.
- Time-varying relations (head of state, CEO, president) — ~140 eval items as r2, ~120 as r1. Ground truth is a 2024 snapshot. Asked user whether to drop.
- User idea: second answer format with first + last letter. Pros/cons given to user; not implemented yet.

## 2026-10-22 — Autonomous session: dataset v2, screening, Olmo eligibility (M1 done), M2 started

**Goal (user):** drop time-varying relations, screen questions for nonsense facts, then continue autonomously; download Llama in parallel.

**What was done (commands; all decisions also in README Decisions table):**
- Time-varying office relations dropped (`ok_timeless`), single-hop noun phrases `np1`/`np2` added, first+last-letter annotation added.
- Screening: `HF_HUB_CACHE=/root/hf_hub_ephemeral .venv/bin/python scripts/screen_facts.py --pool <pre-screen pool> --out data/screen_v2.parquet` (judge Qwen/Qwen3-8B, ~11 min). Judge model downloaded to the ephemeral container disk because of the /workspace quota (see problems).
- `.venv/bin/python scripts/build_dataset.py --version v2 --screen data/screen_v2.parquet` → 24,226 kept items.
- `.venv/bin/python scripts/sample_eval_set.py --pool data/three_hop_pool_v2.parquet --out data/three_hop_eval_v2.parquet` → 3,000 items (30 probe entities × 10).
- `.venv/bin/python scripts/compute_eligibility.py --config configs/olmo3_7b.yaml --items data/three_hop_eval_v2.parquet` → `results/allenai__Olmo-3-7B-Instruct/v2/eligibility.{parquet,json}`.
- Verified citations: Kowalski et al. arXiv:2608.21664 and Lindsey arXiv:2601.01828 exist with matching titles; baseline word list extracted (99 unique).
- Tests: `pytest tests/test_data.py tests/test_eligibility.py` → 12 passed.

**Results (Olmo 3 7B Instruct, eval v2, n = 3,000, greedy):**
- Three-hop accuracy 14.4% (bootstrap 95% CI 13.1–15.7%) vs null-prompt chance 8.7% (8.2–9.3%). TwoHopFact 19.6%, Hopping Too Late 9.0%.
- hop1 known 15.8%, hop2 27.6%, hop3 (spelling) 99.6%; all hops 7.0% (211 items); accuracy on all-hops-known items 32%.
- Eligible: **431 baseline_correct (primary), 68 baseline_correct & hops_known (strict)**. Only 19 of the 300 probe-entity items are baseline_correct.
- First+last-letter format: 0/1,301 — Olmo answers with a single letter, ignoring the two-letter instruction. Not useful for this model.
- 99.9% of next-token probability is on letter tokens (answers are well-formatted).
- Manually checked raw outputs: grading of hops and letters behaves as intended (errors are real model errors, e.g. Robinson Crusoe → "United States").

**Open questions / known problems:**
- **Disk quota on /workspace (~50 GB, 48 GB used)**: Llama download failed half-way (2/4 shards, partial files left in HF cache); Qwen3-8B judge put on ephemeral disk. Planned activation storage (~70–140 GB) will not fit. Needs user decision (free uv cache 11 GB / delete models / bigger volume).
- Screening precision ~40% on a 25-fact spot check.
- Very few probe-entity items are baseline_correct for Olmo (19) → per-entity probe evaluation on the eligible set will be thin for this model.
