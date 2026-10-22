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

## 2026-10-22 — M2 on Olmo 3 7B: sanity checks FAIL (handoff summary)

**Bottom line:** For Olmo 3 7B, the spec's concept directions (and an entity-centred variant) do **not** pass the M2 sanity checks. Steering conveys entity *type*/attributes but never the entity name. Ablating the bridge (e2) direction hurts accuracy no more than ablating a random same-type entity's direction. The user has proposed moving to a more capable model (GLM-4.7-Flash); M3+ not started.

**What was done:**
- `.venv/bin/python scripts/compute_directions.py --config configs/olmo3_7b.yaml --items data/three_hop_eval_v2.parquet --pool data/three_hop_pool_v2.parquet`
  → `results/allenai__Olmo-3-7B-Instruct/v2/{directions_e2.safetensors (not in git, 677 entities × 32 layers), steering_check.parquet, bridge_necessary.parquet/.json}`.
- Diagnostics (scripts in `scripts/diagnostics/`, outputs in `results/allenai__Olmo-3-7B-Instruct/v2/diagnostics/`):
  `steer_centered.py` (steering, original vs entity-centred directions, layers 10/16), `ablate_centred.py` (ablation specificity + no-ablation re-run), `readout.py` (does projection identify the true e2 in question activations; printed only).

**Results (all on Olmo 3 7B, eval v2):**
1. Concept directions share a large common component: mean pairwise cosine between different entities' directions = 0.55 at layer 16 (Lindsey baseline-word subtraction). Centring on the mean over the 677 entities removes it (cosine -0.001).
2. Steering at layer 16 (and 10), 8 entities, added norm 0.5–4 × mean residual norm (18.5): **0/8 entities named at any scale**, for both direction types. Generations do shift towards the entity's type/attributes (Rowling/Le Guin → "her birth name", ExxonMobil → "world's largest company", id Software → "1992"); ≥ 2–4× breaks fluency.
3. Ablation on the 431 baseline_correct items (one item at a time): no ablation 97.7% (10 items flip between batched and single runs — near-ties); e2 ablation, band [0.25, 0.75), last 8 positions: 86.8% vs same-type control 84.5% (spec directions) / 90.0% vs 89.8% (centred). Similar across all bands/positions tried. On the 80 confident items (p_target > 0.8): 100% under every ablation. On the strict set (68): no e2-specific drop. → **Sanity check "ablating e2 hurts, random same-type does not" fails.** `bridge_necessary` as defined = 18 items, not meaningful given the control.
4. Readout: true e2 ranks at ~0.8–0.9 percentile among all 677 directions at many layers/positions, but only ~0.6–0.8 among same-type entities (best ~0.8 at layers 24–31 at the position after `<|im_end|>`), and already ~0.7 at layer 4–8, so partly lexical/e1-driven. Weak evidence that e2 is linearly visible, not conclusive (e1/e3 confounds not controlled).

**Interpretation (belief, not verified):** Most Olmo "correct" answers are likely guesses (accuracy 14.4% vs chance 8.7%), so there may be little bridge computation to disrupt; and "Tell me about X" directions may not match how X is represented as a latent intermediate. A more capable model separates these explanations.

**Needs user attention first:**
- Disk: /workspace quota ~50 GB, 48 GB used. GLM-4.7-Flash alone is 62.5 GB.
- Batched vs single-item runs flip ~2% of near-tie answers: eligibility and experiments should use identical batching (or require a margin).
- Chat-template kwargs (e.g. `enable_thinking=False` for GLM) are not yet supported by `format_prompts`.

## 2026-10-22 (new pod: 1× A100 80GB, larger /workspace) — GLM-4.7-Flash (autonomous)

**What was done:**
- `HF_HOME=/workspace/.cache/huggingface` added to `/workspace/configs/.bashrc` (new pod didn't set it; configs repo change not committed).
- `model.py`: config field `chat_template_kwargs` (passed to the chat template by `format_prompts`), used for `enable_thinking: false`. `configs/glm47_flash.yaml`.
- `compute_directions.py`: added a no-ablation single-item reference (`correct_none`); `bridge_necessary` now requires it.
- GLM smoke: 55.8 GiB GPU, 47 layers, hidden 2048, ~107 items/s recording all layers (batch 32). Template `[gMASK]<sop><|user|>…<|assistant|></think>`, no system prompt.
- `MULTIHOP_TEST_MODEL=zai-org/GLM-4.7-Flash pytest tests/test_model.py` → 8 passed, 2 FAILED, 1 skipped. Failures are real and left as is:
  ablating at the last 8 positions changes activations at *earlier* positions from layer 28 onward (rel. change 1e-4 at L28 growing to ~5% at the last layers; vs 15% at ablated positions); identical reruns are bit-identical; band starts at L12 with zero leak up to L27.
  Most likely numerical (mixture-of-experts token grouping changes bf16 rounding), not causal. Implication: small late-layer differences between conditions in MoE models are partly noise.
- `scripts/compute_eligibility.py --config configs/glm47_flash.yaml --items data/three_hop_eval_v2.parquet` (~15 min).

**Results (GLM-4.7-Flash, eval v2, n = 3,000):** accuracy **11.5%** (95% CI 10.4–12.7%) vs null chance 7.1%; hop1 37.4%, hop2 52.7%, hop3 99.5%, all hops 23.0%; accuracy on all-hops-known items 21.2%.
21% of answers are the first letter of e2 (bridge-letter shortcut). Eligible: 345 primary, 146 strict. Two-letter format 0.1%.
GLM knows more hops than Olmo (23% vs 7% all hops) but composes them worse. Did not run M2 on GLM (user: switch model if accuracy very low).

**Decision (user instruction):** switch to the most capable open model that fits the 80GB A100 in bf16. Chosen: `Qwen/Qwen3.8-27B` (dense, 55.6 GB; strongest small self-hostable model per web aggregators; dense avoids MoE noise). Alternative: `google/gemma-4-31B-it`. Caveat: screening judge (Qwen3-8B) is now the same family as an evaluated model.

## 2026-10-22 — Qwen3.8-27B (autonomous, continued)

**What was done:**
- Model choice: web search for strongest open model fitting 1× A100 80GB in bf16 (no FP8 on A100). Candidates checked on HF: `Qwen/Qwen3.8-27B` (dense, 55.6 GB), `google/gemma-4-31B-it`, `Qwen/Qwen3.6-35B-A3B` (71.9 GB MoE, tight). Chose Qwen3.8-27B. Sources: respan.ai, fireworks.ai, computingforgeeks.com, github.com/xigh/open-weight-models (aggregator numbers, different harnesses).
- Architecture: multimodal wrapper (`Qwen3_5ForConditionalGeneration`), 64 text layers, hidden 5120, 3 of every 4 layers linear attention (gated delta rule), every 4th full attention. Fast kernels (`flash-linear-attention`, `causal_conv1d`) not installed → PyTorch fallback (~34 items/s recording all layers). Not installed to avoid environment changes; could speed things up.
- Smoke (`scripts/smoke_model.py configs/qwen38_27b.yaml`): 50.1 GiB GPU; **3/3 example questions correct** (P, S, T) with p > 0.94. With thinking disabled the prompt ends `<|im_start|>assistant\n<think>\n\n</think>\n\n`, so **all K=8 read positions are template tokens** (no question tokens). Kept K=8 for now; "all positions" ablation variants cover earlier tokens.
- **Padding problem found:** left-padded batches changed Qwen3.8 activations by up to 5.9% relative (growing with amount of padding: 0 pad → 1.6%, 8 → 3.8%, 30 → 5.9%). Fix: `record`/`greedy_generate` now batch prompts by exact token length (no padding), in input order; callers pass whole lists + `batch_size`. Batched vs single is now bit-identical. This also removes the Olmo near-tie batching flips for future runs (Olmo/GLM results above were computed with padded batches).
- `get_layers` / new `hidden_size()` support multimodal wrappers. `test_record_matches_hidden_states` now uses unpadded single-prompt references (still exact equality).
- Tests: Qwen3.8 `tests/test_model.py` 10 passed, 1 skipped (fp32 test, too large); small model 11 passed.
- `scripts/compute_eligibility.py --config configs/qwen38_27b.yaml --items data/three_hop_eval_v2.parquet` → `results/Qwen__Qwen3.8-27B/v2/eligibility.*` (no-padding batching).

**Results (Qwen3.8-27B, eval v2, n = 3,000):** accuracy **23.2%** (95% CI 21.8–24.7%) vs null chance 10.4% (9.8–11.0%); TwoHopFact 26.9%, Hopping Too Late 19.5%.
hop1 28.2%, hop2 43.4%, hop3 100%; all hops 15.7%; **accuracy on all-hops-known items 48.5%**. Bridge-letter answers 12% (GLM 21%).
Eligible: **697 primary, 229 strict**; 48/300 probe-entity items correct. Two-letter format 0.5% (not useful for any model so far).
Comparison: Olmo 3 7B 14.4% (chance 8.7%), GLM-4.7-Flash 11.5% (chance 7.1%). Qwen3.8-27B is the working model from here on.
