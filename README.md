# Activation monitor evasion with latent multi-hop questions

Can an LLM, instructed in-context only, suppress the representation of a concept it needs to answer a
question (the bridge entity in a latent three-hop question), while still answering correctly — and does
that evade activation monitors? See `SPEC.md` for the full design.

## Setup (RunPod)

```bash
bash /workspace/pod-setup.sh          # after each pod start (links dotfiles; bashrc sets UV_CACHE_DIR)
cd /workspace/activation_monitors
uv sync                               # creates .venv (torch 2.8.0+cu128; Blackwell GPUs need CUDA >= 12.8)
hf auth login                         # Llama 3.1 is gated
```

## Decisions (deviations and clarifications of SPEC.md)

| Date | Topic | Decision |
|---|---|---|
| 2026-10-22 | Probes | Run **both**: (a) per-entity probes (e.g. "California") for entities that appear as `e2` in several items; (b) entity-type probes (e.g. "a state"). (b) may be confounded because the prompt itself mentions same-type entities (e.g. Oregon) — interpret with care. |
| 2026-10-22 | "Answer token" position (Section 7) | Means the forward pass that **generates** the letter, i.e. the last prompt token. This is already the last of the K prompt positions, so no separate position is recorded. |
| 2026-10-22 | Instruction placement | Instruction is inserted **before** "Answer immediately with only the letter.", so all conditions end with identical tokens at the read positions. (Proposed; user did not object.) |
| 2026-10-22 | Same-type distractors | Best effort (e.g. via Wikidata type). Items whose `e2` has no well-defined distractors are **kept**; selectivity correction / decoy are marked unavailable for them. |
| 2026-10-22 | Hooking / multi-GPU | Plain HuggingFace forward hooks + `device_map="auto"` behind a small interface, so the backend can be swapped for very large targets (up to GLM 5.3). Multi-GPU untested so far (dev pod has 1 GPU). |
