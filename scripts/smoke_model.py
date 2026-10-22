"""Smoke checks for a model config: load stats, example greedy answers, read-position tokens, throughput.

Usage: .venv/bin/python scripts/smoke_model.py configs/olmo3_7b.yaml [--batch-sizes 16 32]
"""

import argparse
import time

import torch

from multihop.model import format_prompts, get_layers, hidden_size, greedy_generate, load_config, load_model, record, resolve_store_layers

QUESTIONS = [
    "What is the first letter of the capital of the country where the Eiffel Tower is located? "
    "Answer immediately with only the letter.",  # France -> Paris -> P
    "What is the first letter of the capital of the state directly south of Oregon? "
    "Answer immediately with only the letter.",  # California -> Sacramento -> S
    "What is the first letter of the capital of the country whose national flag is a red circle on a white "
    "background? Answer immediately with only the letter.",  # Japan -> Tokyo -> T
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    ap.add_argument("--n-items", type=int, default=128)
    args = ap.parse_args()
    cfg = load_config(args.config)

    t0 = time.time()
    model, tok = load_model(cfg)
    load_s = time.time() - t0
    n = len(get_layers(model))
    print(f"model {cfg.model_id}: load {load_s:.1f}s, GPU mem allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
          f"n_layers {n}, hidden {hidden_size(model)}, param devices {sorted({str(p.device) for p in model.parameters()})}")

    prompts = format_prompts(tok, QUESTIONS)
    print("\nFull formatted prompt 0:\n" + repr(prompts[0]))
    gens = greedy_generate(model, tok, prompts, max_new_tokens=16)
    rec = record(model, tok, prompts, k=cfg.k_positions)
    for j, q in enumerate(QUESTIONS):
        top = [(tok.decode([i]), round(lp, 3)) for i, lp in zip(rec.topk_ids[j, :5].tolist(), rec.topk_logprobs[j, :5].tolist())]
        print(f"\nQ: {q}\n  greedy (16 new tokens, raw): {gens[j]!r}\n  next token: {rec.next_tokens[j]!r}  top-5 (token, logprob): {top}"
              f"\n  last {cfg.k_positions} prompt tokens: {rec.read_tokens[j]}")

    # Throughput of record() with all-layer storage.
    layers = resolve_store_layers(cfg, n)
    items = format_prompts(tok, [QUESTIONS[i % len(QUESTIONS)] for i in range(args.n_items)])
    for bs in args.batch_sizes or [cfg.batch_size]:
        record(model, tok, items[:bs], k=cfg.k_positions, layers=layers)  # warm-up
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for s in range(0, len(items), bs):
            r = record(model, tok, items[s:s + bs], k=cfg.k_positions, layers=layers)
        torch.cuda.synchronize()
        dt = time.time() - t0
        bytes_per_item = r.acts[0].numel() * r.acts.element_size()
        print(f"\nbatch {bs}: {len(items) / dt:.1f} items/s ({len(items)} items, prompt len ~{len(tok(items[0])['input_ids'])}), "
              f"peak GPU mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
              f"acts per item {tuple(r.acts.shape[1:])} = {bytes_per_item / 2**20:.2f} MiB")


if __name__ == "__main__":
    main()
