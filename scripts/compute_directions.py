"""M2: concept directions for e2 entities, steering sanity check, and the bridge_necessary ablation test.

Usage: python scripts/compute_directions.py --config configs/olmo3_7b.yaml --items data/three_hop_eval_v2.parquet \
           --pool data/three_hop_pool_v2.parquet
Needs results/<model>/<version>/eligibility.parquet (scripts/compute_eligibility.py).
Writes to results/<model>/<version>/: directions_e2.safetensors, steering_check.parquet, bridge_necessary.parquet + .json

bridge_necessary (SPEC 3.3b) uses the PRIMARY setting fixed in advance: band [0.25, 0.75] of depth, last K prompt
positions. Other bands/positions are reported as a sweep only (not used to choose the definition).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file

from multihop.data import grade
from multihop.directions import concept_vectors, load_baseline_words, mean_resid_norm, steering_check
from multihop.model import ablate_direction, format_prompts, layer_at_fraction, layers_in_band, load_config, load_model, n_layers, record

PRIMARY = {"band": (0.25, 0.75), "positions": "lastK"}
SWEEP = [((0.25, 0.75), "lastK"), ((0.25, 0.75), "all"), ((0.0, 1.0), "lastK"), ((0.0, 1.0), "all"),
         ((0.25, 0.5), "all"), ((0.5, 0.75), "all")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--items", required=True)
    p.add_argument("--pool", required=True, help="pool used to draw same-type control entities")
    p.add_argument("--baseline-words", default="configs/baseline_words.txt")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--n-steer", type=int, default=8)
    p.add_argument("--steer-multiples", type=float, nargs="+", default=[0, 0.5, 1, 2, 4])
    args = p.parse_args()

    cfg = load_config(args.config)
    items = pd.read_parquet(args.items)
    version = items["dataset_version"].iloc[0]
    out_dir = Path(args.results_dir) / cfg.model_id.replace("/", "__") / version
    el = pd.read_parquet(out_dir / "eligibility.parquet")
    items = items.merge(el[["item_id", "baseline_correct"]], on="item_id", validate="one_to_one")
    rng = np.random.default_rng(cfg.seed)

    model, tok = load_model(cfg)
    L = n_layers(model)
    all_layers = list(range(L))

    # Control entity per correct item: a random different e2 of the same e2_type from the kept pool.
    pool = pd.read_parquet(args.pool)
    pool = pool[pool["keep"]].drop_duplicates("e2_qid")
    correct = items[items["baseline_correct"]].copy()
    ctrl = []
    for r in correct.itertuples():
        cands = pool[(pool["e2_type"] == r.e2_type) & (pool["e2_qid"] != r.e2_qid)]
        ctrl.append(cands.iloc[rng.integers(len(cands))])
    correct["ctrl_qid"] = [c["e2_qid"] for c in ctrl]
    correct["ctrl_name"] = [c["e2"] for c in ctrl]

    # Concept directions (cached).
    dpath = out_dir / "directions_e2.safetensors"
    vecs = load_file(dpath) if dpath.exists() else {}
    names = dict(zip(correct["e2_qid"], correct["e2"])) | dict(zip(correct["ctrl_qid"], correct["ctrl_name"]))
    probe = items[items["probe_entity"]].drop_duplicates("e2_qid").head(args.n_steer)
    names |= dict(zip(probe["e2_qid"], probe["e2"]))
    todo = [q for q in names if q not in vecs]
    if todo:
        words = load_baseline_words(args.baseline_words)
        cv = concept_vectors(model, tok, [names[q] for q in todo], words, all_layers, cfg.batch_size)
        vecs |= {q: v.to(torch.bfloat16).contiguous() for q, v in zip(todo, cv)}
        save_file(vecs, dpath)
    print(f"{len(vecs)} directions in {dpath}")

    # Steering sanity check at the middle layer.
    mid = layer_at_fraction(0.5, L)
    norm = mean_resid_norm(model, tok, mid, cfg.batch_size, load_baseline_words(args.baseline_words))
    rows = []
    for r in probe.itertuples():
        rows += steering_check(model, tok, r.e2, list(r.e2_aliases), vecs[r.e2_qid].float()[mid], mid,
                               [m * norm for m in args.steer_multiples])
    steer = pd.DataFrame(rows)
    steer["multiple"] = steer["scale"] / norm
    steer.to_parquet(out_dir / "steering_check.parquet", index=False)
    print(f"mean residual norm at layer {mid}: {norm:.1f}")
    print(steer.groupby("multiple")["mentions"].mean().rename("fraction mentioning entity").to_string())

    # Ablation test on baseline-correct items.
    prompts = format_prompts(tok, correct["question"].tolist())
    res = correct[["item_id", "e2_qid", "ctrl_qid"]].copy()
    for band, pos in SWEEP:
        layers = layers_in_band(list(band), L)
        k = cfg.k_positions if pos == "lastK" else 10**6
        for which, col in (("e2", "e2_qid"), ("ctrl", "ctrl_qid")):
            outs = []
            for i in range(len(correct)):  # one item at a time: each item has its own direction
                d = {l: vecs[correct[col].iloc[i]].float()[l] for l in layers}
                with ablate_direction(model, d, layers, k):
                    outs.append(record(model, tok, prompts[i:i + 1], k=1, layers=[0]).next_tokens[0])
            res[f"correct_{which}_{band[0]}-{band[1]}_{pos}"] = [grade(o, l) for o, l in zip(outs, correct["letter"])]
    tag = f"{PRIMARY['band'][0]}-{PRIMARY['band'][1]}_{PRIMARY['positions']}"
    res["bridge_necessary"] = ~res[f"correct_e2_{tag}"] & res[f"correct_ctrl_{tag}"]
    res.to_parquet(out_dir / "bridge_necessary.parquet", index=False)

    summary = {"n_baseline_correct": len(res), "primary": tag, "n_bridge_necessary": int(res["bridge_necessary"].sum()),
               "steer_mid_layer": mid, "steer_resid_norm": norm,
               "accuracy_after_ablation": {c.removeprefix("correct_"): float(res[c].mean()) for c in res.columns if c.startswith("correct_")}}
    (out_dir / "bridge_necessary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
