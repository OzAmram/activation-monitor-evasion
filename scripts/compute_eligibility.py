"""Per-model eligibility + chance baseline on the evaluation set.

Usage: python scripts/compute_eligibility.py --config configs/olmo3_7b.yaml --items data/three_hop_eval_v2.parquet
Writes results/<model>/<dataset_version>/eligibility.parquet (skipped if it exists; pass --force to recompute).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from multihop.eligibility import compute_eligibility
from multihop.model import load_config, load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--items", required=True)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    items = pd.read_parquet(args.items)
    if args.limit:
        items = items.head(args.limit)
    version = items["dataset_version"].iloc[0]
    out_dir = Path(args.results_dir) / cfg.model_id.replace("/", "__") / version
    out = out_dir / ("eligibility.parquet" if not args.limit else f"eligibility_limit{args.limit}.parquet")
    if out.exists() and not args.force:
        print(f"{out} exists; skipping")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_model(cfg)
    el = compute_eligibility(model, tok, items.reset_index(drop=True), cfg.batch_size, cfg.seed)
    el.insert(0, "model_id", cfg.model_id)
    el.insert(1, "dataset_version", version)
    el.to_parquet(out, index=False)

    summary = {
        "model_id": cfg.model_id, "dataset_version": version, "items": args.items, "n": len(el),
        "baseline_accuracy": float(el["baseline_correct"].mean()),
        "chance_null_prompt": float(el["chance_p_target"].mean()),
        "n_null_missing": int(el["chance_p_target"].isna().sum()),
        "mean_letter_mass": float(el["baseline_letter_mass"].mean()),
        "hop1_known": float(el["hop1_known"].mean()), "hop2_known": float(el["hop2_known"].mean()),
        "hop3_known": float(el["hop3_known"].mean()), "hops_known": float(el["hops_known"].mean()),
        "n_baseline_correct": int(el["baseline_correct"].sum()),
        "n_fl_defined": int(el["fl_correct"].notna().sum()),
        "fl_accuracy": float(el["fl_correct"].dropna().astype(bool).mean()) if el["fl_correct"].notna().any() else None,
        "single_letter_accuracy_on_fl_defined": float(el.loc[el["fl_correct"].notna(), "baseline_correct"].mean()) if el["fl_correct"].notna().any() else None,
        "n_baseline_correct_and_hops_known": int((el["baseline_correct"] & el["hops_known"]).sum()),
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
