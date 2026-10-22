"""Screen the single-hop facts of kept items with a judge model (see multihop.screening).

Usage: python scripts/screen_facts.py --pool <pool.pkl|parquet> --out data/screen_<version>.parquet [--judge Qwen/Qwen3-8B]
Output: one row per unique (noun phrase, object) fact with p_yes. Rerunning reuses existing scores.
"""
import argparse
from pathlib import Path

import pandas as pd

from multihop.model import ModelConfig, load_model
from multihop.screening import p_yes, screen_prompts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--judge", default="Qwen/Qwen3-8B")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None, help="screen only the first N facts (for quick checks)")
    args = p.parse_args()

    pool = pd.read_pickle(args.pool) if args.pool.endswith(".pkl") else pd.read_parquet(args.pool)
    k = pool[pool["keep"]]
    facts = pd.concat([k[["np1", "e2"]].set_axis(["np", "obj"], axis=1),
                       k[["np2", "e3"]].set_axis(["np", "obj"], axis=1)]).drop_duplicates().reset_index(drop=True)
    out = Path(args.out)
    if out.exists():
        done = pd.read_parquet(out)
        facts = facts.merge(done[["np", "obj"]], how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
    else:
        done = None
    if args.limit:
        facts = facts.head(args.limit)
    print(f"{len(facts)} facts to screen with {args.judge}")

    model, tok = load_model(ModelConfig(model_id=args.judge))
    facts["p_yes"] = p_yes(model, tok, screen_prompts(tok, list(zip(facts["np"], facts["obj"]))), args.batch_size)
    facts["judge"] = args.judge
    res = pd.concat([done, facts]) if done is not None else facts
    res.to_parquet(out, index=False)
    print(f"wrote {out}: {len(res)} facts, {(res.p_yes < 0.5).sum()} with p_yes < 0.5")


if __name__ == "__main__":
    main()
