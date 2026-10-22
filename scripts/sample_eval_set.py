"""Draw the fixed evaluation subset from a dataset pool (model-independent; every model runs on it).

Usage: python scripts/sample_eval_set.py --pool data/three_hop_pool_v1.parquet --out data/three_hop_eval_v1.parquet
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from multihop.data import sample_eval_set


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-total", type=int, default=3000)
    p.add_argument("--n-probe-entities", type=int, default=30)
    p.add_argument("--per-entity", type=int, default=10)
    p.add_argument("--min-distinct-e3", type=int, default=3)
    p.add_argument("--max-entities-per-type", type=int, default=15)
    p.add_argument("--max-per-relation-pair", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    pool = pd.read_parquet(args.pool)
    ev = sample_eval_set(pool, args.n_total, args.n_probe_entities, args.per_entity, args.min_distinct_e3,
                         args.max_entities_per_type, args.max_per_relation_pair, args.seed)
    out = Path(args.out)
    ev.to_parquet(out, index=False)
    probe = ev[ev["probe_entity"]]
    summary = {
        "args": vars(args),
        "n": len(ev),
        "by_source_and_probe_entity": {f"{s}|{b}": int(n) for (s, b), n in ev.groupby(["source", "probe_entity"]).size().items()},
        "n_relation_pairs": int(ev.groupby(["source", "r1", "r2"]).ngroups),
        "n_distinct_e2": int(ev["e2_qid"].nunique()),
        "probe_entities": sorted(probe["e2"].unique().tolist()),
        "letter_distribution": ev["letter"].value_counts().to_dict(),
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "letter_distribution"}, indent=2))


if __name__ == "__main__":
    main()
