"""Build the versioned three-hop item pool: load both sources, apply model-independent filters, save.

Download the raw sources first (pinned revisions):
    hf download soheeyang/TwoHopFact --repo-type dataset --revision 59a84cd883f71641a03fb6fa50da92a9603e7845 --local-dir data/raw/TwoHopFact
    git clone https://github.com/edenbiran/HoppingTooLate data/raw/HoppingTooLate && git -C data/raw/HoppingTooLate checkout 6c4b43e084a71a83df0bf2cb3a0c0f1b5d4be420

Usage: python scripts/build_dataset.py --version v1
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from multihop.data import load_all
from multihop.filtering import FILTERS, apply_filters

SOURCES = {
    "twohopfact": {"repo": "soheeyang/TwoHopFact", "revision": "59a84cd883f71641a03fb6fa50da92a9603e7845"},
    "hoppingtoolate": {"repo": "https://github.com/edenbiran/HoppingTooLate", "revision": "6c4b43e084a71a83df0bf2cb3a0c0f1b5d4be420"},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--version", required=True)
    p.add_argument("--screen", default=None, help="screening scores from scripts/screen_facts.py (adds ok_screen filter)")
    args = p.parse_args()

    screen = pd.read_parquet(args.screen) if args.screen else None
    df = apply_filters(load_all(Path(args.raw_dir)), screen=screen)
    df["dataset_version"] = args.version
    out = Path(args.out_dir) / f"three_hop_pool_{args.version}.parquet"
    df.to_parquet(out, index=False)

    summary = {
        "version": args.version,
        "sources": SOURCES,
        "n_total": len(df),
        "n_kept": int(df["keep"].sum()),
        "kept_by_source": df[df["keep"]].groupby("source").size().to_dict(),
        "first_failing_filter": df["drop_reason"].value_counts().to_dict(),
        "screen": args.screen,
        "failing_each_filter": {f: int((~df[f]).sum()) for f in FILTERS + (["ok_screen"] if screen is not None else [])},
    }
    (Path(args.out_dir) / f"three_hop_pool_{args.version}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
