"""Ablation specificity with entity-centred concept directions vs the spec's (Lindsey) directions, plus steering.
Usage: python scripts/diagnostics/ablate_centred_generic.py configs/<model>.yaml
Needs results/<model>/v2/{directions_e2.safetensors, bridge_necessary.parquet} from scripts/compute_directions.py.
Writes results/<model>/v2/diagnostics/{ablate_centred.parquet, steer_centred.parquet}.
"""
import json, sys
from pathlib import Path
import pandas as pd, torch
from safetensors.torch import load_file
from multihop.data import grade
from multihop.directions import load_baseline_words, mean_resid_norm, steering_check
from multihop.model import ablate_direction, format_prompts, layer_at_fraction, layers_in_band, load_config, load_model, n_layers, record

cfg = load_config(sys.argv[1]); d = Path("results") / cfg.model_id.replace("/", "__") / "v2"
(d / "diagnostics").mkdir(exist_ok=True)
v = load_file(d / "directions_e2.safetensors"); qids = list(v); qi = {q: i for i, q in enumerate(qids)}
X = torch.stack([v[q].float() for q in qids]); C = X - X.mean(0, keepdim=True)
bn = pd.read_parquet(d / "bridge_necessary.parquet")
bn = bn[bn.correct_none].reset_index(drop=True)            # only items correct without ablation (single runs)
items = pd.read_parquet("data/three_hop_eval_v2.parquet").set_index("item_id").loc[bn.item_id].reset_index()
model, tok = load_model(cfg); L = n_layers(model)
prompts = format_prompts(tok, items.question.tolist())
for name, M in (("lindsey", X), ("centred", C)):
    for l in (layer_at_fraction(0.5, L),):
        xn = torch.nn.functional.normalize(M[:, l], dim=-1); n = len(qids)
        print(f"{name} L{l} mean pairwise cosine: {((xn @ xn.T).sum() - n) / (n * (n - 1)):.3f}")
out = {}
for band, k in (((0.25, 0.75), cfg.k_positions), ((0.25, 0.75), 10**6)):
    layers = layers_in_band(list(band), L)
    for which, col in (("e2", "e2_qid"), ("ctrl", "ctrl_qid")):
        res = []
        for i in range(len(items)):
            with ablate_direction(model, {l: C[qi[bn[col].iloc[i]], l] for l in layers}, layers, k):
                res.append(grade(record(model, tok, prompts[i:i + 1], k=1, layers=[0]).next_tokens[0], items.letter.iloc[i]))
        out[f"centred_{band[0]}-{band[1]}_{'lastK' if k == cfg.k_positions else 'all'}_{which}"] = res
df = pd.DataFrame(out); df.insert(0, "item_id", bn.item_id); df.to_parquet(d / "diagnostics" / "ablate_centred.parquet")
print(json.dumps({c: round(df[c].mean(), 3) for c in df.columns if c != "item_id"}, indent=1))
probe = pd.read_parquet("data/three_hop_eval_v2.parquet").query("probe_entity").drop_duplicates("e2_qid").head(8)
mid = layer_at_fraction(0.5, L); norm = mean_resid_norm(model, tok, mid, cfg.batch_size, load_baseline_words("configs/baseline_words.txt"))
rows = []
for r in probe.itertuples():
    for x in steering_check(model, tok, r.e2, list(r.e2_aliases), C[qi[r.e2_qid], mid], mid, [m * norm for m in (0.5, 1, 1.5, 2, 3)]):
        x["multiple"] = x["scale"] / norm; rows.append(x)
s = pd.DataFrame(rows); s.to_parquet(d / "diagnostics" / "steer_centred.parquet")
print("centred steering, fraction naming entity:", s.groupby("multiple").mentions.mean().round(3).to_dict())
