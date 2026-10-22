# Experiment: entity-centred concept directions (subtract mean over all entities) vs Lindsey baseline-word directions.
import pandas as pd, torch
from safetensors.torch import load_file
from multihop.model import load_config, load_model, layer_at_fraction
from multihop.directions import steering_check, mean_resid_norm, load_baseline_words
cfg = load_config("configs/olmo3_7b.yaml")
d = "results/allenai__Olmo-3-7B-Instruct/v2/"
v = load_file(d + "directions_e2.safetensors"); qids = list(v)
X = torch.stack([v[q].float() for q in qids])          # [n, L, H]
C = X - X.mean(0, keepdim=True)                         # entity-centred
for l in (10, 16):
    xn = torch.nn.functional.normalize(C[:, l], dim=-1); n = len(qids)
    print(f"L{l} mean pairwise cosine, centred: {((xn @ xn.T).sum() - n) / (n * (n - 1)):.3f}")
items = pd.read_parquet("data/three_hop_eval_v2.parquet")
probe = items[items.probe_entity].drop_duplicates("e2_qid").head(8)
model, tok = load_model(cfg)
words = load_baseline_words("configs/baseline_words.txt")
rows = []
for l in (10, 16):
    norm = mean_resid_norm(model, tok, l, 32, words)
    for r in probe.itertuples():
        for kind, M in (("lindsey", X), ("centred", C)):
            res = steering_check(model, tok, r.e2, list(r.e2_aliases), M[qids.index(r.e2_qid), l], l, [m * norm for m in (0.5, 1, 1.5, 2, 3)])
            for x in res: x.update(kind=kind, multiple=x["scale"] / norm)
            rows += res
df = pd.DataFrame(rows); df.to_parquet("results/allenai__Olmo-3-7B-Instruct/v2/diagnostics/steer_centered.parquet")
print(df.groupby(["kind", "layer", "multiple"]).mentions.mean().unstack().round(2).to_string())
