# Experiment: ablation specificity with entity-centred directions vs Lindsey directions; plus no-ablation check.
import numpy as np, pandas as pd, torch, json
from safetensors.torch import load_file
from multihop.data import grade
from multihop.model import load_config, load_model, ablate_direction, format_prompts, record, layers_in_band
cfg = load_config("configs/olmo3_7b.yaml"); d = "results/allenai__Olmo-3-7B-Instruct/v2/"
v = load_file(d + "directions_e2.safetensors"); qids = list(v)
X = torch.stack([v[q].float() for q in qids]); C = X - X.mean(0, keepdim=True)
bn = pd.read_parquet(d + "bridge_necessary.parquet")
items = pd.read_parquet("data/three_hop_eval_v2.parquet").set_index("item_id").loc[bn.item_id].reset_index()
model, tok = load_model(cfg)
prompts = format_prompts(tok, items.question.tolist())
out = {"none": [grade(record(model, tok, prompts[i:i+1], k=1, layers=[0]).next_tokens[0], l) for i, l in enumerate(items.letter)]}
for kind, M in (("lindsey", X), ("centred", C)):
    for band, k in (((0.25, 0.75), 8), ((0.25, 0.75), 10**6), ((0.0, 1.0), 10**6)):
        layers = layers_in_band(list(band), 32)
        for which, col in (("e2", "e2_qid"), ("ctrl", "ctrl_qid")):
            res = []
            for i in range(len(items)):
                dd = {l: M[qids.index(bn[col].iloc[i]), l] for l in layers}
                with ablate_direction(model, dd, layers, k):
                    res.append(grade(record(model, tok, prompts[i:i+1], k=1, layers=[0]).next_tokens[0], items.letter.iloc[i]))
            out[f"{kind}_{band}_{'lastK' if k == 8 else 'all'}_{which}"] = res
df = pd.DataFrame(out); df.to_parquet("results/allenai__Olmo-3-7B-Instruct/v2/diagnostics/ablate_centred.parquet")
print(json.dumps(df.mean().round(3).to_dict(), indent=1))
