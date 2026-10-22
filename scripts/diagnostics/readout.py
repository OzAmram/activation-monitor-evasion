# Diagnostic: does projection onto concept directions identify the true e2 among 677 entities, in baseline question activations?
import pandas as pd, torch
from safetensors.torch import load_file
from multihop.model import load_config, load_model, format_prompts, record
cfg = load_config("configs/olmo3_7b.yaml"); d = "results/allenai__Olmo-3-7B-Instruct/v2/"
v = load_file(d + "directions_e2.safetensors"); qids = list(v)
X = torch.stack([v[q].float() for q in qids]); C = X - X.mean(0, keepdim=True)
el = pd.read_parquet(d + "eligibility.parquet")
items = pd.read_parquet("data/three_hop_eval_v2.parquet").merge(el[["item_id", "baseline_correct"]], on="item_id")
items = items[items.baseline_correct & items.e2_qid.isin(qids)]
model, tok = load_model(cfg)
acts = torch.cat([record(model, tok, format_prompts(tok, items.question.iloc[s:s+16].tolist()), k=8).acts.float()
                  for s in range(0, len(items), 16)])                   # [n, 32, 8, H]
tgt = torch.tensor([qids.index(q) for q in items.e2_qid])
pool = pd.read_parquet("data/three_hop_pool_v2.parquet").drop_duplicates("e2_qid").set_index("e2_qid")
etype = [pool.e2_type.get(q) for q in qids]
same = torch.tensor([[etype[j] == etype[t] for j in range(len(qids))] for t in tgt.tolist()])   # [n, E]
print("same-type candidates per item: median", same.sum(1).median().item(), "min", same.sum(1).min().item())
for kind, M in (("lindsey", X), ("centred", C)):
    U = torch.nn.functional.normalize(M, dim=-1)                       # [E, 32, H]
    proj = torch.einsum("nlkh,elh->nlke", acts - acts.mean(0, keepdim=True), U)   # centre acts over items
    true = proj.gather(-1, tgt[:, None, None, None].expand(-1, 32, 8, 1))
    pct = (proj < true).float().mean(-1)                               # percentile rank of true e2, [n, 32, 8]
    w = same[:, None, None, :].float()
    pct_same = ((proj < true).float() * w).sum(-1) / (w.sum(-1) - 1)  # among same-type entities (excluding itself)
    print(kind, "SAME-TYPE percentile rank:"); print(pd.DataFrame(pct_same.mean(0)[[0,4,8,12,16,20,24,28,31]].numpy().round(2), index=[0,4,8,12,16,20,24,28,31]).to_string())
    continue
    m = pct.mean(0)
    print(f"{kind}: mean percentile rank of true e2 (0.5 = chance); rows = layers 0,4,..,28,31; cols = last 8 positions")
    print(pd.DataFrame(m[[0,4,8,12,16,20,24,28,31]].numpy().round(2), index=[0,4,8,12,16,20,24,28,31]).to_string())
