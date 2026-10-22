"""Per-model eligibility columns (SPEC 3.3b) and the null-prompt chance baseline (SPEC 3.4).

Model-dependent: results are keyed by (model_id, dataset_version, item_id) and never delete items.
`bridge_necessary` needs concept directions and is computed later (directions.py).
"""
import re
import string
import unicodedata

import numpy as np
import pandas as pd
import torch

from multihop.data import QUESTION_TEMPLATE, grade, grade_fl
from multihop.model import format_prompts, greedy_generate, record

HOP_QUESTION = "What is the name of {np}? Answer immediately with only the name."
LETTER_QUESTION = "What is the first letter of the name {name}? Answer immediately with only the letter."


def letter_token_ids(tokenizer) -> dict[str, list[int]]:
    """For each letter A-Z, the single-token ids that start an answer with it ("S", " S", "s", " s")."""
    out = {}
    for L in string.ascii_uppercase:
        ids = set()
        for v in (L, " " + L, L.lower(), " " + L.lower()):
            enc = tokenizer.encode(v, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        if not ids:
            raise ValueError(f"No single-token form of letter {L}")
        out[L] = sorted(ids)
    return out


def letter_probs(last_logits: torch.Tensor, letter_ids: dict[str, list[int]]) -> np.ndarray:
    """[batch, 26] probability of starting the answer with each letter, renormalised over the 26 letters.
    Also returns the total (unnormalised) probability mass on letter tokens, as a format check."""
    logp = torch.log_softmax(last_logits.float(), dim=-1)
    per_letter = torch.stack([torch.logsumexp(logp[:, ids], dim=-1) for ids in letter_ids.values()], dim=-1)
    mass = per_letter.exp().sum(-1)
    return torch.softmax(per_letter, dim=-1).numpy(), mass.numpy()


def null_questions(items: pd.DataFrame, seed: int) -> pd.Series:
    """Replace e1 in each question by the e1 of another item with the same (source, r1, r2) relation pair
    and a different e2 (so the null prompt is well-formed but about an unrelated entity of the same type)."""
    rng = np.random.default_rng(seed)
    out = {}
    for _, g in items.groupby(["source", "r1", "r2"]):
        for r in g.itertuples():
            cands = g[(g["e2_qid"] != r.e2_qid) & (g["e1"] != r.e1)]
            if len(cands) == 0 or r.np.count(r.e1) != 1:
                out[r.Index] = None
                continue
            other = cands.iloc[rng.integers(len(cands))]
            out[r.Index] = QUESTION_TEMPLATE.format(np=r.np.replace(r.e1, other["e1"]))
    return pd.Series(out).reindex(items.index)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9 ]", " ", s)


def names_match(output: str, aliases: list[str]) -> bool:
    """True if any alias (>= 3 chars) appears as whole words in the output, ignoring case/accents/punctuation."""
    out = f" {' '.join(_norm(output).split())} "
    return any(len(a) >= 3 and f" {' '.join(_norm(a).split())} " in out for a in aliases)


def _batched(fn, prompts, batch_size):
    res = []
    for s in range(0, len(prompts), batch_size):
        res += fn(prompts[s:s + batch_size])
    return res


def letter_answers(model, tok, questions: list[str], batch_size: int):
    """Greedy first token + letter distribution at the answer position, for each question."""
    letter_ids = letter_token_ids(tok)
    next_tokens, probs, masses = [], [], []
    for s in range(0, len(questions), batch_size):
        rec = record(model, tok, format_prompts(tok, questions[s:s + batch_size]), k=1, layers=[0])
        p, m = letter_probs(rec.last_logits, letter_ids)
        next_tokens += rec.next_tokens
        probs.append(p)
        masses.append(m)
    return next_tokens, np.concatenate(probs), np.concatenate(masses)


def compute_eligibility(model, tok, items: pd.DataFrame, batch_size: int, seed: int) -> pd.DataFrame:
    """baseline_correct, hops_known (and per-hop columns), and null-prompt chance per item."""
    letters = list(string.ascii_uppercase)
    out = items[["item_id", "letter"]].copy()

    nt, probs, mass = letter_answers(model, tok, items["question"].tolist(), batch_size)
    out["baseline_output"] = nt
    out["baseline_correct"] = [grade(o, l) for o, l in zip(nt, items["letter"])]
    out["baseline_p_target"] = probs[np.arange(len(items)), [letters.index(l) for l in items["letter"]]]
    out["baseline_letter_mass"] = mass

    nq = null_questions(items, seed)
    has = nq.notna().to_numpy()
    out["null_question"] = nq.to_numpy()
    out["chance_p_target"] = np.nan
    if has.any():
        _, nprobs, _ = letter_answers(model, tok, nq[has].tolist(), batch_size)
        tgt = [letters.index(l) for l in items["letter"][has]]
        out.loc[has, "chance_p_target"] = nprobs[np.arange(has.sum()), tgt]

    gen = lambda qs: greedy_generate(model, tok, format_prompts(tok, qs), max_new_tokens=24)

    # Secondary first+last-letter format, only where it is well-defined (ok_fl).
    fl = items["ok_fl"].to_numpy()
    out["fl_output"], out["fl_correct"] = None, pd.NA
    if fl.any():
        gen_fl = lambda qs: greedy_generate(model, tok, format_prompts(tok, qs), max_new_tokens=6)
        o = _batched(gen_fl, items.loc[fl, "question_fl"].tolist(), batch_size)
        out.loc[fl, "fl_output"] = o
        out.loc[fl, "fl_correct"] = [grade_fl(x, a + b) for x, a, b in zip(o, items.loc[fl, "letter"], items.loc[fl, "letter_last"])]
    h1 = _batched(gen, [HOP_QUESTION.format(np=x) for x in items["np1"]], batch_size)
    h2 = _batched(gen, [HOP_QUESTION.format(np=x) for x in items["np2"]], batch_size)
    h3 = _batched(gen, [LETTER_QUESTION.format(name=x) for x in items["e3"]], batch_size)
    out["hop1_output"], out["hop2_output"], out["hop3_output"] = h1, h2, h3
    out["hop1_known"] = [names_match(o, a) for o, a in zip(h1, items["e2_aliases"])]
    out["hop2_known"] = [names_match(o, a) for o, a in zip(h2, items["e3_aliases"])]
    out["hop3_known"] = [grade(o, l) for o, l in zip(h3, items["letter"])]
    out["hops_known"] = out["hop1_known"] & out["hop2_known"] & out["hop3_known"]
    return out
