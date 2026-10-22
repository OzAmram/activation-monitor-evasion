"""Dataset filters (model-independent, SPEC 3.3a).

Every filter adds a boolean `ok_*` column; nothing is deleted here. `keep` is the AND of all filters and
`drop_reason` names the first failing filter (in FILTERS order). Per-model eligibility columns
(SPEC 3.3b) are computed elsewhere and never delete items.
"""
import re
from collections import defaultdict

import pandas as pd

from multihop.data import QUESTION_TEMPLATE_FL, first_letter, last_letter

FILTERS = [
    "ok_letter",        # e3 name starts with an A-Z letter and not with "The "
    "ok_distinct",      # e1, e2, e3 are three different entities
    "ok_unique",        # exactly one e2 for (e1, r1), and all known e3 for (e2, r2) share one letter
    "ok_aliases",       # all aliases of e3 share its first letter
    "ok_collision",     # letter differs from e1, e2, and the shortcut r2(e1)
    "ok_wellformed",    # question names neither e2 nor e3 (nor aliases)
    "ok_dedup",         # first occurrence of the same (e1, e2, e3) fact across sources
    "ok_timeless",      # neither hop is a time-varying office (head of state, CEO, president, ...)
]

# Relations whose value changes with each term of office, so the ground truth depends on the snapshot date.
# Hopping Too Late: Wikidata properties; TwoHopFact: rough relation categories.
TIME_VARYING_RELATIONS = {"P35", "P169", "P1789", "president", "ceo"}


def _strip_the(name: str) -> str:
    return name[4:] if name.lower().startswith("the ") else name


def answer_letter(e3: str) -> str | None:
    """Target letter for hop 3. None if ill-defined (non-letter start, or a leading "The" makes it ambiguous)."""
    if e3.strip().lower().startswith("the "):
        return None
    return first_letter(e3)


def alias_letters(aliases: list[str]) -> set[str]:
    """First letters of all aliases; aliases not starting with a letter (e.g. "$") are ignored, a leading "The" is stripped."""
    return {l for a in aliases if (l := first_letter(_strip_the(a))) is not None}


def mentions(text: str, names: list[str]) -> list[str]:
    """Names occurring in `text` as whole words. Case-insensitive for names of 4+ characters;
    case-sensitive for shorter ones (abbreviations like "US" vs the word "us"). Single characters are skipped."""
    hits = []
    for n in names:
        n = n.strip()
        if len(n) < 2:
            continue
        flags = re.IGNORECASE if len(n) >= 4 else 0
        if re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", text, flags):
            hits.append(n)
    return hits


def apply_filters(df: pd.DataFrame, screen: pd.DataFrame | None = None, screen_threshold: float = 0.5) -> pd.DataFrame:
    """Apply all filters. If `screen` (columns np, obj, p_yes; from scripts/screen_facts.py) is given, also
    require both single-hop facts to be judged sensible (`ok_screen`, p_yes >= screen_threshold)."""
    df = df.copy()
    df["letter"] = [answer_letter(x) for x in df["e3"]]
    df["ok_letter"] = df["letter"].notna()

    df["ok_distinct"] = (df["e1_qid"] != df["e2_qid"]) & (df["e2_qid"] != df["e3_qid"]) & (df["e1_qid"] != df["e3_qid"])

    # Fact tables per source (relation keys are source-specific): (subject, relation) -> objects.
    e2_of = defaultdict(set)   # (source, e1, r1) -> {e2}
    obj_of = defaultdict(set)  # (source, subject, relation) -> {object label}, from both hops
    for r in df.itertuples():
        e2_of[(r.source, r.e1_qid, r.r1)].add(r.e2_qid)
        obj_of[(r.source, r.e1_qid, r.r1)].add(r.e2)
        obj_of[(r.source, r.e2_qid, r.r2)].add(r.e3)

    def unique(r):
        if len(e2_of[(r.source, r.e1_qid, r.r1)]) != 1:
            return False
        return len({answer_letter(o) for o in obj_of[(r.source, r.e2_qid, r.r2)]}) == 1

    df["ok_unique"] = [unique(r) for r in df.itertuples()]

    df["ok_aliases"] = [l is not None and alias_letters(a) == {l} for l, a in zip(df["letter"], df["e3_aliases"])]

    # Shortcut answers: r2 applied directly to e1, where the data contains that fact (e.g. capital of Oregon).
    df["shortcut"] = [sorted(obj_of.get((r.source, r.e1_qid, r.r2), set())) for r in df.itertuples()]

    def collision_free(r):
        if r.letter is None:
            return False
        bad = {first_letter(r.e1), first_letter(r.e2)} | {first_letter(s) for s in r.shortcut}
        return r.letter not in bad

    df["ok_collision"] = [collision_free(r) for r in df.itertuples()]

    df["ok_wellformed"] = [not mentions(q, list(a2) + list(a3)) for q, a2, a3 in zip(df["question"], df["e2_aliases"], df["e3_aliases"])]

    df["ok_dedup"] = ~df.duplicated(subset=["e1_qid", "e2_qid", "e3_qid"], keep="first")

    df["ok_timeless"] = ~(df["r1"].isin(TIME_VARYING_RELATIONS) | df["r2"].isin(TIME_VARYING_RELATIONS))

    # Secondary first+last-letter format: annotate only (not part of `keep`). Valid if the last letter is
    # defined and shared by all e3 aliases (aliases not ending in a letter are ignored).
    df["letter_last"] = [last_letter(x) for x in df["e3"]]
    df["ok_fl"] = [l is not None and {last_letter(a) for a in al} - {None} == {l} for l, al in zip(df["letter_last"], df["e3_aliases"])]
    df["question_fl"] = [QUESTION_TEMPLATE_FL.format(np=x) for x in df["np"]]

    filters = list(FILTERS)
    if screen is not None:
        p = dict(zip(zip(screen["np"], screen["obj"]), screen["p_yes"]))
        df["p_yes_hop1"] = [p.get(k) for k in zip(df["np1"], df["e2"])]
        df["p_yes_hop2"] = [p.get(k) for k in zip(df["np2"], df["e3"])]
        pre = df[filters].all(axis=1)
        missing = pre & (df["p_yes_hop1"].isna() | df["p_yes_hop2"].isna())
        if missing.any():
            raise ValueError(f"{missing.sum()} otherwise-kept items have no screening score; rerun scripts/screen_facts.py")
        df["ok_screen"] = (df["p_yes_hop1"] >= screen_threshold) & (df["p_yes_hop2"] >= screen_threshold)
        filters.append("ok_screen")

    df["keep"] = df[filters].all(axis=1)
    failing = ~df[filters]
    df["drop_reason"] = failing.idxmax(axis=1).where(failing.any(axis=1), None)
    return df
