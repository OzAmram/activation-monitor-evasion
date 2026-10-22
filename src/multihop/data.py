"""Dataset loading, three-hop item construction, and letter grading.

Both sources are normalised to one schema (one row per two-hop fact e1 -r1-> e2 -r2-> e3), and a
third, non-semantic hop is appended: the first letter of e3's name.
"""
import ast
import unicodedata
from pathlib import Path

import pandas as pd

QUESTION_TEMPLATE = "What is the first letter of the name of {np}? Answer immediately with only the letter."

# TwoHopFact gives r2 only as a sentence prefix ("{} was born in the city of"); we need a noun phrase.
# Hand-written, one per r2 sentence template in the release. Unknown templates raise (no silent fallback).
TWOHOPFACT_R2_NP = {
    "{} was played by an actor named": "the actor who played {}",
    "The national anthem of {} is named": "the national anthem of {}",
    "{} was written by an author named": "the author of {}",
    "{} was born in the city of": "the city where {} was born",
    "{} was born in the country of": "the country where {} was born",
    "The capital of {} is": "the capital of {}",
    "{} is in the country of": "the country where {} is located",
    "{} was created by": "the creator of {}",
    "The name of the director of {} is": "the director of {}",
    "The father of {} is named": "the father of {}",
    "{} was founded by the person named": "the founder of {}",
    "{} has its headquarters in the city of": "the city where {} has its headquarters",
    "{} has its headquarters in the country of": "the country where {} has its headquarters",
    "The mother of {} is named": "the mother of {}",
    "{} was released in the country of": "the country of origin of {}",
    "{} is led by president": "the president of {}",
    "The spouse of {} is named": "the spouse of {}",
    "{} is listed on a stock exchange named": "the stock exchange that {} is listed on",
    "In college, {} majored in": "the subject that {} majored in at college",
    "As an undergrad, {} attended the university named": "the university that {} attended as an undergraduate",
}

COLUMNS = [
    "item_id", "source", "source_row",
    "e1", "e1_qid", "e1_type", "r1",
    "e2", "e2_qid", "e2_type", "e2_aliases", "r2",
    "e3", "e3_qid", "e3_type", "e3_aliases",
    "np", "question",
]


def first_letter(name: str) -> str | None:
    """Uppercase first character of `name` after stripping accents, or None if it is not an A-Z letter."""
    s = unicodedata.normalize("NFKD", name.strip())
    s = "".join(c for c in s if not unicodedata.combining(c))
    if not s or not ("A" <= s[0].upper() <= "Z"):
        return None
    return s[0].upper()


def grade(output: str, target_letter: str) -> bool:
    """Exact match of the first non-whitespace character of the model output (case-insensitive)."""
    s = output.lstrip()
    return bool(s) and s[0].upper() == target_letter.upper()


def _flatten_aliases(x) -> list[str]:
    if isinstance(x, str):
        return [x]
    out = []
    for y in x:
        out += _flatten_aliases(y)
    return out


def _parse_aliases(raw, label: str) -> list[str]:
    """Parse a stringified alias list/tuple; always include the canonical label first, deduplicated."""
    aliases = _flatten_aliases(ast.literal_eval(raw)) if isinstance(raw, str) and raw.strip() else []
    return list(dict.fromkeys([label] + [a for a in aliases if a.strip()]))


def load_twohopfact(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path)
    unknown = set(t["r2.template"]) - set(TWOHOPFACT_R2_NP)
    if unknown:
        raise ValueError(f"No noun-phrase template for TwoHopFact r2 templates: {unknown}")
    r1_np = t.apply(lambda r: r["mu.template"].format(r["e1.value"]), axis=1)
    np_ = [TWOHOPFACT_R2_NP[tpl].format(x) for tpl, x in zip(t["r2.template"], r1_np)]
    df = pd.DataFrame({
        "source": "twohopfact",
        "source_row": t["uid"],
        "e1": t["e1.value"], "e1_qid": t["e1.wikidata_qid"], "e1_type": t["e1.category"],
        # Relation keys use the rough category (e.g. "birthcity"), so r2 can be applied to e1 for shortcut checks.
        "r1": t["r1.rough_category"],
        "e2": t["e2.value"], "e2_qid": t["e2.wikidata_qid"], "e2_type": t["e2.rough_category"],
        "e2_aliases": [_parse_aliases(a, l) for a, l in zip(t["e2.aliases"], t["e2.value"])],
        "r2": t["r2.rough_category"],
        "e3": t["e3.value"], "e3_qid": t["e3.wikidata_qid"], "e3_type": t["e3.rough_category"],
        "e3_aliases": [_parse_aliases(a, l) for a, l in zip(t["e3.aliases"], t["e3.value"])],
        "np": np_,
    })
    return _finish(df)


def load_hoppingtoolate(path: Path) -> pd.DataFrame:
    h = pd.read_csv(path)
    df = pd.DataFrame({
        "source": "hoppingtoolate",
        "source_row": h["id"],
        "e1": h["e1_label"], "e1_qid": h["e1"], "e1_type": h["e1_type"],
        "r1": h["r1"],  # Wikidata property id, e.g. P27
        "e2": h["e2_label"], "e2_qid": h["e2"], "e2_type": h["e2_type"],
        "e2_aliases": [_parse_aliases(a, l) for a, l in zip(h["e2_aliases"], h["e2_label"])],
        "r2": h["r2"],
        "e3": h["e3_label"], "e3_qid": h["e3"], "e3_type": h["e3_type"],
        "e3_aliases": [_parse_aliases(a, l) for a, l in zip(h["e3_aliases"], h["e3_label"])],
        # source_prompt is a cloze prefix ending in " is" (e.g. "the capital of ... of Stephen Harper is").
        "np": h["source_prompt"].str.removesuffix(" is"),
    })
    if df["np"].str.endswith(" is").any():
        raise ValueError("HoppingTooLate noun phrase still ends with ' is'")
    return _finish(df)


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df["item_id"] = df["source"] + ":" + df["source_row"].astype(str)
    df["question"] = [QUESTION_TEMPLATE.format(np=x) for x in df["np"]]
    return df[COLUMNS].reset_index(drop=True)


def load_all(raw_dir: Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    return pd.concat([
        load_twohopfact(raw_dir / "TwoHopFact" / "TwoHopFact.csv"),
        load_hoppingtoolate(raw_dir / "HoppingTooLate" / "datasets" / "two_hop.csv"),
    ], ignore_index=True)
