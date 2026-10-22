from pathlib import Path

import pandas as pd
import pytest

from multihop.data import COLUMNS, QUESTION_TEMPLATE, first_letter, grade, load_all
from multihop.filtering import answer_letter, apply_filters, mentions

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def _item(item_id, e1, r1, e2, r2, e3, np_, e2_aliases=(), e3_aliases=(), qids=None):
    q1, q2, q3 = qids or (e1, e2, e3)
    return dict(
        item_id=item_id, source="test", source_row=item_id,
        e1=e1, e1_qid=q1, e1_type="t", r1=r1,
        e2=e2, e2_qid=q2, e2_type="t", e2_aliases=[e2, *e2_aliases], r2=r2,
        e3=e3, e3_qid=q3, e3_type="t", e3_aliases=[e3, *e3_aliases],
        np=np_, question=QUESTION_TEMPLATE.format(np=np_),
    )


def test_grade():
    assert grade("S", "S")
    assert grade("  s\n", "S")
    assert grade("Sacramento", "S")  # first non-whitespace character only
    assert not grade("The answer is S", "S")
    assert not grade("", "S")


def test_first_letter_and_answer_letter():
    assert first_letter("Édouard") == "E"
    assert first_letter("sacramento") == "S"
    assert first_letter("$") is None
    assert first_letter("1984") is None
    assert answer_letter("The Hague") is None  # ambiguous: T or H
    assert answer_letter("Sacramento") == "S"


def test_oregon_rejected_by_shortcut_collision():
    # Oregon -> California -> Sacramento (S); shortcut capital(Oregon) = Salem (S) must be rejected.
    df = pd.DataFrame([
        _item("oregon", "Oregon", "south_of", "California", "capital", "Sacramento",
              "the capital of the state directly south of Oregon"),
        _item("fact", "Willamette River", "flows_in", "Oregon", "capital", "Salem",
              "the capital of the state the Willamette River flows in"),
        _item("nevada", "Nevada", "west_of", "California", "capital", "Sacramento",
              "the capital of the state directly west of Nevada"),
    ])
    out = apply_filters(df).set_index("item_id")
    assert out.loc["oregon", "shortcut"] == ["Salem"]
    assert not out.loc["oregon", "ok_collision"]
    assert out.loc["oregon", "drop_reason"] == "ok_collision"
    # Same chain without a colliding shortcut passes the collision filter (dedup on (e1, e2, e3) differs).
    assert out.loc["nevada", "ok_collision"]


def test_collision_with_e1_and_e2():
    df = pd.DataFrame([
        _item("a", "Spain", "r1", "Canada", "capital", "Ottawa", "x of Spain"),   # passes
        _item("b", "Oman", "r1", "Canada", "capital", "Ottawa", "x of Oman"),     # e1 collides (O)
        _item("c", "Spain", "r1", "Ohio", "capital", "Olympia", "y of Spain"),     # e2 collides (O)
    ])
    out = apply_filters(df).set_index("item_id")
    assert out.loc["a", "ok_collision"]
    assert not out.loc["b", "ok_collision"]
    assert not out.loc["c", "ok_collision"]


def test_alias_consistency():
    df = pd.DataFrame([
        _item("a", "X", "r1", "Y", "r2", "Louisville", "a", e3_aliases=["Derby City"]),
        _item("b", "X2", "r1", "Y2", "r2", "Louisville", "b", e3_aliases=["Louisville, KY", "$"], qids=("X2", "Y2", "L2")),
    ])
    out = apply_filters(df).set_index("item_id")
    assert not out.loc["a", "ok_aliases"]
    assert out.loc["b", "ok_aliases"]


def test_ambiguous_bridge_rejected():
    df = pd.DataFrame([
        _item("a", "Person", "citizenship", "Canada", "capital", "Ottawa", "a"),
        _item("b", "Person", "citizenship", "France", "capital", "Paris", "b"),
    ])
    out = apply_filters(df)
    assert not out["ok_unique"].any()


def test_mentions():
    assert mentions("the capital of California", ["California"])
    assert mentions("the capital of california", ["California"])
    assert not mentions("the capital of Californian wine", ["California"])
    assert not mentions("tell us the answer", ["US"])  # short aliases are case-sensitive
    assert mentions("born in the US", ["US"])


@pytest.mark.skipif(not (RAW / "TwoHopFact").exists(), reason="raw data not downloaded")
def test_real_data_kept_questions_never_name_bridge_or_answer():
    df = apply_filters(load_all(RAW))
    assert list(df.columns[: len(COLUMNS)]) == COLUMNS
    kept = df[df["keep"]]
    assert len(kept) >= 500
    for r in kept.itertuples():
        assert not mentions(r.question, r.e2_aliases + r.e3_aliases), r.item_id
        assert r.question.endswith("Answer immediately with only the letter.")
        assert " is?" not in r.question


@pytest.mark.skipif(not (RAW / "TwoHopFact").exists(), reason="raw data not downloaded")
def test_sample_eval_set_properties():
    from multihop.data import sample_eval_set
    pool = apply_filters(load_all(RAW))
    ev = sample_eval_set(pool, 3000, 40, 10, 3, 10, 40, seed=0)
    assert len(ev) == 3000 and ev["item_id"].is_unique and ev["keep"].all()
    probe = ev[ev["probe_entity"]]
    assert probe.groupby("e2_qid").size().eq(10).all() and probe["e2_qid"].nunique() == 40
    assert probe.groupby("e2_qid")["e3_qid"].nunique().min() >= 3
    assert ev[~ev["probe_entity"]].groupby(["source", "r1", "r2"]).size().max() <= 40
    # Deterministic given the seed.
    assert ev["item_id"].tolist() == sample_eval_set(pool, 3000, 40, 10, 3, 10, 40, seed=0)["item_id"].tolist()
