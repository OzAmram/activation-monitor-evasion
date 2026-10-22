import pandas as pd

from multihop.eligibility import names_match, null_questions


def test_names_match():
    assert names_match("Sacramento", ["Sacramento"])
    assert names_match("The answer is São Paulo.", ["Sao Paulo"])
    assert names_match("george orwell", ["George Orwell", "Eric Blair"])
    assert not names_match("Salem", ["Sacramento"])
    assert not names_match("Parisian", ["Paris"])  # whole words only
    assert not names_match("A", ["A"])  # aliases shorter than 3 characters are ignored


def test_null_questions_swap_e1_within_relation_pair():
    items = pd.DataFrame({
        "source": ["s"] * 3, "r1": ["r1"] * 3, "r2": ["r2"] * 3,
        "e1": ["Oregon", "Texas", "Maine"], "e2_qid": ["CA", "NM", "NH"],
        "np": ["the capital of the state south of Oregon", "the capital of the state west of Texas",
               "the capital of the state west of Maine"],
    })
    nq = null_questions(items, seed=0)
    for q, e1 in zip(nq, items["e1"]):
        assert e1 not in q and any(o in q for o in items["e1"] if o != e1)
