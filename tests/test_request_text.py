"""Proposal section 3: rendering requests as text and parsing them back (request_text.py)."""

import itertools

from trajrisk import benchmark as bm
from trajrisk import request_text as rt


def _scenario(diet="vegetarian", course="main-dish", cuisine=None, max_time=30,
              soft=None):
    return {"hard": {"diet": diet, "course": course, "cuisine": cuisine, "max_time": max_time},
            "soft": soft or {"time": 1.0}}


def test_example_from_the_proposal():
    text = rt.render(_scenario(), 0)
    assert text == ("Recommend a vegetarian main dish that takes no more than 30 minutes. "
                    "Among suitable options, I prefer shorter cooking times.")


def test_parser_recovers_every_combination_on_template_text():
    softs = [{"time": 1.0}, {"calories": 0.7, "protein": 0.3},
             {"protein": 0.4, "health": 0.6}, {"time": 0.5, "ingredients": 0.5}]
    for d, c, k, t, s, tpl in itertools.product(
            [None] + bm.DIETS, [None] + bm.COURSES, [None] + bm.CUISINES,
            [None] + bm.MAX_TIMES, softs, (0, 1, 2)):
        if not any((d, c, k, t)):
            continue
        sc = _scenario(d, c, k, t, s)
        result = rt.compare(rt.parse(rt.render(sc, tpl)), sc)
        assert result["all_correct"], (rt.render(sc, tpl), rt.parse(rt.render(sc, tpl)))


def test_parser_on_free_wording():
    p = rt.parse("Could I get an Indian vegan lunch in under 15 minutes? "
                 "Healthier recipes and fewer calories would be equally nice.")
    assert p["hard"] == {"diet": "vegan", "course": "lunch", "cuisine": "indian", "max_time": 15}
    assert p["soft"] == {"health": 0.5, "calories": 0.5}


def test_compare_flags_a_wrong_order():
    sc = _scenario(soft={"calories": 0.7, "protein": 0.3})
    swapped = {"hard": sc["hard"], "soft": {"protein": 0.7, "calories": 0.3}}
    r = rt.compare(swapped, sc)
    assert r["soft_set"] and not r["soft_order"] and not r["all_correct"]


def test_articles():
    assert rt.render(_scenario(None, "main-dish", "european", None), 0).startswith(
        "Recommend a European main dish.")
    assert rt.render(_scenario(None, "appetizers", None, None), 0).startswith(
        "Recommend an appetizer.")
    assert rt.render(_scenario(None, None, "indian", None), 0).startswith(
        "Recommend an Indian recipe.")
