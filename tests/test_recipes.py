"""Proposal section 3: parsing of recipe attributes (src/trajrisk/recipes.py)."""

import numpy as np
import pandas as pd

from trajrisk import recipes as rc


def test_parse_list_and_missing():
    assert rc.parse_list("['vegan', 'easy']") == ["vegan", "easy"]
    assert rc.parse_list(np.nan) is None
    assert rc.parse_list("not a list") is None


def test_parse_ingredient_names_from_hummus_format():
    text = "{'': [('popcorn, popped', '12  time(s) cups '), ('Butter', '0.5 time(s) cup ')]}"
    assert rc.parse_ingredient_names(text) == ["popcorn, popped", "butter"]


def test_meat_words_use_word_boundaries_and_exceptions():
    assert rc.mentions_meat(["chicken broth"]) is True
    assert rc.mentions_meat(["graham crackers"]) is False      # 'ham' inside a word
    assert rc.mentions_meat(["eggplant", "tofu"]) is False
    assert rc.mentions_meat(None) is None
    assert rc.mentions_animal_product(["peanut butter", "coconut milk"]) is False
    assert rc.mentions_animal_product(["2 eggs"]) is True


def test_missing_tags_stay_unknown():
    recipes = pd.DataFrame({"new_recipe_id": [7, 8], "title": ["a", "b"],
                            "tags": ["['vegetarian', 'main-dish']", np.nan],
                            "ingredients": ["{'': [('rice', '1')]}", np.nan],
                            "duration": [25.0, 0.0], "ingredients_sizes": [1, 2]})
    items = pd.DataFrame({"item_id": [8, 7], "item_idx": [0, 1]})
    a = rc.build_attributes(recipes, items)
    assert a["item_id"].tolist() == [8, 7]
    assert a.loc[1, "tag:vegetarian"] is True and a.loc[1, "tag:vegan"] is False
    assert a.loc[0, "tag:vegetarian"] is None                    # no tags: unknown, not False
    assert np.isnan(a.loc[0, "duration"])                        # 0 minutes: unknown
    assert a.loc[1, "duration"] == 25.0
