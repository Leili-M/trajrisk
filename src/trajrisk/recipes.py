"""Recipe attributes for the controlled benchmark (proposal section 3).

Turns the raw HUMMUS recipe table into one clean row per catalog recipe:
numeric attributes, tag flags and parsed ingredient names. Missing values stay
missing (NaN or None): the proposal asks to treat missing attributes as unknown.
"""

import ast
import re

import numpy as np
import pandas as pd

# Hard-constraint vocabulary: values come straight from the food.com tags.
DIET_TAGS = ["vegetarian", "vegan"]
COURSE_TAGS = ["main-dish", "desserts", "breakfast", "lunch", "appetizers",
               "salads", "soups-stews", "side-dishes", "brunch", "beverages"]
CUISINE_TAGS = ["north-american", "european", "asian", "mexican", "italian",
                "african", "middle-eastern", "south-american", "indian"]
TIME_TAGS = {"15-minutes-or-less": 15, "30-minutes-or-less": 30,
             "60-minutes-or-less": 60, "4-hours-or-less": 240}

# Soft-preference attributes and their raw columns. Nutrition values are used only
# as relative (percentile) preferences, never as absolute thresholds, because the
# data set does not state reliably whether they are per serving or per 100 g.
NUMERIC_COLUMNS = {"duration": "duration", "n_ingredients": "ingredients_sizes",
                   "n_steps": "direction_size", "calories": "calories [cal]",
                   "protein": "protein [g]", "fat": "totalFat [g]",
                   "sugar": "sugars [g]", "fsa_score": "fsa_score",
                   "who_score": "who_score"}

# Words used only to CHECK the diet tags against the ingredient lists.
MEAT_WORDS = ["chicken", "beef", "pork", "bacon", "ham", "sausage", "sausages",
              "turkey", "lamb", "veal", "venison", "duck", "prosciutto",
              "pepperoni", "salami", "chorizo", "steak", "meat", "shrimp",
              "prawn", "prawns", "fish", "salmon", "tuna", "cod", "tilapia",
              "crab", "lobster", "scallop", "scallops", "clam", "clams",
              "mussel", "mussels", "anchovy", "anchovies", "oyster", "oysters",
              "gelatin"]
ANIMAL_WORDS = ["milk", "butter", "cheese", "cream", "egg", "eggs", "yogurt",
                "yoghurt", "honey", "buttermilk", "ghee", "mayonnaise"]
PLANT_EXCEPTIONS = ["peanut butter", "almond butter", "apple butter", "cocoa butter",
                    "nut butter", "coconut milk", "almond milk", "soy milk", "rice milk",
                    "oat milk", "coconut cream", "cream of tartar", "vegan",
                    "egg substitute", "eggplant", "dairy-free", "non-dairy",
                    "vegetarian", "meatless", "soy", "tofu", "seitan", "veggie"]

_MEAT_RE = re.compile(r"\b(" + "|".join(MEAT_WORDS) + r")\b")
_ANIMAL_RE = re.compile(r"\b(" + "|".join(ANIMAL_WORDS) + r")\b")


def parse_list(text):
    """'["a", "b"]' -> ['a', 'b']; missing or broken values -> None."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        val = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    return [str(v) for v in val] if isinstance(val, (list, tuple)) else None


def parse_ingredient_names(text):
    """HUMMUS stores {'group': [(name, amount), ...], ...}; return the names in lower case."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        val = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    groups = val.values() if isinstance(val, dict) else [val]
    names = [str(item[0] if isinstance(item, (list, tuple)) else item).lower()
             for group in groups for item in group]
    return names or None


def _strip_exceptions(name):
    for phrase in PLANT_EXCEPTIONS:
        name = name.replace(phrase, " ")
    return name


def mentions_meat(names):
    if names is None:
        return None
    return any(_MEAT_RE.search(_strip_exceptions(n)) for n in names)


def mentions_animal_product(names):
    if names is None:
        return None
    return any(_MEAT_RE.search(_strip_exceptions(n)) or _ANIMAL_RE.search(_strip_exceptions(n))
               for n in names)


def clean_duration(x):
    """Minutes; zero, negative or longer than two days is treated as unknown."""
    return x if pd.notna(x) and 0 < x <= 2880 else np.nan


def build_attributes(recipes, items):
    """One row per catalog item, in item_idx order.

    recipes: raw pp_recipes rows (must contain new_recipe_id)
    items:   processed items table (item_id = new_recipe_id, item_idx = model index)
    """
    r = recipes.drop_duplicates("new_recipe_id").set_index("new_recipe_id")
    r = r.reindex(items["item_id"].to_numpy())
    out = pd.DataFrame({"item_idx": items["item_idx"].to_numpy(),
                        "item_id": items["item_id"].to_numpy(),
                        "title": r["title"].to_numpy()})
    for name, col in NUMERIC_COLUMNS.items():
        out[name] = pd.to_numeric(r[col], errors="coerce").to_numpy() if col in r else np.nan
    out["duration"] = out["duration"].map(clean_duration)

    tags = r["tags"].map(parse_list).to_numpy()
    out["has_tags"] = [t is not None for t in tags]
    for t in DIET_TAGS + COURSE_TAGS + CUISINE_TAGS + list(TIME_TAGS):
        # True / False when the recipe has tags, None (unknown) when it has none
        out[f"tag:{t}"] = [None if ts is None else (t in ts) for ts in tags]

    names = r["ingredients"].map(parse_ingredient_names).to_numpy()
    out["ingredients"] = [None if n is None else "; ".join(n) for n in names]
    out["mentions_meat"] = [mentions_meat(n) for n in names]
    out["mentions_animal_product"] = [mentions_animal_product(n) for n in names]
    return out
