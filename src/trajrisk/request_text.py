"""Natural-language requests for the controlled benchmark (proposal section 3).

render():  structured scenario  ->  English request, with three sentence templates
parse():   English request      ->  structured constraints and preferences (rule-based)

The proposal asks to create structured specifications first and render text
afterwards, and to treat language understanding as a separate factor. The text
states every hard constraint exactly and the ORDER of the soft preferences
(main preference, secondary preference, or both equally); exact weights are not
expressed in words. The parser therefore recovers weights on a fixed scale:
one preference = 1.0, main / secondary = 0.7 / 0.3, equal = 0.5 / 0.5.
"""

import re

# ---------------------------------------------------------------- phrase banks
DIET = {"vegetarian": "vegetarian", "vegan": "vegan"}
COURSE = {"main-dish": "main dish", "desserts": "dessert", "breakfast": "breakfast",
          "lunch": "lunch", "appetizers": "appetizer", "salads": "salad",
          "soups-stews": "soup or stew", "side-dishes": "side dish"}
CUISINE = {"north-american": "North American", "european": "European", "asian": "Asian",
           "mexican": "Mexican", "italian": "Italian", "indian": "Indian"}
TIME = {15: "15 minutes", 30: "30 minutes", 60: "an hour"}

# Two wordings per preference; the template number picks one of them
SOFT = {"time": ["the quicker the better", "shorter cooking times"],
        "ingredients": ["as few ingredients as possible", "fewer ingredients"],
        "calories": ["as few calories as possible", "fewer calories"],
        "protein": ["as much protein as possible", "more protein"],
        "health": ["as healthy as possible", "healthier recipes"]}


def _article(phrase):
    """'an' before a vowel sound; 'European' starts with a 'y' sound, so it takes 'a'."""
    w = phrase.lower()
    return "an" if w[0] in "aeiou" and not w.startswith(("eu", "uni", "use")) else "a"


def _dish(hard):
    """'a vegetarian Asian dessert' from the diet, cuisine and course constraints."""
    words = [DIET[hard["diet"]]] if hard.get("diet") else []
    words += [CUISINE[hard["cuisine"]]] if hard.get("cuisine") else []
    words += [COURSE[hard["course"]] if hard.get("course") else "recipe"]
    phrase = " ".join(words)
    return f"{_article(phrase)} {phrase}"


def _time(hard, template):
    t = hard.get("max_time")
    if not t:
        return ""
    return [f" that takes no more than {TIME[t]}", f" ready in {TIME[t]} or less",
            f" with a total time of at most {TIME[t]}"][template]


def _soft(soft, template):
    """Order of preferences in words; weights 0.5/0.5 are 'equally'."""
    items = sorted(soft.items(), key=lambda kv: -kv[1])
    w = 0 if template == 1 else 1        # 'the quicker the better' reads best after 'If possible,'
    if len(items) == 1:
        p = SOFT[items[0][0]][w]
        return [f" Among suitable options, I prefer {p}.", f" If possible, {p}.",
                f" My priority is {p}."][template]
    (a, wa), (b, wb) = items
    pa, pb = SOFT[a][w], SOFT[b][w]
    if abs(wa - wb) < 1e-9:
        return [f" Among suitable options, {pa} and {pb} matter equally to me.",
                f" If possible, equally {pa} and {pb}.",
                f" My priorities are {pa} and {pb}, equally important."][template]
    return [f" Among suitable options, I mainly prefer {pa}, and secondly {pb}.",
            f" If possible, mainly {pa}, and secondly {pb}.",
            f" My main priority is {pa}; my second priority is {pb}."][template]


def render(scenario, template):
    """English request for template 0, 1 or 2."""
    hard, soft = scenario["hard"], scenario["soft"]
    opening = ["Recommend ", "I'm looking for ", "Can you suggest "][template]
    end = "?" if template == 2 else "."
    return f"{opening}{_dish(hard)}{_time(hard, template)}{end}{_soft(soft, template)}"


# ---------------------------------------------------------------- parser
def _find(text, bank):
    """Key whose phrase occurs in the text (longest phrase first), or None."""
    for key, phrase in sorted(bank.items(), key=lambda kv: -len(kv[1])):
        if re.search(r"\b" + re.escape(phrase.lower()) + r"\b", text):
            return key
    return None


def _soft_mentions(text):
    """(position, preference) for every preference phrase found in the text."""
    found = []
    for name, phrases in SOFT.items():
        for p in phrases:
            m = re.search(re.escape(p), text)
            if m:
                found.append((m.start(), name))
                break
    return sorted(found)


def parse(text):
    """Structured request recovered from the text: {'hard': {...}, 'soft': {...}}."""
    t = text.lower()
    hard = {"diet": _find(t, DIET), "course": _find(t, COURSE),
            "cuisine": _find(t, CUISINE), "max_time": None}
    m = re.search(r"(\d+)\s*minutes|an hour", t)
    if m:
        hard["max_time"] = int(m.group(1)) if m.group(1) else 60
    prefs = [name for _, name in _soft_mentions(t)]
    if len(prefs) == 1:
        soft = {prefs[0]: 1.0}
    elif len(prefs) >= 2:
        a, b = prefs[:2]
        soft = {a: 0.5, b: 0.5} if "equal" in t else {a: 0.7, b: 0.3}
    else:
        soft = {}
    return {"hard": hard, "soft": soft}


def soft_order(soft):
    """Preferences from most to least important; ties keep a fixed order."""
    return tuple(sorted(soft, key=lambda k: (-round(soft[k], 6), k)))


def compare(parsed, scenario):
    """Which parts of the request were recovered correctly."""
    hard_true = {k: v for k, v in scenario["hard"].items()}
    hard_ok = all(parsed["hard"].get(k) == v for k, v in hard_true.items())
    true_soft, got_soft = scenario["soft"], parsed["soft"]
    same_set = set(true_soft) == set(got_soft)
    equal_true = len(true_soft) == 2 and len(set(round(v, 6) for v in true_soft.values())) == 1
    equal_got = len(got_soft) == 2 and len(set(round(v, 6) for v in got_soft.values())) == 1
    order_ok = same_set and (equal_true == equal_got) and \
        (equal_true or soft_order(true_soft) == soft_order(got_soft))
    return {"hard_exact": hard_ok, "soft_set": same_set, "soft_order": order_ok,
            "all_correct": hard_ok and order_ok}
