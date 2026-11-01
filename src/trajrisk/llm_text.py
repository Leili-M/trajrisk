"""Prompts and output parsing for the language-model steps of the benchmark (section 3).

Two uses of one open instruction-tuned language model:
  paraphrase:  rewrite a template request freely, keeping its meaning
  parse:       read a request and return its constraints as JSON
Only prompt construction and output parsing live here, so they can be tested
without a GPU; scripts/llm_requests.py runs the model.
"""

import json
import re

from trajrisk import request_text as rt

PARAPHRASE_SYSTEM = (
    "You rewrite recipe requests. Write the same request in different, natural words, "
    "as a real person might type it. Keep every requirement exactly: the diet, the type "
    "of dish, the cuisine and the maximum cooking time. Keep the preferences and their "
    "importance: if one preference matters more than the other, your text must still say "
    "which one matters more; if they matter equally, say so. Do not add, drop or soften any "
    "requirement, and do not add new preferences. Answer with the rewritten request only.")

PARSE_SYSTEM = (
    "You extract structured constraints from recipe requests. Answer with one JSON object "
    "and nothing else, with these keys:\n"
    '  "diet": one of ' + json.dumps(list(rt.DIET)) + " or null\n"
    '  "course": one of ' + json.dumps(list(rt.COURSE)) + " or null\n"
    '  "cuisine": one of ' + json.dumps(list(rt.CUISINE)) + " or null\n"
    '  "max_time": maximum total time in minutes as an integer, or null\n'
    '  "preferences": the preferences the request states, from ' + json.dumps(list(rt.SOFT)) +
    ", most important first\n"
    '  "equal": true if the request says the preferences are equally important, else false\n'
    "Rules: use null when a value is not stated. List only preferences that are stated; never "
    "add one. A maximum time is a hard limit, not the preference \"time\": use \"time\" only "
    "when the request also asks for quicker recipes among the suitable ones. \"health\" means "
    "healthier recipes in general; \"calories\" means fewer calories.\n")

# Few-shot examples for the parser (not taken from the benchmark scenarios)
PARSE_EXAMPLES = [
    ("Any vegan breakfast ideas that take at most 15 minutes? Mostly I want lots of protein, "
     "and after that as few ingredients as possible.",
     {"diet": "vegan", "course": "breakfast", "cuisine": None, "max_time": 15,
      "preferences": ["protein", "ingredients"], "equal": False}),
    ("I need a Mexican side dish, ideally a light one with fewer calories.",
     {"diet": None, "course": "side-dishes", "cuisine": "mexican", "max_time": None,
      "preferences": ["calories"], "equal": False}),
    ("Show me an Italian salad I can make in under an hour; healthy and quick matter the same to me.",
     {"diet": None, "course": "salads", "cuisine": "italian", "max_time": 60,
      "preferences": ["health", "time"], "equal": True}),
]


def paraphrase_messages(text):
    return [{"role": "system", "content": PARAPHRASE_SYSTEM},
            {"role": "user", "content": text}]


def parse_messages(text):
    msgs = [{"role": "system", "content": PARSE_SYSTEM}]
    for example, answer in PARSE_EXAMPLES:
        msgs += [{"role": "user", "content": example},
                 {"role": "assistant", "content": json.dumps(answer)}]
    return msgs + [{"role": "user", "content": text}]


def clean_paraphrase(raw):
    """First paragraph of the model output, without quotes or a leading label."""
    text = raw.strip().split("\n\n")[0].strip()
    text = re.sub(r"^(rewritten request|request|paraphrase)\s*:\s*", "", text, flags=re.I)
    return text.strip().strip('"').strip()


def _norm(value, bank):
    """Accept a key ('main-dish') or its phrase ('main dish'); anything else -> None."""
    if value is None:
        return None
    v = str(value).strip().lower()
    for key, phrase in bank.items():
        if v in (key.lower(), phrase.lower(), phrase.lower().replace(" ", "-")):
            return key
    return None


def parse_llm_output(raw):
    """Model output -> {'hard': {...}, 'soft': {...}, 'valid_json': bool}.

    Weights follow the same fixed scale as the rule-based parser:
    one preference 1.0; main / secondary 0.7 / 0.3; equal 0.5 / 0.5.
    """
    empty = {"hard": {"diet": None, "course": None, "cuisine": None, "max_time": None},
             "soft": {}, "valid_json": False}
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return empty
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return empty
    try:
        t = d.get("max_time")
        max_time = int(t) if t is not None and str(t).strip() not in ("", "null") else None
    except (TypeError, ValueError):
        max_time = None
    hard = {"diet": _norm(d.get("diet"), rt.DIET), "course": _norm(d.get("course"), rt.COURSE),
            "cuisine": _norm(d.get("cuisine"), rt.CUISINE), "max_time": max_time}
    prefs = [p for p in (d.get("preferences") or []) if p in rt.SOFT]
    prefs = list(dict.fromkeys(prefs))[:2]
    if len(prefs) == 1:
        soft = {prefs[0]: 1.0}
    elif len(prefs) == 2:
        soft = ({prefs[0]: 0.5, prefs[1]: 0.5} if d.get("equal") is True
                else {prefs[0]: 0.7, prefs[1]: 0.3})
    else:
        soft = {}
    return {"hard": hard, "soft": soft, "valid_json": True}
