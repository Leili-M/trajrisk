"""Proposal section 3: prompts and output parsing for the language-model steps (llm_text.py)."""

from trajrisk import llm_text as lt


def test_parse_valid_json_with_phrases_and_extra_text():
    raw = ('Here is the JSON: {"diet": "vegetarian", "course": "main dish", "cuisine": null, '
           '"max_time": "30", "preferences": ["time", "calories"], "equal": false}')
    p = lt.parse_llm_output(raw)
    assert p["valid_json"]
    assert p["hard"] == {"diet": "vegetarian", "course": "main-dish", "cuisine": None, "max_time": 30}
    assert p["soft"] == {"time": 0.7, "calories": 0.3}


def test_parse_equal_and_unknown_values():
    p = lt.parse_llm_output('{"diet": "keto", "course": "desserts", "cuisine": "Asian", '
                            '"max_time": null, "preferences": ["health", "protein", "x"], "equal": true}')
    assert p["hard"]["diet"] is None                       # not in the vocabulary
    assert p["hard"]["cuisine"] == "asian"
    assert p["soft"] == {"health": 0.5, "protein": 0.5}


def test_broken_output_is_marked_invalid():
    p = lt.parse_llm_output("Sorry, I cannot help with that.")
    assert not p["valid_json"] and p["soft"] == {}


def test_clean_paraphrase():
    assert lt.clean_paraphrase('Rewritten request: "Any quick vegan lunch ideas?"\n\nNote: ...') == \
        "Any quick vegan lunch ideas?"


def test_prompts_list_the_vocabulary():
    msgs = lt.parse_messages("x")
    assert "main-dish" in msgs[0]["content"] and "protein" in msgs[0]["content"]
    assert msgs[-1] == {"role": "user", "content": "x"}
    # every few-shot answer is itself valid parser output
    for _, answer in lt.PARSE_EXAMPLES:
        assert lt.parse_llm_output(__import__("json").dumps(answer))["valid_json"]
