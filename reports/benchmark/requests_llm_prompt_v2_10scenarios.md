# Proposal section 3, step 4: free paraphrases and language parsing

Model: Qwen/Qwen2.5-7B-Instruct (revision a09a35458c702b33eeacc393d103063234e8bc28); 2 paraphrases per scenario (temperature 0.9, top-p 0.95); parsing with greedy decoding; 25 s in total.

| Text source | Texts | Parser | Hard constraints exact | Preference order | All correct |
|---|---|---|---|---|---|
| template | 30 | rule-based | 100.0% | 100.0% | 100.0% |
| template | 30 | language model | 100.0% | 100.0% | 100.0% |
| llm | 20 | rule-based | 90.0% | 50.0% | 40.0% |
| llm | 20 | language model | 100.0% | 80.0% | 80.0% |

Language-model parser returned valid JSON for 100.0% of texts.

## Example paraphrases

- template: Recommend a European main dish. Among suitable options, I prefer fewer calories.
  - paraphrase: Suggest a European main dish, with a preference for options that are lower in calories.
  - paraphrase: Suggest a European main dish, prioritizing options with fewer calories.
- template: Recommend a vegetarian North American appetizer. Among suitable options, fewer calories and healthier recipes matter equally to me.
  - paraphrase: Looking for a vegetarian North American appetizer. Between options, those with fewer calories and that are healthier matter equally to me.
  - paraphrase: Suggest a vegetarian North American appetizer. For options that fit, fewer calories and healthier recipes are equally important to me.
- template: Recommend a European side dish that takes no more than an hour. Among suitable options, I mainly prefer more protein, and secondly shorter cooking times.
  - paraphrase: Suggest a European side dish that can be prepared within an hour. If there are suitable options, I'd prefer those with more protein, and secondarily, shorter cooking times are better.
  - paraphrase: Suggest a European side dish that doesn't take longer than an hour to make. Of the appropriate choices, I prefer those with more protein, and secondarily, I care about shorter cooking times.
- template: Recommend a vegetarian North American appetizer. Among suitable options, I prefer more protein.
  - paraphrase: Suggest a vegetarian North American appetizer. If there are multiple options, I'd prefer one with more protein.
  - paraphrase: Suggest a vegetarian North American appetizer. If there are options, I prefer those with more protein.
- template: Recommend a dessert that takes no more than 15 minutes. Among suitable options, I mainly prefer shorter cooking times, and secondly fewer ingredients.
  - paraphrase: Looking for a dessert that can be made in under 15 minutes. Between options, I'd prefer those with shorter cooking times, and if necessary, fewer ingredients.
  - paraphrase: Looking for a dessert that can be made in under 15 minutes. Among options that fit this timeframe, I prefer those with fewer ingredients, but also want to consider shorter cooking times equally.
- template: Recommend a vegetarian North American side dish. Among suitable options, I mainly prefer more protein, and secondly fewer calories.
  - paraphrase: Looking for a vegetarian side dish from North America. Prefer options with more protein over fewer calories.
  - paraphrase: Suggest a vegetarian side dish from North America. Of the suitable options, I prefer those with more protein over fewer calories.

## Paraphrases the language-model parser did not read back correctly (first 10)

- Looking for a vegetarian side dish from North America. Prefer options with more protein over fewer calories.  ->  {'diet': 'vegetarian', 'course': 'side-dishes', 'cuisine': 'north-american', 'max_time': None} {'protein': 1.0}
- Looking for a dessert that can be made in under 15 minutes. Among options that fit this timeframe, I prefer those with fewer ingredients, but also want to consider shorter cooking times equally.  ->  {'diet': None, 'course': 'desserts', 'cuisine': None, 'max_time': 15} {'ingredients': 0.5, 'time': 0.5}
- Suggest a vegetarian side dish from North America. Of the suitable options, I prefer those with more protein over fewer calories.  ->  {'diet': 'vegetarian', 'course': 'side-dishes', 'cuisine': 'north-american', 'max_time': None} {'protein': 1.0}
- Can you suggest a vegetarian lunch that doesn't take longer than an hour to prepare? For options that fit this criteria, I prefer those with more protein and about the same cooking time.  ->  {'diet': 'vegetarian', 'course': 'lunch', 'cuisine': None, 'max_time': 60} {'protein': 0.7, 'time': 0.3}
