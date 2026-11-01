# Proposal section 3, step 4: free paraphrases and language parsing

Model: Qwen/Qwen2.5-7B-Instruct (revision a09a35458c702b33eeacc393d103063234e8bc28); 2 paraphrases per scenario (temperature 0.9, top-p 0.95); parsing with greedy decoding; 276 s in total.

| Text source | Texts | Parser | Hard constraints exact | Preference order | All correct |
|---|---|---|---|---|---|
| template | 1,800 | rule-based | 100.0% | 100.0% | 100.0% |
| template | 1,800 | language model | 99.3% | 99.2% | 98.5% |
| llm | 1,200 | rule-based | 97.3% | 34.9% | 34.4% |
| llm | 1,200 | language model | 98.0% | 84.7% | 82.9% |

Language-model parser returned valid JSON for 100.0% of texts.

## Example paraphrases

- template: Recommend a European main dish. Among suitable options, I prefer fewer calories.
  - paraphrase: Suggest a European main dish, with a preference for options that have fewer calories.
  - paraphrase: Suggest a European main dish. If there are suitable options, I'd prefer one with fewer calories.
- template: Recommend a vegetarian North American appetizer. Among suitable options, fewer calories and healthier recipes matter equally to me.
  - paraphrase: Looking for a vegetarian North American appetizer. Between suitable choices, options with fewer calories and those that are healthier matter equally to me.
  - paraphrase: Suggest a vegetarian North American appetizer. For the best choice, consider options with fewer calories and overall healthiness equally.
- template: Recommend a European side dish that takes no more than an hour. Among suitable options, I mainly prefer more protein, and secondly shorter cooking times.
  - paraphrase: Suggest a European side dish that can be prepared within an hour. Preferably, it should have more protein and, if needed, cook faster.
  - paraphrase: Suggest a European side dish that doesn't take longer than an hour to prepare. Of the suitable options, I prefer those with more protein, and also consider shorter cooking times important.
- template: Recommend a vegetarian North American appetizer. Among suitable options, I prefer more protein.
  - paraphrase: Suggest a vegetarian North American appetizer. If there are multiple options, I'd prefer one with more protein.
  - paraphrase: Suggest a vegetarian North American appetizer. If there are several options, I'd prefer one with more protein.
- template: Recommend a dessert that takes no more than 15 minutes. Among suitable options, I mainly prefer shorter cooking times, and secondly fewer ingredients.
  - paraphrase: Looking for a dessert that can be made in under 15 minutes. Prefer recipes with shorter cooking times, and if there's a tie, opt for those with fewer ingredients.
  - paraphrase: Looking for a dessert that can be made in 15 minutes or less. Prefer options with shorter prep times and fewer ingredients.
- template: Recommend a vegetarian North American side dish. Among suitable options, I mainly prefer more protein, and secondly fewer calories.
  - paraphrase: Looking for a vegetarian North American side dish. Ideally, it should have more protein and fewer calories.
  - paraphrase: Suggest a vegetarian North American side dish. Among appropriate options, I prefer those with more protein and fewer calories.

## Paraphrases the language-model parser did not read back correctly (first 10)

- Looking for a dessert that can be made in under 15 minutes. Prefer recipes with shorter cooking times, and if there's a tie, opt for those with fewer ingredients.  ->  {'diet': None, 'course': 'desserts', 'cuisine': None, 'max_time': 15} {'time': 0.5, 'ingredients': 0.5}
- Suggest a vegetarian European main course. Of the appropriate choices, I prefer those with more protein over fewer calories.  ->  {'diet': 'vegetarian', 'course': 'main-dish', 'cuisine': 'european', 'max_time': None} {'protein': 1.0}
- Can you suggest a quick vegetarian European dish that doesn't exceed 15 minutes to prepare and has fewer calories?  ->  {'diet': 'vegetarian', 'course': None, 'cuisine': 'european', 'max_time': 15} {'time': 0.7, 'calories': 0.3}
- Suggest a North American salad that can be made within a short cooking time.  ->  {'diet': None, 'course': 'salads', 'cuisine': 'north-american', 'max_time': 30} {'time': 1.0}
- Looking for a quick vegetarian recipe under 15 minutes, with a preference for using fewer ingredients.  ->  {'diet': 'vegetarian', 'course': None, 'cuisine': None, 'max_time': 15} {'time': 0.7, 'ingredients': 0.3}
- Looking for a quick lunch option with fewer ingredients, if possible.  ->  {'diet': None, 'course': 'lunch', 'cuisine': None, 'max_time': 30} {'time': 0.7, 'ingredients': 0.3}
- Looking for a dessert option that's both healthier and lower in calories.  ->  {'diet': None, 'course': 'desserts', 'cuisine': None, 'max_time': None} {'health': 0.5, 'calories': 0.5}
- Looking for a Mexican dish that can be made in under an hour. I prioritize lower calorie options, but also care about having good protein content.  ->  {'diet': None, 'course': 'main-dish', 'cuisine': 'mexican', 'max_time': 60} {'calories': 0.7, 'protein': 0.3}
- Looking for a salad that doesn't take longer than 15 minutes to prepare. Of the options, I'd prefer higher protein content over lower calorie count.  ->  {'diet': None, 'course': 'salads', 'cuisine': None, 'max_time': 15} {'protein': 1.0}
- Suggest a salad that can be made in under 30 minutes, with a preference for using fewer ingredients.  ->  {'diet': None, 'course': 'salads', 'cuisine': None, 'max_time': 30} {'time': 0.7, 'ingredients': 0.3}
