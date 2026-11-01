# Proposal section 3, step 4: free paraphrases and language parsing

Model: Qwen/Qwen2.5-7B-Instruct (revision a09a35458c702b33eeacc393d103063234e8bc28); 2 paraphrases per scenario (temperature 0.9, top-p 0.95); parsing with greedy decoding; 47 s in total.

| Text source | Texts | Parser | Hard constraints exact | Preference order | All correct |
|---|---|---|---|---|---|
| template | 30 | rule-based | 100.0% | 100.0% | 100.0% |
| template | 30 | language model | 100.0% | 83.3% | 83.3% |
| llm | 20 | rule-based | 90.0% | 15.0% | 10.0% |
| llm | 20 | language model | 100.0% | 45.0% | 45.0% |

Language-model parser returned valid JSON for 100.0% of texts.

## Example paraphrases

- template: Recommend a European main dish. Among suitable options, I prefer fewer calories.
  - paraphrase: Suggest a European main dish with lower calorie content.
  - paraphrase: Suggest a European main dish with lower calorie options.
- template: Recommend a vegetarian North American appetizer. Among suitable options, fewer calories and healthier recipes matter equally to me.
  - paraphrase: Looking for a vegetarian North American appetizer with a focus on both fewer calories and healthier options.
  - paraphrase: Suggest a low-calorie, healthy vegetarian appetizer from North American cuisine. Both criteria are equally important.
- template: Recommend a European side dish that takes no more than an hour. Among suitable options, I mainly prefer more protein, and secondly shorter cooking times.
  - paraphrase: Suggest a European side dish that can be prepared in under an hour. I have a preference for options with more protein, and also value dishes that cook quickly.
  - paraphrase: Suggest a European side dish that doesn't take over an hour to prepare. Ideally, focus on options with more protein, and consider quicker cooking times too.
- template: Recommend a vegetarian North American appetizer. Among suitable options, I prefer more protein.
  - paraphrase: Suggest a vegetarian North American appetizer that offers good protein content.
  - paraphrase: Suggest a vegetarian North American appetizer that has a good amount of protein.
- template: Recommend a dessert that takes no more than 15 minutes. Among suitable options, I mainly prefer shorter cooking times, and secondly fewer ingredients.
  - paraphrase: Suggest a dessert that can be made in under 15 minutes. Ideally, it should have a shorter cooking time and use fewer ingredients.
  - paraphrase: Suggest a dessert that can be made in under 15 minutes. Among options, I prefer those with shorter cooking times, and if necessary, fewer ingredients.
- template: Recommend a vegetarian North American side dish. Among suitable options, I mainly prefer more protein, and secondly fewer calories.
  - paraphrase: Suggest a vegetarian side dish from North America. I prefer options with more protein and fewer calories.
  - paraphrase: Suggest a vegetarian side dish from North America that has a good amount of protein and is lower in calories.

## Paraphrases the language-model parser did not read back correctly (first 10)

- Suggest a European main dish with lower calorie content.  ->  {'diet': None, 'course': 'main-dish', 'cuisine': 'european', 'max_time': None} {'calories': 0.7, 'ingredients': 0.3}
- Looking for a vegetarian North American appetizer with a focus on both fewer calories and healthier options.  ->  {'diet': 'vegetarian', 'course': 'appetizers', 'cuisine': 'north-american', 'max_time': None} {'calories': 0.7, 'health': 0.3}
- Suggest a vegetarian North American appetizer that offers good protein content.  ->  {'diet': 'vegetarian', 'course': 'appetizers', 'cuisine': 'north-american', 'max_time': None} {'protein': 0.7, 'ingredients': 0.3}
- Suggest a healthy European appetizer that can be prepared in under 30 minutes.  ->  {'diet': None, 'course': 'appetizers', 'cuisine': 'european', 'max_time': 30} {'health': 0.7, 'time': 0.3}
- Suggest a recipe that can be prepared in under 15 minutes. I lean towards options with fewer ingredients and prefer those that cook quickly as well.  ->  {'diet': None, 'course': None, 'cuisine': None, 'max_time': 15} {'time': 0.7, 'ingredients': 0.3}
- Suggest a vegetarian North American lunch with a focus on higher protein content.  ->  {'diet': 'vegetarian', 'course': 'lunch', 'cuisine': 'north-american', 'max_time': None} {'protein': 0.7, 'health': 0.3}
- Suggest a European main dish with lower calorie options.  ->  {'diet': None, 'course': 'main-dish', 'cuisine': 'european', 'max_time': None} {'calories': 0.7, 'ingredients': 0.3}
- Suggest a European side dish that doesn't take over an hour to prepare. Ideally, focus on options with more protein, and consider quicker cooking times too.  ->  {'diet': None, 'course': 'side-dishes', 'cuisine': 'european', 'max_time': 60} {'time': 0.7, 'protein': 0.3}
- Suggest a vegetarian North American appetizer that has a good amount of protein.  ->  {'diet': 'vegetarian', 'course': 'appetizers', 'cuisine': 'north-american', 'max_time': None} {'protein': 0.7, 'ingredients': 0.3}
- Suggest a healthy European appetizer that doesn't take longer than 30 minutes to prepare.  ->  {'diet': None, 'course': 'appetizers', 'cuisine': 'european', 'max_time': 30} {'health': 0.7, 'time': 0.3}
