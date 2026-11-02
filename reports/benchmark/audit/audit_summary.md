# Proposal section 3, step 5: human audit

Reviewed scenarios: 100 (stratified random sample; one reviewer).

| Check | Judged | Pass |
|---|---|---|
| The request is clear and unambiguous | 30 | 100.0% |
| The example recipes really satisfy every hard constraint (empty for scenarios with no feasible recipe) | 20 | 100.0% |
| The recipe data contain what is needed to judge the request | 30 | 100.0% |
| Paraphrase 1 keeps every requirement and the preference order | 30 | 80.0% |
| Paraphrase 2 keeps every requirement and the preference order | 30 | 83.3% |

Paraphrases judged: 60; meaning kept: 81.7%.
Among paraphrases that kept their meaning, the language-model parser was correct for 77.6% and the rule-based parser for 57.1%.

## Reviewer notes

- test-0028: healthy moved into the request wording; order kept
- test-0031: para2 loses the direction (does not say healthier / fewer calories); example 2 (Tabouli) is Lebanese but tagged asian
- test-0032: both paraphrases say “under an hour” instead of “no more than an hour” (boundary only)
- test-0044: para2: English part loses the order (“both”); the order appears only in an added Chinese sentence (language switch by the LLM)
- test-0075: para2 says “under an hour” (boundary only); example 1 (hummus) is Middle Eastern but tagged asian
- test-0207: para1 adds an odd comparison (“a bit more important”) but keeps the meaning
- test-0208: both paraphrases phrase it as “healthier over fewer calories”: order kept, but fewer calories reads as a rival rather than a second preference
- test-0209: para1 “under an hour” (boundary only); para2 “more protein over fewer calories” (order kept, second preference weakened)
- test-0210: 2 of 3 examples are Lebanese but tagged asian (only 10 feasible recipes); para1 “under an hour” (boundary only)
- test-0211: para1: “healthy” in the request and “healthier over fewer calories” (order kept); para2: preferences only in an added Chinese sentence (language switch)
- test-0212: para1 loses the direction (does not say fewer calories / more protein); examples are snacks tagged as lunch
- test-0222: para1 “under 30 minutes” (boundary only); example 2 (hummus) is Middle Eastern but tagged asian
- test-0223: para1: English part loses the order, order only in an added Chinese sentence; ingredient lists contain $template$ placeholders that hide some ingredient names
- test-0224: para1 “under 30 minutes” (boundary only); para2 “the amount of protein” loses the direction; example 1 is a dressing tagged as salad
- test-0229: para2: “healthy” moved into the request wording; order kept
- test-0363: both paraphrases “under 15 minutes” (boundary only)
- test-0364: para1 “also consider cooking time” loses the direction (shorter)
- test-0368: para1 “healthiness and protein content” loses the direction
- test-0372: para1 “under 30 minutes” (boundary only)
- test-0377: para1 adds a requirement (“still need decent amounts of protein”); para2 “lower calories and higher protein” loses the order
- test-0379: para1 “shorter over lower calorie” (order kept)
- test-0383: para1 switches to Chinese for the second preference
