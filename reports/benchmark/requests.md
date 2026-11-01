# Proposal section 3, step 3: natural-language requests

Scenarios: 600; texts: 1,800 (three templates per scenario).

Parser accuracy on template text (consistency check, should be 100%):

| Template | Hard constraints exact | Preference set | Preference order | All correct |
|---|---|---|---|---|
| 0 | 100.0% | 100.0% | 100.0% | 100.0% |
| 1 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2 | 100.0% | 100.0% | 100.0% | 100.0% |

## Examples

**valid-0000** hard {'diet': None, 'course': 'main-dish', 'cuisine': 'european', 'max_time': None}, soft {'calories': 1.0}
- Recommend a European main dish. Among suitable options, I prefer fewer calories.
- I'm looking for a European main dish. If possible, as few calories as possible.
- Can you suggest a European main dish? My priority is fewer calories.

**valid-0001** hard {'diet': 'vegetarian', 'course': 'appetizers', 'cuisine': 'north-american', 'max_time': None}, soft {'calories': 0.5, 'health': 0.5}
- Recommend a vegetarian North American appetizer. Among suitable options, fewer calories and healthier recipes matter equally to me.
- I'm looking for a vegetarian North American appetizer. If possible, equally as few calories as possible and as healthy as possible.
- Can you suggest a vegetarian North American appetizer? My priorities are fewer calories and healthier recipes, equally important.

**valid-0002** hard {'diet': None, 'course': 'side-dishes', 'cuisine': 'european', 'max_time': 60}, soft {'protein': 0.7, 'time': 0.3}
- Recommend a European side dish that takes no more than an hour. Among suitable options, I mainly prefer more protein, and secondly shorter cooking times.
- I'm looking for a European side dish ready in an hour or less. If possible, mainly as much protein as possible, and secondly the quicker the better.
- Can you suggest a European side dish with a total time of at most an hour? My main priority is more protein; my second priority is shorter cooking times.

**valid-0003** hard {'diet': 'vegetarian', 'course': 'appetizers', 'cuisine': 'north-american', 'max_time': None}, soft {'protein': 1.0}
- Recommend a vegetarian North American appetizer. Among suitable options, I prefer more protein.
- I'm looking for a vegetarian North American appetizer. If possible, as much protein as possible.
- Can you suggest a vegetarian North American appetizer? My priority is more protein.

