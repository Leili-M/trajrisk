# Proposal section 3, step 1: recipe attributes of the HUMMUS catalog

Catalog recipes: 43,529

| Attribute | Recipes with a value | Share |
|---|---|---|
| tags | 35,575 | 81.7% |
| ingredients | 43,529 | 100.0% |
| duration | 42,953 | 98.7% |
| n_ingredients | 43,529 | 100.0% |
| n_steps | 43,529 | 100.0% |
| calories | 43,529 | 100.0% |
| protein | 43,529 | 100.0% |
| fat | 43,529 | 100.0% |
| sugar | 43,529 | 100.0% |
| fsa_score | 42,324 | 97.2% |
| who_score | 43,439 | 99.8% |

## Hard-constraint values (recipes carrying the tag)

| Tag | Recipes |
|---|---|
| vegetarian | 6,005 |
| vegan | 1,599 |
| main-dish | 11,647 |
| desserts | 6,102 |
| breakfast | 2,330 |
| lunch | 4,523 |
| appetizers | 3,074 |
| salads | 2,187 |
| soups-stews | 1,522 |
| side-dishes | 4,973 |
| brunch | 3,148 |
| beverages | 1,801 |
| north-american | 9,013 |
| european | 4,232 |
| asian | 2,092 |
| mexican | 1,240 |
| italian | 1,305 |
| african | 424 |
| middle-eastern | 307 |
| south-american | 153 |
| indian | 354 |
| 15-minutes-or-less | 7,114 |
| 30-minutes-or-less | 8,507 |
| 60-minutes-or-less | 10,659 |
| 4-hours-or-less | 7,252 |

## Can the tags be trusted?

- Tagged vegetarian: 6,005; of these, ingredient list mentions a meat or fish word: 146 (2.4%)
- Tagged vegan: 1,599; of these, mentions an animal product: 208 (13.0%)
- Recipes with tags but without the vegetarian tag whose ingredients mention no meat word: 15,978 (tags are incomplete, so 'not tagged' is not the same as 'not vegetarian')

| Time tag | Tagged recipes with a duration | Duration within the limit |
|---|---|---|
| 15-minutes-or-less | 6,929 | 99.7% |
| 30-minutes-or-less | 8,507 | 99.9% |
| 60-minutes-or-less | 10,659 | 100.0% |
| 4-hours-or-less | 7,252 | 100.0% |

Examples of vegetarian-tagged recipes with a meat word (for manual inspection):

- Perfect Potato, Leek and Onion Soup With Garlic Cheese Toasts: extra virgin olive oil; large garlic cloves, chopped; grated swiss cheese; slices sourdough bread; chopped fresh chives or 1/2 cup $template2$, chopped; extra v
- Chef Flower's Cracked Wheat Pilaf  - Kibrisli Bulgur Pilavi: olive oil; onion, medium size and very finely diced; vermicelli, crushed (please do not substitute to another pasta); bulgur (cracked wheat); chicken stock or 1
- Mexican Eggs and Rice: cooked rice (leftover); eggs; oil or 1 tablespoon bacon grease; grated cheddar cheese; salsa
- An Avocado-Licious Sandwich: slices whole wheat bread, toasted; small avocado, sliced; roma tomato, sliced; slices lightlife turkey slices (or tofurkey...or real turkey if that's your thing
- Asian Green Beans in a Snap: cut green beans, well drained (french cut or regular in a can); minced onion (instant dried minced onion works well too); garlic clove, minced; soy sauce (depen
- Thick Spinach Soup: oil; butter; onion, chopped; potatoes, diced; spinach, chopped; pints vegetable stock or 1 1/2 pints chicken stock; grated nutmeg; salt; fresh ground black pepp
- Black Bean and Avocado Soup: olive oil; red onion, diced; garlic cloves, minced; ground cumin (i use the full 2 tsp); cayenne powder (to taste); ground cinnamon; salt; sugar; chopped fresh 
- Leek, Mushroom and Lemon Risotto: leek; cremini mushroom; olive oil; garlic cloves, minced; butter; small yellow onion, chopped; arborio rice; fresh vegetable stock or 5 cups fresh chicken stock
- Hungarian Noodles With Vegetarian Sausage: olive oil; vegetarian sausages (morningstar or boca are good); large sweet onion, quartered and thinly sliced; package shredded mixed cabbage (or 1 lb. you shre
- Crispy Noodle Salad With Sweet and Sour Dressing: mixed baby greens, 1 sack; fresh bean sprout; shredded carrot; radishes, sliced; scallions, chopped; red bell pepper, cut into thin strips; fried chinese noodle
- Family Favorite Healthy Minestrone Soup!: vegetable stock or 4 cups vegetarian chicken stock; also use fresh-i usually one can along with chopped fresh tomatoes) or (14 1/2 ounce) cans $template2$ (can 
- Mixed Beans With Bacon and Almonds: or 4 slices bacon; package frozen sugar snap peas; package frozen green beans; package frozen butter beans; package frozen lima beans; slivered almonds; salt & 
- Broccoli Kugel: frozen chopped broccoli, thawed; margarine; flour; water; parve chicken soup powder; mayonnaise; onion soup mix; eggs; pepper; pine nuts
- Lone Star Steakhouse Spicy Ranch Dip Copycat: prepared ranch salad dressing; onion powder; lemon pepper; crushed red pepper flakes; cayenne pepper; chili powder; beef bouillon powder; cocoa powder; ground c
- Arroz Brasileiro (Brazilian Rice): olive oil; onions, finely chopped; rice; hot chicken broth (to make this vegetarian or vegan i use 3 cups hot water and 3 full tsp of mckays vegan chicken seaso

## Soft preferences: rank correlation between attributes

Preferences are oriented so that higher means preferred. A negative value means the two preferences conflict: recipes that are good for one tend to be bad for the other.

| Pair | Spearman |
|---|---|
| shorter time vs fewer ingredients | +0.36 |
| shorter time vs fewer calories | +0.27 |
| shorter time vs more protein | -0.31 |
| shorter time vs healthier (FSA) | +0.19 |
| fewer ingredients vs fewer calories | +0.25 |
| fewer ingredients vs more protein | -0.30 |
| fewer ingredients vs healthier (FSA) | +0.17 |
| fewer calories vs more protein | -0.72 |
| fewer calories vs healthier (FSA) | +0.62 |
| more protein vs healthier (FSA) | -0.37 |

## Feasible recipes per combination of 2-4 hard constraints

| Feasible recipes | Combinations |
|---|---|
| 0 | 51 |
| 1-9 | 332 |
| 10-50 | 453 |
| 51-200 | 254 |
| >200 | 205 |

Full grid: reports/benchmark/feasible_grid.csv
