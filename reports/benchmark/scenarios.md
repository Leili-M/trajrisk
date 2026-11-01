# Proposal section 3, step 2: scenario specifications

Scenarios: 600, one per user (595 users; users in both splits: 5).

| Split | none | few | many |
|---|---|---|---|
| test | 40 | 180 | 180 |
| valid | 20 | 90 | 90 |

| Split | single | aligned | conflicting |
|---|---|---|---|
| test | 134 | 133 | 133 |
| valid | 67 | 67 | 66 |

Feasible recipes per scenario type (min / median / max):

- few: 10 / 23 / 50
- many: 203 / 583 / 18639
- none: 0 / 0 / 0

Scenarios with no feasible recipe (abstention expected): 60
Held-out combinations in validation: 0 (must be 0); in test: 132
Held-out preference pairs: [('time', 'calories'), ('protein', 'health')]; held-out course: soups-stews

Available hard-constraint combinations per type: few 215, many 194, none 29

## Examples

- valid-0000 (many, single): hard [course=main-dish, cuisine=european]; soft [calories 1.0]; feasible 1,727, unknown 7,954
- valid-0091 (few, aligned): hard [diet=vegetarian, course=desserts, cuisine=asian]; soft [health 0.6, calories 0.4]; feasible 33, unknown 5,088
- valid-0002 (many, conflicting): hard [course=side-dishes, cuisine=european, max_time=60]; soft [protein 0.7, time 0.3]; feasible 592, unknown 5,869
- valid-0180 (none, single): hard [diet=vegan, course=desserts, cuisine=italian]; soft [time 1.0]; feasible 0, unknown 1,395
- valid-0092 (few, conflicting): hard [course=salads, cuisine=italian, max_time=30]; soft [calories 0.6, protein 0.4]; feasible 46, unknown 3,351
