# Audit guide for the controlled benchmark (proposal section 3)

The sheet `reports/benchmark/audit/audit_sheet.csv` contains a stratified random sample
of scenarios: equal numbers from every combination of feasible-set type
(none / few / many) and preference type (single / aligned / conflicting).

For every row, fill the five check columns with **1** (yes) or **0** (no). Leave a
cell empty only where the guide says so. Use `notes` for anything unusual.
Judge each row on its own; do not change the sheet in any other way.

| Column | Question | 1 means |
|---|---|---|
| `clear` | Is the template request clear and unambiguous? | A cook would know exactly what is asked. |
| `feasible_ok` | Do all three example recipes really satisfy every hard constraint in `hard`? Check the title, time, tags and ingredients. | No example breaks a constraint. Leave empty when `n_feasible` is 0. |
| `attributes_ok` | Does the recipe data contain what is needed to judge this request (time, tags, ingredients, nutrition)? | Every constraint and preference can be checked from the data shown. |
| `para1_same` | Does paraphrase 1 keep every hard constraint and the order of the preferences (main / secondary / equal) of the template? | Nothing added, dropped, softened or reordered. |
| `para2_same` | Same question for paraphrase 2. | As above. |

Small changes in wording are fine ("ready in 30 minutes or less" vs "under 30 minutes").
A paraphrase fails if, for example, it drops "vegetarian", turns "mainly A, secondly B"
into "A and B" or "equally A and B", adds a preference, or changes a time limit.

The reviewer was the author of the project (one person). This is reported as a
limitation: there is no second reviewer, so agreement between reviewers cannot be measured.
