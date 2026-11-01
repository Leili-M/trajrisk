# Proposal section 1: checks on the processed data

## movielens

Catalog rule: items with at least 5 positive training interactions (2,934 items).

| Check | Result |
|---|---|
| interactions inside split boundaries | PASS |
| cases inside split boundaries | PASS |
| no case in training period | PASS |
| catalog items | 2,934 |
| min positive training interactions per item | 5 |
| every item has k training positives | PASS |
| every target in catalog | PASS |
| cases checked | 60,000 |
| history items all strictly earlier | PASS |
| history size matches recomputation | PASS |
| n prior matches earlier positive events | PASS |
| every case has history | PASS |
| target never hidden as seen | PASS |
| target never in history | PASS |

| Split | Interactions | Positive | Positive on catalog items | Cases with earlier history | Users |
|---|---|---|---|---|---|
| valid_model | 50,009 | 28,217 | 28,080 | 27,651 | 485 |
| valid_risk | 50,011 | 26,925 | 26,785 | 26,399 | 520 |
| test | 200,041 | 112,260 | 111,426 | 110,457 | 1,759 |

## base

Catalog rule: items with at least 5 positive training interactions (43,529 items).

| Check | Result |
|---|---|
| interactions inside split boundaries | PASS |
| cases inside split boundaries | PASS |
| no case in training period | PASS |
| catalog items | 43,529 |
| min positive training interactions per item | 5 |
| every item has k training positives | PASS |
| every target in catalog | PASS |
| cases checked | 60,000 |
| history items all strictly earlier | PASS |
| history size matches recomputation | PASS |
| n prior matches earlier positive events | PASS |
| every case has history | PASS |
| target never hidden as seen | PASS |
| target never in history | PASS |

| Split | Interactions | Positive | Positive on catalog items | Cases with earlier history | Users |
|---|---|---|---|---|---|
| valid_model | 68,148 | 57,867 | 30,602 | 24,317 | 7,284 |
| valid_risk | 72,612 | 63,424 | 30,417 | 23,891 | 6,613 |
| test | 281,814 | 223,509 | 119,954 | 60,458 | 13,182 |

## amazon_video_games

Catalog rule: items with at least 5 positive training interactions (13,608 items).

| Check | Result |
|---|---|
| interactions inside split boundaries | PASS |
| cases inside split boundaries | PASS |
| no case in training period | PASS |
| catalog items | 13,608 |
| min positive training interactions per item | 5 |
| every item has k training positives | PASS |
| every target in catalog | PASS |
| cases checked | 53,920 |
| history items all strictly earlier | PASS |
| history size matches recomputation | PASS |
| n prior matches earlier positive events | PASS |
| every case has history | PASS |
| target never hidden as seen | PASS |
| target never in history | PASS |

| Split | Interactions | Positive | Positive on catalog items | Cases with earlier history | Users |
|---|---|---|---|---|---|
| valid_model | 40,729 | 33,630 | 22,348 | 19,085 | 9,369 |
| valid_risk | 40,729 | 33,827 | 17,668 | 14,835 | 7,668 |
| test | 162,918 | 130,457 | 32,527 | 26,350 | 13,977 |

## amazon_books

Catalog rule: items with at least 20 positive training interactions (57,262 items).

| Check | Result |
|---|---|
| interactions inside split boundaries | PASS |
| cases inside split boundaries | PASS |
| no case in training period | PASS |
| catalog items | 57,262 |
| min positive training interactions per item | 20 |
| every item has k training positives | PASS |
| every target in catalog | PASS |
| cases checked | 60,000 |
| history items all strictly earlier | PASS |
| history size matches recomputation | PASS |
| n prior matches earlier positive events | PASS |
| every case has history | PASS |
| target never hidden as seen | PASS |
| target never in history | PASS |

| Split | Interactions | Positive | Positive on catalog items | Cases with earlier history | Users |
|---|---|---|---|---|---|
| valid_model | 474,415 | 412,143 | 154,278 | 136,077 | 74,562 |
| valid_risk | 474,415 | 413,948 | 121,175 | 104,919 | 60,269 |
| test | 1,897,660 | 1,642,459 | 294,167 | 253,631 | 122,397 |
