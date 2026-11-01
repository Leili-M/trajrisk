# Proposal section 2: dataset audit

| Dataset | Events | Users | Items | Min events/user | Min events/item | Identical rows | Repeated user-item pairs | Positive share |
|---|---|---|---|---|---|---|---|---|
| movielens | 1,000,209 | 6,040 | 3,706 | 20 | 1 | 0 | 0 | 57.5% |
| base | 1,916,424 | 302,412 | 507,335 | 1 | 1 | 0 | 0 | 88.5% |
| amazon_video_games | 814,586 | 94,762 | 25,612 | 5 | 5 | 0 | 0 | 80.4% |
| amazon_books | 9,488,297 | 776,370 | 495,063 | 5 | 5 | 0 | 0 | 84.7% |

## movielens

### Source files

| File | Bytes | SHA-256 |
|---|---|---|
| download_manifest.json | 219 |  |
| ml-1m.zip | 5,917,549 | a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20 |

Download manifest:

```json
{
  "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
  "size_mb": 5.64,
  "sha256": "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
  "downloaded_utc": "2026-10-21T11:55:50+00:00"
}
```

### Ratings and duplicates

Raw rating counts: {'1.0': 56174, '2.0': 107557, '3.0': 261197, '4.0': 348971, '5.0': 226310}  
Rows dropped by the rating rule: 0  
Positive ratings: [4.0, 5.0], share 57.5%  
Duplicates: {'identical_rows': 0, 'same_user_item_same_time': 0, 'user_item_pairs_seen_more_than_once': 0, 'extra_rows_from_repeated_pairs': 0}  
Period: 2000-04-25 23:05:32 to 2003-02-28 17:49:50

### Frozen split

| Part | From | To | Events | Share of events | Share of time | Users |
|---|---|---|---|---|---|---|
| train | 2000-04-25 23:05:32 | 2000-11-22 03:06:26 | 700,148 | 70.0% | 20.2% | 4,870 |
| valid_model | 2000-11-22 03:06:26 | 2000-11-26 06:42:19 | 50,009 | 5.0% | 0.4% | 504 |
| valid_risk | 2000-11-26 06:42:19 | 2000-12-02 14:52:18 | 50,011 | 5.0% | 0.6% | 557 |
| test | 2000-12-02 14:52:18 | 2003-02-28 17:49:50 | 200,041 | 20.0% | 78.8% | 1,783 |

Test events with an earlier positive (before catalog filter): event split 111,291 events / 1,762 users; time-period split (test from 2002-08-04 23:40:58.400000) 5,666 events / 328 users.

### Attrition

**valid_model**

| Step | Events | Users |
|---|---|---|
| all events | 50,009 | 504 |
| after rating rule | 50,009 | 504 |
| positive | 28,217 | 488 |
| positive on catalog items | 28,080 | 488 |
| with an earlier positive (cases) | 27,651 | 485 |

Positive catalog events repeating an earlier interaction: 0; cases repeating an earlier interaction: 0

**valid_risk**

| Step | Events | Users |
|---|---|---|
| all events | 50,011 | 557 |
| after rating rule | 50,011 | 557 |
| positive | 26,925 | 523 |
| positive on catalog items | 26,785 | 521 |
| with an earlier positive (cases) | 26,399 | 520 |

Positive catalog events repeating an earlier interaction: 0; cases repeating an earlier interaction: 0

**test**

| Step | Events | Users |
|---|---|---|
| all events | 200,041 | 1,783 |
| after rating rule | 200,041 | 1,783 |
| positive | 112,260 | 1,762 |
| positive on catalog items | 111,426 | 1,759 |
| with an earlier positive (cases) | 110,457 | 1,759 |

Positive catalog events repeating an earlier interaction: 0; cases repeating an earlier interaction: 0

### Attribute coverage of catalog items

Catalog items: 2,934, found in metadata: 2,934

| Attribute | Share with a value |
|---|---|
| title | 100.0% |
| genres | 100.0% |

## base

### Source files

| File | Bytes | SHA-256 |
|---|---|---|
| download_manifest.json | 868 |  |
| pp_members.zip | 14,178,071 | 2d177ec3fdaf5ee828f799e96ff795154c9c3a72b450e824766596c9047496ad |
| pp_recipes.zip | 316,978,383 | b37ab3e4133c447a36427d6429fad1ed6b459a130123257627137716895d544c |
| pp_reviews.zip | 210,415,143 | c5a1f6ce8db5d20fac1922fdfe6a3f50d59f12409f605158fbadda07fb395246 |

Download manifest:

```json
{
  "downloaded_utc": "2026-10-21T11:55:47+00:00",
  "files": {
    "pp_reviews.zip": {
      "url": "https://gitlab.com/felix134/connected-recipe-data-set/-/raw/master/data/hummus_data/preprocessed/pp_reviews.zip",
      "size_mb": 200.67,
      "sha256": "c5a1f6ce8db5d20fac1922fdfe6a3f50d59f12409f605158fbadda07fb395246"
    },
    "pp_recipes.zip": {
      "url": "https://gitlab.com/felix134/connected-recipe-data-set/-/raw/master/data/hummus_data/preprocessed/pp_recipes.zip",
      "size_mb": 302.29,
      "sha256": "b37ab3e4133c447a36427d6429fad1ed6b459a130123257627137716895d544c"
    },
    "pp_members.zip": {
      "url": "https://gitlab.com/felix134/connected-recipe-data-set/-/raw/master/data/hummus_data/preprocessed/pp_members.zip",
      "size_mb": 13.52,
      "sha256": "2d177ec3fdaf5ee828f799e96ff795154c9c3a72b450e824766596c9047496ad"
    }
  }
}
```

### Ratings and duplicates

Raw rating counts: {'1.0': 16919, '2.0': 17778, '2.5': 76876, '3.0': 50564, '4.0': 229869, '5.0': 1017908, '6.0': 506510}  
Rows dropped by the rating rule: 506,510  
Positive ratings: [4.0, 5.0], share 88.5%  
Duplicates: {'identical_rows': 0, 'same_user_item_same_time': 0, 'user_item_pairs_seen_more_than_once': 0, 'extra_rows_from_repeated_pairs': 0}  
Period: 2000-01-25 00:00:00 to 2022-09-19 00:00:00

### Frozen split

| Part | From | To | Events | Share of events | Share of time | Users |
|---|---|---|---|---|---|---|
| train | 2000-01-25 00:00:00 | 2010-07-05 00:00:00 | 987,340 | 70.0% | 46.1% | 121,584 |
| valid_model | 2010-07-05 00:00:00 | 2011-04-02 00:00:00 | 68,148 | 4.8% | 3.3% | 21,395 |
| valid_risk | 2011-04-02 00:00:00 | 2012-02-23 00:00:00 | 72,612 | 5.2% | 4.0% | 21,563 |
| test | 2012-02-23 00:00:00 | 2022-09-19 00:00:00 | 281,814 | 20.0% | 46.7% | 144,198 |

Test events with an earlier positive (before catalog filter): event split 128,527 events / 19,351 users; time-period split (test from 2018-03-09 09:36:00) 12,658 events / 3,792 users.

### Attrition

**valid_model**

| Step | Events | Users |
|---|---|---|
| all events | 97,328 | 26,953 |
| after rating rule | 68,148 | 21,395 |
| positive | 57,867 | 17,469 |
| positive on catalog items | 30,602 | 12,370 |
| with an earlier positive (cases) | 24,317 | 7,284 |

Positive catalog events repeating an earlier interaction: 0; cases repeating an earlier interaction: 0

**valid_risk**

| Step | Events | Users |
|---|---|---|
| all events | 106,323 | 27,641 |
| after rating rule | 72,612 | 21,563 |
| positive | 63,424 | 17,501 |
| positive on catalog items | 30,417 | 12,156 |
| with an earlier positive (cases) | 23,891 | 6,613 |

Positive catalog events repeating an earlier interaction: 0; cases repeating an earlier interaction: 0

**test**

| Step | Events | Users |
|---|---|---|
| all events | 439,720 | 164,110 |
| after rating rule | 281,814 | 144,198 |
| positive | 223,509 | 103,793 |
| positive on catalog items | 119,954 | 66,508 |
| with an earlier positive (cases) | 60,458 | 13,182 |

Positive catalog events repeating an earlier interaction: 0; cases repeating an earlier interaction: 0

### Attribute coverage of catalog items

Catalog items: 43,529, found in metadata: 43,529

| Attribute | Share with a value |
|---|---|
| Unnamed: 0 | 100.0% |
| recipe_id | 100.0% |
| title | 100.0% |
| description | 97.0% |
| author_id | 100.0% |
| duration | 98.7% |
| directions | 100.0% |
| ingredients | 100.0% |
| serves | 83.4% |
| last_changed_date | 100.0% |
| food_kg_locator | 100.0% |
| recipe_url | 100.0% |
| tags | 81.7% |
| new_author_id | 100.0% |
| average_rating | 100.0% |
| number_of_ratings | 100.0% |
| servingsPerRecipe | 100.0% |
| servingSize [g] | 100.0% |
| calories [cal] | 100.0% |
| caloriesFromFat [cal] | 100.0% |
| totalFat [g] | 100.0% |
| saturatedFat [g] | 100.0% |
| cholesterol [mg] | 100.0% |
| sodium [mg] | 100.0% |
| totalCarbohydrate [g] | 100.0% |
| dietaryFiber [g] | 100.0% |
| sugars [g] | 100.0% |
| protein [g] | 100.0% |
| direction_size | 100.0% |
| ingredients_sizes | 100.0% |
| who_score | 99.8% |
| fsa_score | 97.2% |
| nutri_score | 100.0% |
| normalization_comment | 2.6% |
| ingredient_food_kg_urls | 83.9% |
| ingredient_food_kg_names | 83.9% |

## amazon_video_games

### Source files

| File | Bytes | SHA-256 |
|---|---|---|
| Video_Games.csv.gz | 12,801,967 | a2bde5f3b945960d161538c200dd87845e6ee471b46da96410dde61613c6901c |
| download_manifest.json | 304 |  |
| meta_Video_Games.jsonl.gz | 103,096,082 | ced9696b0874121327a438bfe9d3d01aba45da158e58f94f4920352d75fc1fc9 |

Download manifest:

```json
{
  "source": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Video_Games.csv.gz",
  "file": "Video_Games.csv.gz",
  "bytes": 12801967,
  "sha256": "a2bde5f3b945960d161538c200dd87845e6ee471b46da96410dde61613c6901c",
  "checked_utc": "2026-10-21T12:57:22+00:00"
}
```

### Ratings and duplicates

Raw rating counts: {'1.0': 52563, '2.0': 36809, '3.0': 70347, '4.0': 134800, '5.0': 520067}  
Rows dropped by the rating rule: 0  
Positive ratings: [4.0, 5.0], share 80.4%  
Duplicates: {'identical_rows': 0, 'same_user_item_same_time': 0, 'user_item_pairs_seen_more_than_once': 0, 'extra_rows_from_repeated_pairs': 0}  
Period: 1999-10-22 21:43:15 to 2023-09-02 18:14:28.630000

### Frozen split

| Part | From | To | Events | Share of events | Share of time | Users |
|---|---|---|---|---|---|---|
| train | 1999-10-22 21:43:15 | 2018-08-21 20:11:02.455000 | 570,210 | 70.0% | 78.9% | 79,333 |
| valid_model | 2018-08-21 20:11:02.455000 | 2019-04-07 12:15:06.767000 | 40,729 | 5.0% | 2.6% | 17,586 |
| valid_risk | 2019-04-07 12:15:06.767000 | 2019-12-10 03:23:20.749000 | 40,729 | 5.0% | 2.8% | 17,207 |
| test | 2019-12-10 03:23:20.749000 | 2023-09-02 18:14:28.630000 | 162,918 | 20.0% | 15.6% | 41,569 |

Test events with an earlier positive (before catalog filter): event split 120,841 events / 35,892 users; time-period split (test from 2018-11-24 14:08:13.904000) 171,098 events / 45,335 users.

### Attrition

**valid_model**

| Step | Events | Users |
|---|---|---|
| all events | 40,729 | 17,586 |
| after rating rule | 40,729 | 17,586 |
| positive | 33,630 | 14,855 |
| positive on catalog items | 22,348 | 10,633 |
| with an earlier positive (cases) | 19,085 | 9,369 |

Positive catalog events repeating an earlier interaction: 0; cases repeating an earlier interaction: 0

**valid_risk**

| Step | Events | Users |
|---|---|---|
| all events | 40,729 | 17,207 |
| after rating rule | 40,729 | 17,207 |
| positive | 33,827 | 14,680 |
| positive on catalog items | 17,668 | 8,864 |
| with an earlier positive (cases) | 14,835 | 7,668 |

Positive catalog events repeating an earlier interaction: 0; cases repeating an earlier interaction: 0

**test**

| Step | Events | Users |
|---|---|---|
| all events | 162,918 | 41,569 |
| after rating rule | 162,918 | 41,569 |
| positive | 130,457 | 36,283 |
| positive on catalog items | 32,527 | 16,654 |
| with an earlier positive (cases) | 26,350 | 13,977 |

Positive catalog events repeating an earlier interaction: 0; cases repeating an earlier interaction: 0

### Attribute coverage of catalog items

Catalog items: 13,608, found in metadata: 13,608

| Attribute | Share with a value |
|---|---|
| title | 100.0% |
| description | 87.8% |
| features | 86.5% |
| categories | 96.5% |
| main_category | 98.0% |
| details | 97.4% |
| store | 99.5% |
| price | 65.1% |
| images | 100.0% |
| videos | 30.6% |

average_rating and rating_number are not used (they include future reviews)

## amazon_books

### Source files

| File | Bytes | SHA-256 |
|---|---|---|
| Books.csv.gz | 144,479,502 | 28c5104d0c2f7a7e842da3431606e21a245d8fb317e3434cd20b5c5a7b521615 |
| download_manifest.json | 293 |  |
| meta_Books.jsonl.gz | 4,942,125,770 | d41f2348d3e0f3099d2a7ee45b304970282199f92a7ccdfe8ba4b1f753009a10 |
| meta_Books_kept.jsonl.gz | 1,155,523,717 | 2454d45c92111e714127e8feb3ee8d502cbc5e3b7b3f910149f05e6b2c357193 |

Download manifest:

```json
{
  "source": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Books.csv.gz",
  "file": "Books.csv.gz",
  "bytes": 144479502,
  "sha256": "28c5104d0c2f7a7e842da3431606e21a245d8fb317e3434cd20b5c5a7b521615",
  "checked_utc": "2026-10-22T13:49:56+00:00"
}
```

### Ratings and duplicates

Raw rating counts: {'1.0': 274462, '2.0': 356264, '3.0': 818836, '4.0': 1920017, '5.0': 6118718}  
Rows dropped by the rating rule: 0  
Positive ratings: [4.0, 5.0], share 84.7%  
Duplicates: {'identical_rows': 0, 'same_user_item_same_time': 0, 'user_item_pairs_seen_more_than_once': 0, 'extra_rows_from_repeated_pairs': 0}  
Period: 1996-08-17 16:01:38 to 2023-09-12 00:22:22.061000

### Frozen split

| Part | From | To | Events | Share of events | Share of time | Users |
|---|---|---|---|---|---|---|
| train | 1996-08-17 16:01:38 | 2018-04-09 13:36:20.892000 | 6,641,807 | 70.0% | 80.0% | 699,115 |
| valid_model | 2018-04-09 13:36:20.892000 | 2018-12-07 22:22:31.655000 | 474,415 | 5.0% | 2.5% | 177,043 |
| valid_risk | 2018-12-07 22:22:31.655000 | 2019-07-31 13:10:12.512000 | 474,415 | 5.0% | 2.4% | 177,760 |
| test | 2019-07-31 13:10:12.512000 | 2023-09-12 00:22:22.061000 | 1,897,660 | 20.0% | 15.2% | 358,036 |

Test events with an earlier positive (before catalog filter): event split 1,596,311 events / 329,618 users; time-period split (test from 2018-04-13 17:54:13.248800) 2,374,791 events / 428,013 users.

### Attrition

**valid_model**

| Step | Events | Users |
|---|---|---|
| all events | 474,415 | 177,043 |
| after rating rule | 474,415 | 177,043 |
| positive | 412,143 | 160,239 |
| positive on catalog items | 154,278 | 84,474 |
| with an earlier positive (cases) | 136,077 | 74,562 |

Positive catalog events repeating an earlier interaction: 0; cases repeating an earlier interaction: 0

**valid_risk**

| Step | Events | Users |
|---|---|---|
| all events | 474,415 | 177,760 |
| after rating rule | 474,415 | 177,760 |
| positive | 413,948 | 161,236 |
| positive on catalog items | 121,175 | 69,084 |
| with an earlier positive (cases) | 104,919 | 60,269 |

Positive catalog events repeating an earlier interaction: 0; cases repeating an earlier interaction: 0

**test**

| Step | Events | Users |
|---|---|---|
| all events | 1,897,660 | 358,036 |
| after rating rule | 1,897,660 | 358,036 |
| positive | 1,642,459 | 330,997 |
| positive on catalog items | 294,167 | 140,864 |
| with an earlier positive (cases) | 253,631 | 122,397 |

Positive catalog events repeating an earlier interaction: 0; cases repeating an earlier interaction: 0

### Attribute coverage of catalog items

Catalog items: 57,262, found in metadata: 57,262

| Attribute | Share with a value |
|---|---|
| title | 100.0% |
| description | 85.7% |
| features | 99.6% |
| categories | 99.6% |
| main_category | 100.0% |
| details | 96.8% |
| store | 99.8% |
| price | 90.0% |
| images | 89.5% |
| videos | 12.9% |

average_rating and rating_number are not used (they include future reviews)
