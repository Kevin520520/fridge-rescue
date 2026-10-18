# Fridge Rescue corpus data card

## Dataset summary

- Source dataset: Epicurious Recipes (13k)
- Source repository: https://github.com/josephrmartinez/recipe-dataset
- Pinned raw file: https://raw.githubusercontent.com/josephrmartinez/recipe-dataset/c5a1e8dab69d46841d914be1f20cca6fbe4b6260/13k-recipes.csv
- Pinned revision: `c5a1e8dab69d46841d914be1f20cca6fbe4b6260`
- Source file: `epicurious_recipes.csv`
- Source SHA-256: `d0e3977da7c1dc287ed75d6770f4064a7caeaa20ef6f64708b730e02f4d5f17a`
- Source and derived-corpus license: CC BY-SA 3.0
- Curated artifacts: 2000
- Fixed seed: 6012
- Curation version: 3.1
- Curation rules SHA-256: `b1622d52c4aa886c90480cdba384afa85d07a0ef54db402367f38e0ed3a19cc0`

## Selection and cleaning

The CSV is parsed with Python's CSV reader so embedded newlines remain attached to their records.
`Ingredients` must be a valid Python list containing only non-blank strings. Each selected recipe
contains at least 3 ingredient lines and 2 newline-delimited
instruction steps. The minimum source-measurement coverage is 60%. High-precision
phrases remove non-food and pet-food records, reviewed source exclusions are applied by source row
ID, source rows containing markup, missing quantities or unresolved sub-recipe references are
rejected, standalone equipment/metadata/narrative lines are removed, and exact content duplicates
are removed.

Every retained value in `ingredient_lines_original` comes from the source list. Quantities and units
are never guessed. The only text correction restores spacing in unambiguous joined mixed fractions
(for example, `11/2 cups` becomes `1 1/2 cups`) and its count is recorded. The source-line count,
removed non-ingredient-line count and format-correction count remain in every curated row.
`ingredients_normalized` is a lowercase/whitespace-normalized copy of the same complete lines for
matching. Each row also contains the fixed source-file hash and a hash of its original CSV fields.

The final sample is allocated with square-root weighting across derived broad-category and recipe-
complexity strata. Within each stratum, selection uses a fixed-seed SHA-256 priority.

## Important limitations

- Source text is community-published recipe content; deterministic checks improve structural and
  scope quality but do not prove that every recipe has been kitchen-tested.
- Some seasonings are legitimately listed without a numeric quantity (for example, salt to taste).
  Their source wording is preserved and the per-record measurement coverage is reported.
- This source does not provide dependable time, rating, review-count, serving or yield fields; the
  corresponding curated fields are null rather than inferred.
- Equipment and dietary statuses are conservative text-derived labels. `unknown` is not a safety
  guarantee, and allergy-related use still requires human checking.
- The corpus is primarily English and does not represent every cuisine or household.

## Final category and complexity distribution

- `beverage|complex`: 7
- `beverage|moderate`: 22
- `beverage|simple`: 19
- `bread_baking|complex`: 58
- `bread_baking|moderate`: 76
- `bread_baking|simple`: 31
- `breakfast|complex`: 14
- `breakfast|moderate`: 29
- `breakfast|simple`: 2
- `dessert|complex`: 80
- `dessert|moderate`: 77
- `dessert|simple`: 37
- `grain_pasta|complex`: 50
- `grain_pasta|moderate`: 75
- `grain_pasta|simple`: 33
- `main_protein|complex`: 130
- `main_protein|moderate`: 148
- `main_protein|simple`: 54
- `other|complex`: 96
- `other|moderate`: 185
- `other|simple`: 121
- `salad|complex`: 55
- `salad|moderate`: 91
- `salad|simple`: 27
- `sauce_condiment|complex`: 76
- `sauce_condiment|moderate`: 94
- `sauce_condiment|simple`: 43
- `snack_appetizer|moderate`: 1
- `soup|complex`: 42
- `soup|moderate`: 53
- `soup|simple`: 3
- `vegetable_legume|complex`: 57
- `vegetable_legume|moderate`: 76
- `vegetable_legume|simple`: 38
