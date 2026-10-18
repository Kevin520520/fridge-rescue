# V3 retrieval policy scan

Low relevance is defined before inspection as a Top-10 candidate using less than 50% of the user's non-pantry input ingredients. The last two columns expose low recipe coverage and extra shopping burden.

| Split | Policy | No result | False no result | Full-input hit | Low input relevance | Coverage <.25 | Missing >6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev | coverage_015_missing_8 | 3/24 | 0 | 18/21 (0.857) | 15/201 (0.075) | 27/201 | 36/201 |
| dev | coverage_020_missing_7 | 3/24 | 0 | 17/21 (0.810) | 17/198 (0.086) | 21/198 | 25/198 |
| dev | coverage_020_missing_8 | 3/24 | 0 | 18/21 (0.857) | 15/198 (0.076) | 23/198 | 36/198 |
| dev | coverage_025_missing_6 | 3/24 | 0 | 13/21 (0.619) | 27/192 (0.141) | 0/192 | 0/192 |
| dev | two_tier_020_7_rescue_015_8 | 3/24 | 0 | 18/21 (0.857) | 17/198 (0.086) | 20/198 | 25/198 |
| final | coverage_015_missing_8 | 2/20 | 0 | 15/18 (0.833) | 14/162 (0.086) | 9/162 | 40/162 |
| final | coverage_020_missing_7 | 2/20 | 0 | 14/18 (0.778) | 19/162 (0.117) | 8/162 | 29/162 |
| final | coverage_020_missing_8 | 2/20 | 0 | 15/18 (0.833) | 14/162 (0.086) | 9/162 | 40/162 |
| final | coverage_025_missing_6 | 2/20 | 0 | 13/18 (0.722) | 37/162 (0.228) | 0/162 | 0/162 |
| final | two_tier_020_7_rescue_015_8 | 2/20 | 0 | 15/18 (0.833) | 18/162 (0.111) | 8/162 | 30/162 |

## Chinese and English equivalence

- D01 vs D02 / coverage_025_missing_6: ranked candidate IDs identical = True; English best = 1.0; Chinese best = 1.0.
- D01 vs D02 / coverage_020_missing_7: ranked candidate IDs identical = True; English best = 1.0; Chinese best = 1.0.
- D01 vs D02 / coverage_020_missing_8: ranked candidate IDs identical = True; English best = 1.0; Chinese best = 1.0.
- D01 vs D02 / two_tier_020_7_rescue_015_8: ranked candidate IDs identical = True; English best = 1.0; Chinese best = 1.0.
- D01 vs D02 / coverage_015_missing_8: ranked candidate IDs identical = True; English best = 1.0; Chinese best = 1.0.

## Focus cases

- D01 / coverage_025_missing_6: candidates=10, best_input_utilization=1.000, full_input_hit=True, low_relevance=0.
- D02 / coverage_025_missing_6: candidates=10, best_input_utilization=1.000, full_input_hit=True, low_relevance=0.
- D04 / coverage_025_missing_6: candidates=5, best_input_utilization=0.500, full_input_hit=False, low_relevance=0.
- D05 / coverage_025_missing_6: candidates=10, best_input_utilization=0.750, full_input_hit=False, low_relevance=0.
- D06 / coverage_025_missing_6: candidates=10, best_input_utilization=0.500, full_input_hit=False, low_relevance=7.
- D07 / coverage_025_missing_6: candidates=10, best_input_utilization=0.667, full_input_hit=False, low_relevance=0.
- D10 / coverage_025_missing_6: candidates=10, best_input_utilization=0.750, full_input_hit=False, low_relevance=0.
- D16 / coverage_025_missing_6: candidates=7, best_input_utilization=0.500, full_input_hit=False, low_relevance=0.
- F12 / coverage_025_missing_6: candidates=10, best_input_utilization=0.667, full_input_hit=False, low_relevance=9.
- F18 / coverage_025_missing_6: candidates=0, best_input_utilization=none, full_input_hit=False, low_relevance=0.
- F19 / coverage_025_missing_6: candidates=0, best_input_utilization=none, full_input_hit=False, low_relevance=0.
- D01 / coverage_020_missing_7: candidates=10, best_input_utilization=1.000, full_input_hit=True, low_relevance=0.
- D02 / coverage_020_missing_7: candidates=10, best_input_utilization=1.000, full_input_hit=True, low_relevance=0.
- D04 / coverage_020_missing_7: candidates=7, best_input_utilization=1.000, full_input_hit=True, low_relevance=0.
- D05 / coverage_020_missing_7: candidates=10, best_input_utilization=1.000, full_input_hit=True, low_relevance=0.
- D06 / coverage_020_missing_7: candidates=10, best_input_utilization=0.750, full_input_hit=False, low_relevance=0.
- D07 / coverage_020_missing_7: candidates=10, best_input_utilization=0.667, full_input_hit=False, low_relevance=0.
- D10 / coverage_020_missing_7: candidates=10, best_input_utilization=0.750, full_input_hit=False, low_relevance=0.
- D16 / coverage_020_missing_7: candidates=10, best_input_utilization=0.500, full_input_hit=False, low_relevance=0.
- F12 / coverage_020_missing_7: candidates=10, best_input_utilization=0.667, full_input_hit=False, low_relevance=8.
- F18 / coverage_020_missing_7: candidates=0, best_input_utilization=none, full_input_hit=False, low_relevance=0.
- F19 / coverage_020_missing_7: candidates=0, best_input_utilization=none, full_input_hit=False, low_relevance=0.
- D01 / coverage_020_missing_8: candidates=10, best_input_utilization=1.000, full_input_hit=True, low_relevance=0.
- D02 / coverage_020_missing_8: candidates=10, best_input_utilization=1.000, full_input_hit=True, low_relevance=0.
- D04 / coverage_020_missing_8: candidates=7, best_input_utilization=1.000, full_input_hit=True, low_relevance=0.
- D05 / coverage_020_missing_8: candidates=10, best_input_utilization=1.000, full_input_hit=True, low_relevance=0.
- D06 / coverage_020_missing_8: candidates=10, best_input_utilization=1.000, full_input_hit=True, low_relevance=0.
- D07 / coverage_020_missing_8: candidates=10, best_input_utilization=0.667, full_input_hit=False, low_relevance=0.
- D10 / coverage_020_missing_8: candidates=10, best_input_utilization=0.750, full_input_hit=False, low_relevance=0.
- D16 / coverage_020_missing_8: candidates=10, best_input_utilization=0.500, full_input_hit=False, low_relevance=0.
- F12 / coverage_020_missing_8: candidates=10, best_input_utilization=0.667, full_input_hit=False, low_relevance=8.
- F18 / coverage_020_missing_8: candidates=0, best_input_utilization=none, full_input_hit=False, low_relevance=0.
- F19 / coverage_020_missing_8: candidates=0, best_input_utilization=none, full_input_hit=False, low_relevance=0.
- D01 / two_tier_020_7_rescue_015_8: candidates=10, best_input_utilization=1.000, full_input_hit=True, low_relevance=0.
- D02 / two_tier_020_7_rescue_015_8: candidates=10, best_input_utilization=1.000, full_input_hit=True, low_relevance=0.
- D04 / two_tier_020_7_rescue_015_8: candidates=7, best_input_utilization=1.000, full_input_hit=True, low_relevance=0.
- D05 / two_tier_020_7_rescue_015_8: candidates=10, best_input_utilization=1.000, full_input_hit=True, low_relevance=0.
- D06 / two_tier_020_7_rescue_015_8: candidates=10, best_input_utilization=1.000, full_input_hit=True, low_relevance=0.
- D07 / two_tier_020_7_rescue_015_8: candidates=10, best_input_utilization=0.667, full_input_hit=False, low_relevance=0.
- D10 / two_tier_020_7_rescue_015_8: candidates=10, best_input_utilization=0.750, full_input_hit=False, low_relevance=0.
- D16 / two_tier_020_7_rescue_015_8: candidates=10, best_input_utilization=0.500, full_input_hit=False, low_relevance=0.
- F12 / two_tier_020_7_rescue_015_8: candidates=10, best_input_utilization=0.667, full_input_hit=False, low_relevance=8.
- F18 / two_tier_020_7_rescue_015_8: candidates=0, best_input_utilization=none, full_input_hit=False, low_relevance=0.
- F19 / two_tier_020_7_rescue_015_8: candidates=0, best_input_utilization=none, full_input_hit=False, low_relevance=0.
- D01 / coverage_015_missing_8: candidates=10, best_input_utilization=1.000, full_input_hit=True, low_relevance=0.
- D02 / coverage_015_missing_8: candidates=10, best_input_utilization=1.000, full_input_hit=True, low_relevance=0.
- D04 / coverage_015_missing_8: candidates=10, best_input_utilization=1.000, full_input_hit=True, low_relevance=0.
- D05 / coverage_015_missing_8: candidates=10, best_input_utilization=1.000, full_input_hit=True, low_relevance=0.
- D06 / coverage_015_missing_8: candidates=10, best_input_utilization=1.000, full_input_hit=True, low_relevance=0.
- D07 / coverage_015_missing_8: candidates=10, best_input_utilization=0.667, full_input_hit=False, low_relevance=0.
- D10 / coverage_015_missing_8: candidates=10, best_input_utilization=0.750, full_input_hit=False, low_relevance=0.
- D16 / coverage_015_missing_8: candidates=10, best_input_utilization=0.500, full_input_hit=False, low_relevance=0.
- F12 / coverage_015_missing_8: candidates=10, best_input_utilization=0.667, full_input_hit=False, low_relevance=8.
- F18 / coverage_015_missing_8: candidates=0, best_input_utilization=none, full_input_hit=False, low_relevance=0.
- F19 / coverage_015_missing_8: candidates=0, best_input_utilization=none, full_input_hit=False, low_relevance=0.
