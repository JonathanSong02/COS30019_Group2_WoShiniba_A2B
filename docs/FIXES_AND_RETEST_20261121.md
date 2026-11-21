# A2B routing and preprocessing corrections — 21 November 2026

Based on Group 2's 21 November `(6)(1)` project snapshot (Anson commit `b7c1b09`). This copy has not been pushed to GitHub.

## Corrected issues

1. **Strict flow capacity** (`main.py`): A predicted per-approach flow above 1500 veh/h makes only the affected directed edges ineligible. Other origin–destination trips proceed on the feasible network. If no feasible route remains the program returns a detailed capacity error. The result includes `capacity_excluded_sites`, `capacity_excluded_link_count`, `network_incomplete` and `route_scope`. An optional experimental Clamp is unchanged; no silent clamping is performed in Reject.
2. **Missing numeric detector readings** (`data_processing/scats_preprocess.py`): aggregation checks both the number of expected approaches and the count of valid numeric readings. A missing detector count makes that site's 15-minute sum missing instead of artificially low.
3. **GUI** (`gui/app.py`): explanatory partial-network warning distinguishes missing forecasts from capacity-excluded edges. Its JSON results convert integer site keys to text for browser compatibility. The route-comparison tab notes the restriction.
4. **Tests**: 5 new unit regressions and two new A2B acceptance scenarios (TC27 and TC28). TC15 now tests strict-edge exclusion rather than global rejection. A2B acceptance records contain input, expected result, actual result, status and remarks for all 28 cases.

## Verification completed here

- `python -m unittest discover -s tests -v`: **163/163 passed**.
- `python run_a2b_testcases.py`: **28/28 passed**.
- `python run_15_testcases.py`: **90/90 passed** (15 A2A cases, six algorithms).
- A real Regression Tree peak-time route `2000 -> 3002` at `2006-10-30T08:00` returns five feasible strict routes while excluding three edges supplied by over-capacity site 2825. Experimental Clamp keeps five routes and marks 2825 as capped.

## Data and model compatibility

The saved `data/processed/data_quality_report.md` reports zero negative or missing raw numeric counts. Therefore the newly corrected numeric-NaN branch is not expected to alter the original training set, but the raw XLS could not be regenerated in this environment because `xlrd` was not installed and package access was unavailable. On the student's computer: `python -m pip install -r requirements.txt`, then rerun `python -m data_processing.scats_preprocess --input "data/raw/Scats Data October 2006.xls" --out data/processed` in a **separate copy or output folder**, and compare results before replacing saved training artefacts. No model retraining or re-evaluation was performed here.

## Known limitations and submission tasks

- Strict Reject returns the fastest routes **within the eligible road network**, not globally across over-capacity roads. The GUI displays this qualification.
- One future quarter-hour forecast is multiplied by four for hourly flow; flow is equally divided among approaches. Road distances and turn restrictions are partly estimated.
- Gradio should be manually demonstrated on the team's Windows environment, including six search algorithms and all three ML models.
- The supplied report is a working submission draft: group members must verify their contribution text and existing signatures, confirm Regression Tree/tutor flow convention approval, and complete the official separate Who Did What document.
- Updated Canvas deadline: **29 November 2026, 23:59 (Malaysia time)**.
