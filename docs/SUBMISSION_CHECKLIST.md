# COS30019 A2B — Final Group 2 Submission Checklist (20 Nov 2026)

Due: **Sunday 29 November 2026, 23:59** (updated Canvas time; the older PDF says 22 November). Check the official PDF and Canvas rubric as the authority.

## Required functionality (rubric 100 marks)

- [ ] 9 Data processing: explain and demonstrate extraction, cleaning, SCATS time series construction and nonleaky chronological split.
- [ ] 21 Algorithms: trained LSTM, GRU and approved third method (Regression Tree); show actual trained model predictions. Obtain/retain tutor confirmation of third method.
- [ ] 15 Evaluation: reproducible matched testing with MAE, RMSE and NRMSE, per-site/horizon analyses, sample size and objective interpretation.
- [ ] 15 Integration: GUI input/settings/visualisation and `config.json`; map SCATS intersections; predicted traffic → hourly flow → speed → edge seconds → up to five best paths.
- [ ] 10 Testing: at least 15 documented distinct A2B scenarios with expected, actual, result and limitations; current `run_a2b_testcases.py` contains 26.
- [ ] 15 Report: 8–10 content pages excluding signed cover and TOC; include instructions, introduction, features/bugs/missing, testing, results/insights, conclusion, acknowledgements and references.
- [ ] 15 Research: explain demonstrated initiatives and data-backed findings (e.g. per-approach versus whole-site capacity, rush-hour behaviour, route comparison and road graph review).

## Final checks before commit/submission

- [ ] Use latest Git `main` and do **not** overwrite Darren's `main.py`, `routing/travel_time.py`, `gui/route_map.py` or regenerated route comparisons with older ZIP copies.
- [ ] Run `python -m unittest discover -s tests -v` and `python run_a2b_testcases.py`; document failures honestly if any.
- [ ] Launch `python -m gui.app` on a fresh team Windows PC. Show Traffic Prediction, Route Guidance and Model Comparison tabs with screenshots and real 2006 test timestamp.
- [ ] Install dependencies in a fresh virtual environment; the repository currently pins versions, so verify those versions are installable on the team's Windows/Python setup.
- [ ] Demonstrate GRU / LSTM / Regression Tree at 2006-10-30T20:00:00 and distinguish historical simulation from live navigation.
- [ ] Document 4× single-quarter-hour approximation, per-approach averaging, road-distance limitations and strict versus experimental Clamp policy.
- [ ] Ask Darren for the records of Dr. Joel's reported 19 November flow-convention consultation; also confirm Regression Tree approval.
- [ ] Add signed cover with correct member names and IDs, and verify the specific tasks in the contribution statement with all three teammates.
- [ ] Attach **COS30019 – Who did What.docx**, as required by Canvas.
- [ ] Keep the report within 8–10 content pages, caption figures/tables, and cite resources meaningfully.
- [ ] Submit final source/code, trained models, datasets needed to run, documented tests, report named by Team ID, and official contribution document using the accepted Canvas submission method.
