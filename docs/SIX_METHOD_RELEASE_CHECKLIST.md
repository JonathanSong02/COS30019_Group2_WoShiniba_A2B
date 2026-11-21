# 21 November 2026: six-method routing release

The active A2B route engine uses DFS, BFS, GBFS, A*, IDDFS and weighted IDA*, **not** Yen/Dijkstra. The original six A2A `algorithms/*.py` modules and `search.py` remain intact for independent A2A marking.

## What was verified

* 163/163 Python unit tests passed in the reviewed environment, including 10 additional six-method integration tests.
* 28/28 recorded A2B acceptance scenarios passed, covering data, three predictors, traffic conversion, graph, routing and GUI.
* Both Gradio routing areas expose six-method dropdowns; configuration sets default `AS`.
* A2B CLI `--search-algorithm CUS2` returned 5 routes for SCATS 2000 to 3002 (30 October 2006, 20:00), best 12.282 min.
* Brute-force enumeration on small directed weighted graphs independently matched the A* and weighted IDA* top-five paths (including cycles, ties, zero-cost links and randomized DAGs).
* The four-pair results of the actual Regression Tree model are in `results/search_comparison/six_search_methods.csv`.

## Run from project root (PowerShell)

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_a2b_testcases.py
python -m evaluation.evaluate_search_methods
python -m gui.app
```

Example CLI:

```powershell
python main.py route --origin 2000 --destination 3002 --time 2006-10-30T20:00:00 --model regression_tree --horizon 15 --hourly-policy steady_4x --search-algorithm AS
```

## Before merge and submission

1. Recheck latest teammate commits (`git status` then `git pull --ff-only` on a clean working tree) and review the diff of `main.py`, `gui/app.py`, `routing/top_k_routes.py` rather than replacing your live repository folder without reviewing.
2. Confirm that the selected search method changes routes in both Route Guidance and Model Comparison.
3. Do **not** claim that DFS, BFS, GBFS or IDDFS find true shortest travel-time routes. Only A* and weighted IDA* do so under the input nonnegative-weight assumptions.
4. Use the revised report **working draft** under `docs/` and check contributions, signatures, tutor approval and screenshots against final committed code.
5. Unresolved unrelated audit items: capacity over-limit at an unrelated site can reject route computation, and partly missing raw numeric detector observations require separate review. Neither is silently claimed fixed here.
6. Submit the updated Canvas-required *Who Did What* template separately and retain evidence for member contributions.
