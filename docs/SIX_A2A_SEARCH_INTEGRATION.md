# Assignment 2B: six Assignment 2A search methods

This project no longer uses Yen's algorithm or Dijkstra for the **live A2B top-five route calculation**. `routing/top_k_routes.py` now implements the six A2A search **strategies** on the same map-reviewed, directed, travel-time-weighted SCATS graph built from traffic predictions. The original A2A `search.py`, `algorithms/*.py`, and 15 text cases are preserved for independent A2A verification. An A2A research extension (`visit_all.py`) and legacy testing references may still mention Dijkstra; they are not called by A2B routing.

| GUI selection | Strategy | Route selection | Are reported top-five globally minimum travel-time paths? |
|---|---|---|---|
| DFS | Depth-First Search, LIFO | First five goal paths encountered under DFS; sorted afterwards by estimated travel time | **No** |
| BFS | Breadth-First Search, FIFO | First five goal paths in increasing *hop count* | **No**, unless all edges have identical time |
| GBFS | Greedy Best-First Search | First five paths found with smallest geometric h(s) expanded first | **No** |
| AS | A*, best f=g+h first | Five minimum-cost loopless paths | **Yes**, with nonnegative times and admissible h |
| CUS1 | Iterative Deepening DFS (IDDFS) | Five distinct paths encountered at increasing depth limits | **No**, optimises hops rather than time |
| CUS2 | Iterative Deepening A* (weighted IDA*) | Five minimum-cost loopless paths using increasing f thresholds | **Yes**, with nonnegative times and admissible h |

**Clarification:** Original 2A `algorithms/cus2.py` deliberately minimises **moves** rather than weighted path cost. Its A2B counterpart uses `f=g_travel_time+h_travel_time` so that it correctly optimises traffic-based travel time, and does **not** change the original A2A implementation.

## Admissible heuristic

For the SCATS graph, `h(s) = alpha * Euclidean(s, destination)` where `alpha` is the smallest `edge_travel_time / Euclidean_edge_distance` over links with a positive geometric distance. The coordinates and ratio use the same coordinate units, so each edge time is at least alpha times its geometric displacement. By the triangle inequality, `h` never exceeds the remaining travel time. This makes the A* and weighted IDA* top-five results optimal under the input edge weights. The algorithm does **not** use road distance as the edge weight; it still uses the lecturer's traffic-to-travel-time estimates.

Every result uses a path-specific cycle check, so no route visits the same SCATS intersection twice. All six respect road direction and weight validity. DFS also prunes branches when no simple path can reach the destination without revisiting a node; this pruning cannot change its set of reachable goal paths. Search budgets raise an explicit error instead of silently returning incomplete routes.

## Reproduction

```powershell
python -m unittest discover -s tests -v
python run_a2b_testcases.py
python -m evaluation.evaluate_search_methods
python -m gui.app
python main.py route --origin 2000 --destination 3002 --time 2006-10-30T20:00:00 --model regression_tree --horizon 15 --hourly-policy steady_4x --search-algorithm AS
```

The GUI Route Guidance and Model Comparison tabs expose the six search options. The default `AS` ensures the ordinary top-five traffic route display is actually the five cheapest simple routes, as required. To compare algorithms fairly, use the **same origin, destination, model and timestamp**, and avoid suggesting DFS/BFS/GBFS/CUS1 achieve global travel-time optimality.

Evidence: `tests/test_six_search_a2b.py` checks all six, including independent exhaustive path enumeration on small random graphs; `results/search_comparison/six_search_methods.csv` records identical-model experiments for four real SCATS origin/destination pairs.

## Still applicable limitations

* Model horizon availability depends on saved checkpoints. Full-network 30/60 minute GRU/LSTM checkpoints are not available.
* Road distances are approximations; the 4× flow conversion and even per-approach allocation are assumptions.
* The strict global capacity policy and raw-detector missing-value preprocessing findings from the earlier system audit are **not fixed by this search-algorithm migration**.
* **Revise your A2B report:** replace all Yen/Dijkstra routing descriptions with the six-method method/results and explain why A* or weighted IDA* is selected for optimal top-five routing.
