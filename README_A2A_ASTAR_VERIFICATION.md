# A2A A* compatibility verification (optional test addition)

This adds `tests/test_a2a_astar_scats.py` only. It does **not** change `main.py`,
`routing/`, saved checkpoints or model evaluation results.

## Install and run

Copy the `tests/` folder into the root of the existing COS30019 A2B project.
Run from that project directory in PowerShell:

```powershell
python -m unittest discover -s tests -p test_a2a_astar_scats.py -v
python -m unittest discover -s tests -v
```

The new tests compare the shortest route found by the **original A2A A***
(`algorithms/astar.py`) against route 1 returned by A2B A*
(`routing/top_k_routes.py`) using A2B's SCATS Graph adapter and flow-to-travel-time
weights. They check three origin/destination pairs, in both origin-flow and
destination-flow variants.

**Important:** Traffic inputs are *invented and strictly within capacity*.
They test algorithm interoperability, **not model accuracy**, current real
routing times or the final choice of traffic aggregation method. Document this
scope accurately. When production travel-time conversion changes, run the
end-to-end route tests separately using the actual trained ML predictions.
