# Jonathan's `main.py` – COS30019 Assignment 2B

This is a **new entry point**, added without rewriting Darren's preprocessing or Anson's LSTM. The original A2A files remain unchanged.

## 1. What is working today

The latest uploaded project includes trained GRU models for **SCATS site 2000**, with horizons 15, 30, and 60 minutes and lookback 8. You can now call them through `main.py`:

```powershell
python main.py predict --model gru --site 2000 --time 2006-10-20T10:00:00 --horizon 30
```

`--time` means the **last observed** quarter-hour; the predicted target time is `--time + horizon`. Output is the number of vehicles in **one** 15-minute period. It is **not** an hourly total.

The shared prediction wrapper can also call `models.lstm_model.predict_lstm()` when matching trained LSTM files are available. Darren's Regression Tree is **not yet wired up** and will raise an explicit error instead of silently swapping models.

## 2. Real routing is awaiting Darren's verified road links

The current `data/processed/scats_sites.csv` contains site coordinates but **does not prove which roads are connected**. Ask Darren to provide `data/processed/scats_road_links.csv` with actual directed roads and distances in kilometres:

```csv
from_site,to_site,distance_km
```

**Do not fill this file by connecting nearest locations.** An edge in one direction does not automatically imply an edge in the other direction.

After adding the real graph and training model checkpoints for **all sites required by the graph**, run:

```powershell
python main.py route --model gru --origin 2000 --destination 3002 --time 2006-10-20T10:00:00 --horizon 15 --roads data/processed/scats_road_links.csv --hourly-policy steady_4x
```

This will **fail clearly** if a site lacks a trained model, or if a predicted hourly volume exceeds the conversion formula's stated 1,500 vehicles/hour assumption. That is intentional. Do not use fake fallback predictions.

## 3. Two modelling constraints to clarify

- **Hourly aggregation:** The current models predict a single 15-minute count. `--hourly-policy steady_4x` *approximates* the hourly count as four times that predicted count; it does **not** forecast all four quarter-hours. Document this limitation. For the actual system, prefer a true four-period forecast/aggregation after agreeing with the team.
- **Over-capacity flow:** The lecturer's quadratic assumes at most 1,500 vehicles/hour. By default, values above this raise an error. `--capacity-policy clamp` is an explicit *sensitivity/demo approximation* that caps flow at 1,500; it should **not** be silently used as the scientifically correct result. Discuss the real dataset's high flow with Dr. Joel.
- **Origin/destination convention:** The flow conversion PDF says use traffic at the **starting** site; the main A2B assignment brief says **destination**. The CLI defaults to `origin`; use `--flow-site destination` only if confirmed.

## 4. Tests

```powershell
python -m unittest discover -s tests -v
```

The `tests/test_main_integration.py` tests use synthetic tiny networks **only** to validate software contracts: no SCATS road links are invented. Your GRU checkpoints are real site-2000 trained weights.

## 5. Next step after backend works

1. Darren provides verified road links, approved flow convention, Regression Tree.
2. Each model supports all sites/horizons needed for the chosen network.
3. Add a GUI that calls `predict_traffic()` / `compute_traffic_routes()` and displays explicit errors for unsupported inputs.
4. Document the real 15 test cases and report measured route results separately from synthetic unit tests.
