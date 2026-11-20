"""Reproducible comparison of all six A2A search methods on identical A2B traffic.

Run from the project root:
    python -m evaluation.evaluate_search_methods

All six methods search the SAME 40-site graph and SAME Regression Tree
predictions at 2006-10-30T20:00:00; this isolates search behaviour.
"""
import csv
from pathlib import Path
from time import perf_counter

import main

OUTPUT = Path(__file__).resolve().parents[1] / 'results/search_comparison/six_search_methods.csv'
TRIPS = ((2000, 3002), (970, 4812), (2827, 4273), (4821, 3001))
METHODS = ('DFS', 'BFS', 'GBFS', 'AS', 'CUS1', 'CUS2')
TIME = '2006-10-30T20:00:00'


def generate():
    roads = main.load_verified_road_links(main.DEFAULT_ROADS)
    sites = main.load_site_coordinates(main.DEFAULT_SITES)
    rows = []
    # Predictor values are cached across methods, ensuring identical input.
    cache = {}
    def predictor(site, when):
        key = (site, when)
        if key not in cache:
            cache[key] = main.predict_traffic(site, when, model_name='regression_tree')
        return cache[key]
    for origin, destination in TRIPS:
        for algorithm in METHODS:
            started = perf_counter()
            result = main.compute_traffic_routes(
                origin, destination, TIME, roads, sites, predictor,
                k=5, search_algorithm=algorithm)
            elapsed_ms = (perf_counter() - started) * 1000.0
            first = result['routes'][0] if result['routes'] else None
            rows.append({
                'origin': origin, 'destination': destination,
                'last_observed': TIME, 'model': 'regression_tree',
                'algorithm': algorithm,
                'lowest_travel_time_guaranteed': result['assumptions']['search_cost_optimal'],
                'route_count': len(result['routes']),
                'nodes_expanded': result['assumptions']['search_nodes_expanded'],
                'elapsed_ms_end_to_end': round(elapsed_ms, 3),
                'best_discovered_route_min': first['travel_time_minutes'] if first else '',
                'best_discovered_route': '->'.join(map(str,first['scats_sites'])) if first else '',
            })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == '__main__':
    rows = generate()
    for row in rows:
        print(f"{row['origin']}->{row['destination']}: {row['algorithm']:<4}  "
              f"routes={row['route_count']}  expanded={row['nodes_expanded']}  "
              f"fastest discovered={row['best_discovered_route_min']} min")
    print('Saved:', OUTPUT)
