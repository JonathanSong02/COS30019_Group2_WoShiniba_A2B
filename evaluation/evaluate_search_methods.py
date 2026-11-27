"""Reproducible comparison of all six A2A search methods on identical A2B traffic.

Run from the project root:
    python -m evaluation.evaluate_search_methods            # four report trips
    python -m evaluation.evaluate_search_methods --sample 30  # 30 random trips x 2 times

All six methods search the SAME 40-site graph and SAME Regression Tree
predictions at 2006-10-30T20:00:00; this isolates search behaviour.
"""
import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from time import perf_counter

import main

OUTPUT = Path(__file__).resolve().parents[1] / 'results/search_comparison/six_search_methods.csv'
SAMPLE_OUTPUT = OUTPUT.with_name('six_search_methods_sample.csv')
SAMPLE_TIMES = ('2006-10-30T03:00:00', '2006-10-30T20:00:00')
TRIPS = ((2000, 3002), (970, 4812), (2827, 4273), (4821, 3001))
METHODS = ('DFS', 'BFS', 'GBFS', 'AS', 'CUS1', 'CUS2')
TIME = '2006-10-30T20:00:00'


def _run_trip(origin, destination, when, roads, sites, predictor):
    """Run all six methods on one trip and return one row per method."""
    rows = []
    for algorithm in METHODS:
        started = perf_counter()
        try:
            result = main.compute_traffic_routes(
                origin, destination, when, roads, sites, predictor,
                k=5, search_algorithm=algorithm)
            error = ''
        except (ValueError, RuntimeError) as exc:  # e.g. search budget or no feasible route
            result, error = None, str(exc)
        elapsed_ms = (perf_counter() - started) * 1000.0
        first = result['routes'][0] if result and result['routes'] else None
        rows.append({
            'origin': origin, 'destination': destination,
            'last_observed': when, 'model': 'regression_tree',
            'algorithm': algorithm,
            'lowest_travel_time_guaranteed': result['assumptions']['search_cost_optimal'] if result else '',
            'route_count': len(result['routes']) if result else 0,
            'nodes_expanded': result['assumptions']['search_nodes_expanded'] if result else '',
            'elapsed_ms_end_to_end': round(elapsed_ms, 3),
            'best_discovered_route_min': first['travel_time_minutes'] if first else '',
            'best_discovered_route': '->'.join(map(str, first['scats_sites'])) if first else '',
            'error': error,
        })
    return rows


def _setup():
    roads = main.load_verified_road_links(main.DEFAULT_ROADS)
    sites = main.load_site_coordinates(main.DEFAULT_SITES)
    # Predictor values are cached across methods, ensuring identical input.
    cache = {}

    def predictor(site, when):
        key = (site, when)
        if key not in cache:
            cache[key] = main.predict_traffic(site, when, model_name='regression_tree')
        return cache[key]
    return roads, sites, predictor


def _write(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate():
    """Four fixed report trips at 20:00 (identical inputs for every method)."""
    roads, sites, predictor = _setup()
    rows = []
    for origin, destination in TRIPS:
        rows.extend(_run_trip(origin, destination, TIME, roads, sites, predictor))
    _write(rows, OUTPUT)
    return rows


def generate_sample(trip_count=30, seed=42):
    """Random origin/destination pairs at an off-peak and an evening time.

    Summarises, per method, how often its best route equals the A* optimum,
    the largest extra time, and the median nodes expanded and run time.
    """
    roads, sites, predictor = _setup()
    site_ids = sorted(sites)
    rng = random.Random(seed)
    pairs = set()
    while len(pairs) < trip_count:
        origin, destination = rng.sample(site_ids, 2)
        pairs.add((origin, destination))
    rows = []
    for when in SAMPLE_TIMES:
        for origin, destination in sorted(pairs):
            rows.extend(_run_trip(origin, destination, when, roads, sites, predictor))
    _write(rows, SAMPLE_OUTPUT)

    optimum = {(r['origin'], r['destination'], r['last_observed']): r['best_discovered_route_min']
               for r in rows if r['algorithm'] == 'AS' and r['best_discovered_route_min'] != ''}
    summary = {'trips': len(optimum), 'times': list(SAMPLE_TIMES), 'seed': seed, 'methods': {}}
    for algorithm in METHODS:
        ok = [r for r in rows if r['algorithm'] == algorithm and r['best_discovered_route_min'] != ''
              and (r['origin'], r['destination'], r['last_observed']) in optimum]
        gaps = [r['best_discovered_route_min'] - optimum[(r['origin'], r['destination'], r['last_observed'])]
                for r in ok]
        summary['methods'][algorithm] = {
            'trips_completed': len(ok),
            'best_route_equals_optimum_percent': 100.0 * sum(g < 1e-6 for g in gaps) / len(ok) if ok else None,
            'mean_extra_minutes': statistics.mean(gaps) if gaps else None,
            'max_extra_minutes': max(gaps) if gaps else None,
            'median_nodes_expanded': statistics.median(r['nodes_expanded'] for r in ok) if ok else None,
            'median_elapsed_ms': statistics.median(r['elapsed_ms_end_to_end'] for r in ok) if ok else None,
        }
    SAMPLE_OUTPUT.with_name('six_search_methods_sample_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare the six A2A search methods on A2B traffic')
    parser.add_argument('--sample', type=int, default=0,
                        help='number of random origin/destination pairs (default: the four report trips)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.sample:
        print(json.dumps(generate_sample(args.sample, args.seed), indent=2))
    else:
        rows = generate()
        for row in rows:
            print(f"{row['origin']}->{row['destination']}: {row['algorithm']:<4}  "
                  f"routes={row['route_count']}  expanded={row['nodes_expanded']}  "
                  f"fastest discovered={row['best_discovered_route_min']} min")
        print('Saved:', OUTPUT)
