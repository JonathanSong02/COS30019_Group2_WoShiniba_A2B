"""Proof of concept for A2B top-five routes using A2A input graph.

Example:
    python demo_routing.py PathFinder-test.txt 2 5
NOTE: A2A sample graph weights are abstract costs, NOT traffic travel times.
"""
import argparse
from graph import load_graph
from routing.top_k_routes import find_top_k_routes

def main():
    parser = argparse.ArgumentParser(description='Route search prototype (not final TBRGS)')
    parser.add_argument('file', help='A2A .txt graph file')
    parser.add_argument('origin', type=int)
    parser.add_argument('destination', type=int)
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()
    graph = load_graph(args.file)
    routes = find_top_k_routes(graph, args.origin, args.destination, args.k)
    if not routes:
        print('No route available.')
    for position, (path, cost) in enumerate(routes, start=1):
        print(f'{position}. {" -> ".join(map(str, path))}   cost={cost:g}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
