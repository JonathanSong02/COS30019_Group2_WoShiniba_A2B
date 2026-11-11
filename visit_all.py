"""COS30019 A2A OPTIONAL RESEARCH: minimum-cost walk visiting ALL destinations.

Usage (does not change the six required search methods):
    python visit_all.py PathFinder-test.txt
    python visit_all.py research_cases/research01_greedy_trap.txt

Exact Dijkstra search over an augmented state (node ID, visited-goal mask).
A destination is marked visited whenever the route passes through it.
Vertices may be revisited: on a directed graph this can be essential.
Requires non-negative edge costs (as accepted by graph.py).
Standard library only; Python 3.8+; Windows 10/11 compatible.
"""
import heapq
import itertools
import sys
from graph import load_graph


def solve_visit_all(graph):
    """Return (path or None, minimum_cost or None, states_created, expanded).

    'states_created' counts queue insertions, including the initial state,
    and is NOT the six-method search-tree node counter.
    """
    goals = sorted(graph.destinations)
    goal_bit = {node: 1 << index for index, node in enumerate(goals)}
    finished = (1 << len(goals)) - 1
    start_mask = goal_bit.get(graph.origin, 0)
    start = (graph.origin, start_mask)

    best = {start: 0}
    parents = {start: None}
    sequence = itertools.count()
    frontier = [(0, next(sequence), start)]
    created, expanded = 1, 0

    while frontier:
        cost, _, (node, mask) = heapq.heappop(frontier)
        key = (node, mask)
        if cost != best.get(key):
            continue  # Ignore superseded queue entries
        expanded += 1
        if mask == finished:
            result_path = []
            while key is not None:
                result_path.append(key[0])
                key = parents[key]
            result_path.reverse()
            return result_path, cost, created, expanded

        # Graph.neighbours is already sorted by ascending node ID.
        for neighbour, edge_cost in graph.neighbours(node):
            if edge_cost < 0:
                raise ValueError('Visit-all Dijkstra requires non-negative costs')
            new_mask = mask | goal_bit.get(neighbour, 0)
            child = (neighbour, new_mask)
            new_cost = cost + edge_cost
            # Strictly improve only. For ties retain the earliest-created route.
            if new_cost < best.get(child, float('inf')):
                best[child] = new_cost
                parents[child] = (node, mask)
                heapq.heappush(frontier, (new_cost, next(sequence), child))
                created += 1

    return None, None, created, expanded


def main(argv):
    if len(argv) != 2:
        print('Usage: python visit_all.py <filename>')
        return 1
    filename = argv[1]
    try:
        graph = load_graph(filename)
        path, cost, created, expanded = solve_visit_all(graph)
    except (ValueError, OSError) as exc:
        print('Problem loading or solving graph: {}'.format(exc))
        return 1

    print('File: {}'.format(filename))
    print('Mode: VISIT_ALL (optional research extension)')
    print('Origin: {}'.format(graph.origin))
    print('Destinations to visit: {}'.format(' '.join(map(str, sorted(graph.destinations)))))
    if path is None:
        print('No route visits every destination.')
    else:
        print('Shortest total cost: {}'.format(cost))
        print('Path: {}'.format(' -> '.join(map(str, path))))
        print('Moves: {}'.format(len(path) - 1))
    print('Queue entries created: {}'.format(created))
    print('Expanded augmented states: {}'.format(expanded))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
