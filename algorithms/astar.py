"""
astar.py
A* search: expand the node with the smallest f(n) = g(n) + h(n), where
g(n) is the total edge cost from the origin and h(n) is a lower bound on
the remaining cost to the nearest destination.

Team conventions (keep these the same in every algorithm):
  - nodes_created counts every Node object made, including the root.
  - The goal test happens when a node is taken OUT of the frontier.
  - A neighbour already on the current path is skipped (cycle check).

Tie-breaking:
  The frontier is ordered by (f, creation_order). Children are created in
  ascending node-id order because graph.neighbours() is sorted, so:
    - equal-f siblings  -> smaller node id first (it was created first);
    - equal-f nodes on different branches -> the one added first.
  The node id is deliberately NOT part of the key (see gbfs.py).
"""

import heapq
from itertools import count

from node import Node


def _heuristic_scale(graph):
    """Return a factor that keeps straight-line distance admissible.

    If any edge costs less than its geometric length, raw Euclidean
    distance could overestimate the remaining cost. Scaling h by the
    smallest cost/length ratio (capped at 1) makes every edge cost at
    least scale * its length, so by the triangle inequality the scaled
    distance to the nearest goal never overestimates. When every edge
    costs at least its length (as in the assignment sample), scale = 1.
    """
    scale = 1.0
    for source, neighbours in graph.edges.items():
        for destination, cost in neighbours:
            distance = graph.distance(source, destination)
            if distance > 0:
                scale = min(scale, cost / distance)
    return scale


def search(graph):
    """Return (goal_node, nodes_created), or (None, nodes_created)."""
    scale = _heuristic_scale(graph)
    root = Node(graph.origin)
    nodes_created = 1
    creation_order = count()   # increasing stamp for chronological ties

    def f(search_node):
        return search_node.path_cost + scale * graph.heuristic(search_node.state)

    # Each frontier item: (f, creation_order, search_node)
    frontier = [(f(root), next(creation_order), root)]

    while frontier:
        _, _, node = heapq.heappop(frontier)

        # Goal test when the node is removed from the frontier. With an
        # admissible heuristic this guarantees the lowest-cost path.
        if node.state in graph.destinations:
            return node, nodes_created

        for neighbour, edge_cost in graph.neighbours(node.state):
            # Tree-search cycle check: skip states already on this path.
            if node.on_path(neighbour):
                continue

            child = Node(neighbour, node, node.path_cost + edge_cost)
            nodes_created += 1
            heapq.heappush(frontier, (f(child), next(creation_order), child))

    return None, nodes_created
