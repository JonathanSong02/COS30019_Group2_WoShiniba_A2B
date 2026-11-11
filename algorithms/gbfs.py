"""
gbfs.py
Greedy best-first search: expand the node with the smallest h(n), the
straight-line distance to the NEAREST destination. Path cost is ignored.

Team conventions (keep these the same in every algorithm):
  - nodes_created counts every Node object made, including the root.
  - The goal test happens when a node is taken OUT of the frontier.
  - A neighbour already on the current path is skipped (cycle check).

Tie-breaking (assignment NOTE 2):
  The frontier is ordered by (h, creation_order). Children are created in
  ascending node-id order because graph.neighbours() is sorted, so:
    - equal-h siblings  -> smaller node id first (it was created first);
    - equal-h nodes on different branches -> the one added first.
  The node id is deliberately NOT part of the key: putting it before
  creation_order would let a later, smaller-id node jump ahead of an
  earlier node on another branch, which breaks the chronological rule.
"""

import heapq
from itertools import count

from node import Node


def search(graph):
    """Return (goal_node, nodes_created), or (None, nodes_created)."""
    root = Node(graph.origin)
    nodes_created = 1
    creation_order = count()   # increasing stamp for chronological ties

    # Each frontier item: (h, creation_order, search_node)
    frontier = [(graph.heuristic(root.state), next(creation_order), root)]

    while frontier:
        _, _, node = heapq.heappop(frontier)

        # Goal test when the node is removed from the frontier.
        if node.state in graph.destinations:
            return node, nodes_created

        for neighbour, cost in graph.neighbours(node.state):
            # Tree-search cycle check: skip states already on this path.
            if node.on_path(neighbour):
                continue

            child = Node(neighbour, node, node.path_cost + cost)
            nodes_created += 1
            heapq.heappush(
                frontier,
                (graph.heuristic(child.state), next(creation_order), child),
            )

    return None, nodes_created
