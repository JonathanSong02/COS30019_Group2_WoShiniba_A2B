"""
bfs.py
Breadth-first search: expand the shallowest node first (FIFO queue).

Team conventions (keep these the same in every algorithm):
  - nodes_created counts every Node object made, including the root.
  - The goal test happens when a node is taken OUT of the frontier.
  - A neighbour already on the current path is skipped (cycle check).
"""

from collections import deque

from node import Node


def search(graph):
    root = Node(graph.origin)
    nodes_created = 1
    frontier = deque([root])

    while frontier:
        node = frontier.popleft()

        if node.state in graph.destinations:
            return node, nodes_created

        # neighbours() is already in ascending order, and a FIFO queue keeps
        # chronological order, which satisfies the tie-breaking rule.
        for neighbour, cost in graph.neighbours(node.state):
            if node.on_path(neighbour):
                continue
            frontier.append(Node(neighbour, node, node.path_cost + cost))
            nodes_created += 1

    return None, nodes_created
