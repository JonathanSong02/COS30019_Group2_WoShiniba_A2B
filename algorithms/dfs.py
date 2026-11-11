"""
dfs.py
Depth-first search: expand the deepest node first (LIFO stack).

Team conventions (keep these the same in every algorithm):
  - nodes_created counts every Node object made, including the root.
  - The goal test happens when a node is taken OUT of the frontier.
  - A neighbour already on the current path is skipped (cycle check).
"""

from node import Node


def search(graph):
    root = Node(graph.origin)
    nodes_created = 1
    frontier = [root]

    while frontier:
        node = frontier.pop()

        if node.state in graph.destinations:
            return node, nodes_created

        # neighbours() is in ascending order. Pushing them in reverse puts the
        # smallest id on top of the stack, so it is expanded first. Across
        # branches the stack is chronological (most recently added first),
        # which is what depth-first order means.
        children = []
        for neighbour, cost in graph.neighbours(node.state):
            if node.on_path(neighbour):
                continue
            children.append(Node(neighbour, node, node.path_cost + cost))
            nodes_created += 1
        frontier.extend(reversed(children))

    return None, nodes_created
