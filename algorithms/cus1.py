"""CUS1: iterative deepening depth-first search (IDDFS)."""

from node import Node


def _depth_limited_search(graph, node, limit, counter):
    """Return (goal, cutoff), updating the shared Node creation counter."""
    # A node is goal-tested when selected for expansion, matching BFS.
    if node.state in graph.destinations:
        return node, False

    if node.depth == limit:
        # A cutoff is real only when an unvisited successor could be explored
        # at a deeper limit. This lets IDDFS stop on exhausted finite graphs.
        cutoff = any(
            not node.on_path(neighbour)
            for neighbour, _ in graph.neighbours(node.state)
        )
        return None, cutoff

    cutoff_occurred = False
    # neighbours() is sorted, so recursive DFS preserves ascending-number
    # tie-breaking and the chronological order in which children are created.
    for neighbour, cost in graph.neighbours(node.state):
        if node.on_path(neighbour):
            continue

        child = Node(neighbour, node, node.path_cost + cost)
        counter[0] += 1
        goal, cutoff = _depth_limited_search(graph, child, limit, counter)
        if goal is not None:
            return goal, False
        cutoff_occurred = cutoff_occurred or cutoff

    return None, cutoff_occurred


def search(graph):
    """Return the shallowest reachable destination and all Nodes created."""
    nodes_created = [0]
    depth_limit = 0

    while True:
        # Iterative deepening rebuilds and counts a new root on every pass.
        root = Node(graph.origin)
        nodes_created[0] += 1
        goal, cutoff = _depth_limited_search(
            graph, root, depth_limit, nodes_created
        )

        if goal is not None:
            return goal, nodes_created[0]
        if not cutoff:
            return None, nodes_created[0]

        depth_limit += 1
