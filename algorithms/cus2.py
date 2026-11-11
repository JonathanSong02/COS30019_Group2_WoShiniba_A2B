"""CUS2: IDA* optimized for the minimum number of moves."""

import math

from node import Node


def _maximum_geometric_edge_length(graph):
    """Return the longest edge length measured from node coordinates."""
    maximum = 0.0
    for from_node, edges in graph.edges.items():
        for to_node, _ in edges:
            maximum = max(maximum, graph.distance(from_node, to_node))
    return maximum


def _move_heuristic(graph, state, maximum_edge_length):
    """Admissible lower bound on the moves to the nearest destination."""
    if state in graph.destinations:
        return 0
    # With no positive-length geometric edge, zero is the safe lower bound.
    if maximum_edge_length <= 0 or not graph.destinations:
        return 0

    distance = min(graph.distance(state, goal) for goal in graph.destinations)
    return math.ceil(distance / maximum_edge_length)


def _threshold_search(graph, node, threshold, maximum_edge_length, counter):
    """Return (goal, smallest_exceeded_f) for one IDA* threshold pass."""
    # CUS2 deliberately uses depth rather than weighted path_cost for g(n).
    f_cost = node.depth + _move_heuristic(
        graph, node.state, maximum_edge_length
    )
    if f_cost > threshold:
        return None, f_cost

    # Passing the threshold selects the node for expansion and goal testing.
    if node.state in graph.destinations:
        return node, math.inf

    smallest_exceeded = math.inf
    # Sorted neighbours give ascending node-number ties; recursive visitation
    # retains chronological order among children with otherwise equal priority.
    for neighbour, edge_cost in graph.neighbours(node.state):
        if node.on_path(neighbour):
            continue

        child = Node(neighbour, node, node.path_cost + edge_cost)
        counter[0] += 1
        goal, exceeded = _threshold_search(
            graph, child, threshold, maximum_edge_length, counter
        )
        if goal is not None:
            return goal, math.inf
        smallest_exceeded = min(smallest_exceeded, exceeded)

    return None, smallest_exceeded


def search(graph):
    """Return a destination reached in the minimum number of moves."""
    maximum_edge_length = _maximum_geometric_edge_length(graph)
    threshold = _move_heuristic(
        graph, graph.origin, maximum_edge_length
    )
    nodes_created = [0]

    while True:
        # Every threshold iteration creates and counts a fresh search-tree root.
        root = Node(graph.origin)
        nodes_created[0] += 1
        goal, next_threshold = _threshold_search(
            graph, root, threshold, maximum_edge_length, nodes_created
        )

        if goal is not None:
            return goal, nodes_created[0]
        if math.isinf(next_threshold):
            return None, nodes_created[0]

        # IDA* advances to the smallest f-value that exceeded the old bound.
        threshold = next_threshold
