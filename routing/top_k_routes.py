"""Compute up to k lowest-cost simple paths on a non-negative directed graph.

This module reuses A2A's Graph.neighbours() adjacency API. Graph weights are
whatever units the caller supplies: for A2B, *seconds of travel time*.
It does not predict traffic, build roads, or convert SCATS volumes.

Yen's algorithm enumerates loopless alternatives, with Dijkstra as the
shortest-path subroutine. Paths are unique by their SCATS-node sequence.
"""

import heapq
import itertools
import math


def _check_graph(graph):
    """Validate that every edge has a finite, non-negative numeric cost."""
    known = set(graph.coords)
    for source, edges in graph.edges.items():
        if source not in known:
            raise ValueError(f"Unknown source SCATS site: {source}")
        for destination, cost in edges:
            if destination not in known:
                raise ValueError(f"Unknown destination SCATS site: {destination}")
            if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
                raise ValueError(f"Invalid edge weight {source}->{destination}: {cost!r}")


def _dijkstra(graph, origin, destination, banned_nodes=frozenset(), banned_edges=frozenset()):
    if origin in banned_nodes or destination in banned_nodes:
        return None
    serial = itertools.count()
    heap = [(0.0, next(serial), origin, (origin,))]
    best = {origin: 0.0}
    settled = set()
    while heap:
        cost, _, current, path = heapq.heappop(heap)
        if current in settled or cost > best[current]:
            continue
        settled.add(current)
        if current == destination:
            return list(path), cost
        for neighbour, edge_cost in sorted(graph.neighbours(current)):
            if (current, neighbour) in banned_edges or neighbour in banned_nodes:
                continue
            if neighbour in settled:
                continue
            new_cost = cost + edge_cost
            if new_cost < best.get(neighbour, float('inf')):
                best[neighbour] = new_cost
                heapq.heappush(heap, (new_cost, next(serial), neighbour, path + (neighbour,)))
    return None


def _route_cost(graph, path):
    total = 0.0
    for current, neighbour in zip(path, path[1:]):
        options = [weight for candidate, weight in graph.neighbours(current)
                   if candidate == neighbour]
        if not options:
            raise ValueError(f"Missing road edge {current}->{neighbour}")
        total += min(options)
    return total


def find_top_k_routes(graph, origin, destination, k=5):
    """Return [(path_of_site_ids, total_weight), ...], sorted by increasing cost.

    Up to five routes (the A2B requirement). If unreachable, return [].
    If origin == destination, return [([origin], 0.0)]. No loops in paths.
    For the real SCATS graph, edge weights must already be in seconds.
    """
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 5:
        raise ValueError('k must be an integer from 1 to 5')
    if origin not in graph.coords or destination not in graph.coords:
        raise ValueError('Origin and destination must be valid graph/SCATS site IDs')
    _check_graph(graph)

    first = _dijkstra(graph, origin, destination)
    if first is None:
        return []
    accepted = [first]
    accepted_paths = {tuple(first[0])}
    candidates = []
    candidate_paths = set()
    serial = itertools.count()

    while len(accepted) < k:
        previous_path, _ = accepted[-1]
        for index in range(len(previous_path) - 1):
            root = previous_path[:index + 1]
            spur_origin = root[-1]
            removed_edges = set()
            for path, _ in accepted:
                if len(path) > index + 1 and path[:index + 1] == root:
                    removed_edges.add((path[index], path[index + 1]))
            # Ban earlier root nodes to guarantee a loopless route.
            spur = _dijkstra(graph, spur_origin, destination,
                             banned_nodes=set(root[:-1]),
                             banned_edges=removed_edges)
            if spur is None:
                continue
            combined = tuple(root[:-1] + spur[0])
            if combined in accepted_paths or combined in candidate_paths:
                continue
            total_cost = _route_cost(graph, combined)
            heapq.heappush(candidates, (total_cost, next(serial), combined))
            candidate_paths.add(combined)

        if not candidates:
            break
        cost, _, path_tuple = heapq.heappop(candidates)
        candidate_paths.discard(path_tuple)
        accepted_paths.add(path_tuple)
        accepted.append((list(path_tuple), cost))

    return accepted
