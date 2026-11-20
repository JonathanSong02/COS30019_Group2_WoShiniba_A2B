"""A2B routing with the SIX A2A tree-search strategies (no Yen/Dijkstra).

All strategies explore *simple* directed paths through the predicted-travel-time
weighted SCATS graph. DFS/BFS/GBFS/CUS1 select paths according to their own
frontier discipline, not guaranteed lowest travel time. A* (AS) and weighted
IDA* (CUS2) use an admissible cost-distance lower bound and return the exact
k cheapest simple paths for nonnegative edge costs.

The original A2A implementations in algorithms/ are deliberately preserved,
including CUS2's A2A minimum-**moves** objective. Here CUS2 is the weighted
travel-**time** generalisation needed for A2B; otherwise it would not be
optimal for the assignment's predicted-time edge weights.
"""
from __future__ import annotations

from collections import deque
import heapq
from itertools import count
from functools import lru_cache
import math

METHODS = ("DFS", "BFS", "GBFS", "AS", "CUS1", "CUS2")
# Upper bound is explicit: never silently report a partial search as complete.
MAX_EXPANSIONS = 500_000


def _validated(graph, origin, destination, k, algorithm):
    if algorithm not in METHODS:
        raise ValueError(f"Unknown search algorithm {algorithm!r}; choose {', '.join(METHODS)}")
    if type(k) is not int or not 1 <= k <= 5:
        raise ValueError("k must be an integer from 1 to 5")
    if origin not in graph.coords or destination not in graph.coords:
        raise ValueError("Origin and destination must be valid graph/SCATS site IDs")
    adjacency = {}
    reverse = {s: [] for s in graph.coords}
    for source, edges in graph.edges.items():
        if source not in graph.coords:
            raise ValueError(f"Unknown source SCATS site: {source}")
        minimal = {}
        for target, cost in edges:
            if target not in graph.coords:
                raise ValueError(f"Unknown destination SCATS site: {target}")
            if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
                raise ValueError(f"Invalid edge weight {source}->{target}: {cost!r}")
            minimal[target] = min(float(cost), minimal.get(target, math.inf))
        adjacency[source] = tuple(sorted(minimal.items()))
        for target in minimal:
            reverse[target].append(source)
    for source in graph.coords:
        adjacency.setdefault(source, ())
    # Ignore dead-end branches that provably cannot reach the selected goal.
    can_reach = {destination}
    queue = deque([destination])
    while queue:
        for predecessor in reverse[queue.popleft()]:
            if predecessor not in can_reach:
                can_reach.add(predecessor)
                queue.append(predecessor)
    return adjacency, can_reach


def _admissible_h(graph, adjacency, destination):
    """Cost lower bound from the minimum edge-cost / geometric-distance ratio.

    With h(s) = alpha * Euclidean(s,goal) and every edge c >= alpha *
    Euclidean(edge), triangle inequality gives consistency and admissibility.
    This works even if geographic coordinates are degrees rather than km.
    """
    scale = math.inf
    for source, edges in adjacency.items():
        for target, cost in edges:
            distance = graph.distance(source, target)
            if distance > 0:
                scale = min(scale, cost / distance)
    if not math.isfinite(scale):
        scale = 0.0
    return lambda site: scale * graph.distance(site, destination)


def _limit(counter):
    if counter[0] >= MAX_EXPANSIONS:
        raise ValueError("Search expansion limit reached; no incomplete routes are returned. "
                         "Try A* (AS) or fewer routes for this graph.")
    counter[0] += 1


def _frontier_routes(adjacency, reach, origin, destination, k, method, heuristic, expanded):
    serial = count()
    start = ((origin,), 0.0)
    if method == "DFS":
        frontier = [start]
        get = frontier.pop
        def add(children):
            frontier.extend(reversed(children))
    elif method == "BFS":
        frontier = deque([start])
        get = frontier.popleft
        def add(children):
            frontier.extend(children)
    else:
        frontier = [(0.0 if method == "AS" else heuristic(origin), next(serial), start)]
        def get():
            return heapq.heappop(frontier)[2]
        def add(children):
            for path, cost in children:
                priority = (cost + heuristic(path[-1])) if method == "AS" else heuristic(path[-1])
                heapq.heappush(frontier, (priority, next(serial), (path, cost)))
    # In a dense cyclic network DFS can spend exponential time exploring
    # branches whose *remaining unvisited nodes* can no longer reach the goal.
    # This feasibility pruning changes no DFS answer: impossible branches
    # cannot contain any simple route to the destination.
    @lru_cache(maxsize=131072)
    def feasible(current, forbidden):
        waiting = deque([current])
        reached = {current}
        while waiting:
            for next_site, _ in adjacency[waiting.popleft()]:
                if next_site == destination:
                    return True
                if next_site not in forbidden and next_site not in reached:
                    reached.add(next_site)
                    waiting.append(next_site)
        return False

    found = []
    while frontier and (len(found) < k or (
        method == "AS" and frontier[0][0] <=
        sorted((cost for _, cost in found))[k - 1] + 1e-9
    )):
        path, cost = get()
        _limit(expanded)
        current = path[-1]
        if current == destination:
            found.append((list(path), cost))
            continue
        children = [(path + (neighbor,), cost + weight)
                    for neighbor, weight in adjacency[current]
                    if neighbor in reach and neighbor not in path and (
                        method != "DFS" or neighbor == destination or
                        feasible(neighbor, frozenset(path))
                    )]
        add(children)
    # A* goal pop order is optimal-cost order; other methods are only
    # sorted among their discovered candidates, NOT claimed globally optimal.
    return sorted(found, key=lambda route: (route[1], len(route[0]), tuple(route[0])))[:k]


def _iddfs_routes(adjacency, reach, origin, destination, k, expanded):
    found = []
    visited_paths = set()

    def walk(path, cost, depth_remaining):
        _limit(expanded)
        if path[-1] == destination:
            if path not in visited_paths:
                found.append((list(path), cost))
                visited_paths.add(path)
            return len(found) >= k
        if not depth_remaining:
            return False
        for neighbor, weight in adjacency[path[-1]]:
            if neighbor in reach and neighbor not in path:
                if walk(path + (neighbor,), cost + weight, depth_remaining - 1):
                    return True
        return False

    for limit in range(len(adjacency)):
        if walk((origin,), 0.0, limit):
            break
    return sorted(found, key=lambda route: (route[1], len(route[0]), tuple(route[0])))[:k]


def _ida_star_routes(adjacency, reach, origin, destination, k, heuristic, expanded):
    """Weighted IDA*; keep increasing admissible f thresholds until top-k proven."""
    threshold = heuristic(origin)
    found = {}
    # Continue whole threshold pass: IDA* DFS discovers goals in depth order,
    # not cost order, even though f is used as a cutoff.
    while True:
        next_threshold = math.inf

        def walk(path, cost):
            nonlocal next_threshold
            _limit(expanded)
            f = cost + heuristic(path[-1])
            if f > threshold + 1e-9:
                next_threshold = min(next_threshold, f)
                return
            if path[-1] == destination:
                found[path] = cost
                return
            for neighbor, weight in adjacency[path[-1]]:
                if neighbor in reach and neighbor not in path:
                    walk(path + (neighbor,), cost + weight)

        walk((origin,), 0.0)
        ranked = sorted(found.items(), key=lambda item: (item[1], len(item[0]), item[0]))
        if len(ranked) >= k and ranked[k - 1][1] <= threshold + 1e-9:
            return [(list(path), cost) for path, cost in ranked[:k]]
        if math.isinf(next_threshold):
            return [(list(path), cost) for path, cost in ranked[:k]]
        # Floating-point threshold changes must move forward.
        threshold = max(math.nextafter(threshold, math.inf), next_threshold)


def search_routes(graph, origin, destination, k=5, algorithm="AS"):
    """Return (routes, statistics), where routes are (site-path, time-seconds).

    AS and CUS2: cost-optimal top-k. DFS/BFS/GBFS/CUS1: up to k unique
    strategy-discovered routes, ranked by their reported travel time.
    """
    algorithm = str(algorithm).upper()
    adjacency, can_reach = _validated(graph, origin, destination, k, algorithm)
    optimal = algorithm in {"AS", "CUS2"}
    if origin == destination:
        return [([origin], 0.0)], {"algorithm": algorithm, "expanded": 1, "cost_optimal": optimal}
    if origin not in can_reach:
        return [], {"algorithm": algorithm, "expanded": 0, "cost_optimal": optimal}
    expanded = [0]
    h = _admissible_h(graph, adjacency, destination)
    if algorithm in {"DFS", "BFS", "GBFS", "AS"}:
        routes = _frontier_routes(adjacency, can_reach, origin, destination, k, algorithm, h, expanded)
    elif algorithm == "CUS1":
        routes = _iddfs_routes(adjacency, can_reach, origin, destination, k, expanded)
    else:
        routes = _ida_star_routes(adjacency, can_reach, origin, destination, k, h, expanded)
    return routes, {"algorithm": algorithm, "expanded": expanded[0], "cost_optimal": optimal}


def find_top_k_routes(graph, origin, destination, k=5, algorithm="AS"):
    """Legacy call interface retained for tests/integration; default is A*."""
    return search_routes(graph, origin, destination, k, algorithm)[0]
