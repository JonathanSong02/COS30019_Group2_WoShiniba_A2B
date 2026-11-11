"""
run_15_testcases.py
COS30019 A2A: automated test runner for the 15 test cases in testcases/.

Run from the project folder:
    python run_15_testcases.py
Results are printed and saved to A2A_15_Test_Results.csv.

For every test case and every method, the runner checks:
  1. Exact expected output: goal node, number of nodes created and path
     (or "no goal reachable" and the node count for unreachable cases).
     Expected values were worked out by hand and cross-checked against an
     independent reference implementation.
  2. Path validity: starts at the origin, ends at a destination and only
     uses directed edges that exist in the file.
  3. Optimality: BFS, CUS1 and CUS2 must use the fewest moves; A* must
     have the lowest total cost.
It also checks that a problem file saved with a UTF-8 BOM still loads.
"""

import csv
import heapq
import tempfile
from collections import deque
from pathlib import Path

import search
from graph import load_graph

ROOT = Path(__file__).resolve().parent
CASE_DIR = ROOT / "testcases"
RESULT_FILE = ROOT / "A2A_15_Test_Results.csv"
METHODS = ("DFS", "BFS", "GBFS", "AS", "CUS1", "CUS2")
FEWEST_MOVES_METHODS = ("BFS", "CUS1", "CUS2")

# Expected output per test case and method: (goal, nodes_created, path).
# goal and path are None when no destination can be reached.
NO_GOAL = None
EXPECTED = {
    "test01_assignment_sample.txt": {
        "DFS": (5, 7, [2, 1, 3, 5]),   "BFS": (4, 10, [2, 1, 4]),
        "GBFS": (5, 6, [2, 3, 5]),     "AS": (5, 8, [2, 3, 5]),
        "CUS1": (4, 8, [2, 1, 4]),     "CUS2": (4, 4, [2, 1, 4]),
    },
    "test02_direct_vs_indirect_route.txt": {
        "DFS": (3, 4, [1, 2, 3]),      "BFS": (3, 4, [1, 3]),
        "GBFS": (3, 3, [1, 3]),        "AS": (3, 4, [1, 3]),
        "CUS1": (3, 4, [1, 3]),        "CUS2": (3, 3, [1, 3]),
    },
    "test03_unreachable_destination.txt": {
        "DFS": (NO_GOAL, 4, None),     "BFS": (NO_GOAL, 4, None),
        "GBFS": (NO_GOAL, 4, None),    "AS": (NO_GOAL, 4, None),
        "CUS1": (NO_GOAL, 7, None),    "CUS2": (NO_GOAL, 8, None),
    },
    "test04_misleading_heuristic.txt": {
        "DFS": (4, 4, [1, 2, 4]),      "BFS": (4, 5, [1, 2, 4]),
        "GBFS": (4, 4, [1, 3, 4]),     "AS": (4, 5, [1, 2, 4]),
        "CUS1": (4, 7, [1, 2, 4]),     "CUS2": (4, 3, [1, 2, 4]),
    },
    "test05_bidirectional_cycles.txt": {
        "DFS": (5, 6, [1, 2, 3, 4, 5]), "BFS": (5, 9, [1, 3, 4, 5]),
        "GBFS": (5, 6, [1, 3, 4, 5]),   "AS": (5, 7, [1, 3, 4, 5]),
        "CUS1": (5, 18, [1, 3, 4, 5]),  "CUS2": (5, 7, [1, 3, 4, 5]),
    },
    "test06_cycles_dead_end.txt": {
        "DFS": (6, 6, [1, 2, 3, 4, 6]),   "BFS": (6, 6, [1, 2, 3, 4, 6]),
        "GBFS": (6, 6, [1, 2, 3, 4, 6]),  "AS": (6, 6, [1, 2, 3, 4, 6]),
        "CUS1": (6, 18, [1, 2, 3, 4, 6]), "CUS2": (6, 13, [1, 2, 3, 4, 6]),
    },
    "test07_equal_cost_goals_tie.txt": {
        "DFS": (2, 3, [1, 2]),         "BFS": (2, 3, [1, 2]),
        "GBFS": (2, 3, [1, 2]),        "AS": (2, 3, [1, 2]),
        "CUS1": (2, 3, [1, 2]),        "CUS2": (2, 2, [1, 2]),
    },
    "test08_fewest_moves_vs_lowest_cost.txt": {
        "DFS": (4, 5, [1, 2, 4]),      "BFS": (5, 6, [1, 5]),
        "GBFS": (5, 4, [1, 5]),        "AS": (4, 5, [1, 2, 4]),
        "CUS1": (5, 5, [1, 5]),        "CUS2": (5, 4, [1, 5]),
    },
    "test09_cross_branch_goal_tie.txt": {
        "DFS": (5, 4, [1, 2, 5]),      "BFS": (5, 5, [1, 2, 5]),
        "GBFS": (5, 4, [1, 2, 5]),     "AS": (5, 5, [1, 2, 5]),
        "CUS1": (5, 7, [1, 2, 5]),     "CUS2": (5, 3, [1, 2, 5]),
    },
    "test10_diamond_converging_paths.txt": {
        "DFS": (5, 5, [1, 2, 4, 5]),   "BFS": (5, 7, [1, 2, 4, 5]),
        "GBFS": (5, 5, [1, 2, 4, 5]),  "AS": (5, 7, [1, 2, 4, 5]),
        "CUS1": (5, 13, [1, 2, 4, 5]), "CUS2": (5, 4, [1, 2, 4, 5]),
    },
    "test11_shallow_vs_deep_goal.txt": {
        "DFS": (5, 5, [1, 2, 4, 5]),   "BFS": (3, 4, [1, 3]),
        "GBFS": (3, 3, [1, 3]),        "AS": (3, 3, [1, 3]),
        "CUS1": (3, 4, [1, 3]),        "CUS2": (3, 3, [1, 3]),
    },
    "test12_origin_is_destination.txt": {
        "DFS": (1, 1, [1]),            "BFS": (1, 1, [1]),
        "GBFS": (1, 1, [1]),           "AS": (1, 1, [1]),
        "CUS1": (1, 1, [1]),           "CUS2": (1, 1, [1]),
    },
    "test13_one_way_edges.txt": {
        "DFS": (5, 5, [1, 3, 2, 5]),   "BFS": (5, 6, [1, 3, 2, 5]),
        "GBFS": (5, 5, [1, 3, 4, 5]),  "AS": (5, 5, [1, 3, 4, 5]),
        "CUS1": (5, 11, [1, 3, 2, 5]), "CUS2": (5, 8, [1, 3, 2, 5]),
    },
    "test14_zero_heuristic_gapped_ids.txt": {
        "DFS": (9, 5, [1, 2, 6, 9]),   "BFS": (9, 7, [1, 2, 6, 9]),
        "GBFS": (9, 7, [1, 2, 6, 9]),  "AS": (9, 7, [1, 2, 6, 9]),
        "CUS1": (9, 13, [1, 2, 6, 9]), "CUS2": (9, 19, [1, 2, 6, 9]),
    },
    "test15_large_graph_multi_goal.txt": {
        "DFS": (7, 6, [1, 2, 4, 7]),   "BFS": (7, 15, [1, 2, 4, 7]),
        "GBFS": (7, 6, [1, 2, 4, 7]),  "AS": (7, 14, [1, 3, 5, 7]),
        "CUS1": (7, 15, [1, 2, 4, 7]), "CUS2": (7, 4, [1, 2, 4, 7]),
    },
}


def lowest_cost(graph):
    """Dijkstra: lowest total cost to any destination (None if unreachable)."""
    queue = [(0, graph.origin)]
    best = {graph.origin: 0}
    while queue:
        cost, state = heapq.heappop(queue)
        if cost != best[state]:
            continue
        if state in graph.destinations:
            return cost
        for neighbour, edge_cost in graph.neighbours(state):
            new_cost = cost + edge_cost
            if new_cost < best.get(neighbour, float("inf")):
                best[neighbour] = new_cost
                heapq.heappush(queue, (new_cost, neighbour))
    return None


def fewest_moves(graph):
    """Graph BFS: fewest moves to any destination (None if unreachable)."""
    queue = deque([(graph.origin, 0)])
    seen = {graph.origin}
    while queue:
        state, moves = queue.popleft()
        if state in graph.destinations:
            return moves
        for neighbour, _ in graph.neighbours(state):
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append((neighbour, moves + 1))
    return None


def path_cost(graph, path):
    """Total cost of a path; raises ValueError if it is not a legal route."""
    if path[0] != graph.origin:
        raise ValueError("Path does not start at the origin")
    if path[-1] not in graph.destinations:
        raise ValueError("Path does not end at a destination")
    total = 0
    for source, target in zip(path, path[1:]):
        costs = [c for n, c in graph.neighbours(source) if n == target]
        if not costs:
            raise ValueError(f"Illegal directed move {source} -> {target}")
        total += costs[0]
    return total


def check_case(graph, method, expected, min_moves, min_cost):
    """Run one method on one graph; return a result row for the CSV."""
    exp_goal, exp_nodes, exp_path = expected
    goal_node, nodes_created = search.METHODS[method](graph)
    row = {
        "method": method,
        "expected": "no goal" if exp_goal is None
                    else f"{exp_goal} / {exp_nodes} / {' '.join(map(str, exp_path))}",
        "goal": "", "nodes_created": nodes_created, "path": "",
        "moves": "", "path_cost": "",
        "reference_min_moves": min_moves, "reference_min_cost": min_cost,
        "status": "PASS", "remarks": "",
    }
    problems = []

    if goal_node is None:
        if exp_goal is not None:
            problems.append("no goal found but one was expected")
    else:
        path = goal_node.path()
        row.update(goal=goal_node.state, path=" ".join(map(str, path)),
                   moves=len(path) - 1)
        try:
            row["path_cost"] = path_cost(graph, path)
        except ValueError as error:
            problems.append(str(error))
        if exp_goal is None:
            problems.append("goal found but destination should be unreachable")
        elif (goal_node.state, path) != (exp_goal, exp_path):
            problems.append("goal/path differs from expected")
        if method in FEWEST_MOVES_METHODS and row["moves"] != min_moves:
            problems.append("not the fewest moves")
        if method == "AS" and row["path_cost"] != "" \
                and abs(row["path_cost"] - min_cost) > 1e-9:
            problems.append("not the lowest cost")

    if nodes_created != exp_nodes:
        problems.append(f"nodes_created {nodes_created} != expected {exp_nodes}")

    if problems:
        row["status"] = "FAIL"
        row["remarks"] = "; ".join(problems)
    return row


def check_bom_handling():
    """A file saved with a UTF-8 BOM (e.g. by Notepad) must still load."""
    source = CASE_DIR / "test01_assignment_sample.txt"
    with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as tmp:
        tmp.write(b"\xef\xbb\xbf" + source.read_bytes())
    try:
        load_graph(tmp.name)
        return True
    except ValueError:
        return False
    finally:
        Path(tmp.name).unlink()


def main():
    cases = sorted(CASE_DIR.glob("test[0-9][0-9]_*.txt"))
    names = {case.name for case in cases}
    if names != set(EXPECTED):
        print("ERROR: testcases/ does not match the expected 15 files.")
        print("  Missing:", sorted(set(EXPECTED) - names))
        print("  Unexpected:", sorted(names - set(EXPECTED)))
        return 1

    records = []
    for case in cases:
        graph = load_graph(str(case))
        min_moves, min_cost = fewest_moves(graph), lowest_cost(graph)
        for method in METHODS:
            row = {"test_case": case.name}
            row.update(check_case(graph, method, EXPECTED[case.name][method],
                                  min_moves, min_cost))
            records.append(row)
            print(f"{case.stem:40s} {method:4s} {row['status']}"
                  + (f"  ({row['remarks']})" if row["remarks"] else ""))

    bom_ok = check_bom_handling()
    print(f"\nUTF-8 BOM file loads: {'PASS' if bom_ok else 'FAIL'}")

    with RESULT_FILE.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    passed = sum(r["status"] == "PASS" for r in records)
    print(f"{len(cases)} test cases x {len(METHODS)} methods: "
          f"{passed}/{len(records)} PASS")
    print(f"Saved to: {RESULT_FILE.name}")
    return 0 if passed == len(records) and bom_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
