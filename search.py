"""
search.py
Entry point for the Route Finding program.

Usage:
    python search.py <filename> <method>
where <method> is one of DFS, BFS, GBFS, AS, CUS1, CUS2 (not case-sensitive).

Output when a destination is reached:
    filename method
    goal number_of_nodes
    path
Output when no destination can be reached:
    filename method
    No goal is reachable; number_of_nodes
"""

import sys

from graph import load_graph
from algorithms import dfs, bfs, gbfs, astar, cus1, cus2

# Method name (as typed on the command line) -> search function.
# Every search function takes a Graph and returns (goal_node or None, nodes_created).
METHODS = {
    "DFS": dfs.search,
    "BFS": bfs.search,
    "GBFS": gbfs.search,
    "AS": astar.search,
    "CUS1": cus1.search,
    "CUS2": cus2.search,
}


def main(argv):
    if len(argv) != 3:
        print("Usage: python search.py <filename> <method>")
        print("Methods: " + ", ".join(METHODS))
        return 1

    filename, method = argv[1], argv[2].upper()

    if method not in METHODS:
        print(f"Unknown method '{argv[2]}'. Choose from: " + ", ".join(METHODS))
        return 1

    try:
        graph = load_graph(filename)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return 1
    except ValueError as error:
        print(f"Invalid problem file: {error}")
        return 1

    goal_node, nodes_created = METHODS[method](graph)

    print(f"{filename} {method}")
    if goal_node is None:
        print(f"No goal is reachable; {nodes_created}")
    else:
        print(f"{goal_node.state} {nodes_created}")
        print(" ".join(str(state) for state in goal_node.path()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
