"""
graph.py
Graph representation and problem-file parser for the Route Finding problem.

A problem file has four sections:
    Nodes:          one "id: (x,y)" per line
    Edges:          one "(from,to): cost" per line (edges are directed)
    Origin:         a single node id
    Destinations:   node ids separated by semicolons
"""

import math
import re

NODE_PATTERN = re.compile(r"^(\d+)\s*:\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)$")
EDGE_PATTERN = re.compile(r"^\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*:\s*(\d+(?:\.\d+)?)$")
SECTIONS = ("nodes", "edges", "origin", "destinations")


def _to_number(text):
    """Convert '4' to 4 and '4.5' to 4.5 so costs print cleanly."""
    value = float(text)
    return int(value) if value.is_integer() else value


class Graph:
    """Directed, weighted graph with 2D coordinates for every node."""

    def __init__(self):
        self.coords = {}          # node id -> (x, y)
        self.edges = {}           # node id -> list of (neighbour, cost)
        self.origin = None
        self.destinations = set()

    def add_node(self, node_id, x, y):
        self.coords[node_id] = (x, y)
        self.edges.setdefault(node_id, [])

    def add_edge(self, from_node, to_node, cost):
        self.edges.setdefault(from_node, []).append((to_node, cost))

    def finalise(self):
        """Sort each adjacency list by neighbour id.

        The assignment says that when all else is equal, smaller node ids are
        expanded first, so every algorithm can rely on neighbours being in
        ascending order.
        """
        for node_id in self.edges:
            self.edges[node_id].sort(key=lambda pair: pair[0])

    def neighbours(self, node_id):
        """Return [(neighbour, cost), ...] in ascending neighbour order."""
        return self.edges.get(node_id, [])

    def distance(self, a, b):
        """Straight-line (Euclidean) distance between two nodes."""
        (x1, y1), (x2, y2) = self.coords[a], self.coords[b]
        return math.hypot(x2 - x1, y2 - y1)

    def heuristic(self, node_id):
        """Straight-line distance to the NEAREST destination (for GBFS / A*)."""
        return min(self.distance(node_id, d) for d in self.destinations)


def load_graph(filename):
    """Read a problem file and return a Graph. Raises ValueError on bad input."""
    graph = Graph()
    section = None

    with open(filename, "r", encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            # Section headers, e.g. "Nodes:" or "Origin: 2"
            header, _, rest = line.partition(":")
            if header.strip().lower() in SECTIONS and not line.startswith("("):
                section = header.strip().lower()
                line = rest.strip()
                if not line:
                    continue

            if section == "nodes":
                match = NODE_PATTERN.match(line)
                if not match:
                    raise ValueError(f"Line {line_no}: bad node line '{line}'")
                graph.add_node(int(match.group(1)),
                               _to_number(match.group(2)),
                               _to_number(match.group(3)))

            elif section == "edges":
                match = EDGE_PATTERN.match(line)
                if not match:
                    raise ValueError(f"Line {line_no}: bad edge line '{line}'")
                graph.add_edge(int(match.group(1)), int(match.group(2)),
                               _to_number(match.group(3)))

            elif section == "origin":
                graph.origin = int(line)

            elif section == "destinations":
                for part in line.split(";"):
                    if part.strip():
                        graph.destinations.add(int(part.strip()))

            else:
                raise ValueError(f"Line {line_no}: content before any section header")

    _validate(graph)
    graph.finalise()
    return graph


def _validate(graph):
    if graph.origin is None:
        raise ValueError("No origin given")
    if not graph.destinations:
        raise ValueError("No destinations given")
    known = set(graph.coords)
    for node_id in [graph.origin, *graph.destinations]:
        if node_id not in known:
            raise ValueError(f"Node {node_id} is used but has no coordinates")
    for from_node, pairs in graph.edges.items():
        for to_node, _ in pairs:
            if from_node not in known or to_node not in known:
                raise ValueError(f"Edge ({from_node},{to_node}) uses an unknown node")
