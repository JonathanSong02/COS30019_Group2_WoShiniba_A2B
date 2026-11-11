"""
node.py
A node in the search TREE (not the graph). Several tree nodes can hold the
same graph state, reached by different paths.
"""


class Node:
    def __init__(self, state, parent=None, path_cost=0):
        self.state = state                    # graph node id
        self.parent = parent                  # Node we came from (None for root)
        self.path_cost = path_cost            # g(n): total edge cost from origin
        self.depth = 0 if parent is None else parent.depth + 1   # number of moves

    def path(self):
        """List of graph node ids from the origin to this node."""
        states = []
        node = self
        while node is not None:
            states.append(node.state)
            node = node.parent
        return list(reversed(states))

    def on_path(self, state):
        """True if `state` already appears on the path to this node.

        Used to stop tree search from looping around cycles such as 1 -> 3 -> 1.
        """
        node = self
        while node is not None:
            if node.state == state:
                return True
            node = node.parent
        return False

    def __repr__(self):
        return f"Node({self.state}, g={self.path_cost}, depth={self.depth})"
