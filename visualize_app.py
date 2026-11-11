"""
visualize_app.py
Desktop version of the map viewer (study aid, optional). Opens a window
instead of writing an HTML page.

Usage:
    python visualize_app.py                      all maps: PathFinder-test.txt,
                                                 testcases/*.txt, research_cases/*.txt
    python visualize_app.py <file> [<file> ...]  only the files given

The window loads each problem with graph.py, runs all six search methods
(plus the visit-all research solver if visit_all.py is present) and draws
the map. Pick a problem file and a method, then step through the search
with the buttons, the slider, a click on the log, or the keyboard
(Left/Right = back/next, Home/End = first/last, Space = play/pause).

How the step-through works: while a method runs, the Node class inside each
algorithm module is temporarily swapped for a subclass that logs every
search-tree node in the order it is created. The algorithms themselves are
not modified, and the log length equals the program's number_of_nodes.

Standard library only (tkinter); Python 3.8+; Windows 10/11 compatible.
"""

import glob
import math
import os
import sys
import tkinter as tk
from tkinter import ttk

from graph import load_graph
from node import Node
from algorithms import dfs, bfs, gbfs, astar, cus1, cus2

try:  # The research extension is optional.
    from visit_all import solve_visit_all
except ImportError:
    solve_visit_all = None

METHODS = [("DFS", dfs), ("BFS", bfs), ("GBFS", gbfs),
           ("AS", astar), ("CUS1", cus1), ("CUS2", cus2)]

# Upper limit on logged tree nodes per method, so huge searches
# (e.g. IDDFS on a big graph) do not make the log list enormous.
MAX_LOGGED_NODES = 5000

HERE = os.path.dirname(os.path.abspath(__file__))

METHOD_INFO = [
    ("DFS", "DFS", "Depth-first: keeps going deeper along one branch and only backs up when it hits a dead end."),
    ("BFS", "BFS", "Breadth-first: creates every node one move away, then two moves away, and so on."),
    ("GBFS", "GBFS", "Greedy best-first: always expands the node that looks closest to a destination (straight-line distance)."),
    ("AS", "AS", "A*: expands the node with the lowest cost so far plus the straight-line estimate of the cost still to go."),
    ("CUS1", "CUS1", "Iterative deepening DFS: runs depth-limited DFS again and again, raising the depth limit by one each round."),
    ("CUS2", "CUS2", "IDA*: repeated depth-first passes with a growing limit on moves so far plus estimated moves to go."),
    ("VISITALL", "Visit all goals", "Research extension: the cheapest route that passes through every destination (Dijkstra over node plus goals visited)."),
]
ABOUT = {key: about for key, _, about in METHOD_INFO}

# Colours.
PAPER = "#f6f8fa"
PANEL = "#ffffff"
GRID = "#dde4eb"
GRID_STRONG = "#b4c1ce"
INK = "#22384f"
MUTED = "#5c6f82"
FADED = "#a3b0bd"
EDGE = "#3f6b8e"
ORIGIN = "#2f8f5b"
DEST = "#c23b3b"
EXPLORED = "#d4e5f5"
MARK = "#f2b33d"
MARK_SOFT = "#f9dc9e"

FONT = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
R = 14                      # node radius in pixels
PLAY_DELAY_MS = 700


# ---------- Running the searches ----------
def run_with_log(module, graph):
    """Run module.search(graph) and log every Node it creates, in order.

    Returns a dict with the result and the creation log, where each log entry
    is [state, index_of_parent_entry] (-1 for the root).
    """
    log = []

    class RecordingNode(Node):
        def __init__(self, state, parent=None, path_cost=0):
            super().__init__(state, parent, path_cost)
            self.log_index = len(log)
            if len(log) < MAX_LOGGED_NODES:
                parent_index = -1 if parent is None else parent.log_index
                log.append([state, parent_index])

    original = module.Node
    module.Node = RecordingNode          # Swap in the logging subclass...
    try:
        goal_node, nodes_created = module.search(graph)
    finally:
        module.Node = original           # ...and always put the original back.

    result = {
        "nodesCreated": nodes_created,
        "log": log,
        "truncated": nodes_created > len(log),
        "goal": None, "path": [], "cost": None,
    }
    if goal_node is not None:
        result["goal"] = goal_node.state
        result["path"] = goal_node.path()
        result["cost"] = goal_node.path_cost
    return result


def describe_problem(filename):
    """Load one problem file and gather everything the viewer needs."""
    entry = {"file": os.path.relpath(filename, HERE).replace("\\", "/")}
    try:
        graph = load_graph(filename)
    except (OSError, ValueError) as error:
        entry["error"] = str(error)
        return entry

    entry["nodes"] = [{"id": n, "x": x, "y": y}
                      for n, (x, y) in sorted(graph.coords.items())]
    entry["edges"] = [{"from": a, "to": b, "cost": c}
                      for a in sorted(graph.edges)
                      for b, c in graph.neighbours(a)]
    entry["origin"] = graph.origin
    entry["destinations"] = sorted(graph.destinations)

    results = {}
    for name, module in METHODS:
        results[name] = run_with_log(module, graph)

    if solve_visit_all is not None:
        path, cost, created, _ = solve_visit_all(graph)
        # Its "log" is the route itself, so the viewer can walk it move by move.
        results["VISITALL"] = {"nodesCreated": created,
                               "log": [[n, i - 1] for i, n in enumerate(path or [])],
                               "truncated": False,
                               "goal": path[-1] if path else None,
                               "path": path or [], "cost": cost}
    entry["results"] = results
    return entry


def default_files():
    files = [os.path.join(HERE, "PathFinder-test.txt")]
    files += sorted(glob.glob(os.path.join(HERE, "testcases", "*.txt")))
    files += sorted(glob.glob(os.path.join(HERE, "research_cases", "*.txt")))
    return [f for f in files if os.path.isfile(f)]


# ---------- Layout: data coordinates -> screen pixels ----------
def layout(m, width, height):
    xs = [n["x"] for n in m["nodes"]]
    ys = [n["y"] for n in m["nodes"]]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    # Give tiny or single-point maps some room, then pad one unit on each side.
    if max_x - min_x < 4:
        c = (min_x + max_x) / 2
        min_x, max_x = c - 2, c + 2
    if max_y - min_y < 4:
        c = (min_y + max_y) / 2
        min_y, max_y = c - 2, c + 2
    min_x, max_x = math.floor(min_x - 1), math.ceil(max_x + 1)
    min_y, max_y = math.floor(min_y - 1), math.ceil(max_y + 1)

    margin = 40
    scale = max(1, min(150, (width - 2 * margin) / (max_x - min_x),
                       (height - 2 * margin) / (max_y - min_y)))
    # Centre the drawing in the canvas.
    ox = (width - (max_x - min_x) * scale) / 2
    oy = (height - (max_y - min_y) * scale) / 2

    def sx(x):
        return ox + (x - min_x) * scale

    def sy(y):
        return oy + (max_y - y) * scale        # y axis points up, like the brief

    # Nodes sharing a coordinate are spread on a small circle so all are visible.
    groups = {}
    for n in m["nodes"]:
        groups.setdefault((n["x"], n["y"]), []).append(n)
    pos, spread = {}, False
    for group in groups.values():
        if len(group) == 1:
            pos[group[0]["id"]] = (sx(group[0]["x"]), sy(group[0]["y"]))
            continue
        spread = True
        if len(group) == len(m["nodes"]):
            radius = min(width, height) * 0.3
        else:
            radius = max(30, (len(group) * 2.6 * R) / (2 * math.pi))
        for i, n in enumerate(group):
            a = 2 * math.pi * i / len(group) - math.pi / 2
            pos[n["id"]] = (sx(n["x"]) + radius * math.cos(a), sy(n["y"]) + radius * math.sin(a))
    return {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y,
            "sx": sx, "sy": sy, "pos": pos, "spread": spread}


# ---------- Edge geometry ----------
# One-way edge: one arrow. Two-way with equal cost: one plain line (like the brief's figure).
# Two-way with different costs: two arrows side by side, each with its own cost.
def edge_shapes(m, pos):
    lookup = {(e["from"], e["to"]): e["cost"] for e in m["edges"]}
    segments = [(e["from"], e["to"]) for e in m["edges"]]
    node_pts = list(pos.values())
    shapes, by_key, labels, aliases = [], {}, [], []

    def dist_to_segment(x, y, p, q):
        dx, dy = q[0] - p[0], q[1] - p[1]
        len2 = dx * dx + dy * dy or 1
        t = max(0, min(1, ((x - p[0]) * dx + (y - p[1]) * dy) / len2))
        return math.hypot(x - (p[0] + t * dx), y - (p[1] + t * dy))

    # Put each cost label at the middle of its edge, or slide it along the edge if that
    # spot is too close to another label, a node, or a different edge.
    def place_label(f, t, px, py, gap):
        a, b = pos[f], pos[t]
        for frac in (0.5, 0.4, 0.6, 0.3, 0.7):
            x = a[0] + (b[0] - a[0]) * frac + px * gap
            y = a[1] + (b[1] - a[1]) * frac + py * gap
            clear = (all(math.hypot(lx - x, ly - y) > 22 for lx, ly in labels) and
                     all(math.hypot(nx - x, ny - y) > R + 10 for nx, ny in node_pts) and
                     all({sf, st} == {f, t} or dist_to_segment(x, y, pos[sf], pos[st]) > 18
                         for sf, st in segments))
            if clear:
                labels.append((x, y))
                return x, y
        x = (a[0] + b[0]) / 2 + px * gap      # nothing clear: use the middle
        y = (a[1] + b[1]) / 2 + py * gap
        labels.append((x, y))
        return x, y

    for e in m["edges"]:
        f, t, cost = e["from"], e["to"], e["cost"]
        back = lookup.get((t, f))
        two_way = back is not None
        if two_way and back == cost and f > t:
            aliases.append(((f, t), (t, f)))   # already drawn as one line
            continue
        a, b = pos[f], pos[t]
        if f == t:
            shape = {"loop": True, "x": a[0], "y": a[1], "cost": cost}
        else:
            dx, dy = b[0] - a[0], b[1] - a[1]
            length = math.hypot(dx, dy) or 1
            ux, uy = dx / length, dy / length
            px, py = -uy, ux
            merged = two_way and back == cost
            off = 5 if two_way and not merged else 0
            lx, ly = place_label(f, t, px, py, off + 11)
            shape = {"loop": False,
                     "x1": a[0] + ux * R + px * off, "y1": a[1] + uy * R + py * off,
                     "x2": b[0] - ux * (R + 2) + px * off, "y2": b[1] - uy * (R + 2) + py * off,
                     "lx": lx, "ly": ly, "cost": cost, "arrow": not merged}
        shapes.append(shape)
        by_key[(f, t)] = shape
    for key, other in aliases:
        by_key[key] = by_key.get(other)
    return shapes, by_key


def fmt(value):
    return str(value)


class Viewer:
    def __init__(self, root, maps):
        self.root = root
        self.maps = maps
        self.map_index = 0
        self.method = tk.StringVar(value="BFS")
        self.step = 0
        self.timer = None
        self.log_key = None
        self.build_ui()
        self.choose(0, "BFS")

    # ---------- Window layout ----------
    def build_ui(self):
        root = self.root
        root.title("Route Finding map viewer")
        root.geometry("1240x800")
        root.minsize(860, 560)
        root.configure(bg=PAPER)

        style = ttk.Style()
        style.theme_use("clam")                    # same look on Windows and elsewhere
        style.configure(".", font=FONT)
        style.configure("TFrame", background=PAPER)
        style.configure("TLabel", background=PAPER, foreground=INK)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Panel.TLabel", background=PANEL, foreground=INK)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
        # Make the selected method button clearly stand out.
        style.configure("Toolbutton", background=PAPER, padding=(6, 4))
        style.map("Toolbutton", background=[("selected", INK), ("active", GRID)],
                  foreground=[("selected", PANEL)])

        top = ttk.Frame(root, padding=(16, 12, 16, 6))
        top.pack(fill="x")
        ttk.Label(top, text="Route Finding map viewer", font=("Segoe UI", 15, "bold")).pack(side="left")
        ttk.Label(top, text="Problem file", foreground=MUTED).pack(side="left", padx=(28, 6))
        self.combo = ttk.Combobox(top, state="readonly", width=46,
                                  values=[m["file"] for m in self.maps])
        self.combo.pack(side="left")
        self.combo.bind("<<ComboboxSelected>>", self.on_map_selected)
        # Map facts get their own line so long file names never push them off screen.
        self.facts = ttk.Label(root, foreground=MUTED, padding=(16, 0, 16, 0))
        self.facts.pack(fill="x")

        body = ttk.Frame(root, padding=(16, 4, 16, 16))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew")
        self.canvas = tk.Canvas(left, bg=PAPER, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.draw())
        self.build_legend(left)
        self.note = ttk.Label(left, foreground=MUTED)
        self.note.pack(anchor="w")

        side = ttk.Frame(body, style="Panel.TFrame", padding=14, relief="solid", borderwidth=1)
        side.grid(row=0, column=1, sticky="ns", padx=(16, 0))

        tabs = ttk.Frame(side, style="Panel.TFrame")
        tabs.pack(fill="x")
        tabs.columnconfigure((0, 1, 2, 3), uniform="tabs", weight=1)
        shown = [info for info in METHOD_INFO
                 if any(info[0] in m.get("results", {}) for m in self.maps)]
        for i, (key, label, _) in enumerate(shown):
            ttk.Radiobutton(tabs, text=label, value=key, variable=self.method,
                            style="Toolbutton",
                            command=lambda: self.choose(self.map_index, self.method.get())
                            ).grid(row=i // 4, column=i % 4, sticky="ew", padx=2, pady=2)

        self.about = ttk.Label(side, style="Muted.TLabel", wraplength=330, justify="left")
        self.about.pack(fill="x", pady=(10, 8))
        self.summary = ttk.Frame(side, style="Panel.TFrame")
        self.summary.pack(fill="x", pady=(0, 10))

        self.slider = tk.Scale(side, from_=0, to=0, orient="horizontal", showvalue=False,
                               bg=PANEL, highlightthickness=0, troughcolor=GRID,
                               activebackground=MARK, command=self.on_slide)
        self.slider.pack(fill="x")
        buttons = ttk.Frame(side, style="Panel.TFrame")
        buttons.pack(fill="x", pady=(4, 0))
        self.buttons = {}
        for name, label, action in (
                ("first", "First", lambda: self.jump(0)),
                ("prev", "Back", lambda: self.jump(self.step - 1)),
                ("play", "Play", self.toggle_play),
                ("next", "Next", lambda: self.jump(self.step + 1)),
                ("last", "Last", lambda: self.jump(10 ** 9))):
            b = ttk.Button(buttons, text=label, width=7, command=action)
            b.pack(side="left", padx=(0, 4))
            self.buttons[name] = b

        self.step_text = ttk.Label(side, style="Panel.TLabel", wraplength=330, justify="left")
        self.step_text.pack(fill="x", pady=(8, 8))

        log_frame = ttk.Frame(side, style="Panel.TFrame")
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Listbox(log_frame, font=("Consolas", 10), activestyle="none",
                              exportselection=False, borderwidth=1, relief="solid",
                              highlightthickness=0, fg=INK, bg=PANEL,
                              selectbackground=MARK_SOFT, selectforeground=INK)
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.log.bind("<<ListboxSelect>>", self.on_log_click)

        # Keyboard shortcuts. Widgets that use these keys themselves are skipped.
        for key, action in (("<Right>", lambda: self.jump(self.step + 1)),
                            ("<Left>", lambda: self.jump(self.step - 1)),
                            ("<Home>", lambda: self.jump(0)),
                            ("<End>", lambda: self.jump(10 ** 9))):
            root.bind(key, lambda e, a=action: self.on_key(e, a, ("TCombobox", "Listbox", "Scale")))
        root.bind("<space>", lambda e: self.on_key(
            e, self.toggle_play, ("TCombobox", "Listbox", "Scale", "TButton", "TRadiobutton")))

    def build_legend(self, parent):
        legend = ttk.Frame(parent)
        legend.pack(fill="x", pady=(6, 2))

        def item(col, row, text, draw):
            cell = ttk.Frame(legend)
            cell.grid(row=row, column=col, sticky="w", padx=(0, 18))
            c = tk.Canvas(cell, width=32, height=22, bg=PAPER, highlightthickness=0)
            c.pack(side="left")
            draw(c)
            ttk.Label(cell, text=text, foreground=MUTED).pack(side="left")

        # Two columns, three rows, so every item fits next to the map.
        item(0, 0, "Origin", lambda c: c.create_oval(9, 4, 23, 18, fill=ORIGIN, outline=""))
        item(1, 0, "Destination", lambda c: (
            c.create_oval(10, 5, 22, 17, fill=PANEL, outline=INK),
            c.create_oval(7, 2, 25, 20, outline=DEST, width=2)))
        item(0, 1, "One-way edge", lambda c: c.create_line(
            3, 11, 29, 11, fill=EDGE, width=1.6, arrow="last", arrowshape=(8, 10, 3)))
        item(1, 1, "Two-way edge, same cost", lambda c: c.create_line(3, 11, 29, 11, fill=EDGE, width=1.6))
        item(0, 2, "Created or visited (×n = more than once)",
             lambda c: c.create_oval(9, 4, 23, 18, fill=EXPLORED, outline=INK))
        item(1, 2, "Path returned", lambda c: c.create_line(
            5, 11, 27, 11, fill=MARK_SOFT, width=10, capstyle="round"))

    # ---------- State ----------
    def current_map(self):
        return self.maps[self.map_index]

    def result(self):
        return self.current_map().get("results", {}).get(self.method.get())

    def choose(self, map_index, method):
        self.stop_play()
        self.map_index = map_index
        self.method.set(method)
        self.combo.current(map_index)
        res = self.result()
        self.step = len(res["log"]) if res else 0     # open on the finished search
        self.render()

    def set_step(self, step):
        res = self.result()
        n = len(res["log"]) if res else 0
        self.step = max(0, min(n, step))
        if self.step == n:
            self.stop_play()
        self.render()

    def jump(self, step):
        self.stop_play()
        self.set_step(step)

    def stop_play(self):
        if self.timer is not None:
            self.root.after_cancel(self.timer)
            self.timer = None

    def toggle_play(self):
        if self.timer is not None:
            self.stop_play()
            self.render()
            return
        res = self.result()
        if not res or not res["log"]:
            return
        if self.step >= len(res["log"]):
            self.step = 0
        self.timer = self.root.after(PLAY_DELAY_MS, self.tick)
        self.render()

    def tick(self):
        self.timer = self.root.after(PLAY_DELAY_MS, self.tick)
        self.set_step(self.step + 1)               # stops the timer at the last step

    # ---------- Events ----------
    def on_map_selected(self, _event):
        self.choose(self.combo.current(), self.method.get())
        self.root.focus_set()                      # so the arrow keys step again

    def on_slide(self, value):
        step = int(float(value))
        if step != self.step:
            self.jump(step)

    def on_log_click(self, _event):
        selection = self.log.curselection()
        self.log.selection_clear(0, "end")
        if selection:
            self.jump(selection[0] + 1)

    def on_key(self, event, action, skip_classes):
        try:
            if event.widget.winfo_class() in skip_classes:
                return
        except AttributeError:
            return
        action()

    # ---------- Drawing ----------
    def render(self):
        self.draw()
        self.update_panel()

    def draw(self):
        cv = self.canvas
        cv.delete("all")
        width, height = cv.winfo_width(), cv.winfo_height()
        if width < 50 or height < 50:
            return                                 # window not laid out yet
        m = self.current_map()
        if "error" in m:
            cv.create_text(20, 40, anchor="w", fill=DEST, font=FONT,
                           text="Could not load this file: " + m["error"])
            self.note.config(text="")
            return
        res = self.result()
        if res is None:
            return
        L = layout(m, width, height)
        pos = L["pos"]
        shapes, by_key = edge_shapes(m, pos)
        sx, sy = L["sx"], L["sy"]

        # Graph paper: grid lines every unit (or every few units on big maps).
        step_u = 1
        while (L["max_x"] - L["min_x"]) / step_u > 24 or (L["max_y"] - L["min_y"]) / step_u > 24:
            step_u *= 2
        x = math.ceil(L["min_x"] / step_u) * step_u
        while x <= L["max_x"]:
            cv.create_line(sx(x), sy(L["max_y"]), sx(x), sy(L["min_y"]),
                           fill=GRID_STRONG if x == 0 else GRID, width=1.5 if x == 0 else 1)
            cv.create_text(sx(x), sy(L["min_y"]) + 14, text=str(x), fill=MUTED, font=("Segoe UI", 8))
            x += step_u
        y = math.ceil(L["min_y"] / step_u) * step_u
        while y <= L["max_y"]:
            cv.create_line(sx(L["min_x"]), sy(y), sx(L["max_x"]), sy(y),
                           fill=GRID_STRONG if y == 0 else GRID, width=1.5 if y == 0 else 1)
            cv.create_text(sx(L["min_x"]) - 8, sy(y), anchor="e", text=str(y), fill=MUTED,
                           font=("Segoe UI", 8))
            y += step_u

        # What the selected method has done up to the current step.
        log = res["log"]
        at_end = self.step >= len(log)
        created = {}
        for state, _ in log[:self.step]:
            created[state] = created.get(state, 0) + 1
        now = log[self.step - 1] if self.step > 0 and log else None
        now_parent = log[now[1]][0] if now and now[1] >= 0 else None
        # Search methods show the returned path once finished; visit-all draws its
        # route move by move, because each step there is a move along the route.
        walk = self.method.get() == "VISITALL"
        if walk:
            path_moves = max(0, self.step - 1)
        else:
            path_moves = len(res["path"]) - 1 if at_end else 0

        # Highlighter under the solution path.
        for i in range(max(0, path_moves)):
            s = by_key.get((res["path"][i], res["path"][i + 1]))
            if s and not s["loop"]:
                cv.create_line(s["x1"], s["y1"], s["x2"], s["y2"], fill=MARK_SOFT,
                               width=12, capstyle="round")

        # Edges and their costs.
        now_edge = by_key.get((now_parent, now[0])) if now and now_parent is not None else None
        for s in shapes:
            active = s is now_edge and not at_end
            colour = MARK if active else EDGE
            if s["loop"]:
                x0, y0 = s["x"], s["y"]
                cv.create_line(x0 - 6, y0 - R, x0 - 30, y0 - 55, x0 + 30, y0 - 55, x0 + 6, y0 - R,
                               smooth="raw", fill=colour, width=1.6, arrow="last",
                               arrowshape=(9, 11, 4))
                self.cost_label(x0, y0 - 52, s["cost"])
                continue
            cv.create_line(s["x1"], s["y1"], s["x2"], s["y2"], fill=colour,
                           width=3.5 if active else 1.6,
                           arrow="last" if s["arrow"] else "none", arrowshape=(10, 12, 4))
            self.cost_label(s["lx"], s["ly"], s["cost"])

        # Nodes.
        for n in m["nodes"]:
            px, py = pos[n["id"]]
            is_origin = n["id"] == m["origin"]
            is_dest = n["id"] in m["destinations"]
            is_now = now is not None and now[0] == n["id"] and not at_end
            if is_dest:
                cv.create_oval(px - R - 4, py - R - 4, px + R + 4, py + R + 4, outline=DEST, width=2.5)
            fill = ORIGIN if is_origin else EXPLORED if created.get(n["id"]) else PANEL
            cv.create_oval(px - R, py - R, px + R, py + R, fill=fill,
                           outline=MARK if is_now else INK, width=4 if is_now else 1.5)
            cv.create_text(px, py, text=str(n["id"]), font=FONT_BOLD,
                           fill=PANEL if is_origin else INK)
            if created.get(n["id"], 0) > 1:
                cv.create_text(px + R + 4, py - R + 2, anchor="w", font=("Segoe UI", 8),
                               fill=MUTED, text="×" + str(created[n["id"]]))

        self.note.config(text="Nodes sharing coordinates are spread out slightly so all are visible."
                         if L["spread"] else "")

    def cost_label(self, x, y, cost):
        """Edge cost with a paper-coloured backing so lines do not cross the digits."""
        cv = self.canvas
        text = cv.create_text(x, y, text=fmt(cost), font=("Segoe UI", 9, "bold"), fill=INK)
        x1, y1, x2, y2 = cv.bbox(text)
        back = cv.create_rectangle(x1 - 2, y1, x2 + 2, y2, fill=PAPER, outline="")
        cv.tag_lower(back, text)

    # ---------- Side panel ----------
    def update_panel(self):
        m = self.current_map()
        method = self.method.get()
        if "error" in m:
            self.facts.config(text="")
        else:
            self.facts.config(text=f"Origin {m['origin']}, destinations "
                                   f"{', '.join(map(str, m['destinations']))}, "
                                   f"{len(m['nodes'])} nodes, {len(m['edges'])} directed edges")

        for widget in self.summary.winfo_children():
            widget.destroy()
        res = self.result()
        if res is None:
            self.about.config(text="")
            self.step_text.config(text="")
            self.log.delete(0, "end")
            self.log_key = None
            self.set_controls_enabled(False)
            return

        walk = method == "VISITALL"
        self.about.config(text=ABOUT[method])
        created_label = "Queue entries" if walk else "Nodes created"
        if res["goal"] is None:
            rows = [("Result" if walk else "Goal", "No route found"),
                    (created_label, res["nodesCreated"])]
        else:
            rows = [("Ends at" if walk else "Goal", res["goal"]),
                    (created_label, res["nodesCreated"]),
                    ("Path", " → ".join(map(str, res["path"]))),
                    ("Moves", len(res["path"]) - 1),
                    ("Path cost", fmt(res["cost"]))]
        for i, (key, value) in enumerate(rows):
            ttk.Label(self.summary, text=key, style="Muted.TLabel").grid(
                row=i, column=0, sticky="nw", padx=(0, 14))
            ttk.Label(self.summary, text=str(value), style="Panel.TLabel", font=FONT_BOLD,
                      wraplength=240, justify="left").grid(row=i, column=1, sticky="w")

        log = res["log"]
        self.set_controls_enabled(bool(log))
        self.slider.config(to=len(log))
        self.slider.set(self.step)
        self.buttons["play"].config(text="Pause" if self.timer is not None else "Play")
        self.step_text.config(text=self.describe_step(m, res, walk))

        # Rebuild the log list only when the method or map changes.
        key = (self.map_index, method)
        if self.log_key != key:
            self.log_key = key
            self.log.delete(0, "end")
            for i, (state, parent) in enumerate(log):
                origin = " (start)" if parent < 0 else f" from {log[parent][0]}"
                self.log.insert("end", f"{i + 1:>5}   {state}{origin}")
        for i in range(len(log)):
            if i + 1 == self.step:
                self.log.itemconfig(i, background=MARK_SOFT, foreground=INK)
            else:
                self.log.itemconfig(i, background=PANEL, foreground=INK if i < self.step else FADED)
        if self.step > 0:
            self.log.see(self.step - 1)

    def set_controls_enabled(self, enabled):
        for b in self.buttons.values():
            b.state(["!disabled"] if enabled else ["disabled"])
        self.slider.config(state="normal" if enabled else "disabled")

    def describe_step(self, m, res, walk):
        log = res["log"]
        n = len(log)
        if not n:
            return "No route passes through every destination, so there is nothing to step through."
        if self.step == 0:
            text = ("Press Next or Play to walk the route one move at a time." if walk
                    else f"Press Next or Play to watch {n} search-tree nodes being created.")
        elif walk:
            path = res["path"]
            costs = {}
            for e in m["edges"]:
                k = (e["from"], e["to"])
                costs[k] = min(costs.get(k, float("inf")), e["cost"])
            total = sum(costs[(path[i - 1], path[i])] for i in range(1, self.step))
            if self.step == 1:
                text = f"Start at {path[0]}."
            else:
                text = (f"Move {self.step - 1} of {len(path) - 1}: {path[self.step - 2]} → "
                        f"{path[self.step - 1]}, cost so far {fmt(total)}.")
            if self.step == n:
                text += " Every destination has been visited."
        else:
            state, parent = log[self.step - 1]
            text = f"Node {self.step} of {n}: created {state}" + (
                " (the start)" if parent < 0 else f" from {log[parent][0]}") + "."
            if self.step == n:
                text += (" Search finished without reaching a destination." if res["goal"] is None
                         else " Search finished; the highlighted route is the path returned.")
        if res["truncated"]:
            text += f" Only the first {n} of {res['nodesCreated']} nodes are shown."
        return text


def main(argv):
    files = argv[1:] or default_files()
    if not files:
        print("No problem files found. Usage: python visualize_app.py [<file> ...]")
        return 1

    maps = []
    for filename in files:
        if not os.path.isfile(filename):
            print(f"File not found: {filename}")
            continue
        maps.append(describe_problem(filename))
        print(f"  {maps[-1]['file']}: {maps[-1].get('error', 'ok')}")
    if not maps:
        return 1

    root = tk.Tk()
    Viewer(root, maps)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
