"""
viz_tikz_fhypergraph.py
=======================
Generate a TikZ F-hypergraph diagram from real SCOTUS G.pkl data.

Layout: pure-Python Reingold-Tilford style hierarchical tree layout.
        No graphviz, no external layout tool.
Style:  Gallo, Longo, Pallottino & Nguyen (1993) -- curved hyperarcs
        fan out from a filled connector node to head nodes using
        TikZ Bezier curves (bend left / bend right).

F-arc anatomy (matching Fig. 2(b) of Gallo et al.):
  tail node  --straight-->  connector disc  --curved-->  head nodes

Requires only: Python stdlib + pickle + networkx (for tree traversal only).
The .tex output compiles with: pdflatex scotus_timeline.tex
"""

import argparse
import os
import pickle
import sys
from datetime import datetime
from collections import defaultdict

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)


def processed_dir(dataset):
    return os.path.join(BASE_DIR, "data", "processed", dataset)


def to_int(k):
    try: return int(k)
    except: return k


def parse_tau(s):
    if not s: return None
    try: return datetime.strptime(str(s)[:10], "%Y-%m-%d")
    except: return None


def load_data(dataset):
    pdir = processed_dir(dataset)
    with open(os.path.join(pdir, "G.pkl"), "rb") as f:
        G = pickle.load(f)
    meta_path = os.path.join(pdir, "opinions_meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            om = pickle.load(f)
    else:
        om = {}
    return G, om


def select_subgraph(G, om, max_nodes, max_heads):
    """Select the Salerno chain (7 F-arcs, 20 nodes, 1890-1992)."""
    G_int = {to_int(k): v for k, v in G.items()}

    tau_map = {}
    for oid, m in om.items():
        name = m.get("case_name", "")
        if name and name != "?":
            t = parse_tau(m.get("tau", ""))
            if t: tau_map[to_int(oid)] = t
    for u, meta in G_int.items():
        if u not in tau_map:
            name = meta.get("case_name", "")
            if name and name != "?":
                t = parse_tau(meta.get("tau", ""))
                if t: tau_map[u] = t

    arcs = {}
    for u, meta in G_int.items():
        if u not in tau_map: continue
        F = [to_int(v) for v in meta.get("F", frozenset()) if to_int(v) in tau_map]
        if 2 <= len(F) <= max_heads:
            arcs[u] = frozenset(F)

    start = 112772
    visited = set(); result = {}; queue = [start]; all_nodes = set()
    while queue and len(all_nodes) < max_nodes:
        u = queue.pop(0)
        if u in visited or u not in arcs: continue
        visited.add(u)
        F = arcs[u]
        new = ({u} | F) - all_nodes
        if len(all_nodes) + len(new) > max_nodes:
            F_trim = frozenset(list(F)[:max_nodes - len(all_nodes) - 1])
            if not F_trim: continue
            F = F_trim
        result[u] = F; all_nodes |= {u} | F
        for v in sorted(F, key=lambda x: tau_map.get(x, datetime.min)):
            if v in arcs and v not in visited: queue.append(v)

    dates = [tau_map[n] for n in all_nodes]
    print(f"[viz] Subgraph: {len(result)} F-arcs, {len(all_nodes)} nodes")
    print(f"[viz] Date range: {min(dates).date()} - {max(dates).date()}")
    return result, tau_map, all_nodes


# ---------------------------------------------------------------------------
# Pure-Python hierarchical layout (Reingold-Tilford inspired)
# ---------------------------------------------------------------------------
def build_tree(chosen_arcs, all_nodes):
    """
    Build a rooted tree from the F-arc chain.
    Root = the tail node that is not a head of any other arc.
    Children of a node = its head nodes (in the F-arc where it is tail).
    Shared nodes (appearing as both tail and head) are handled by treating
    the first occurrence as the canonical position.
    """
    # Find root: a tail that is not a head of any arc
    all_heads = set()
    for F in chosen_arcs.values():
        all_heads |= F
    roots = [u for u in chosen_arcs if u not in all_heads]
    if not roots:
        roots = [next(iter(chosen_arcs))]
    root = roots[0]

    # Build children map (tree, no revisits)
    children = defaultdict(list)
    placed = set()

    def dfs(u):
        placed.add(u)
        if u in chosen_arcs:
            for v in sorted(chosen_arcs[u], key=lambda x: x):
                if v not in placed:
                    children[u].append(v)
                    dfs(v)
    dfs(root)

    # Any remaining nodes (isolated or cycle-broken) attach to root
    for n in all_nodes:
        if n not in placed:
            children[root].append(n)
            placed.add(n)

    return root, children


def assign_positions(root, children, all_nodes, x_sep=2.2, y_sep=2.0):
    """
    Assign (x, y) positions: depth on the Y axis (root at top, leaves at bottom),
    sibling spread on the X axis.  This gives a compact top-down tree matching
    the style of Gallo et al. Fig. 1.
    """
    # Assign x by in-order traversal (leaves get integer x slots)
    leaf_counter = [0]
    leaf_x = {}
    def assign_leaf_x(u):
        if not children[u]:
            leaf_x[u] = leaf_counter[0]
            leaf_counter[0] += 1
        else:
            for c in children[u]:
                assign_leaf_x(c)
    assign_leaf_x(root)

    # Internal node x = mean of children x
    node_x = {}
    def assign_x(u):
        if not children[u]:
            node_x[u] = leaf_x[u]
        else:
            for c in children[u]:
                assign_x(c)
            node_x[u] = sum(node_x[c] for c in children[u]) / len(children[u])
    assign_x(root)

    # Assign y by depth (root at top = max_depth, leaves at 0)
    node_depth = {}
    def assign_depth(u, d=0):
        node_depth[u] = d
        for c in children[u]:
            assign_depth(c, d + 1)
    assign_depth(root)
    max_depth = max(node_depth.values()) if node_depth else 1

    # Scale to cm: x spreads horizontally, y goes downward (root at top)
    pos = {}
    for n in all_nodes:
        if n in node_x and n in node_depth:
            px = node_x[n] * x_sep
            py = (max_depth - node_depth[n]) * y_sep  # root at top
            pos[n] = (px, py)
        else:
            pos[n] = (0.0, 0.0)

    return pos


def short_label(case_name, max_words=2):
    if not case_name: return ""
    for sep in [" v.", " V.", " vs."]:
        if sep in case_name:
            case_name = case_name.split(sep)[0].strip()
            break
    words = case_name.split()
    label = " ".join(words[:max_words])
    for ch in ["&", "%", "$", "#", "_", "{", "}", "~", "^", "\\"]:
        label = label.replace(ch, "\\" + ch)
    return label


def generate_tikz(chosen_arcs, om, tau_map, all_nodes):
    root, children = build_tree(chosen_arcs, all_nodes)
    pos = assign_positions(root, children, all_nodes, x_sep=1.8, y_sep=2.4)

    node_ids = {n: f"n{i}" for i, n in enumerate(sorted(all_nodes))}
    citing_set = set(chosen_arcs.keys())

    L = []
    L.append(r"\documentclass[tikz, border=14pt]{standalone}")
    L.append(r"\usepackage{tikz}")
    L.append(r"\usetikzlibrary{arrows.meta, bending}")
    L.append(r"\begin{document}")
    L.append(r"\begin{tikzpicture}[")
    L.append(r"  opinion/.style={circle, draw=black, fill=white,")
    L.append(r"    line width=0.7pt, minimum size=18pt, inner sep=0pt},")
    L.append(r"  citing/.style={circle, draw=black, fill=black!12,")
    L.append(r"    line width=0.9pt, minimum size=18pt, inner sep=0pt},")
    L.append(r"  conn/.style={circle, fill=black, minimum size=6pt, inner sep=0pt},")
    L.append(r"  tail/.style={->, >=Stealth, line width=0.7pt,")
    L.append(r"    shorten >=2pt, shorten <=2pt},")
    L.append(r"  lbl/.style={font=\tiny, text=black!70, align=center, text width=1.7cm}")
    L.append(r"]")
    L.append("")

    # Opinion nodes
    L.append("% Opinion nodes")
    for i, n in enumerate(sorted(all_nodes)):
        xp, yp = pos[n]
        style = "citing" if n in citing_set else "opinion"
        tid = node_ids[n]
        L.append(f"\\node[{style}] ({tid}) at ({xp:.3f}cm,{yp:.3f}cm) {{}};")
        name = om.get(str(n), om.get(n, {})).get("case_name", "")
        label = short_label(name)
        yr = tau_map[n].year if n in tau_map else ""
        # Year inside the node circle
        L.append(f"\\node[font=\\tiny, text=black!50] at ({xp:.3f}cm,{yp:.3f}cm) {{{yr}}};")
        if label:
            # Label alternates above/below based on sorted index
            offset = 0.70 if i % 2 == 0 else -0.70
            L.append(f"\\node[lbl] at ({xp:.3f}cm,{yp+offset:.3f}cm) {{{label}}};")
    L.append("")

    # F-arcs: tail -> connector (straight) -> heads (curved Bezier)
    L.append("% F-arcs")
    for i, (u, F) in enumerate(chosen_arcs.items()):
        if u not in pos: continue
        ux, uy = pos[u]

        head_list = [v for v in F if v in pos]
        if not head_list: continue

        # Connector placed at midpoint between tail and centroid of heads
        hx_mean = sum(pos[v][0] for v in head_list) / len(head_list)
        hy_mean = sum(pos[v][1] for v in head_list) / len(head_list)
        cx = (ux + hx_mean) / 2.0
        cy = (uy + hy_mean) / 2.0

        cid = f"c{i}"
        L.append(f"% E_{i+1}  ({u})")
        L.append(f"\\node[conn] ({cid}) at ({cx:.3f}cm,{cy:.3f}cm) {{}};")
        # Straight arrow: tail -> connector
        L.append(f"\\draw[tail] ({node_ids[u]}) -- ({cid});")

        # Curved arrows: connector -> each head
        # Sort heads by y-position so top/bottom bends are consistent
        head_list_sorted = sorted(head_list, key=lambda v: pos[v][1])
        n_heads = len(head_list_sorted)
        mid_idx = n_heads // 2
        for j, v in enumerate(head_list_sorted):
            dy = pos[v][1] - cy
            if n_heads == 1 or (n_heads % 2 == 1 and j == mid_idx):
                # Middle or sole head: straight
                L.append(f"\\draw[tail] ({cid}) -- ({node_ids[v]});")
            elif dy > 0.2:
                bend = min(max(int(abs(dy) * 7), 15), 40)
                L.append(
                    f"\\draw[->, >=Stealth, line width=0.7pt, "
                    f"shorten >=2pt, shorten <=1pt, "
                    f"bend left={bend}] ({cid}) to ({node_ids[v]});"
                )
            elif dy < -0.2:
                bend = min(max(int(abs(dy) * 7), 15), 40)
                L.append(
                    f"\\draw[->, >=Stealth, line width=0.7pt, "
                    f"shorten >=2pt, shorten <=1pt, "
                    f"bend right={bend}] ({cid}) to ({node_ids[v]});"
                )
            else:
                L.append(f"\\draw[tail] ({cid}) -- ({node_ids[v]});")

        # Arc label
        L.append(
            f"\\node[font=\\scriptsize, text=black!50] at "
            f"({cx+0.05:.3f}cm,{cy+0.32:.3f}cm) {{$E_{{{i+1}}}$}};"
        )
        L.append("")

    L.append(r"\end{tikzpicture}")
    L.append(r"\end{document}")
    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scotus")
    parser.add_argument("--max-nodes", type=int, default=20)
    parser.add_argument("--max-heads", type=int, default=4)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    G, om = load_data(args.dataset)
    chosen_arcs, tau_map, all_nodes = select_subgraph(
        G, om, args.max_nodes, args.max_heads
    )
    tikz = generate_tikz(chosen_arcs, om, tau_map, all_nodes)

    if args.output is None:
        out_dir = os.path.join(BASE_DIR, "data", "viz")
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir, f"{args.dataset}_timeline.tex")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(tikz)

    print(f"[viz] Written to {args.output}")
    print(f"[viz] Compile with: pdflatex {args.output}")


if __name__ == "__main__":
    main()
