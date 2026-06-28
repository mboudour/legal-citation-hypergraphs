"""
viz_fhypergraph.py  –  SCOTUS F-hypergraph, Gallo et al. (1993) Fig. 4 style.
Uses pygraphviz dot layout + matplotlib for drawing.

The subgraph is hardcoded (10 opinion nodes + 3 F-arc nodes, post-2000).

── Tweaking positions ───────────────────────────────────────────────────────
Run once with --print-pos to print the computed pos dict to stdout:
    python viz_fhypergraph.py --print-pos

Copy the printed dict into POS_OVERRIDE below and edit any coordinates you
want to move.  When POS_OVERRIDE is non-empty it takes priority over the
automatic dot layout.

── Tweaking curves ──────────────────────────────────────────────────────────
Edit BEND_IN / BEND_BRANCH dicts below.  Positive = curves left, negative right.

Usage:
    python viz_fhypergraph.py [--output fhypergraph_scotus.pdf] [--print-pos]
"""

import argparse, math, os, pprint
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

# ── Hardcoded subgraph ────────────────────────────────────────────────────────

ARCS = [
    {"id": "E1", "tau": "2016-01-25", "tail": "james",
     "heads": ["hughes", "christian", "rivers"]},
    {"id": "E2", "tau": "1980-11-10", "tail": "hughes",
     "heads": ["haines", "conley", "wolff", "estelle", "carey", "bell"]},
    {"id": "E3", "tau": "1972-02-22", "tail": "haines",
     "heads": ["conley"]},
]

LABELS = {
    "james":     "James v.\nCity of Boise",
    "hughes":    "Hughes v.\nRowe",
    "rivers":    "Rivers v.\nRoadway Express",
    "christian": "Christiansburg\nv. EEOC",
    "haines":    "Haines v.\nKerner",
    "conley":    "Conley v.\nGibson",
    "wolff":     "Wolff v.\nMcDonnell",
    "estelle":   "Estelle v.\nGamble",
    "carey":     "Carey v.\nPiphus",
    "bell":      "Bell v.\nWolfish",
}

DATES = {
    "james":     "2016-01-25", "hughes":    "1980-11-10",
    "rivers":    "1994-04-26", "christian": "1978-01-23",
    "haines":    "1972-02-22", "conley":    "1957-11-18",
    "wolff":     "1974-06-26", "estelle":   "1976-11-30",
    "carey":     "1978-03-21", "bell":      "1979-05-14",
}

TAIL_NODES = {"james", "hughes", "haines"}

# ── Per-arc bend overrides (tweak these to reshape curves) ───────────────────
# Positive = curves left, negative = curves right
BEND_IN = {
    "E1":  0.30,   # incoming arc james → C_E1
    "E2": -0.28,   # incoming arc hughes → C_E2
    "E3":  0.25,   # incoming arc haines → C_E3
}
BEND_BRANCH = {
    # (arc_id, head_id) -> bend value; default alternates ±0.22
    ("E1", "hughes"):    -0.20,
    ("E1", "christian"):  0.00,
    ("E1", "rivers"):     0.25,
    ("E2", "haines"):    -0.30,
    ("E2", "conley"):    -0.40,
    ("E2", "wolff"):     -0.20,
    ("E2", "estelle"):    0.00,
    ("E2", "carey"):      0.20,
    ("E2", "bell"):       0.35,
    ("E3", "conley"):     0.20,
}

# ── Position overrides ────────────────────────────────────────────────────────
# Leave empty ({}) to use automatic dot layout.
# Run with --print-pos to get the auto-computed dict, then paste and edit here.
# F-arc connector nodes are keyed "C_E1", "C_E2", "C_E3".
POS_OVERRIDE = {
    "conley": np.array([1.5000, 8.5000])
    # Example (paste output of --print-pos here and edit):
    # "james":     np.array([4.00, 0.00]),
    # "C_E1":      np.array([3.50, 1.80]),
    # ...
}

# ── Layout via pygraphviz dot ─────────────────────────────────────────────────

def get_layout():
    """Compute positions for opinion nodes; connector nodes are derived later."""
    try:
        import pygraphviz as pgv
        A = pgv.AGraph(directed=True, strict=False)
        A.graph_attr.update(rankdir="TB", nodesep="1.4", ranksep="2.0",
                            splines="none")
        A.node_attr.update(shape="circle", width="0.5", fixedsize="true")
        for n in LABELS:
            A.add_node(n)
        for arc in ARCS:
            for h in arc["heads"]:
                A.add_edge(arc["tail"], h)
        A.layout(prog="dot")
        pos = {}
        for n in LABELS:
            xy = A.get_node(n).attr["pos"].split(",")
            pos[n] = np.array([float(xy[0]), float(xy[1])])
        # Flip y so root is at top
        max_y = max(v[1] for v in pos.values())
        pos = {n: np.array([x, max_y - y]) for n, (x, y) in pos.items()}
        # Normalise to roughly unit scale
        span = max(max(v[0] for v in pos.values()) - min(v[0] for v in pos.values()),
                   max(v[1] for v in pos.values()) - min(v[1] for v in pos.values()))
        if span > 0:
            pos = {n: v / span * 8.0 for n, v in pos.items()}
        return pos
    except Exception as e:
        print(f"pygraphviz layout failed ({e}), using manual positions")
        return {
            "james":     np.array([4.0, 0.0]),
            "hughes":    np.array([2.5, 2.5]),
            "rivers":    np.array([6.5, 2.5]),
            "christian": np.array([4.5, 2.5]),
            "haines":    np.array([0.5, 5.0]),
            "conley":    np.array([0.5, 7.5]),
            "wolff":     np.array([2.0, 6.5]),
            "estelle":   np.array([3.5, 7.0]),
            "carey":     np.array([5.0, 6.5]),
            "bell":      np.array([6.5, 5.5]),
        }


def compute_connector_positions(pos):
    """Derive connector node positions from opinion node positions."""
    conn_pos = {}
    for arc in ARCS:
        aid   = arc["id"]
        tail  = arc["tail"]
        heads = arc["heads"]
        src   = pos[tail]
        head_pos  = np.array([pos[h] for h in heads])
        centroid  = head_pos.mean(axis=0)
        C = src + 0.62 * (centroid - src)
        if np.linalg.norm(C - src) < 3.5 * NODE_R:
            d  = centroid - src
            dn = np.linalg.norm(d)
            C  = src + (3.5 * NODE_R) * d / dn if dn > 1e-9 else src
        conn_pos[f"C_{aid}"] = C
    return conn_pos

# ── Bézier helpers ────────────────────────────────────────────────────────────

def qbez(p0, cp, p2, t):
    return (1-t)**2 * p0 + 2*(1-t)*t * cp + t**2 * p2

def qtan(p0, cp, p2, t):
    return 2*(1-t)*(cp - p0) + 2*t*(p2 - cp)

def ctrl(a, b, bend):
    mid  = (a + b) / 2.0
    d    = b - a
    perp = np.array([-d[1], d[0]])
    n    = np.linalg.norm(perp)
    return mid + (bend * np.linalg.norm(d) * perp / n) if n > 1e-9 else mid

def curve_pts(p0, p2, bend, r0, r2, n=120):
    """Return array of points for a Bézier clipped at node boundaries."""
    cp  = ctrl(p0, p2, bend)
    ts  = np.linspace(0, 1, n)
    pts = np.array([qbez(p0, cp, p2, t) for t in ts])
    i0  = next((i for i, p in enumerate(pts) if np.linalg.norm(p - p0) >= r0), 0)
    i1  = next((i for i, p in enumerate(reversed(pts))
                if np.linalg.norm(p - p2) >= r2), 0)
    i1  = len(pts) - 1 - i1
    return pts[i0:i1+1], cp

# ── Sizes ─────────────────────────────────────────────────────────────────────

NODE_R = 0.18   # all node circles same radius (data units)
CONN_R = 0.18   # F-arc connector node radius (same as opinion nodes)

# ── Main drawing ──────────────────────────────────────────────────────────────

def draw(pos, conn_pos, output):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_aspect('equal')
    ax.axis('off')

    # ═══════════════════════════════════════════════════════════════════════════
    # PASS 1 – Draw all nodes FIRST at high zorder so they cover arrowhead tips
    # ═══════════════════════════════════════════════════════════════════════════

    # F-arc connector nodes (filled black circles)
    for arc in ARCS:
        aid = arc["id"]
        tau = arc["tau"]
        C   = conn_pos[f"C_{aid}"]
        ax.add_patch(plt.Circle(C, CONN_R, color='black', zorder=10))
        ax.text(C[0], C[1] + CONN_R + 0.12,
                f"Decision F-arc\n({tau})",
                fontsize=7.0, ha='center', va='bottom',
                fontfamily='serif', fontweight='bold',
                multialignment='center', zorder=12,
                bbox=dict(boxstyle='round,pad=0.10', fc='lightyellow',
                          ec='#999999', lw=0.5, alpha=0.97))

    # Opinion nodes
    for node, p in pos.items():
        is_tail = node in TAIL_NODES
        ax.add_patch(plt.Circle(p, NODE_R,
                                color='#cccccc' if is_tail else 'white',
                                ec='black', lw=2.2 if is_tail else 1.0,
                                zorder=10))
        ax.text(p[0], p[1] + NODE_R + 0.10, LABELS[node],
                fontsize=7.5, ha='center', va='bottom',
                fontstyle='italic', fontfamily='serif',
                multialignment='center', zorder=12,
                bbox=dict(boxstyle='round,pad=0.08', fc='white',
                          ec='none', alpha=0.9))
        ax.text(p[0], p[1] - NODE_R - 0.08, DATES[node],
                fontsize=6.5, ha='center', va='top',
                color='#555555', fontfamily='serif', zorder=12)

    # ═══════════════════════════════════════════════════════════════════════════
    # PASS 2 – Draw edges and arrowheads BELOW nodes (lower zorder)
    # Arrowheads are FancyArrow patches (zorder=3) so nodes (zorder=10) cover them
    # ═══════════════════════════════════════════════════════════════════════════

    def add_arrowhead(tip, direction, hw, hl, lw_edge, zo):
        """Draw a filled triangle arrowhead at `tip` pointing in `direction`."""
        d = direction / (np.linalg.norm(direction) + 1e-9)
        # Start the arrow shaft just behind the tip so the patch covers it
        start = tip - d * hl
        ax.add_patch(FancyArrow(
            start[0], start[1], d[0] * hl, d[1] * hl,
            width=lw_edge * 0.01,          # shaft width (nearly invisible)
            head_width=hw,
            head_length=hl,
            length_includes_head=True,
            color='black',
            zorder=zo
        ))

    for arc in ARCS:
        aid   = arc["id"]
        tail  = arc["tail"]
        heads = arc["heads"]

        src = pos[tail]
        C   = conn_pos[f"C_{aid}"]

        # Incoming arc: tail → C (curve stops at node boundary)
        pts_in, cp_in = curve_pts(src, C, BEND_IN[aid], NODE_R, CONN_R)
        ax.plot(pts_in[:, 0], pts_in[:, 1], color='black', lw=2.0, zorder=3,
                solid_capstyle='round')
        # Arrowhead tip exactly at the connector node boundary
        tang = qtan(src, cp_in, C, 1.0)
        tang /= np.linalg.norm(tang) + 1e-9
        tip_in = C - tang * CONN_R          # point on circle boundary
        add_arrowhead(tip_in, tang, hw=0.22, hl=0.18, lw_edge=2.0, zo=3)

        # Outgoing branches: C → each head
        for h in heads:
            dst  = pos[h]
            bend = BEND_BRANCH.get((aid, h), 0.22 if heads.index(h) % 2 == 0 else -0.22)
            pts_br, cp_br = curve_pts(C, dst, bend, CONN_R, NODE_R)
            ax.plot(pts_br[:, 0], pts_br[:, 1], color='black', lw=1.5, zorder=3,
                    solid_capstyle='round')
            # Arrowhead tip exactly at the opinion node boundary
            tang_br = qtan(C, cp_br, dst, 1.0)
            tang_br /= np.linalg.norm(tang_br) + 1e-9
            tip_br = dst - tang_br * NODE_R
            add_arrowhead(tip_br, tang_br, hw=0.18, hl=0.15, lw_edge=1.5, zo=3)

    # ── Auto-scale ────────────────────────────────────────────────────────────
    all_pts = list(pos.values()) + list(conn_pos.values())
    xs  = [v[0] for v in all_pts]
    ys  = [v[1] for v in all_pts]
    pad = 1.8
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    plt.tight_layout()
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output, bbox_inches='tight', dpi=200)
    print(f"Saved: {output}")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',    default='fhypergraph_scotus.pdf')
    p.add_argument('--print-pos', action='store_true',
                   help='Print the computed pos dict (opinion + connector nodes) and exit')
    args = p.parse_args()

    print("Computing layout ...")
    # Always start from the full auto layout, then apply any overrides on top
    pos      = get_layout()
    conn_pos = compute_connector_positions(pos)
    if POS_OVERRIDE:
        for k, v in POS_OVERRIDE.items():
            if k.startswith('C_'):
                conn_pos[k] = v
            else:
                pos[k] = v
        # Recompute connector positions for any arc whose tail/heads moved,
        # unless the connector itself was explicitly overridden
        auto_conn = compute_connector_positions(pos)
        for k, v in auto_conn.items():
            if k not in POS_OVERRIDE:
                conn_pos[k] = v

    if args.print_pos:
        full_pos = {**pos, **conn_pos}
        print("\n# ── Paste into POS_OVERRIDE and edit as needed ──────────────────────────")
        print("POS_OVERRIDE = {")
        for k, v in sorted(full_pos.items()):
            print(f'    "{k}": np.array([{v[0]:.4f}, {v[1]:.4f}]),')
        print("}")
        return

    print("Drawing ...")
    draw(pos, conn_pos, args.output)


if __name__ == '__main__':
    main()
