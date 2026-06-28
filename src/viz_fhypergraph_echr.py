"""
viz_fhypergraph_echr.py  –  ECHR F-hypergraph example, Gallo et al. (1993) style.

Chain illustrated:
  MGN Limited v. UK (2011)
    → Decision F-arc (2012-06-19)
      → Krone Verlag GmbH v. Austria (2012)
        → Decision F-arc (2012-06-19)
          → 6 press-freedom head cases (2012-2013)

Same visual style as viz_fhypergraph.py (SCOTUS version).
"""

import argparse, math, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

# ── Hardcoded subgraph ────────────────────────────────────────────────────────

ARCS = [
    # E1: MGN Limited v. UK cites Krone Verlag (single-citation F-arc)
    {"id": "E1", "tau": "2011-01-18",
     "tail": "mgn",
     "heads": ["krone"]},

    # E2: Krone Verlag v. Austria cites 6 press-freedom cases
    {"id": "E2", "tau": "2012-06-19",
     "tail": "krone",
     "heads": ["ooo_vesti", "falter", "ristamaki", "pauliukiene", "delfi", "faber"]},
]

LABELS = {
    "mgn":         "MGN Limited\nv. UK",
    "krone":       "Krone Verlag\nv. Austria",
    "ooo_vesti":   "OOO Vesti\nv. Russia",
    "falter":      "Falter Zeitschriften\nv. Austria",
    "ristamaki":   "Ristamäki & Korvola\nv. Finland",
    "pauliukiene": "Pauliukienė\nv. Lithuania",
    "delfi":       "Delfi AS\nv. Estonia",
    "faber":       "Fáber\nv. Hungary",
}

DATES = {
    "mgn":         "2011-01-18",
    "krone":       "2012-06-19",
    "ooo_vesti":   "2013-05-30",
    "falter":      "2012-09-18",
    "ristamaki":   "2013-10-29",
    "pauliukiene": "2013-11-05",
    "delfi":       "2013-10-10",
    "faber":       "2012-07-24",
}

TAIL_NODES = {"mgn", "krone"}

# ── Per-arc bend overrides ────────────────────────────────────────────────────
BEND_IN = {
    "E1":  0.20,
    "E2": -0.25,
}
BEND_BRANCH = {
    ("E1", "krone"):       0.00,
    ("E2", "ooo_vesti"):  -0.35,
    ("E2", "falter"):     -0.22,
    ("E2", "ristamaki"):  -0.10,
    ("E2", "pauliukiene"): 0.10,
    ("E2", "delfi"):       0.22,
    ("E2", "faber"):       0.35,
}

# ── Position overrides ────────────────────────────────────────────────────────
POS_OVERRIDE = {}

# ── Layout via pygraphviz dot ─────────────────────────────────────────────────

def get_layout():
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
        max_y = max(v[1] for v in pos.values())
        pos = {n: np.array([x, max_y - y]) for n, (x, y) in pos.items()}
        span = max(
            max(v[0] for v in pos.values()) - min(v[0] for v in pos.values()),
            max(v[1] for v in pos.values()) - min(v[1] for v in pos.values())
        )
        if span > 0:
            pos = {n: v / span * 8.0 for n, v in pos.items()}
        return pos
    except Exception as e:
        print(f"pygraphviz layout failed ({e}), using manual positions")
        return {
            "mgn":         np.array([4.0, 0.0]),
            "krone":       np.array([4.0, 2.5]),
            "ooo_vesti":   np.array([0.5, 5.5]),
            "falter":      np.array([1.8, 5.5]),
            "ristamaki":   np.array([3.1, 5.5]),
            "pauliukiene": np.array([4.4, 5.5]),
            "delfi":       np.array([5.7, 5.5]),
            "faber":       np.array([7.0, 5.5]),
        }


NODE_R = 0.18
CONN_R = 0.18


def compute_connector_positions(pos):
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
    cp  = ctrl(p0, p2, bend)
    ts  = np.linspace(0, 1, n)
    pts = np.array([qbez(p0, cp, p2, t) for t in ts])
    i0  = next((i for i, p in enumerate(pts) if np.linalg.norm(p - p0) >= r0), 0)
    i1  = next((i for i, p in enumerate(reversed(pts))
                if np.linalg.norm(p - p2) >= r2), 0)
    i1  = len(pts) - 1 - i1
    return pts[i0:i1+1], cp


def draw(pos, conn_pos, output):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_aspect('equal')
    ax.axis('off')

    # ── PASS 1: nodes first (high zorder) ────────────────────────────────────

    # F-arc connector nodes (filled black)
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

    # ── PASS 2: edges and arrowheads below nodes ──────────────────────────────

    def add_arrowhead(tip, direction, hw, hl, lw_edge, zo):
        d = direction / (np.linalg.norm(direction) + 1e-9)
        start = tip - d * hl
        ax.add_patch(FancyArrow(
            start[0], start[1], d[0] * hl, d[1] * hl,
            width=lw_edge * 0.01,
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
        src   = pos[tail]
        C     = conn_pos[f"C_{aid}"]

        # Incoming arc: tail → C
        pts_in, cp_in = curve_pts(src, C, BEND_IN[aid], NODE_R, CONN_R)
        ax.plot(pts_in[:, 0], pts_in[:, 1], color='black', lw=2.0, zorder=3,
                solid_capstyle='round')
        tang = qtan(src, cp_in, C, 1.0)
        tang /= np.linalg.norm(tang) + 1e-9
        tip_in = C - tang * CONN_R
        add_arrowhead(tip_in, tang, hw=0.22, hl=0.18, lw_edge=2.0, zo=3)

        # Outgoing branches: C → each head
        for h in heads:
            dst  = pos[h]
            bend = BEND_BRANCH.get((aid, h), 0.22 if heads.index(h) % 2 == 0 else -0.22)
            pts_br, cp_br = curve_pts(C, dst, bend, CONN_R, NODE_R)
            ax.plot(pts_br[:, 0], pts_br[:, 1], color='black', lw=1.5, zorder=3,
                    solid_capstyle='round')
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
    p.add_argument('--output', default='fhypergraph_echr.pdf')
    p.add_argument('--print-pos', action='store_true')
    args = p.parse_args()

    print("Computing layout ...")
    pos      = get_layout()
    conn_pos = compute_connector_positions(pos)
    if POS_OVERRIDE:
        for k, v in POS_OVERRIDE.items():
            if k.startswith('C_'):
                conn_pos[k] = v
            else:
                pos[k] = v
        auto_conn = compute_connector_positions(pos)
        for k, v in auto_conn.items():
            if k not in POS_OVERRIDE:
                conn_pos[k] = v

    if args.print_pos:
        full_pos = {**pos, **conn_pos}
        print("\nPOS_OVERRIDE = {")
        for k, v in sorted(full_pos.items()):
            print(f'    "{k}": np.array([{v[0]:.4f}, {v[1]:.4f}]),')
        print("}")
        return

    print("Drawing ...")
    draw(pos, conn_pos, args.output)


if __name__ == '__main__':
    main()
