"""Nonlinear elliptic PDE on a mechanical L-bracket built by CSG.

Domain:  an L-bracket is constructed from trimesh primitives — a
horizontal flange + a vertical flange joined along their edge, with
four bolt-holes drilled through each flange via boolean subtraction.
The resulting mesh is "CAD-like" without a STEP file: real cylinders
cut from real boxes.

Shows the ultimate practical case: your CAD workflow produces an STL /
OBJ (or you construct one programmatically), and the same meshless
solver runs on it with zero fuss.
"""

from __future__ import annotations

import numpy as np

from _geometry_demo import GeometryDemo, default_argparser, run_and_plot


def build_bracket():
    import trimesh

    # Horizontal flange: box of size (L, W, T) centred at (L/2, W/2, T/2).
    L, W, T = 1.0, 0.6, 0.08
    # Vertical flange: same length/thickness, height H.
    H = 0.6
    horiz = trimesh.creation.box(extents=(L, W, T))
    horiz.apply_translation((L / 2, W / 2, T / 2))

    vert = trimesh.creation.box(extents=(L, T, H))
    vert.apply_translation((L / 2, T / 2, H / 2))

    bracket = trimesh.boolean.union([horiz, vert])

    # Drill 4 bolt-holes through each flange.
    r_hole = 0.04
    holes = []
    for (bx, by) in [(0.20, 0.20), (0.20, 0.45), (0.80, 0.20), (0.80, 0.45)]:
        c = trimesh.creation.cylinder(radius=r_hole, height=T * 3)
        c.apply_translation((bx, by, T / 2))
        holes.append(c)
    for (bx, bz) in [(0.20, 0.20), (0.20, 0.45), (0.80, 0.20), (0.80, 0.45)]:
        c = trimesh.creation.cylinder(radius=r_hole, height=T * 3)
        # Rotate so axis is along +y
        R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        c.apply_transform(R)
        c.apply_translation((bx, T / 2, bz))
        holes.append(c)

    bracket = trimesh.boolean.difference([bracket, *holes])
    # Recentre so the bbox is symmetric around origin-ish.
    centre = (bracket.bounds[0] + bracket.bounds[1]) / 2
    bracket.vertices = bracket.vertices - centre
    return bracket


def sample_interior(mesh, n, rng):
    pts = np.empty((0, 3))
    mn, mx = mesh.bounds
    while pts.shape[0] < n:
        batch = rng.uniform(mn + 0.005, mx - 0.005, (6 * n, 3))
        pts = np.vstack([pts, batch[mesh.contains(batch)]])
    return pts[:n]


def boundary_points(mesh, n, rng):
    pts, _ = mesh.sample(n, return_index=True)
    return np.asarray(pts, dtype=np.float64)


def mesh_outline(mesh, stride=18):
    verts = mesh.vertices
    faces = mesh.faces[::stride]
    return [verts[[f[0], f[1], f[2], f[0]]] for f in faces]


def u_exact(x):
    return float(np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]) * np.cos(np.pi * x[2]))


def rhs(x):
    return float(3 * np.pi ** 2 * u_exact(x) + u_exact(x) ** 3)


def main():
    parser = default_argparser()
    parser.add_argument('--bdy', type=int, default=3000)
    args = parser.parse_args()

    mesh = build_bracket()
    print(f'[bracket] verts={len(mesh.vertices)}, faces={len(mesh.faces)}, '
          f'volume={mesh.volume:.4f}')

    rng = np.random.default_rng(args.seed)
    X_dom = sample_interior(mesh, args.N_interior, rng)
    X_bdy = boundary_points(mesh, args.bdy, rng)

    demo = GeometryDemo(
        name='bracket', X_domain=X_dom, X_boundary=X_bdy,
        outline=mesh_outline(mesh, stride=18),
        u_exact=u_exact, rhs=rhs,
    )
    run_and_plot(demo, args)


if __name__ == '__main__':
    main()
