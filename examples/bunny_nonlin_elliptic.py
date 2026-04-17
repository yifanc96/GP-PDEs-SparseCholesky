"""Nonlinear elliptic PDE inside the Stanford bunny.

Domain:  the volume enclosed by the Stanford Bunny triangle mesh
         (examples/data/bunny.obj, 2503 verts / 4968 faces). Rejection-
         sample the bunny's bounding box and keep points reported
         "inside" by trimesh; sample the mesh surface for boundary points.

The most literal "mesh-free on mesh-defined geometry" demo: raw OBJ
coordinates go in, a volumetric PDE comes out, no tet-meshing required.
"""

from __future__ import annotations

import pathlib

import numpy as np

from _geometry_demo import GeometryDemo, default_argparser, run_and_plot


DATA = pathlib.Path(__file__).parent / 'data' / 'bunny.obj'


def _load_bunny():
    import trimesh
    m = trimesh.load(str(DATA), process=True)
    # Shift + scale into a bounding box roughly [-0.5, 0.5]³.
    centre = (m.bounds[0] + m.bounds[1]) / 2
    scale = 1.0 / np.max(m.bounds[1] - m.bounds[0])
    m.vertices = (m.vertices - centre) * (2.0 * scale)
    return m


def sample_interior(mesh, n: int, rng):
    pts = np.empty((0, 3))
    mn, mx = mesh.bounds
    while pts.shape[0] < n:
        batch = rng.uniform(mn + 0.01, mx - 0.01, (6 * n, 3))
        inside = mesh.contains(batch)
        pts = np.vstack([pts, batch[inside]])
    return pts[:n]


def boundary_points(mesh, n: int, rng):
    pts, _face_idx = mesh.sample(n, return_index=True)
    return np.asarray(pts, dtype=np.float64)


def mesh_outline(mesh, stride: int = 12):
    """Render every `stride`-th triangle outline for the 3D plot."""
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

    mesh = _load_bunny()
    print(f'[bunny]   verts={len(mesh.vertices)}, faces={len(mesh.faces)}')

    rng = np.random.default_rng(args.seed)
    X_dom = sample_interior(mesh, args.N_interior, rng)
    X_bdy = boundary_points(mesh, args.bdy, rng)

    demo = GeometryDemo(
        name='bunny', X_domain=X_dom, X_boundary=X_bdy,
        outline=mesh_outline(mesh, stride=12),
        u_exact=u_exact, rhs=rhs,
    )
    run_and_plot(demo, args)


if __name__ == '__main__':
    main()
