"""Synthetic meshes that put each artistic column to a constructive test.

`docs/LABELING_GUIDE.md` records the criteria as the research group stated
them; each column in `features/artistic.py` is this project's guess at how to
measure those criteria. A guess is worth what its counterexamples say, so this
module builds meshes whose construction is known — a tessellation is uniform or
it is not, a surface is creased or it is not — and the tests beside it assert
that the matching column moves when that property moves.

Two prototype families stand in for the ends of the criteria's spectrum:

- uniform tessellations (`icosphere`, `torus`): dense, evenly spaced, the
  signature of automatic remeshing and scan reconstruction;
- hand-made-looking meshes (`box`, `prism`, `tent`): coarse, creased, sparse
  where the surface is flat.

They are prototypes, not ground truth. Agreeing with them shows that a column
measures the property it is named after; it does *not* show that the property
separates real scanned assets from real hand-made ones — only the blind audit
can say that. The tests record the caveats measurement has already turned up:
a uniform quad grid gives q = sqrt(3)/2, below any "too regular" threshold, so
quad-remeshed assets are invisible to the equilateral column; and a fold is an
extrinsic bend with zero intrinsic curvature, so the curvature-density column
cannot see "dense at creases" even when the tessellation is crowded there.

Everything here is plain numpy — no trimesh, no filesystem, no fixtures.
"""

from __future__ import annotations

import math

import numpy as np


def graded_axis(size: int, grading: float = 1.0) -> np.ndarray:
    """Coordinates in [-1, 1] whose spacing tightens towards zero.

    `grading = 1` is uniform; larger values crowd points near the centre. Used
    to make tessellation density follow curvature without moving the surface.
    """
    positions = np.linspace(-1.0, 1.0, size)
    return np.sign(positions) * np.abs(positions) ** grading


def grid_from_axes(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A flat grid over the given coordinates, split into right triangles.

    Every face is half an axis-aligned rectangle, so this whole family sits at
    q = sqrt(3)/2 no matter how the coordinates are spaced. Passing uneven
    coordinates is how the spacing contrasts keep the topology and move only
    the spacing.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    vertices = np.array([[x, y, 0.0] for y in ys for x in xs])
    faces = []
    columns = len(xs)
    for row in range(len(ys) - 1):
        for column in range(columns - 1):
            corner = row * columns + column
            faces.append([corner, corner + 1, corner + columns + 1])
            faces.append([corner, corner + columns + 1, corner + columns])
    return vertices, np.array(faces, dtype=np.int64)


def triangular_patch(
    rows: int, columns: int, height: float
) -> tuple[np.ndarray, np.ndarray]:
    """A parallelogram of a triangular lattice with row spacing `height`.

    `height = sqrt(3)/2` makes every triangle equilateral — the "auto-remeshed"
    end of the face-shape criterion. Squashing the same combinatorial patch
    towards `height = 0` turns the triangles into slivers without changing
    anything else, so a test can watch the equilateral column alone move.
    """
    vertices = [
        [column + (0.5 if row % 2 else 0.0), row * height, 0.0]
        for row in range(rows)
        for column in range(columns)
    ]
    faces = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            a = row * columns + column
            b = a + 1
            c = a + columns
            d = c + 1
            if row % 2 == 0:
                faces += [[a, b, c], [b, d, c]]
            else:
                faces += [[a, b, d], [a, d, c]]
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int64)


def tent(
    columns: int, rows: int, grading: float = 1.0, length: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """A roof: two flat planes meeting at a ridge along x = 0.

    The surface is developable, so its discrete curvature is exactly zero off
    the ridge and positive on it. That makes it a clean test of whether
    tessellation density *tracks* curvature independently of the shape:
    `grading = 1` keeps the column spacing uniform, larger values crowd columns
    towards the ridge. Use an odd `columns` count so a vertex sits on the
    ridge; otherwise the fold is cut off into a flat strip.
    """
    xs = graded_axis(columns, grading)
    ys = np.linspace(0.0, length, rows)
    vertices = np.array([[x, y, 1.0 - abs(x)] for y in ys for x in xs])
    faces = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            corner = row * columns + column
            faces.append([corner, corner + 1, corner + columns + 1])
            faces.append([corner, corner + columns + 1, corner + columns])
    return vertices, np.array(faces, dtype=np.int64)


def bump_disc(
    rings: int = 12, sectors: int = 24, grading: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """A circular plate with one smooth bump, tessellated from the centre out.

    Unlike a fold, a bump carries genuine intrinsic curvature — the angle
    defect around its crown is not zero — so the curvature-density column has
    something real to measure. A polar grid is already densest at the centre
    for free (cells shrink with the radius), so `grading` is what moves the
    contrast: values above one crowd rings towards the curvature, values below
    one crowd them away from it, and 1.0 is the plain polar grid.
    """
    radii = np.linspace(0.0, 1.0, rings + 1) ** grading
    vertices = [[0.0, 0.0, 1.0]]
    for r in radii[1:]:
        for k in range(sectors):
            angle = 2.0 * np.pi * k / sectors
            vertices.append([r * np.cos(angle), r * np.sin(angle), (1.0 - r * r) ** 2])
    faces = []
    for k in range(sectors):
        faces.append([0, 1 + k, 1 + (k + 1) % sectors])
    for i in range(rings - 1):
        inner = 1 + i * sectors
        outer = inner + sectors
        for k in range(sectors):
            j = (k + 1) % sectors
            faces.append([inner + k, inner + j, outer + j])
            faces.append([inner + k, outer + j, outer + k])
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int64)


def subdivide(
    vertices: np.ndarray,
    faces: np.ndarray,
    levels: int = 1,
    project=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear 1-to-4 refinement.

    Midpoints are left where they fall unless `project` is given; the sphere
    generators pass a normalisation so their midpoints stay on the sphere.
    """
    points = [row for row in np.asarray(vertices, dtype=np.float64)]
    current = np.asarray(faces, dtype=np.int64)

    def midpoint(cache: dict[tuple[int, int], int], left: int, right: int) -> int:
        key = (min(left, right), max(left, right))
        if key not in cache:
            middle = (points[left] + points[right]) / 2.0
            if project is not None:
                middle = project(middle)
            points.append(middle)
            cache[key] = len(points) - 1
        return cache[key]

    for _ in range(levels):
        cache: dict[tuple[int, int], int] = {}
        refined = []
        for a, b, c in current:
            ab = midpoint(cache, int(a), int(b))
            bc = midpoint(cache, int(b), int(c))
            ca = midpoint(cache, int(c), int(a))
            refined += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
        current = np.array(refined, dtype=np.int64)
    return np.array(points, dtype=np.float64), current


def icosphere(subdivisions: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """A subdivided icosahedron projected onto the unit sphere.

    `subdivisions = 0` is the raw icosahedron — twenty equilateral triangles,
    the coarsest closed shape that still has no boundary. Higher levels give
    the uniform, near-equilateral, evenly spaced tessellation that automatic
    remeshing produces.
    """
    t = (1.0 + math.sqrt(5.0)) / 2.0
    base = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float64,
    )
    base /= np.linalg.norm(base, axis=1)[:, None]
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )
    normalise = lambda point: point / np.linalg.norm(point)  # noqa: E731
    return subdivide(base, faces, subdivisions, project=normalise)


def torus(
    major_segments: int = 24,
    minor_segments: int = 12,
    major_radius: float = 1.0,
    minor_radius: float = 0.4,
    minor_grading: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """A closed uniform quad grid bent into a ring.

    An evenly spaced ring grid is the "no structure" end of vertex placement,
    but it is not neutral for the curvature-density column: even spacing in
    angle makes the cells at the inner equator smaller, and the Gaussian
    curvature there is negative, so a plain torus reads a strong *negative*
    correlation. `minor_grading` crowds the minor rings towards the outer
    equator (values above one, where the curvature is positive) or towards the
    inner one (below one); 1.0 is the plain ring. That makes the torus a clean
    place to watch density and curvature line up or oppose.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, major_segments, endpoint=False)
    u = np.linspace(0.0, 1.0, minor_segments, endpoint=False)
    phi = 2.0 * np.pi * u**minor_grading
    vertices = [
        [
            (major_radius + minor_radius * np.cos(p)) * np.cos(t),
            (major_radius + minor_radius * np.cos(p)) * np.sin(t),
            minor_radius * np.sin(p),
        ]
        for t in theta
        for p in phi
    ]
    faces = []
    for i in range(major_segments):
        for j in range(minor_segments):
            a = i * minor_segments + j
            b = i * minor_segments + (j + 1) % minor_segments
            c = ((i + 1) % major_segments) * minor_segments + j
            d = ((i + 1) % major_segments) * minor_segments + (j + 1) % minor_segments
            faces += [[a, b, d], [a, d, c]]
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int64)


def box(size: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """A cube: eight vertices, twelve triangles.

    Every vertex sits on a crease and no face is wasted — the criteria's
    "good" extreme, and coarse enough that removing any vertex loses shape.
    """
    corners = size * np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return corners, faces


def prism(
    sides: int = 6, height: float = 1.6, radius: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """A low-poly cylinder: two rims, cap fans, elongated side triangles.

    The rims are this family's version of the concentric rings visible in the
    reference low-poly assets — a circle approximated by few straight segments,
    long thin triangles between them, which is what a hand modeller leaves
    behind and a uniform remesh never does.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = [[radius * np.cos(a), radius * np.sin(a)] for a in angles]
    vertices = [[0.0, 0.0, 0.0], [0.0, 0.0, height]]
    vertices += [[x, y, 0.0] for x, y in ring]
    vertices += [[x, y, height] for x, y in ring]
    hub_bottom, hub_top = 0, 1
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        bottom_i, bottom_j = 2 + i, 2 + j
        top_i, top_j = 2 + sides + i, 2 + sides + j
        faces.append([hub_bottom, bottom_j, bottom_i])
        faces.append([hub_top, top_i, top_j])
        faces.append([bottom_i, bottom_j, top_j])
        faces.append([bottom_i, top_j, top_i])
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int64)
