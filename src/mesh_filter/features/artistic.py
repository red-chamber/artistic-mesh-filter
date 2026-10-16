"""Features built for the stated artistic-mesh criteria.

The operative definition (docs/LABELING_GUIDE.md) has three parts: face and
vertex counts must stay small, faces must not be too regular, and vertex
placement must look orderly. The first is already counted by `basic.py`; this
module covers the other two.

Two quantities here point in a direction that is easy to get backwards:

- `equilateral_ratio` (in `basic.py`) is high for **bad** meshes. Automatic
  remeshing and scan reconstruction chase uniform sampling, so they produce
  near-equilateral triangles. Hand-made low-poly meshes have elongated, oriented
  triangles instead.
- `qem_redundancy_median` is high for **good** meshes. It measures how much
  geometry is lost when a vertex is removed: a mesh that spends vertices only
  where they are needed loses a lot, a scanned surface with vertices piled up on
  flat regions loses almost nothing.

Every quantity here is invariant to translation, rotation and uniform scaling,
so nothing on this list can learn the scale gap between the sources (Toys4K is
normalised to a longest edge of exactly 2.0; Objaverse spans 0.7 to 12160).

Config switches in `configs/features/artistic_v1.yaml` carry the same names as
the columns they produce, so a switch never has to be translated.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mesh_filter.features.basic import GeometryError

DEGREE_BINS = 12
MAX_RING_DEGREE = 32
EPSILON = 1.0e-12
# Below this many vertices a Spearman coefficient is noise dressed up as a
# number: eight vertices with two distinct values already give +-0.94.
MIN_CORRELATION_VERTICES = 30
# Coplanar faces merge into one patch when their normals differ by at most this
# angle; the same tolerance decides whether two patch-border edges are collinear.
# 3 degrees is the group's suggestion, and it is the single cosine that CGAL's
# planar-patch remeshing applies to both tests.
FLAT_ANGLE_TOLERANCE_DEG = 3.0

COLUMNS = [
    "vertex_degree_mode_share",
    "vertex_degree_entropy",
    "one_ring_edge_length_cv_median",
    "local_edge_direction_anisotropy_mean",
    "neighboring_direction_consistency_mean",
    "curvature_density_correlation",
    "qem_redundancy_median",
    "redundant_face_share",
]


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.sort(
        np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1
    )
    return np.unique(edges, axis=0)


def _incidence(
    vertex_count: int, unique_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group edges by endpoint: sorted vertex ids, edge ids, and group sizes."""
    vertex_ids = np.concatenate((unique_edges[:, 0], unique_edges[:, 1]))
    edge_ids = np.concatenate((np.arange(len(unique_edges)), np.arange(len(unique_edges))))
    order = np.argsort(vertex_ids, kind="stable")
    vertex_sorted = vertex_ids[order]
    edge_sorted = edge_ids[order]
    counts = np.bincount(vertex_sorted, minlength=vertex_count)
    starts = np.zeros(vertex_count, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]
    position = np.arange(len(vertex_sorted)) - starts[vertex_sorted]
    return vertex_sorted, edge_sorted, position


def _edge_directions(
    vertices: np.ndarray, unique_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    deltas = vertices[unique_edges[:, 1]] - vertices[unique_edges[:, 0]]
    lengths = np.linalg.norm(deltas, axis=1)
    safe = np.where(lengths > 0, lengths, 1.0)
    return deltas / safe[:, None], lengths


def _degree_columns(vertex_count: int, unique_edges: np.ndarray) -> dict[str, float]:
    """How concentrated the vertex-degree distribution is.

    Hand-made meshes cluster around a few degrees (four and six dominate a
    quad-turned-triangle grid); scanned surfaces spread out because their
    tessellation follows no grid.
    """
    degrees = np.bincount(unique_edges.ravel(), minlength=vertex_count)
    in_use = degrees[degrees > 0]
    if len(in_use) == 0:
        raise GeometryError("no vertex is used by a face")
    counts = np.bincount(np.minimum(in_use, DEGREE_BINS), minlength=DEGREE_BINS + 1)
    shares = counts / counts.sum()
    nonzero = shares[shares > 0]
    return {
        "vertex_degree_mode_share": float(shares.max()),
        "vertex_degree_entropy": float(
            -(nonzero * np.log(nonzero)).sum() / np.log(DEGREE_BINS)
        ),
    }


def _one_ring_cv(
    vertex_count: int, unique_edges: np.ndarray, lengths: np.ndarray
) -> float:
    """Median over vertices of the coefficient of variation of incident edge lengths."""
    starts, ends = unique_edges[:, 0], unique_edges[:, 1]
    degree = np.bincount(np.concatenate((starts, ends)), minlength=vertex_count)
    in_use = degree > 0
    safe_degree = np.where(in_use, degree, 1)

    total = np.bincount(starts, weights=lengths, minlength=vertex_count)
    total += np.bincount(ends, weights=lengths, minlength=vertex_count)
    total_square = np.bincount(starts, weights=lengths**2, minlength=vertex_count)
    total_square += np.bincount(ends, weights=lengths**2, minlength=vertex_count)

    mean = total / safe_degree
    variance = np.maximum(total_square / safe_degree - mean**2, 0.0)
    cv = np.divide(np.sqrt(variance), mean, out=np.zeros_like(mean), where=mean > 0)
    return float(np.median(cv[in_use]))


def _direction_anisotropy(
    vertex_count: int, unique_edges: np.ndarray, directions: np.ndarray
) -> float:
    """How unequal the eigenvalue spectrum of the local edge-direction tensor is.

    A vertex sitting on a ring, an outline or a limb edge has its edges pointing
    along one direction and scores high; a vertex buried in a uniform
    tessellation has directions sprayed evenly and scores near zero.
    """
    tensors = np.zeros((vertex_count, 3, 3), dtype=np.float64)
    outer = directions[:, :, None] * directions[:, None, :]
    for end in (0, 1):
        np.add.at(tensors, unique_edges[:, end], outer)

    eigenvalues = np.linalg.eigvalsh(tensors)  # ascending, batched over vertices
    largest, middle = eigenvalues[:, 2], eigenvalues[:, 1]
    in_use = largest > EPSILON
    if not in_use.any():
        return 0.0
    anisotropy = np.divide(
        largest - middle, largest, out=np.zeros_like(largest), where=in_use
    )
    return float(anisotropy[in_use].mean())


def _neighboring_consistency(
    vertex_count: int, unique_edges: np.ndarray, directions: np.ndarray
) -> float:
    """Mean absolute cosine between pairs of edges meeting at the same vertex.

    Collinear pairs (a chain passing through the vertex) push this up; an even
    spray of directions pushes it down.
    """
    vertex_sorted, edge_sorted, position = _incidence(vertex_count, unique_edges)
    keep = position < MAX_RING_DEGREE
    padded = np.zeros((vertex_count, MAX_RING_DEGREE, 3), dtype=np.float64)
    padded[vertex_sorted[keep], position[keep]] = directions[edge_sorted[keep]]

    counts = np.bincount(vertex_sorted, minlength=vertex_count)
    mask = np.arange(MAX_RING_DEGREE)[None, :] < counts[:, None]
    pairs = mask[:, :, None] & mask[:, None, :]
    pairs &= np.triu(np.ones((MAX_RING_DEGREE, MAX_RING_DEGREE), dtype=bool), k=1)[None]

    gram = np.abs(np.einsum("vid,vjd->vij", padded, padded))
    pair_counts = pairs.sum(axis=(1, 2))
    usable = pair_counts > 0
    if not usable.any():
        raise GeometryError("no vertex has two incident edges")
    totals = (gram * pairs).sum(axis=(1, 2))
    return float((totals[usable] / pair_counts[usable]).mean())


def _face_angles(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    sides = [
        np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1),
        np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1),
    ]

    def angle(opposite: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        denominator = 2.0 * first * second
        cosine = np.divide(
            first**2 + second**2 - opposite**2,
            denominator,
            out=np.ones_like(opposite),
            where=denominator > 0,
        )
        return np.arccos(np.clip(cosine, -1.0, 1.0))

    angles = np.zeros((len(faces), 3), dtype=np.float64)
    for corner in range(3):
        others = [index for index in range(3) if index != corner]
        angles[:, corner] = angle(sides[corner], sides[others[0]], sides[others[1]])
    return areas, angles


def _mid_ranks(values: np.ndarray) -> np.ndarray:
    """Spearman ranks with tied values sharing their average rank.

    Plain index-order ranks make a tie-heavy column decide the correlation by
    floating-point noise — enough to change the answer under a rotation.
    """
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    midpoints = starts + (counts - 1) / 2.0
    del unique
    return midpoints[inverse].astype(np.float64)


def _curvature_density_correlation(
    vertices: np.ndarray, faces: np.ndarray
) -> float:
    """Spearman correlation between local curvature and local face density.

    Hand-made meshes spend geometry where the shape bends: flat regions stay
    sparse, creases and rims get crowded. A scanned surface is dense everywhere,
    so curvature and density are unrelated. A positive correlation is the
    signature the criteria look for.
    """
    areas, angles = _face_angles(vertices, faces)
    keep = areas > EPSILON
    if keep.sum() < 4:
        raise GeometryError("too few non-degenerate faces")
    faces, areas, angles = faces[keep], areas[keep], angles[keep]

    vertex_count = len(vertices)
    angle_sum = np.bincount(faces.ravel(), weights=angles.ravel(), minlength=vertex_count)
    area_sum = np.zeros(vertex_count, dtype=np.float64)
    np.add.at(area_sum, faces.ravel(), np.repeat(areas / 3.0, 3))

    edges = np.sort(
        np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1
    )
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    boundary = np.zeros(vertex_count, dtype=bool)
    border = np.unique(edges, axis=0)[edge_counts == 1]
    if len(border):
        boundary[border.ravel()] = True

    # A boundary vertex is missing half its neighbourhood, so its defect is
    # measured against pi rather than 2*pi.
    target = np.where(boundary, np.pi, 2.0 * np.pi)
    curvature = np.divide(
        target - angle_sum, area_sum, out=np.zeros(vertex_count), where=area_sum > 0
    )
    density = np.divide(
        1.0, area_sum, out=np.zeros(vertex_count), where=area_sum > 0
    )

    # Boundary vertices have half a neighbourhood, so their defect is measured
    # against pi and is not comparable with an interior vertex's. Correlate on
    # interior vertices; only fall back to all of them if there are too few.
    sample = (area_sum > 0) & ~boundary
    if sample.sum() < 10:
        sample = area_sum > 0
    if sample.sum() < MIN_CORRELATION_VERTICES:
        return float("nan")

    sampled_curvature = curvature[sample]
    sampled_density = density[sample]
    for values in (sampled_curvature, sampled_density):
        spread = float(values.max() - values.min())
        scale = float(np.abs(values).max())
        if spread <= 1.0e-9 * max(scale, 1.0):
            # A flat sheet has one curvature value everywhere, a uniform grid one
            # density value everywhere. The correlation is then decided by
            # floating-point noise, so an empty cell says "not measurable"
            # instead of inventing a number that changes with every rotation.
            return float("nan")

    left = _mid_ranks(sampled_curvature)
    right = _mid_ranks(sampled_density)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.sqrt((left**2).sum() * (right**2).sum()))
    if denominator <= 0:
        return 0.0
    return float((left * right).sum() / denominator)


def _qem_redundancy_median(
    vertices: np.ndarray, faces: np.ndarray, unique_edges: np.ndarray
) -> float:
    """How much geometry each vertex is holding on to.

    The quadric of every face plane is accumulated at its three corners, then
    evaluated where the vertex would go if it were collapsed: the centroid of its
    one-ring neighbours. Evaluating the quadric at the vertex itself is useless —
    a vertex lies on all of its own incident planes, so the error is identically
    zero for every mesh. Moving to the neighbour centroid is what makes the number
    mean "what is lost if this vertex is removed".

    A vertex in a flat, oversampled region scores near zero: its neighbours are
    on the same plane, so removing it changes nothing. A vertex carrying a crease
    or a corner scores high.

    The error is divided by the square of the vertex's mean incident edge length,
    not by the bounding-box diagonal. Dividing by the box made every value on the
    real corpus collapse to zero — a fine tessellation of a smooth surface has a
    vanishing collapse error compared with the object's size — and the column
    carried no information. Against the local spacing the number instead reads as
    "how much shape this vertex holds relative to how finely it is surrounded",
    which is the criterion's "uses a face where two would do". Invariant to scale,
    and comparable between meshes.
    """
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    keep = areas > EPSILON
    if keep.sum() < 4:
        raise GeometryError("too few non-degenerate faces")
    faces = faces[keep]
    triangles = vertices[faces]
    normals = cross[keep] / (2.0 * areas[keep])[:, None]

    offsets = -np.einsum("ij,ij->i", normals, triangles[:, 0])
    planes = np.concatenate((normals, offsets[:, None]), axis=1)

    quadrics = planes[:, :, None] * planes[:, None, :]
    vertex_count = len(vertices)
    accumulated = np.zeros((vertex_count, 4, 4), dtype=np.float64)
    for corner in range(3):
        np.add.at(accumulated, faces[:, corner], quadrics)

    incidence_vertices, incidence_edges, _ = _incidence(vertex_count, unique_edges)
    neighbours = np.where(
        unique_edges[incidence_edges, 0] == incidence_vertices,
        unique_edges[incidence_edges, 1],
        unique_edges[incidence_edges, 0],
    )
    neighbour_sum = np.zeros((vertex_count, 3), dtype=np.float64)
    np.add.at(neighbour_sum, incidence_vertices, vertices[neighbours])
    neighbour_count = np.bincount(incidence_vertices, minlength=vertex_count)
    has_neighbours = neighbour_count > 0
    target = np.divide(
        neighbour_sum,
        np.maximum(neighbour_count, 1)[:, None],
        out=vertices.copy(),
        where=has_neighbours[:, None],
    )

    homogeneous = np.concatenate((target, np.ones((vertex_count, 1))), axis=1)
    error = np.einsum("vi,vij,vj->v", homogeneous, accumulated, homogeneous)

    local_lengths = vertices[unique_edges[:, 1]] - vertices[unique_edges[:, 0]]
    lengths = np.linalg.norm(local_lengths, axis=1)
    length_sum = np.bincount(
        incidence_vertices, weights=lengths[incidence_edges], minlength=vertex_count
    )
    mean_length = np.divide(
        length_sum,
        np.maximum(neighbour_count, 1),
        out=np.ones(vertex_count),
        where=has_neighbours,
    )

    in_use = has_neighbours.copy()
    in_use[faces.ravel()] = True
    in_use &= mean_length > 0
    if not in_use.any():
        raise GeometryError("no vertex is used by a face")
    return float(np.median(error[in_use] / mean_length[in_use] ** 2))


def _redundant_face_share(
    vertices: np.ndarray,
    faces: np.ndarray,
    tolerance_deg: float = FLAT_ANGLE_TOLERANCE_DEG,
) -> float:
    """The fraction of faces that could be removed without changing the shape.

    The recipe is CGAL's planar-patch remeshing (`remesh_planar_patches`):
    group edge-adjacent faces into patches of (nearly) one plane, walk each
    patch border in runs of (nearly) collinear edges, and keep a vertex only at
    the ends of those runs and where three or more patches meet — the *corners*.
    A patch with c corners needs c - 2 triangles no matter how finely it is
    tessellated inside, so everything above that is faces spent on nothing: the
    group's "a rectangle needs two triangles, this one used eight".

    A dense but curved surface scores zero here. No two of its faces are
    coplanar, so every patch is a single triangle and every patch is already
    minimal — a sphere's faces are not redundant, however many there are. That
    is what this column adds to the redundancy proxies: it separates "faces
    spent where the shape needs them" from "faces spent on flat ground".

    The minimum is exact for simply connected patches; a patch with a hole is
    triangulated in c + 2h - 2 triangles rather than c - 2, so the share is
    mildly optimistic on such patches. That direction is deliberate: this
    column only ever argues *for* redundancy.
    """
    tolerance_cos = math.cos(math.radians(tolerance_deg))
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    normals = np.divide(
        cross, lengths[:, None], out=np.zeros_like(cross), where=lengths[:, None] > 0
    )

    pairs = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edge_keys = np.sort(pairs, axis=1)
    unique_edges, inverse, counts = np.unique(
        edge_keys, axis=0, return_inverse=True, return_counts=True
    )
    face_ids = np.tile(np.arange(len(faces)), 3)
    order = np.argsort(inverse, kind="stable")
    sorted_faces = face_ids[order]
    starts = np.searchsorted(inverse[order], np.arange(len(unique_edges)), side="left")

    parent = np.arange(len(faces), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    finite = np.isfinite(normals).all(axis=1) & (lengths > 0)
    coplanar = np.zeros(len(unique_edges), dtype=bool)
    for edge in np.nonzero(counts == 2)[0]:
        first = int(sorted_faces[starts[edge]])
        second = int(sorted_faces[starts[edge] + 1])
        if not (finite[first] and finite[second]):
            continue
        if float(normals[first] @ normals[second]) >= tolerance_cos:
            coplanar[edge] = True
            union(first, second)

    _, patch_of_faces = np.unique([find(face) for face in range(len(faces))], return_inverse=True)
    patch_count = int(patch_of_faces.max()) + 1

    # Every border edge contributes its two endpoints as boundary points of the
    # patch(es) it borders; a boundary edge borders one patch, a non-coplanar
    # interior edge borders both of its faces' patches.
    border_edges = np.nonzero((counts == 1) | ((counts == 2) & ~coplanar))[0]
    if not len(border_edges):
        return 0.0  # a closed surface with one patch cannot exist, but be safe
    edge_a = unique_edges[border_edges, 0]
    edge_b = unique_edges[border_edges, 1]
    deltas = vertices[edge_b] - vertices[edge_a]
    delta_lengths = np.linalg.norm(deltas, axis=1)[:, None]
    units = np.divide(deltas, delta_lengths, out=np.zeros_like(deltas), where=delta_lengths > 0)

    boundary = counts[border_edges] == 1
    patch_arrays = []
    vertex_arrays = []
    direction_arrays = []
    for mask, sides in ((boundary, 1), (~boundary, 2)):
        if not mask.any():
            continue
        selected = np.nonzero(mask)[0]
        edges = border_edges[selected]
        faces_a = sorted_faces[starts[edges]]
        patches_a = patch_of_faces[faces_a]
        patch_arrays.append(patches_a)
        vertex_arrays.append(edge_a[selected])
        direction_arrays.append(units[selected])
        patch_arrays.append(patches_a)
        vertex_arrays.append(edge_b[selected])
        direction_arrays.append(-units[selected])
        if sides == 2:
            patches_b = patch_of_faces[sorted_faces[starts[edges] + 1]]
            patch_arrays.append(patches_b)
            vertex_arrays.append(edge_a[selected])
            direction_arrays.append(units[selected])
            patch_arrays.append(patches_b)
            vertex_arrays.append(edge_b[selected])
            direction_arrays.append(-units[selected])

    patch_labels = np.concatenate(patch_arrays)
    vertex_labels = np.concatenate(vertex_arrays)
    directions = np.concatenate(direction_arrays, axis=0)
    order = np.lexsort((vertex_labels, patch_labels))
    patch_labels, vertex_labels, directions = (
        patch_labels[order],
        vertex_labels[order],
        directions[order],
    )
    keys = np.stack((patch_labels, vertex_labels), axis=1)
    _, _, group_sizes = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    group_starts = np.cumsum(group_sizes) - group_sizes
    is_corner = np.ones(len(group_sizes), dtype=bool)
    straight_groups = np.nonzero(group_sizes == 2)[0]
    if len(straight_groups):
        first = directions[group_starts[straight_groups]]
        second = directions[group_starts[straight_groups] + 1]
        dots = np.einsum("ij,ij->i", first, second)
        is_corner[straight_groups] = dots > -tolerance_cos
    group_patches = patch_labels[group_starts]
    corners_per_patch = np.bincount(group_patches[is_corner], minlength=patch_count)
    minimal_faces = np.maximum(corners_per_patch - 2, 1)
    share = 1.0 - float(minimal_faces.sum()) / len(faces)
    return float(min(max(share, 0.0), 1.0))


def compute_artistic_features(
    vertices: np.ndarray,
    faces: np.ndarray,
    enabled: frozenset[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Compute whichever artistic features the config switched on.

    Passing `enabled=None` computes every column in COLUMNS.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise GeometryError("vertices must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise GeometryError("faces must have shape (M, 3)")
    if len(vertices) < 4 or len(faces) < 4:
        raise GeometryError("artistic features need at least four vertices and four faces")

    wanted = set(COLUMNS) if enabled is None else set(enabled)
    vertex_count = len(vertices)
    unique_edges = _unique_edges(faces)
    directions, lengths = _edge_directions(vertices, unique_edges)

    features: dict[str, Any] = {}
    # (built below, then any "not measurable" value is blanked out)
    if {"vertex_degree_mode_share", "vertex_degree_entropy"} & wanted:
        degree_columns = _degree_columns(vertex_count, unique_edges)
        for name, value in degree_columns.items():
            if name in wanted:
                features[name] = value
    if "one_ring_edge_length_cv_median" in wanted:
        features["one_ring_edge_length_cv_median"] = _one_ring_cv(
            vertex_count, unique_edges, lengths
        )
    if "local_edge_direction_anisotropy_mean" in wanted:
        features["local_edge_direction_anisotropy_mean"] = _direction_anisotropy(
            vertex_count, unique_edges, directions
        )
    if "neighboring_direction_consistency_mean" in wanted:
        features["neighboring_direction_consistency_mean"] = _neighboring_consistency(
            vertex_count, unique_edges, directions
        )
    if "curvature_density_correlation" in wanted:
        features["curvature_density_correlation"] = _curvature_density_correlation(
            vertices, faces
        )
    if "qem_redundancy_median" in wanted:
        features["qem_redundancy_median"] = _qem_redundancy_median(
            vertices, faces, unique_edges
        )
    if "redundant_face_share" in wanted:
        features["redundant_face_share"] = _redundant_face_share(vertices, faces)
    return {
        name: ("" if isinstance(value, float) and math.isnan(value) else value)
        for name, value in features.items()
    }
