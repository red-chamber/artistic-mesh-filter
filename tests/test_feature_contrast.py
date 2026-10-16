"""Constructive contrasts for the artistic features.

Every test here builds a mesh whose construction is known and moves exactly one
property, then asserts that the column claiming to measure that property moves
with it. That is weaker than validating against real assets — only the blind
audit can do that — but it is what catches a column measuring something other
than its name, before a screening threshold gets fitted to it.

Two measured caveats live here rather than in a paragraph, because they change
how the columns may be read on real data: a fold carries no intrinsic
curvature, so the curvature-density column is blind to creases; and the most
degree-concentrated meshes are the uniform ones, so that column tracks
regularity rather than craft.
"""

import math

import numpy as np
import pytest

from mesh_filter.contrast import (
    box,
    bump_disc,
    grid_from_axes,
    icosphere,
    prism,
    tent,
    torus,
    triangular_patch,
)
from mesh_filter.features.artistic import compute_artistic_features
from mesh_filter.features.basic import compute_basic_features


def _columns(vertices: np.ndarray, faces: np.ndarray) -> dict[str, object]:
    """Every basic and artistic column for one mesh, as the feature table has them."""
    basic = compute_basic_features(vertices, faces)
    artistic = compute_artistic_features(vertices, faces)
    return {**basic, **artistic}


def test_equilateral_median_tracks_triangle_shape_at_a_fixed_outline() -> None:
    """Same lattice and same outline; only the triangles change.

    `triangular_patch` at height sqrt(3)/2 is the equilateral lattice. The same
    patch at a much smaller row height packs many more rows into the same box,
    so the outline and the combinatorics stay put while every triangle turns
    into a sliver — and the median is the column that has to move.
    """
    tall_vertices, tall_faces = triangular_patch(10, 13, math.sqrt(3.0) / 2.0)
    flat_vertices, flat_faces = triangular_patch(390, 13, 0.02)
    assert tall_vertices[:, 1].max() == pytest.approx(
        flat_vertices[:, 1].max(), rel=0.01
    )

    tall = _columns(tall_vertices, tall_faces)
    flat = _columns(flat_vertices, flat_faces)

    assert tall["equilateral_median"] == pytest.approx(1.0)
    assert tall["sliver_ratio"] == 0.0
    assert flat["equilateral_median"] < 0.1
    assert flat["sliver_ratio"] == pytest.approx(1.0)


def test_qem_redundancy_falls_as_a_surface_is_refined() -> None:
    """Each refinement adds vertices that a flat neighbourhood can lose for free."""
    values = [_columns(*icosphere(level))["qem_redundancy_median"] for level in range(4)]

    assert values == sorted(values, reverse=True)
    assert values[0] > 0.5
    assert values[-1] < 0.05


def test_qem_redundancy_is_high_on_coarse_creased_shapes() -> None:
    """A box spends eight vertices on twelve triangles and every one holds a corner."""
    sphere = _columns(*icosphere(2))["qem_redundancy_median"]

    for vertices, faces in (box(), prism(6), prism(12)):
        assert _columns(vertices, faces)["qem_redundancy_median"] > sphere


def test_curvature_density_needs_intrinsic_curvature_to_exist() -> None:
    """A fold is flat in the intrinsic sense, so the column reads blank on it.

    Both tents carry a crease, and the graded one crowds its tessellation along
    that crease — visually exactly the "dense at bends" the criteria describe —
    but the angle defect along a fold is zero, so there is no curvature to
    correlate with and the honest answer is "not measurable". Whatever this
    column detects on real assets, it is not creases.
    """
    assert _columns(*tent(15, 6, grading=1.0))["curvature_density_correlation"] == ""
    assert _columns(*tent(15, 6, grading=3.0))["curvature_density_correlation"] == ""

    # A bump does carry curvature, and there the column produces a number.
    assert _columns(*bump_disc())["curvature_density_correlation"] != ""


def test_curvature_density_responds_to_where_the_tessellation_is_crowded() -> None:
    """Move the rings and the sign moves with them.

    A polar grid is already densest at the centre for free, so the honest
    contrast is between crowding the rings further towards the bump's curvature
    and crowding them away from it: same shape, same ring count, one
    reparametrisation apart. The torus pair shows the same effect on a closed
    shape, where even spacing reads strongly negative because the inner
    equator is both denser and negatively curved.
    """
    follows = _columns(*bump_disc(grading=2.5))["curvature_density_correlation"]
    against = _columns(*bump_disc(grading=0.5))["curvature_density_correlation"]

    assert follows > 0.5
    assert against < -0.3

    plain = _columns(*torus(24, 12))["curvature_density_correlation"]
    crowded = _columns(*torus(24, 12, minor_grading=2.5))[
        "curvature_density_correlation"
    ]

    assert plain < -0.5
    assert crowded > 0.2


def test_ring_cv_responds_to_local_spacing_irregularity() -> None:
    """Alternating wide and narrow rows make every vertex see mixed edge lengths.

    The two grids keep the same topology — ten rows, ten columns, the same
    triangulation — so the spacing is the only thing that differs.
    """
    even = _columns(*grid_from_axes(np.arange(10.0), np.arange(10.0)))
    gaps = np.concatenate([np.tile([1.0, 0.25], 4), [1.0]])
    uneven = _columns(*grid_from_axes(np.arange(10.0), np.concatenate([[0.0], np.cumsum(gaps)])))

    ratio = uneven["one_ring_edge_length_cv_median"] / even[
        "one_ring_edge_length_cv_median"
    ]
    assert ratio > 1.5


def test_degree_concentration_measures_regularity_not_craft() -> None:
    """Recorded non-separation: the claimed direction does not hold on prototypes.

    The guide expects hand-made meshes to cluster their vertex degrees, but here
    the *most* concentrated meshes are the uniform ones — a subdivided sphere is
    almost entirely degree six, and a ring grid exactly so (entropy zero). The
    column measures how regular the tessellation is, and both ends of the
    spectrum can be regular, so it must not be read as a craft detector alone.
    """
    sphere = _columns(*icosphere(3))["vertex_degree_mode_share"]
    boxed = _columns(*box())["vertex_degree_mode_share"]

    assert sphere > boxed
    assert _columns(*torus(24, 12))["vertex_degree_entropy"] == pytest.approx(
        0.0, abs=1e-9
    )


def test_prototypes_separate_on_shape_and_redundancy() -> None:
    """The two ends of the spectrum, as far as the columns can tell them apart.

    Near-equilateral and quick to give vertices away: the uniform end. Elongated
    and vertex-hungry: the hand-made end. One caveat is asserted with them: a
    uniform quad grid sits at q = sqrt(3)/2 — the same place as a box — so both
    stay under the "too regular" threshold and a quad-remeshed asset cannot be
    told from a hand-made one by the equilateral column alone.
    """
    sphere = _columns(*icosphere(2))

    for vertices, faces in (box(), prism(6), prism(12)):
        made = _columns(vertices, faces)
        assert sphere["equilateral_median"] > made["equilateral_median"]
        assert sphere["qem_redundancy_median"] < made["qem_redundancy_median"]

    assert _columns(*torus(24, 12))["equilateral_median"] < 0.9
    assert _columns(*box())["equilateral_median"] < 0.9
