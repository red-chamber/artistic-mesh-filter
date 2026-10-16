"""The standalone filter: a folder in, a verdict table out."""

from pathlib import Path

import numpy as np
import pytest

from mesh_filter.contrast import box, grid_from_axes
from mesh_filter.manifest import read_csv
from mesh_filter.screening import (
    load_rules,
    required_artistic_columns,
    screen_directory,
)

RULES_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "screening" / "artistic_v1.yaml"
)


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    lines = [f"v {x:.6g} {y:.6g} {z:.6g}" for x, y, z in vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_filter_screens_a_folder_and_names_the_reasons(tmp_path) -> None:
    """A minimal cube is clean; a rectangle of eight triangles is waste."""
    _write_obj(tmp_path / "cube.obj", *box(1.0))
    _write_obj(tmp_path / "plate.obj", *grid_from_axes(np.arange(3.0), np.arange(3.0)))
    output = tmp_path.parent / "verdicts.csv"

    written, report = screen_directory(tmp_path, output, RULES_PATH)

    rows = {row["path"]: row for row in read_csv(written)}
    assert rows["cube.obj"]["verdict"] == "clean"
    assert rows["plate.obj"]["verdict"] == "rejected"
    assert "redundant_faces" in rows["plate.obj"]["failed_rules"]
    assert float(rows["plate.obj"]["redundant_face_share"]) > 0.7
    assert rows["cube.obj"]["failed_rules"] == ""
    assert report["verdicts"] == {"rejected": 1, "flagged": 0, "clean": 1}
    assert report["assets"] == 2
    assert report["failures"] == 0


def test_filter_skips_other_extensions_and_isolates_a_broken_file(tmp_path) -> None:
    _write_obj(tmp_path / "good.obj", *box(1.0))
    (tmp_path / "note.txt").write_text("not a mesh", encoding="utf-8")
    (tmp_path / "broken.obj").write_text("this is not an obj file\n12345\n", encoding="ascii")

    written, report = screen_directory(tmp_path, tmp_path.parent / "verdicts.csv", RULES_PATH)

    assert [row["path"] for row in read_csv(written)] == ["good.obj"]
    assert report["skipped_non_mesh_files"] == 1
    assert report["failures"] == 1
    assert report["failure_examples"][0]["path"] == "broken.obj"


def test_filter_refuses_a_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        screen_directory(tmp_path / "nope", tmp_path / "out.csv", RULES_PATH)


def test_filter_single_file_prints_its_verdict_and_reasons(tmp_path, capsys) -> None:
    """One file in: the answer goes to the terminal, not just into a CSV."""
    _write_obj(tmp_path / "plate.obj", *grid_from_axes(np.arange(3.0), np.arange(3.0)))
    _write_obj(tmp_path / "cube.obj", *box(1.0))

    _, report = screen_directory(
        tmp_path / "plate.obj", tmp_path.parent / "one.csv", RULES_PATH
    )
    verdict = capsys.readouterr().out
    assert "rejected（不合格）" in verdict
    assert "redundant_faces" in verdict
    assert "冗余面占比" in verdict
    assert report["assets"] == 1

    _, report = screen_directory(
        tmp_path / "cube.obj", tmp_path.parent / "two.csv", RULES_PATH
    )
    clean = capsys.readouterr().out
    assert "clean（通过）" in clean
    # A cube has no measurable curvature correlation; that is "not measurable",
    # shown as such, not laundered into a failed check.
    assert "无法测量: density_follows_curvature" in clean


def test_filter_single_unparseable_file_says_so(tmp_path, capsys) -> None:
    (tmp_path / "broken.obj").write_text("not an obj\n", encoding="ascii")

    _, report = screen_directory(
        tmp_path / "broken.obj", tmp_path.parent / "out.csv", RULES_PATH
    )

    output = capsys.readouterr().out
    assert "无法解析" in output
    assert report["failures"] == 1


def test_filter_only_computes_the_columns_the_rules_read() -> None:
    rules, _ = load_rules(RULES_PATH)

    columns = required_artistic_columns(rules)

    assert {
        "redundant_face_share",
        "qem_redundancy_median",
        "curvature_density_correlation",
        "vertex_degree_mode_share",
    } <= columns
    assert "local_edge_direction_anisotropy_mean" not in columns
