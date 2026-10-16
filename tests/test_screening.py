"""The screening rules, tested against the statements they encode.

The rule file is this project's operational definition of the criteria, so these
tests read the shipped `configs/screening/artistic_v1.yaml` rather than a private
copy: a threshold edited by mistake — or rule semantics that drift — should
break here before the corpus gets re-ranked.
"""

from pathlib import Path

import pytest

from mesh_filter.config import ConfigError, load_project_config
from mesh_filter.manifest import read_csv, write_csv
from mesh_filter.screening import (
    evaluate_rules,
    load_rules,
    overall_verdict,
    run_screening,
)

RULES_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "screening" / "artistic_v1.yaml"
)


@pytest.fixture(scope="module")
def rules():
    loaded, _ = load_rules(RULES_PATH)
    return loaded


def _row(**overrides: str) -> dict[str, str]:
    """A row that passes every rule in the shipped config."""
    row = {
        "parse_status": "parsed",
        "zero_area_face_ratio": "0.0",
        "duplicate_face_count": "0",
        "nonmanifold_edge_count": "0",
        "face_count": "1200",
        "small_angle_ratio": "0.05",
        "area_coefficient_of_variation": "1.5",
        "qem_redundancy_median": "0.05",
        "redundant_face_share": "0.0",
        "equilateral_median": "0.5",
        "vertex_degree_mode_share": "0.5",
        "curvature_density_correlation": "0.5",
    }
    row.update(overrides)
    return row


def test_a_clean_row_passes_every_rule(rules) -> None:
    verdicts = evaluate_rules(_row(), rules)

    assert set(verdicts.values()) == {"pass"}
    assert overall_verdict(verdicts, rules) == "clean"


def test_equilateral_alone_is_not_a_rejection_reason(rules) -> None:
    """Criterion 2 as the group restated it: regular faces are not the problem.

    A mesh whose triangles are almost all near-equilateral but whose vertices
    each still hold shape must not be rejected — the signature the criteria
    distrust is "regular *and* redundant", not regular on its own.
    """
    row = _row(equilateral_median="0.98", qem_redundancy_median="0.05")

    verdicts = evaluate_rules(row, rules)

    assert verdicts["uniform_remesh_signature"] == "pass"
    assert overall_verdict(verdicts, rules) == "clean"


def test_regular_faces_plus_redundant_vertices_are_flagged(rules) -> None:
    """The combination the criteria describe: 为了规则形状堆冗余的面。"""
    row = _row(equilateral_median="0.98", qem_redundancy_median="0.0")

    verdicts = evaluate_rules(row, rules)

    assert verdicts["uniform_remesh_signature"] == "fail"
    assert verdicts["redundant_vertices"] == "fail"
    assert overall_verdict(verdicts, rules) == "flagged"


def test_redundant_faces_gate_on_the_calibrated_share(rules) -> None:
    """判据二的核心度量：可以删掉的面成为拒绝理由，等边度不是。

    Calibrated on the first 45-sample round: every mesh the human accepted sat
    at or below 0.075 of removable faces; the strongest rejection read 0.55.
    """
    clean = evaluate_rules(_row(redundant_face_share="0.01"), rules)
    assert clean["redundant_faces"] == "pass"
    assert clean["redundant_faces_elevated"] == "pass"

    elevated = evaluate_rules(_row(redundant_face_share="0.08"), rules)
    assert elevated["redundant_faces"] == "pass"
    assert elevated["redundant_faces_elevated"] == "fail"
    assert overall_verdict(elevated, rules) == "flagged"

    wasteful = evaluate_rules(_row(redundant_face_share="0.30"), rules)
    assert wasteful["redundant_faces"] == "fail"
    assert overall_verdict(wasteful, rules) == "rejected"


def test_face_count_bounds_follow_the_groups_papers(rules) -> None:
    """上界 20k 来自论文；下界经首轮校准从 500 降到 50。

    The 500-face floor is a training-data rule in LATO and MeshRipple, not a
    quality rule: the first calibration round found it killing good meshes (a
    12-triangle cuboid, a 196-face fan). 50 keeps only the scrap guard.
    """
    for count in ("49", "20001"):
        verdicts = evaluate_rules(_row(face_count=count), rules)
        assert verdicts["face_count_range"] == "fail"
        assert overall_verdict(verdicts, rules) == "rejected"
    for count in ("50", "500", "20000"):
        assert evaluate_rules(_row(face_count=count), rules)["face_count_range"] == "pass"


def test_narrow_faces_rule_matches_meshripple(rules) -> None:
    """最小内角 <5° 的面超过两成即丢——课题组论文附录里的原话。"""
    assert evaluate_rules(_row(small_angle_ratio="0.2"), rules)["narrow_faces"] == "pass"

    verdicts = evaluate_rules(_row(small_angle_ratio="0.21"), rules)

    assert verdicts["narrow_faces"] == "fail"
    assert overall_verdict(verdicts, rules) == "rejected"


def test_validity_defects_reject(rules) -> None:
    for overrides in (
        {"parse_status": "failed"},
        {"zero_area_face_ratio": "0.5"},
        {"duplicate_face_count": "3"},
    ):
        verdicts = evaluate_rules(_row(**overrides), rules)
        assert overall_verdict(verdicts, rules) == "rejected", overrides


def test_nonmanifold_edges_are_a_flag_not_a_gate(rules) -> None:
    """First calibration round: this defect does not drive the human's judgement.

    Four sampled meshes tripped it — all on a single edge — and the human
    accepted two of them and rejected the other two, so it cannot gate.
    """
    verdicts = evaluate_rules(_row(nonmanifold_edge_count="1"), rules)

    assert verdicts["nonmanifold_edges"] == "fail"
    assert overall_verdict(verdicts, rules) == "flagged"


def test_a_blank_flag_column_is_unknown_not_failed(rules) -> None:
    """A column that could not be measured must not be laundered into a verdict."""
    row = _row(curvature_density_correlation="")

    verdicts = evaluate_rules(row, rules)

    assert verdicts["density_follows_curvature"] == "unknown"
    assert overall_verdict(verdicts, rules) == "clean"


def test_a_reject_outranks_any_number_of_flags(rules) -> None:
    row = _row(
        face_count="40",
        area_coefficient_of_variation="0.01",
        equilateral_median="0.99",
        qem_redundancy_median="0.0",
        vertex_degree_mode_share="0.1",
        curvature_density_correlation="-1.0",
    )

    verdicts = evaluate_rules(row, rules)

    assert overall_verdict(verdicts, rules) == "rejected"


def test_malformed_rule_files_are_refused(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "rules:\n"
        "  - name: x\n"
        "    severity: sometimes\n"
        "    conditions:\n"
        "      - {column: a, op: max, value: 1}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_rules(bad)


def test_screen_writes_the_table_and_counts_by_source(make_config) -> None:
    config_path = make_config(
        sources={},
        outputs={
            "manifest": "data/manifests/assets.csv",
            "features": "data/manifests/features.csv",
        },
    )
    config = load_project_config(config_path)
    clean = {"asset_id": "a1", "source_dataset": "toys4k", "label": "1", **_row()}
    broken = {
        "asset_id": "a2",
        "source_dataset": "objaverse",
        "label": "0",
        **_row(face_count="40", curvature_density_correlation=""),
    }
    write_csv(config.output_path("features"), [clean, broken], fieldnames=list(clean))

    table_path, report = run_screening(config, RULES_PATH)

    rows = {row["asset_id"]: row for row in read_csv(table_path)}
    assert rows["a1"]["verdict"] == "clean"
    assert rows["a2"]["verdict"] == "rejected"
    assert rows["a2"]["failed_rules"] == "face_count_range"
    assert rows["a2"]["unknown_rules"] == "density_follows_curvature"
    assert report["verdicts"]["toys4k"] == {"rejected": 0, "flagged": 0, "clean": 1}
    assert report["assets"] == 2
    assert report["unapplied"], "the unapplied rules must stay visible in the report"
