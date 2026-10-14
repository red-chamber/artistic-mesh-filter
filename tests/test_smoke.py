import json
from pathlib import Path

from mesh_filter.config import load_project_config
from mesh_filter.smoke import run_smoke_test


def _sources(positive: Path, negative: Path) -> dict:
    def entry(root: Path, label: int) -> dict:
        return {
            "root": root.as_posix(),
            "label": label,
            "label_origin": "synthetic_fixture",
            "selection_rule": "test_only",
        }

    return {"toys4k": entry(positive, 1), "objaverse": entry(negative, 0)}


def _config(make_config, positive: Path, negative: Path, **extra):
    payload = {"formats": {"extensions": [".obj"]}, **extra}
    return make_config(
        sources=_sources(positive, negative),
        outputs={"smoke_report": "reports/smoke.json"},
        extra=payload,
    )


def test_smoke_passes_on_usable_assets(
    make_config, source_tree, valid_mesh: str, second_mesh: str
) -> None:
    positive = source_tree("positive", {"a.obj": valid_mesh, "b.obj": second_mesh})
    negative = source_tree("negative", {"c.obj": valid_mesh, "d.obj": second_mesh})

    output, report = run_smoke_test(load_project_config(_config(make_config, positive, negative)))

    assert report["verdict"] == "pass"
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["verdict"] == "pass"
    assert report["sources"]["toys4k"]["parsed"] == 2
    assert report["sources"]["toys4k"]["estimated_full_pass_seconds"] is not None


def test_smoke_fails_when_a_source_is_empty(make_config, source_tree, valid_mesh: str) -> None:
    positive = source_tree("positive", {"a.obj": valid_mesh})

    _, report = run_smoke_test(
        load_project_config(_config(make_config, positive, Path("does/not/exist")))
    )

    assert report["verdict"] == "fail"
    failed = {check["name"] for check in report["checks"] if check["status"] == "fail"}
    assert "objaverse.ingestable_not_empty" in failed


def test_smoke_reports_quarantine_buckets(
    make_config, source_tree, valid_mesh: str, invalid_mesh: str
) -> None:
    positive = source_tree("positive", {"good.obj": valid_mesh, "broken.obj": invalid_mesh})
    negative = source_tree("negative", {"good.obj": valid_mesh})

    _, report = run_smoke_test(
        load_project_config(_config(make_config, positive, negative, smoke={"min_parse_rate": 0.0}))
    )

    assert report["sources"]["toys4k"]["failed"] == 1
    assert report["quarantine_plan"]
    bucket = next(iter(report["quarantine_plan"]))
    assert report["quarantine_plan"][bucket]["count"] == 1


def test_smoke_parse_rate_gate_can_fail(
    make_config, source_tree, valid_mesh: str, invalid_mesh: str
) -> None:
    positive = source_tree("positive", {"good.obj": valid_mesh, "broken.obj": invalid_mesh})
    negative = source_tree("negative", {"good.obj": valid_mesh})

    _, report = run_smoke_test(
        load_project_config(_config(make_config, positive, negative, smoke={"min_parse_rate": 1.0}))
    )

    assert report["verdict"] == "fail"


def test_smoke_sample_is_deterministic(
    make_config, source_tree, valid_mesh: str, second_mesh: str
) -> None:
    positive = source_tree("positive", {"a.obj": valid_mesh, "b.obj": second_mesh})
    negative = source_tree("negative", {"c.obj": valid_mesh})
    config = load_project_config(_config(make_config, positive, negative))

    _, first = run_smoke_test(config, sample_per_source=2)
    _, second = run_smoke_test(config, sample_per_source=2)

    def sample_ids(report) -> dict:
        return {
            source: [(row["relpath"], row["parse_status"]) for row in rows]
            for source, rows in report["samples"].items()
        }

    # Timing and memory differ between runs; which files were picked must not.
    assert sample_ids(first) == sample_ids(second)


def _config_with_limits(make_config, positive: Path, negative: Path, limits: dict):
    sources = _sources(positive, negative)
    sources["objaverse"]["selection_limits"] = limits
    return make_config(
        sources=sources,
        outputs={"smoke_report": "reports/smoke.json"},
        extra={"formats": {"extensions": [".obj"]}},
    )


def test_smoke_enforces_declared_selection_limits(
    make_config, source_tree, valid_mesh: str
) -> None:
    """The negative set's screening rule is part of its label definition."""
    positive = source_tree("positive", {"a.obj": valid_mesh})
    negative = source_tree("negative", {"big.obj": valid_mesh})  # cube: 12 faces

    config = _config_with_limits(make_config, positive, negative, {"max_face_count": 4})
    _, report = run_smoke_test(load_project_config(config))

    assert report["verdict"] == "fail"
    failed = {check["name"] for check in report["checks"] if check["status"] == "fail"}
    assert "objaverse.selection_limits" in failed
    assert report["sources"]["objaverse"]["selection_limit_violation_count"] == 1
    assert report["sources"]["objaverse"]["selection_limit_violations"][0]["face_count"] == 12


def test_smoke_accepts_assets_inside_selection_limits(
    make_config, source_tree, valid_mesh: str
) -> None:
    positive = source_tree("positive", {"a.obj": valid_mesh})
    negative = source_tree("negative", {"small.obj": valid_mesh})

    config = _config_with_limits(make_config, positive, negative, {"max_face_count": 100})
    _, report = run_smoke_test(load_project_config(config))

    assert report["verdict"] == "pass"
    check = next(item for item in report["checks"] if item["name"] == "objaverse.selection_limits")
    assert check["status"] == "pass"
