import json
from pathlib import Path

from mesh_filter.config import load_project_config
from mesh_filter.probe import probe_sources
from mesh_filter.quarantine import BUCKETS


def _sources(positive: Path, negative: Path) -> dict:
    def entry(root: Path, label: int) -> dict:
        return {
            "root": root.as_posix(),
            "label": label,
            "label_origin": "synthetic_fixture",
            "selection_rule": "test_only",
        }

    return {"toys4k": entry(positive, 1), "objaverse": entry(negative, 0)}


def _probe(make_config, positive: Path, negative: Path):
    config_path = make_config(
        sources=_sources(positive, negative),
        outputs={"probe_report": "reports/probe.json"},
        extra={"formats": {"extensions": [".obj"]}},
    )
    return probe_sources(load_project_config(config_path), 10)


def test_probe_separates_parsed_from_quarantined(
    make_config, source_tree, valid_mesh: str, invalid_mesh: str
) -> None:
    positive = source_tree("positive", {"good.obj": valid_mesh, "broken.obj": invalid_mesh})
    negative = source_tree("negative", {"good.obj": valid_mesh})

    output, report = _probe(make_config, positive, negative)

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["sources"]
    aggregate = report["sources"]["toys4k"]["aggregate"]
    assert aggregate["parsed"] == 1
    assert aggregate["failed"] == 1
    broken = next(
        row for row in report["sources"]["toys4k"]["sampled"] if row["parse_status"] != "parsed"
    )
    assert broken["error_code"]
    assert broken["quarantine_bucket"] in BUCKETS


def test_probe_reports_extensions_outside_the_whitelist(
    make_config, source_tree, valid_mesh: str
) -> None:
    """Formats missing from the whitelist would silently ingest nothing."""
    positive = source_tree("positive", {"good.obj": valid_mesh, "notes.txt": "not a mesh\n"})
    negative = source_tree("negative", {"good.obj": valid_mesh})

    _, report = _probe(make_config, positive, negative)

    toys4k = report["sources"]["toys4k"]
    assert toys4k["ingestable_count"] == 1
    assert ".txt" in toys4k["unconfigured_extensions"]
    assert any("invisible" in note for note in toys4k["notes"])


def test_probe_survives_a_missing_source_directory(
    make_config, source_tree, valid_mesh: str
) -> None:
    positive = source_tree("positive", {"good.obj": valid_mesh})

    _, report = _probe(make_config, positive, Path("does/not/exist"))

    objaverse = report["sources"]["objaverse"]
    assert objaverse["file_count"] == 0
    assert objaverse["notes"]


def test_probe_records_structure_and_scale(make_config, source_tree, valid_mesh: str) -> None:
    positive = source_tree("positive", {"good.obj": valid_mesh})
    negative = source_tree("negative", {"good.obj": valid_mesh})

    _, report = _probe(make_config, positive, negative)

    record = report["sources"]["toys4k"]["sampled"][0]
    assert record["parse_status"] == "parsed"
    assert record["part_count"] >= 1
    assert record["vertex_count"] > 0
    assert record["longest_extent"] > 0
    assert record["peak_python_mb"] >= 0
