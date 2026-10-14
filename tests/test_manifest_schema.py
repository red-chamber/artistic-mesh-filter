import json
from pathlib import Path

from mesh_filter.manifest import MANIFEST_FIELDS

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "data" / "schemas" / "manifest.schema.json"


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_schema_covers_every_manifest_column() -> None:
    """The schema is documentation only if it stays in sync with the writer."""
    schema = _schema()
    assert set(schema["properties"]) == set(MANIFEST_FIELDS)
    assert set(schema["required"]) == set(MANIFEST_FIELDS)


def test_schema_enums_match_pipeline_values() -> None:
    properties = _schema()["properties"]
    assert properties["label"]["enum"] == [0, 1]
    assert properties["parse_status"]["enum"] == ["not_run", "parsed", "failed"]
    assert properties["split"]["enum"] == ["", "train", "val", "test"]
    assert properties["label_origin"]["enum"] == [
        "dataset_proxy",
        "human_review",
        "synthetic_fixture",
    ]
