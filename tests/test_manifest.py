from pathlib import Path

from mesh_filter.manifest import make_asset_id, read_csv, write_csv


def test_asset_id_is_stable_and_source_specific() -> None:
    assert make_asset_id("toys4k", "a.obj") == make_asset_id("toys4k", "a.obj")
    assert make_asset_id("toys4k", "a.obj") != make_asset_id("objaverse", "a.obj")


def test_csv_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "records.csv"
    rows = [{"asset_id": "abc", "label": 1}]
    write_csv(path, rows, ["asset_id", "label"])
    assert read_csv(path) == [{"asset_id": "abc", "label": "1"}]
