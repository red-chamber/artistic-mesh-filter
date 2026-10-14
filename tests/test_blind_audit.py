
from mesh_filter.config import load_project_config
from mesh_filter.evaluation.blind_audit import BLIND_FIELDS, build_audit_gold
from mesh_filter.manifest import read_csv

OUTPUTS = {
    "features": "data/features.csv",
    "split_root": "data/splits/v1",
    "audit_gold": "out/audit_gold.csv",
    "audit_gold_key": "out/audit_gold_key.csv",
}


def _config(make_config, extra: dict | None = None):
    return load_project_config(
        make_config(
            sources={},
            outputs=OUTPUTS,
            extra={"audit_gold": {"size": 8, "seed": 7, **(extra or {})}},
        )
    )


def _two_bin_population() -> list[tuple]:
    """Both labels in two face-count bins, three assets per stratum."""
    specs = []
    for bin_faces in (100, 9000):
        for label in (0, 1):
            for index in range(3):
                specs.append(
                    (f"a{bin_faces}l{label}n{index}", label, "test", bin_faces + index, index + 10)
                )
    return specs


def test_blind_sheet_hides_source_and_label(make_config, write_eval_tables) -> None:
    """The reviewer's file must not contain the answer it is meant to produce."""
    write_eval_tables(_two_bin_population())

    blind_path, _ = build_audit_gold(_config(make_config))

    rows = read_csv(blind_path)
    assert list(rows[0]) == BLIND_FIELDS
    for forbidden in ("source_dataset", "label", "raw_relpath", "face_count"):
        assert forbidden not in rows[0]
    assert all(row["human_label"] == "" for row in rows)


def test_key_maps_anonymous_names_back(make_config, write_eval_tables) -> None:
    write_eval_tables(_two_bin_population())

    blind_path, summary = build_audit_gold(_config(make_config))
    key_rows = read_csv(summary["key_output"])
    blind_rows = read_csv(blind_path)

    assert len(key_rows) == len(blind_rows)
    assert {row["asset_id"] for row in key_rows} == {row["asset_id"] for row in blind_rows}
    pairs = {(row["anonymous_name"], row["asset_id"]) for row in blind_rows}
    assert pairs == {(row["anonymous_name"], row["asset_id"]) for row in key_rows}
    assert {row["source_dataset"] for row in key_rows} == {"toys4k", "objaverse"}
    # The key must be joinable back to the strata the sample was drawn from.
    assert all(row["face_count_bin"] for row in key_rows)
    assert all(row["face_count"] for row in key_rows)


def test_every_stratum_is_sampled_when_the_budget_allows(
    make_config, write_eval_tables
) -> None:
    write_eval_tables(_two_bin_population())

    _, summary = build_audit_gold(_config(make_config), size=8)

    assert summary["sampled"] == 8
    assert len(summary["strata"]) == 4
    assert all(entry["sampled"] >= 1 for entry in summary["strata"])
    assert sum(entry["sampled"] for entry in summary["strata"]) == 8


def test_request_larger_than_population_returns_everything(
    make_config, write_eval_tables
) -> None:
    write_eval_tables(_two_bin_population())

    _, summary = build_audit_gold(_config(make_config), size=1000)

    assert summary["requested"] == 1000
    assert summary["sampled"] == 12
    assert all(entry["sampled"] == entry["available"] for entry in summary["strata"])


def test_anonymous_names_are_sequential(make_config, write_eval_tables) -> None:
    write_eval_tables(_two_bin_population())

    blind_path, _ = build_audit_gold(_config(make_config), size=4)

    names = [row["anonymous_name"] for row in read_csv(blind_path)]
    assert names == [f"A{index:04d}" for index in range(1, 5)]


def test_output_is_deterministic(make_config, write_eval_tables) -> None:
    write_eval_tables(_two_bin_population())
    config = _config(make_config)

    first, _ = build_audit_gold(config, size=6)
    first_text = first.read_text(encoding="utf-8")
    second, _ = build_audit_gold(config, size=6)

    assert second.read_text(encoding="utf-8") == first_text


def test_skipped_assets_are_counted(make_config, write_eval_tables) -> None:
    write_eval_tables(_two_bin_population())

    _, summary = build_audit_gold(_config(make_config), size=4)

    assert summary["skipped"] == {"unparsed": 0, "incomplete": 0}
    assert summary["population"] == 12
