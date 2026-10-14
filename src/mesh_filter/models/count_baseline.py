from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mesh_filter.config import ProjectConfig, load_yaml
from mesh_filter.evaluation.metrics import binary_metrics
from mesh_filter.manifest import read_csv, write_csv


def _read_feature_matrix(
    feature_path: Path,
    split_path: Path,
    feature_columns: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    features = {row["asset_id"]: row for row in read_csv(feature_path)}
    ids: list[str] = []
    matrix: list[list[float]] = []
    labels: list[int] = []
    for split_row in read_csv(split_path):
        asset_id = split_row["asset_id"]
        feature_row = features.get(asset_id)
        if not feature_row or feature_row.get("parse_status") != "parsed":
            continue
        try:
            vector = [float(feature_row[column]) for column in feature_columns]
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(vector).all():
            continue
        ids.append(asset_id)
        matrix.append(vector)
        labels.append(int(split_row["label"]))
    return ids, np.asarray(matrix, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def train_count_baseline(
    config: ProjectConfig,
    experiment_config_path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    from joblib import dump
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    experiment = load_yaml(experiment_config_path)
    columns = list(experiment["feature_columns"])
    seed = int(experiment.get("seed", 42))
    model_options = experiment.get("model", {})
    split_root = config.output_path("split_root")
    feature_path = config.output_path("features")

    _, x_train, y_train = _read_feature_matrix(feature_path, split_root / "train.csv", columns)
    if len(x_train) == 0 or len(np.unique(y_train)) < 2:
        raise ValueError("Training split must contain parsed samples from both labels")

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight=model_options.get("class_weight", "balanced"),
                    max_iter=int(model_options.get("max_iter", 2000)),
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-count-baseline")
    run_root = config.output_path("run_root") / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    dump(model, run_root / "model.joblib")

    all_metrics: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    k = int(experiment.get("evaluation", {}).get("precision_at_k", 50))
    # `matched_test` is a diagnostic subset of the pool, not a training split.
    # Reporting it next to the ordinary test score is the whole point of
    # building it: a score that survives matching is not a polygon-count effect.
    for split_name in ("val", "test", "matched_test"):
        split_path = split_root / f"{split_name}.csv"
        if not split_path.exists():
            continue
        ids, matrix, labels = _read_feature_matrix(feature_path, split_path, columns)
        if len(matrix) == 0:
            all_metrics[split_name] = {"sample_count": 0}
            continue
        scores = model.predict_proba(matrix)[:, 1]
        all_metrics[split_name] = binary_metrics(labels, scores, k=k)
        prediction_rows.extend(
            {
                "asset_id": asset_id,
                "split": split_name,
                "label": int(label),
                "score": float(score),
            }
            for asset_id, label, score in zip(ids, labels, scores, strict=True)
        )

    with (run_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, ensure_ascii=False, indent=2)
    with (run_root / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"experiment": experiment, "paths_config": config.values},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    write_csv(
        run_root / "predictions.csv",
        prediction_rows,
        ["asset_id", "split", "label", "score"],
    )
    return run_root, all_metrics
