"""Rule-based screening for artistic meshes.

The three criteria in `docs/LABELING_GUIDE.md` are judgements, not measurements,
so they land here as an explicit rule set in `configs/screening/*.yaml`. Every
rule carries its provenance — a paper, a criterion, a standard validity check —
and the thresholds can be read, argued with, and re-tuned without touching
code.

Two severities:

- `reject` — a hard gate. The validity defects and the count limits the group's
  own papers filter on (LATO and MeshRipple both drop meshes outside [500, 20k]
  faces; MeshRipple also drops meshes whose narrow faces pass 20%);
- `flag` — recorded and counted, never a gate. The criterion-2 and criterion-3
  thresholds are not calibrated yet, so acting on them now would dress up a
  guess as a filter. The first job of a screening run is to report what the
  rules do to the corpus and rank it; the extremes then go to a human, and the
  thresholds move after that.

Criterion 2 is the reason the flag set looks the way it does. As the group
restated it, near-equilateral triangles are not bad by themselves — spending
extra faces to keep triangles regular is. So no rule rejects on equilateral-ness
alone: `uniform_remesh_signature` fires only on "near-equilateral *and*
redundant", and `redundant_vertices` / `homogeneous_faces` carry the redundancy
side by themselves.

Rules that are known but not yet applicable (the vertex-to-face ratio's merge
convention, MeshRipple's pruned component count, boundary length in normalised
units, self-intersection) stay listed in the rule file under `unapplied`, and the
report repeats them so the gap is visible rather than forgotten.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mesh_filter.config import ConfigError, ProjectConfig, load_yaml
from mesh_filter.manifest import read_csv, write_csv

OPS = ("equals", "min", "max")
SEVERITIES = ("reject", "flag")
MATCHES = ("all", "any")

TABLE_FIELDS = [
    "asset_id",
    "source_dataset",
    "label",
    "verdict",
    "flags_passed",
    "flags_failed",
    "flags_unknown",
    "failed_rules",
    "unknown_rules",
]


@dataclass(frozen=True)
class Condition:
    column: str
    op: str
    value: Any


@dataclass(frozen=True)
class Rule:
    name: str
    severity: str
    provenance: str
    match: str
    conditions: tuple[Condition, ...]


def load_rules(path: str | Path) -> tuple[list[Rule], list[dict[str, str]]]:
    """Read and validate a rule file. Returns (rules, unapplied)."""
    values = load_yaml(path)
    raw_rules = values.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ConfigError(f"{path}: expected a non-empty `rules` list")
    rules: list[Rule] = []
    seen: set[str] = set()
    for entry in raw_rules:
        name = entry.get("name")
        if not name or name in seen:
            raise ConfigError(f"{path}: rule names must be present and unique ({name!r})")
        seen.add(name)
        severity = entry.get("severity")
        if severity not in SEVERITIES:
            raise ConfigError(f"{path}: rule {name!r} has severity {severity!r}")
        match = entry.get("match", "all")
        if match not in MATCHES:
            raise ConfigError(f"{path}: rule {name!r} has match {match!r}")
        raw_conditions = entry.get("conditions")
        if not isinstance(raw_conditions, list) or not raw_conditions:
            raise ConfigError(f"{path}: rule {name!r} has no conditions")
        conditions = []
        for raw in raw_conditions:
            if raw.get("op") not in OPS:
                raise ConfigError(f"{path}: rule {name!r} uses op {raw.get('op')!r}")
            if not raw.get("column"):
                raise ConfigError(f"{path}: rule {name!r} condition lacks a column")
            conditions.append(
                Condition(column=str(raw["column"]), op=str(raw["op"]), value=raw.get("value"))
            )
        rules.append(
            Rule(
                name=str(name),
                severity=str(severity),
                provenance=str(entry.get("provenance", "")),
                match=str(match),
                conditions=tuple(conditions),
            )
        )
    unapplied = [dict(entry) for entry in values.get("unapplied", [])]
    return rules, unapplied


def _condition_verdict(row: dict[str, str], condition: Condition) -> str:
    raw = row.get(condition.column)
    if raw is None or raw == "":
        return "unknown"
    if condition.op == "equals":
        return "pass" if str(raw) == str(condition.value) else "fail"
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return "unknown"
    if condition.op == "min":
        return "pass" if number >= float(condition.value) else "fail"
    return "pass" if number <= float(condition.value) else "fail"


def evaluate_rules(row: dict[str, str], rules: list[Rule]) -> dict[str, str]:
    """Per-rule verdicts for one feature row.

    `match: all` passes when every condition passes; `match: any` passes when at
    least one does — which is how "fails only when both things are true at once"
    is written. A failing condition settles the rule regardless of the rest; a
    rule with no failing condition but a blank one is `unknown`, not `pass`.
    """
    verdicts: dict[str, str] = {}
    for rule in rules:
        outcomes = [_condition_verdict(row, condition) for condition in rule.conditions]
        if rule.match == "any":
            if "pass" in outcomes:
                verdicts[rule.name] = "pass"
            elif "unknown" in outcomes:
                verdicts[rule.name] = "unknown"
            else:
                verdicts[rule.name] = "fail"
        elif "fail" in outcomes:
            verdicts[rule.name] = "fail"
        elif "unknown" in outcomes:
            verdicts[rule.name] = "unknown"
        else:
            verdicts[rule.name] = "pass"
    return verdicts


def overall_verdict(verdicts: dict[str, str], rules: list[Rule]) -> str:
    """`rejected` beats `flagged` beats `clean`."""
    severity = {rule.name: rule.severity for rule in rules}
    if any(
        verdict == "fail" and severity[name] == "reject"
        for name, verdict in verdicts.items()
    ):
        return "rejected"
    if any(
        verdict == "fail" and severity[name] == "flag" for name, verdict in verdicts.items()
    ):
        return "flagged"
    return "clean"


def screening_paths(config: ProjectConfig) -> tuple[Path, Path]:
    """Where the table and the report go: `<output root>/screening/`.

    Derived from `outputs.manifest` (which every config has) so both machines
    can run this without editing their private path files.
    """
    root = config.output_path("manifest").parent.parent / "screening"
    return root / "screening_v1.csv", root / "screening_report.json"


def run_screening(
    config: ProjectConfig, rules_path: str | Path
) -> tuple[Path, dict[str, Any]]:
    rules, unapplied = load_rules(rules_path)
    rows = read_csv(config.output_path("features"))
    flag_names = [rule.name for rule in rules if rule.severity == "flag"]

    table_rows: list[dict[str, Any]] = []
    rule_counts: dict[str, dict[str, dict[str, int]]] = {
        rule.name: {} for rule in rules
    }
    verdict_counts: dict[str, dict[str, int]] = {}
    flag_scores: dict[str, list[int]] = {}
    for row in rows:
        verdicts = evaluate_rules(row, rules)
        verdict = overall_verdict(verdicts, rules)
        source = row.get("source_dataset", "unknown")
        for name, rule_verdict in verdicts.items():
            counts = rule_counts[name].setdefault(source, {"pass": 0, "fail": 0, "unknown": 0})
            counts[rule_verdict] += 1
        verdict_counts.setdefault(source, {"rejected": 0, "flagged": 0, "clean": 0})
        verdict_counts[source][verdict] += 1
        passed = sum(1 for name in flag_names if verdicts[name] == "pass")
        flag_scores.setdefault(source, []).append(passed)
        table_rows.append(
            {
                "asset_id": row.get("asset_id", ""),
                "source_dataset": source,
                "label": row.get("label", ""),
                "verdict": verdict,
                "flags_passed": passed,
                "flags_failed": sum(1 for name in flag_names if verdicts[name] == "fail"),
                "flags_unknown": sum(1 for name in flag_names if verdicts[name] == "unknown"),
                "failed_rules": ";".join(
                    name for name, value in verdicts.items() if value == "fail"
                ),
                "unknown_rules": ";".join(
                    name for name, value in verdicts.items() if value == "unknown"
                ),
            }
        )

    table_path, report_path = screening_paths(config)
    write_csv(table_path, table_rows, fieldnames=TABLE_FIELDS)

    score_summary = {
        source: {
            "n": len(scores),
            "mean_flags_passed": round(sum(scores) / len(scores), 3) if scores else None,
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
        }
        for source, scores in sorted(flag_scores.items())
    }
    report: dict[str, Any] = {
        "output": str(table_path),
        "report": str(report_path),
        "rules_config": str(Path(rules_path)),
        "assets": len(rows),
        "flag_rule_count": len(flag_names),
        "verdicts": verdict_counts,
        "rules": [
            {
                "name": rule.name,
                "severity": rule.severity,
                "match": rule.match,
                "provenance": rule.provenance,
                "conditions": [
                    {"column": c.column, "op": c.op, "value": c.value}
                    for c in rule.conditions
                ],
                "counts": rule_counts[rule.name],
            }
            for rule in rules
        ],
        "flag_score": score_summary,
        "unapplied": unapplied,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return table_path, report
