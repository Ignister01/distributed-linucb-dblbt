"""Static and behavioral locality guarantees."""

import ast
from dataclasses import fields
from pathlib import Path

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "dblbt_fcn"
FORBIDDEN_IDENTIFIERS = (
    "global",
    "jain",
    "other_queue",
    "true_node_count",
    "technology_totals",
)


@pytest.mark.parametrize("module_name", ["observation.py", "reward.py"])
def test_local_modules_do_not_reference_forbidden_identifiers(
    module_name: str,
) -> None:
    tree = ast.parse((SOURCE_ROOT / module_name).read_text(encoding="utf-8"))
    identifiers = [
        node.id if isinstance(node, ast.Name) else node.attr
        for node in ast.walk(tree)
        if isinstance(node, (ast.Name, ast.Attribute))
    ]

    assert not {
        identifier
        for identifier in identifiers
        if any(token in identifier.lower() for token in FORBIDDEN_IDENTIFIERS)
    }


def test_attempt_record_exposes_only_own_local_measurements() -> None:
    from dblbt_fcn.observation import AttemptRecord

    assert tuple(field.name for field in fields(AttemptRecord)) == (
        "outcome",
        "elapsed_us",
        "busy_us",
        "interruptions",
        "access_delay_us",
        "queue_occupancy_ratio",
        "arrivals",
        "retries",
        "effective_data_us",
    )
