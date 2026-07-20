"""Strict canonical raw-record loading and per-run reduction."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from pathlib import Path
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from .experiment import (
    JobSpec,
    _load_completed_job_manifest_metadata,
    artifact_paths,
)
from .config import adaptive_arms
from .io import validated_jsonl_gz_stream
from .metrics import evaluation_utility, jain, nearest_rank_p95
from .provenance import ExecutionProvenance


_ROW_FIELDS = {
    "schema_version",
    "record_type",
    "run_id",
    "scenario_id",
    "policy",
    "seed",
    "config_hash",
    "round_id",
    "tx_start_us",
    "round_end_us",
    "kind",
    "node_ids",
    "technologies",
    "collision_size",
    "reservation_us",
    "effective_data_us",
    "background_busy_us",
    "active_node_ids",
    "backlogged_node_ids",
    "senders",
    "decisions",
    "training_samples",
}
_SENDER_FIELDS = {
    "node_id",
    "technology",
    "selected_backoff_before",
    "next_selected_backoff",
    "interruptions_before",
    "retries_after",
    "db_initialized",
    "deterministic_countdown",
    "delay_us",
    "effective_data_us",
}
_DECISION_FIELDS = {
    "round_id",
    "node_id",
    "previous_arm",
    "arm",
    "profile",
    "context",
    "reward",
    "components",
}
_PROFILE_FIELDS = {"kappa", "alpha", "beta", "m", "b_init"}
_COMPONENT_FIELDS = {
    "airtime_utility",
    "delay_utility",
    "share_utility",
    "reward",
}
_TRAINING_FIELDS = {
    "context",
    "arm",
    "local_reward",
    "local_sequence",
    "node_id",
    "pretraining_seed",
}


def _mapping(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} has invalid fields")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an exact integer >= {minimum}")
    return value


def _number(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite integer or float")
    return float(value)


def _string(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty trimmed string")
    return value


def _strings(value: object, label: str) -> list[str]:
    if type(value) is not list:
        raise ValueError(f"{label} must be a list")
    return [_string(item, label) for item in value]


def _validate_sender(value: object) -> dict[str, Any]:
    sender = _mapping(value, _SENDER_FIELDS, "sender")
    _string(sender["node_id"], "sender node_id")
    _string(sender["technology"], "sender technology")
    for field in (
        "selected_backoff_before",
        "next_selected_backoff",
        "interruptions_before",
        "retries_after",
        "effective_data_us",
    ):
        _integer(sender[field], f"sender {field}")
    if type(sender["db_initialized"]) is not bool:
        raise ValueError("sender db_initialized must be an exact boolean")
    if type(sender["deterministic_countdown"]) is not bool:
        raise ValueError(
            "sender deterministic_countdown must be an exact boolean"
        )
    if sender["delay_us"] is not None:
        _integer(sender["delay_us"], "sender delay_us")
    return sender


def _validate_decision(
    value: object,
    row_id: int,
    active: set[str],
    job: JobSpec,
    provenance: ExecutionProvenance,
) -> dict[str, Any]:
    decision = _mapping(value, _DECISION_FIELDS, "decision")
    decision_round = _integer(decision["round_id"], "decision round_id")
    if decision_round != row_id + 1:
        raise ValueError("decision round_id does not match row")
    if decision_round % 32 != 0:
        raise ValueError("decision round_id must be a completed 32-round boundary")
    node_id = _string(decision["node_id"], "decision node_id")
    if node_id not in active:
        raise ValueError("decision node must be active")
    previous = decision["previous_arm"]
    if previous is not None and not (
        type(previous) is int and 0 <= previous < 24
    ):
        raise ValueError("decision previous_arm is invalid")
    arm = _integer(decision["arm"], "decision arm")
    if arm >= 24:
        raise ValueError("decision arm is invalid")
    profile = _mapping(decision["profile"], _PROFILE_FIELDS, "profile")
    for field in _PROFILE_FIELDS:
        _integer(profile[field], f"profile {field}")
    if profile != adaptive_arms()[arm].model_dump(mode="json"):
        raise ValueError("decision profile does not match arm")
    required_arm: int | None = None
    if job.policy == "fixed_oracle":
        if provenance.mode != "fixed_oracle" or provenance.oracle_arm is None:
            raise ValueError("fixed Oracle manifest provenance is missing")
        required_arm = provenance.oracle_arm
        arm_label = "Oracle arm"
    elif job.policy == "pretrain_arm":
        if provenance.mode != "pretrain_arm" or job.arm_id is None:
            raise ValueError("pretrain manifest provenance is missing")
        required_arm = job.arm_id
        arm_label = "pretrain arm"
    if required_arm is not None and (
        arm != required_arm
        or (previous is not None and previous != required_arm)
    ):
        raise ValueError(f"decision does not match {arm_label}")
    if type(decision["context"]) is not list or len(decision["context"]) != 11:
        raise ValueError("decision context must contain exactly 11 values")
    for item in decision["context"]:
        context_value = _number(item, "decision context")
        if not 0.0 <= context_value <= 1.0:
            raise ValueError("decision context must be normalized")
    if (decision["reward"] is None) != (decision["components"] is None):
        raise ValueError("decision reward and components must appear together")
    if decision["reward"] is not None:
        reward = _number(decision["reward"], "decision reward")
        components = _mapping(
            decision["components"], _COMPONENT_FIELDS, "components"
        )
        for field, item in components.items():
            component = _number(item, "decision component")
            if field != "reward" and not 0.0 <= component <= 1.0:
                raise ValueError("decision utility component must be in 0..1")
        if not math.isclose(
            float(components["reward"]), reward, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("decision reward does not match components")
        base = sum(float(components[field]) for field in _COMPONENT_FIELDS - {"reward"}) / 3
        collision_weight = {
            "collision_weight_0.125": 0.125,
            "collision_weight_0.5": 0.5,
        }.get(job.ablation, 0.25)
        if reward < base - collision_weight - 1e-12 or reward > base + 1e-12:
            raise ValueError("decision reward is outside its feasible range")
    return decision


def _validate_training_sample(value: object) -> dict[str, Any]:
    sample = _mapping(value, _TRAINING_FIELDS, "training sample")
    if type(sample["context"]) is not list or len(sample["context"]) != 11:
        raise ValueError("training context must contain exactly 11 values")
    for item in sample["context"]:
        _number(item, "training context")
    arm = _integer(sample["arm"], "training arm")
    if arm >= 24:
        raise ValueError("training arm is invalid")
    _number(sample["local_reward"], "training local_reward")
    _integer(sample["local_sequence"], "training local_sequence")
    _string(sample["node_id"], "training node_id")
    _integer(sample["pretraining_seed"], "training pretraining_seed")
    return sample


def _topology(job: JobSpec) -> dict[str, str]:
    values: dict[str, str] = {}
    for prefix, technology, count in (
        ("wifi", "wifi", job.scenario.wifi_nodes),
        ("nru", "nru", job.scenario.nru_nodes),
        ("legacy-ap", "legacy_ap", job.scenario.legacy_ap_nodes),
        ("legacy-sta", "legacy_sta", job.scenario.legacy_sta_nodes),
    ):
        for index in range(count):
            values[f"{prefix}-{index:03d}"] = technology
    return values


def _unique(values: list[str], label: str) -> set[str]:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    return set(values)


def _validate_row(
    value: object,
    job: JobSpec,
    expected_round: int,
    provenance: ExecutionProvenance,
) -> dict[str, Any]:
    row = _mapping(value, _ROW_FIELDS, "raw row")
    expected_identity = {
        "schema_version": 1,
        "record_type": "contention_round",
        "run_id": job.run_id,
        "scenario_id": job.scenario.id,
        "policy": job.policy,
        "seed": job.seed,
        "config_hash": job.config_hash,
        "round_id": expected_round,
    }
    if any(row[field] != expected for field, expected in expected_identity.items()):
        raise ValueError("raw row identity does not match expected job")
    start = _integer(row["tx_start_us"], "tx_start_us")
    end = _integer(row["round_end_us"], "round_end_us")
    if end <= start:
        raise ValueError("round_end_us must exceed tx_start_us")
    if row["kind"] not in {"success", "collision"}:
        raise ValueError("raw row kind is invalid")
    topology = _topology(job)
    node_ids = _strings(row["node_ids"], "node_ids")
    technologies = _strings(row["technologies"], "technologies")
    if len(node_ids) != len(technologies):
        raise ValueError("node_ids and technologies lengths differ")
    for field in (
        "collision_size",
        "reservation_us",
        "effective_data_us",
        "background_busy_us",
    ):
        _integer(row[field], field)
    node_set = _unique(node_ids, "node_ids")
    active_values = _strings(row["active_node_ids"], "active_node_ids")
    active = _unique(
        active_values,
        "active_node_ids",
    )
    backlogged = _unique(
        _strings(row["backlogged_node_ids"], "backlogged_node_ids"),
        "backlogged_node_ids",
    )
    join_interval = job.scenario.join_interval_rounds
    lifetime = job.scenario.lifetime_rounds
    topology_ids = list(topology)
    if join_interval is None or lifetime is None:
        expected_active = topology_ids
    else:
        cycle = ((len(topology_ids) - 1) * join_interval) + lifetime
        phase = expected_round % cycle
        expected_active = [
            node_id
            for index, node_id in enumerate(topology_ids)
            if index * join_interval <= phase < index * join_interval + lifetime
        ]
    if active_values != expected_active:
        raise ValueError("active nodes do not match configured lifecycle")
    if not backlogged.issubset(active):
        raise ValueError("backlogged nodes must be active")
    if not node_set.issubset(backlogged):
        raise ValueError("sender nodes must be active and backlogged")
    if any(topology[node_id] != technology for node_id, technology in zip(node_ids, technologies, strict=True)):
        raise ValueError("node technology does not match configured topology")
    if type(row["senders"]) is not list:
        raise ValueError("senders must be a list")
    row["senders"] = [_validate_sender(item) for item in row["senders"]]
    sender_ids = [sender["node_id"] for sender in row["senders"]]
    if sender_ids != node_ids:
        raise ValueError("senders must match node_ids in order")
    if any(
        sender["technology"] != technologies[index]
        for index, sender in enumerate(row["senders"])
    ):
        raise ValueError("sender technology does not match row technologies")
    if row["kind"] == "success":
        if len(node_ids) != 1 or row["collision_size"] != 0:
            raise ValueError("success must have one sender and no collision")
        if row["effective_data_us"] <= 0:
            raise ValueError("success must have positive effective data")
        if row["senders"][0]["effective_data_us"] != row["effective_data_us"]:
            raise ValueError("success sender effective data mismatch")
    else:
        if len(node_ids) < 2 or row["collision_size"] != len(node_ids):
            raise ValueError("collision must contain all colliding senders")
        if row["reservation_us"] != 0 or row["effective_data_us"] != 0:
            raise ValueError("collision cannot have reservation or effective data")
        if any(
            sender["delay_us"] is not None or sender["effective_data_us"] != 0
            for sender in row["senders"]
        ):
            raise ValueError("collision sender has invalid data or delay")
    if type(row["decisions"]) is not list:
        raise ValueError("decisions must be a list")
    row["decisions"] = [
        _validate_decision(item, expected_round, active, job, provenance)
        for item in row["decisions"]
    ]
    if job.policy not in {"adaptive_db_lbt", "fixed_oracle", "pretrain_arm"} and row["decisions"]:
        raise ValueError("non-adaptive policy cannot emit decisions")
    if type(row["training_samples"]) is not list:
        raise ValueError("training_samples must be a list")
    row["training_samples"] = [
        _validate_training_sample(item) for item in row["training_samples"]
    ]
    if job.policy != "pretrain_arm" and row["training_samples"]:
        raise ValueError("only pretrain_arm may emit training samples")
    return row


class ValidatedJobRows(Iterator[dict[str, Any]]):
    """One-shot validated row iterator with manifest execution provenance."""

    def __init__(self, job: JobSpec, output_dir: str | Path) -> None:
        self.job = job
        self.output_dir = output_dir
        self.manifest = _load_completed_job_manifest_metadata(job, output_dir)
        self.execution_provenance = self.manifest.execution_provenance
        self._validate_provenance_mode()
        self._rows = self._iter_rows()
        self._closed = False

    def _validate_provenance_mode(self) -> None:
        if self.job.policy == "adaptive_db_lbt":
            expected_modes = (
                {"context_free_builtin"}
                if self.job.ablation == "context_free_ucb"
                else {"adaptive_blank", "adaptive_model"}
            )
        elif self.job.policy == "fixed_oracle":
            expected_modes = {"fixed_oracle"}
        elif self.job.policy == "pretrain_arm":
            expected_modes = {"pretrain_arm"}
        else:
            expected_modes = {"baseline_builtin"}
        if self.execution_provenance.mode not in expected_modes:
            raise ValueError("execution provenance mode does not match job policy")

    def __iter__(self) -> ValidatedJobRows:
        return self

    def __next__(self) -> dict[str, Any]:
        if self._closed:
            raise StopIteration
        try:
            return next(self._rows)
        except BaseException:
            self.close()
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._rows.close()

    def __enter__(self) -> ValidatedJobRows:
        if self._closed:
            raise RuntimeError("validated job row stream is closed")
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: object,
    ) -> bool:
        self.close()
        return False

    def _iter_rows(self) -> Iterator[dict[str, Any]]:
        raw = artifact_paths(self.job, self.output_dir).raw
        previous_context: dict[str, list[float]] = {}
        previous_end: int | None = None
        row_count = 0
        with validated_jsonl_gz_stream(
            raw,
            expected_sha256=self.manifest.record_hash,
            expected_row_count=self.manifest.row_count,
            require_marker=True,
        ) as source:
            for expected_round, value in enumerate(source):
                row = _validate_row(
                    value,
                    self.job,
                    expected_round,
                    self.execution_provenance,
                )
                if self.job.policy == "pretrain_arm":
                    rewarded = {
                        decision["node_id"]: decision
                        for decision in row["decisions"]
                        if decision["reward"] is not None
                    }
                    if len(rewarded) != sum(
                        decision["reward"] is not None
                        for decision in row["decisions"]
                    ):
                        raise ValueError("duplicate rewarded decision node in row")
                    samples = {
                        sample["node_id"].split(":", 1)[-1]: sample
                        for sample in row["training_samples"]
                    }
                    if len(samples) != len(row["training_samples"]):
                        raise ValueError("duplicate training sample node in row")
                    if set(samples) != set(rewarded):
                        raise ValueError(
                            "training samples must correspond to rewarded decisions"
                        )
                    for node_id, sample in samples.items():
                        decision = rewarded[node_id]
                        if sample["node_id"] != f"{self.job.run_id}:{node_id}":
                            raise ValueError(
                                "training sample node provenance mismatch"
                            )
                        if sample["arm"] != decision["previous_arm"]:
                            raise ValueError(
                                "training arm does not match previous decision arm"
                            )
                        if sample["context"] != previous_context.get(node_id):
                            raise ValueError(
                                "training context does not match prior decision"
                            )
                        if sample["local_reward"] != decision["reward"]:
                            raise ValueError(
                                "training reward does not match decision reward"
                            )
                for decision in row["decisions"]:
                    previous_context[decision["node_id"]] = decision["context"]
                if previous_end is not None and row["tx_start_us"] < previous_end:
                    raise ValueError("raw contention rounds overlap or regress")
                previous_end = int(row["round_end_us"])
                row_count += 1
                yield row
        if row_count != self.job.rounds or self.manifest.row_count != row_count:
            raise ValueError("raw row count does not match expected job rounds")


def iter_job_rows(job: JobSpec, output_dir: str | Path) -> ValidatedJobRows:
    """Load one job only through its expected validated sibling artifacts."""
    return ValidatedJobRows(job, output_dir)


def read_job_rows(job: JobSpec, output_dir: str | Path) -> list[dict[str, Any]]:
    """Return validated rows as a compatibility materialization wrapper."""
    return list(iter_job_rows(job, output_dir))


@dataclass(frozen=True, slots=True)
class RunAggregate:
    rounds: int
    elapsed_us: int
    successes: int
    collisions: int
    collision_probability: float
    effective_airtime: float
    mean_delay_us: float
    p95_delay_us: float
    fairness: float
    evaluation_utility: float
    decision_count: int
    switch_count: int
    training_sample_count: int


def _stable_mean(values: list[int]) -> float:
    return float(sum((Fraction(value) for value in values), Fraction()) / len(values))


def aggregate_rows(
    rows: Iterable[dict[str, Any]],
    *,
    observer: Callable[[dict[str, Any]], None] | None = None,
) -> RunAggregate:
    """Reduce validated canonical rows using the preregistered metrics."""
    iterator = iter(rows)
    try:
        return _aggregate_rows(iterator, observer=observer)
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


def _aggregate_rows(
    rows: Iterator[dict[str, Any]],
    *,
    observer: Callable[[dict[str, Any]], None] | None,
) -> RunAggregate:
    rounds = 0
    elapsed_us = 0
    successes = 0
    collisions = 0
    effective_us = 0
    delays: list[int] = []
    per_node: dict[str, int] = {}
    decision_count = 0
    switch_count = 0
    training_count = 0
    for row in rows:
        round_id = row.get("round_id")
        if type(round_id) is not int or round_id != rounds:
            raise ValueError("raw round_id must start at 0 and be contiguous")
        rounds += 1
        elapsed_us = int(row["round_end_us"])
        successes += row["kind"] == "success"
        collisions += row["kind"] == "collision"
        effective_us += int(row["effective_data_us"])
        for sender in row["senders"]:
            if sender["delay_us"] is not None:
                delays.append(int(sender["delay_us"]))
        for node_id in row["active_node_ids"]:
            per_node.setdefault(str(node_id), 0)
        if row["kind"] == "success":
            winner = str(row["node_ids"][0])
            if winner not in per_node:
                raise ValueError("successful node is not declared active")
            per_node[winner] += int(row["effective_data_us"])
        for decision in row["decisions"]:
            decision_count += 1
            switch_count += (
                decision["previous_arm"] is not None
                and decision["previous_arm"] != decision["arm"]
            )
        training_count += len(row["training_samples"])
        if observer is not None:
            observer(row)
    if rounds == 0:
        raise ValueError("raw rows must be nonempty")
    if elapsed_us <= 0:
        raise ValueError("elapsed_us must be positive")
    if effective_us > elapsed_us:
        raise ValueError("effective data exceeds elapsed time")
    if not per_node:
        raise ValueError("raw rows declare no active nodes")
    airtime = effective_us / elapsed_us
    collision_rate = collisions / rounds
    p95 = float(nearest_rank_p95(delays)) if delays else 0.0
    fairness = jain(list(per_node.values()))
    utility = evaluation_utility(airtime, p95 / 1_000, fairness, collision_rate)
    return RunAggregate(
        rounds=rounds,
        elapsed_us=elapsed_us,
        successes=successes,
        collisions=collisions,
        collision_probability=collision_rate,
        effective_airtime=airtime,
        mean_delay_us=_stable_mean(delays) if delays else 0.0,
        p95_delay_us=p95,
        fairness=fairness,
        evaluation_utility=utility,
        decision_count=decision_count,
        switch_count=switch_count,
        training_sample_count=training_count,
    )
