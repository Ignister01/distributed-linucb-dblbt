"""Deterministic execution of one canonical experiment job."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
import platform
import random
import socket
import subprocess

from .adaptive import (
    AdaptiveController,
    ContextFreeUCB,
    DecisionRecord,
    FixedArmSelector,
    LocalStepInput,
)
from .channel import Channel, Node
from .experiment import (
    JobSpec,
    artifact_paths,
    clear_invalid_job_artifacts,
    derive_stream_seed,
    job_artifact_lock,
    job_is_complete,
    load_completed_job_manifest,
    ensure_job_config_sidecar,
)
from .io import (
    RunManifest,
    write_jsonl_gz,
    write_manifest,
)
from .linucb import LinUCB
from .provenance import ExecutionProvenance, execution_provenance, file_sha256
from .nonideal import InterruptionPerturbation, PeriodicBusyProcess
from .traffic import PoissonTraffic
from .types import PolicyKind, Technology


_ADAPTIVE_POLICIES = {"adaptive_db_lbt", "fixed_oracle", "pretrain_arm"}
_POLICY_KINDS = {
    "random_lbt": PolicyKind.RANDOM,
    "primary_db_lbt": PolicyKind.PRIMARY_DB,
    "tmc_db_lbt": PolicyKind.TMC_DB,
}
_ABLATION_MASKS = {
    "no_queue": (6, 7),
    "no_cca_interrupt": (2, 3, 10),
    "no_delay": (4, 5),
}
_QUEUE_CAPACITY = 64
_MAX_IDLE_TICKS = 1_000_000


def _node_identities(job: JobSpec) -> Iterator[tuple[str, Technology]]:
    counts = (
        ("wifi", Technology.WIFI, job.scenario.wifi_nodes),
        ("nru", Technology.NRU, job.scenario.nru_nodes),
        ("legacy-ap", Technology.LEGACY_AP, job.scenario.legacy_ap_nodes),
        ("legacy-sta", Technology.LEGACY_STA, job.scenario.legacy_sta_nodes),
    )
    for prefix, technology, count in counts:
        for index in range(count):
            yield f"{prefix}-{index:03d}", technology


def _build_nodes(job: JobSpec) -> list[Node]:
    if job.policy in _ADAPTIVE_POLICIES:
        base_policy = PolicyKind.ADAPTIVE
    else:
        base_policy = _POLICY_KINDS[job.policy]
    nodes: list[Node] = []
    for node_id, technology in _node_identities(job):
        rng = random.Random(
            derive_stream_seed(
                job.exogenous_seed, node_id, "initial_backoff"
            )
        )
        selected = rng.randint(0, 15)
        policy = (
            PolicyKind.RANDOM
            if technology in (Technology.LEGACY_AP, Technology.LEGACY_STA)
            else base_policy
        )
        nodes.append(
            Node(
                node_id=node_id,
                technology=technology,
                policy_kind=policy,
                selected=selected,
                remaining=selected,
                active=job.scenario.join_interval_rounds is None,
                backlogged=job.scenario.traffic == "saturated",
            )
        )
    return nodes


def _decision_value(decision: DecisionRecord) -> dict[str, object]:
    components = (
        None
        if decision.reward_components is None
        else asdict(decision.reward_components)
    )
    return {
        "round_id": decision.round_id,
        "node_id": decision.node_id,
        "previous_arm": decision.previous_arm,
        "arm": decision.new_arm,
        "profile": decision.profile.model_dump(mode="json"),
        "context": list(decision.context),
        "reward": decision.reward,
        "components": components,
    }


def simulate_job_records(
    job: JobSpec,
    initial_agent: LinUCB | None = None,
    oracle_arm: int | None = None,
) -> Iterator[dict[str, object]]:
    """Yield exactly one canonical raw row per contention round."""
    if not isinstance(job, JobSpec):
        raise TypeError("job must be a JobSpec")
    if initial_agent is not None and not isinstance(initial_agent, LinUCB):
        raise TypeError("initial_agent must be a LinUCB or None")

    run_id = job.run_id
    config_hash = job.config_hash
    nodes = _build_nodes(job)
    timing = job.timing
    channel = Channel(
        nodes,
        seed=job.exogenous_seed,
        slot_us=timing.slot_us,
        tx_us=timing.tx_us,
        wifi_ack_us=timing.wifi_ack_us,
        nru_sync_us=timing.nru_sync_us,
    )
    controller: AdaptiveController | None = None
    if job.policy in _ADAPTIVE_POLICIES:
        if job.policy == "fixed_oracle":
            if type(oracle_arm) is not int or not 0 <= oracle_arm < 24:
                raise ValueError("fixed_oracle requires oracle_arm in range 0..23")
            agent = FixedArmSelector(24, oracle_arm)
        elif job.policy == "pretrain_arm":
            if job.arm_id is None:
                raise ValueError("pretrain_arm requires job.arm_id")
            agent = FixedArmSelector(24, job.arm_id)
        elif job.ablation == "context_free_ucb":
            agent = ContextFreeUCB(24)
        else:
            agent = (
                LinUCB(24, 11, ridge=1.0, exploration=0.5)
                if initial_agent is None
                else initial_agent.clone()
            )
        collision_weight = {
            "collision_weight_0.125": 0.125,
            "collision_weight_0.5": 0.5,
        }.get(job.ablation, 0.25)
        controller = AdaptiveController(
            channel,
            agent,
            feature_mask=_ABLATION_MASKS.get(job.ablation),
            online_updates=(
                job.policy == "adaptive_db_lbt"
                and job.ablation != "frozen_online"
            ),
            collision_weight=collision_weight,
            interruption_perturbations={
                node.node_id: InterruptionPerturbation(
                    job.scenario.interruption_std,
                    random.Random(
                        derive_stream_seed(
                            job.exogenous_seed,
                            node.node_id,
                            "interruption_perturbation",
                        )
                    ),
                )
                for node in nodes
                if node.policy_kind is PolicyKind.ADAPTIVE
            },
        )

    nodes_by_id = {node.node_id: node for node in nodes}
    queue_depths = {
        node.node_id: 1 if job.scenario.traffic == "saturated" else 0
        for node in nodes
    }
    pending_arrivals = {node.node_id: 0 for node in nodes}
    poisson_sources: dict[str, PoissonTraffic] = {}
    if job.scenario.traffic == "poisson":
        rate = job.scenario.poisson_rate_packets_ms
        if rate is None:
            raise RuntimeError("validated poisson job is missing its rate")
        poisson_sources = {
            node.node_id: PoissonTraffic(
                rate,
                random.Random(
                    derive_stream_seed(
                        job.exogenous_seed, node.node_id, "arrivals"
                    )
                ),
            )
            for node in nodes
        }

    def set_active(node: Node, active: bool) -> None:
        if node.policy_kind is PolicyKind.ADAPTIVE:
            if controller is None:
                raise RuntimeError("adaptive node is missing its controller")
            controller.set_active(node.node_id, active)
        else:
            channel.set_active(node.node_id, active)

    def set_backlogged(node: Node, backlogged: bool) -> None:
        if node.policy_kind is PolicyKind.ADAPTIVE:
            if controller is None:
                raise RuntimeError("adaptive node is missing its controller")
            controller.set_backlogged(node.node_id, backlogged)
        else:
            channel.set_backlogged(node.node_id, backlogged)

    def sample_arrivals(elapsed_us: int) -> None:
        for node in nodes:
            if not node.active:
                continue
            source = poisson_sources.get(node.node_id)
            if source is None:
                continue
            arrivals = source.arrivals(elapsed_us)
            if arrivals:
                old_depth = queue_depths[node.node_id]
                queue_depths[node.node_id] = min(
                    old_depth + arrivals, _QUEUE_CAPACITY
                )
                pending_arrivals[node.node_id] += arrivals
                if old_depth == 0:
                    set_backlogged(node, True)

    periodic: PeriodicBusyProcess | None = None
    next_busy_start: int | None = None
    if job.scenario.interference_interval_ms is not None:
        duration = job.scenario.interference_duration_us
        if duration is None:
            raise RuntimeError("validated periodic job is missing its duration")
        periodic = PeriodicBusyProcess(
            period_us=job.scenario.interference_interval_ms * 1_000,
            busy_us=duration,
        )
        next_busy_start = periodic.start_us

    decision_count = 0
    previous_decision_context: dict[str, list[float]] = {}
    local_sequences: dict[str, int] = {}
    traffic_sample_at_us = channel.now_us
    for round_id in range(job.rounds):
        join_interval = job.scenario.join_interval_rounds
        lifetime = job.scenario.lifetime_rounds
        if join_interval is not None and lifetime is not None:
            maximum_start = (len(nodes) - 1) * join_interval
            cycle = maximum_start + lifetime
            phase = round_id % cycle
            for index, node in enumerate(nodes):
                start = index * join_interval
                should_be_active = start <= phase < start + lifetime
                if node.active != should_be_active:
                    set_active(node, should_be_active)

        idle_ticks = 0
        while not any(
            node.active and node.backlogged for node in nodes
        ):
            if not any(node.active for node in nodes):
                raise RuntimeError(
                    "dynamic active windows leave no contender at this round"
                )
            if idle_ticks >= _MAX_IDLE_TICKS:
                raise RuntimeError("poisson queues exceeded the idle tick limit")
            channel.now_us += timing.tx_us
            sample_arrivals(timing.tx_us)
            traffic_sample_at_us = channel.now_us
            idle_ticks += 1

        background_busy_us = 0
        if periodic is not None and next_busy_start is not None:
            boundary_us = channel.now_us
            while next_busy_start <= boundary_us:
                if controller is None:
                    channel.apply_background_busy(periodic.busy_us)
                else:
                    controller.apply_background_busy(periodic.busy_us)
                background_busy_us += periodic.busy_us
                next_busy_start += periodic.period_us

        active_node_ids = [node.node_id for node in nodes if node.active]
        backlogged_node_ids = [
            node.node_id for node in nodes if node.active and node.backlogged
        ]
        snapshots = {
            node.node_id: {
                "selected": node.selected,
                "interruptions": node.db_state.interruptions,
                "delay_count": len(node.access_delays_us),
            }
            for node in nodes
            if node.active and node.backlogged
        }
        if controller is None:
            result = channel.step()
            decisions: list[DecisionRecord] = []
        else:
            local_inputs = {
                node.node_id: LocalStepInput(
                    (
                        1.0
                        if not poisson_sources
                        else min(
                            queue_depths[node.node_id] / _QUEUE_CAPACITY,
                            1.0,
                        )
                    ),
                    pending_arrivals[node.node_id],
                )
                for node in nodes
                if node.policy_kind is PolicyKind.ADAPTIVE
            }
            result = controller.step(local_inputs)
            for node_id in local_inputs:
                pending_arrivals[node_id] = 0
            decisions = list(controller.decisions_since(decision_count))
            decision_count += len(decisions)

        if poisson_sources and result.kind == "success":
            successful = nodes_by_id[result.node_ids[0]]
            depth = queue_depths[successful.node_id]
            if depth <= 0:
                raise RuntimeError("successful poisson sender has no packet")
            queue_depths[successful.node_id] = depth - 1
            if depth == 1:
                set_backlogged(successful, False)

        elapsed_since_sample = channel.now_us - traffic_sample_at_us
        sample_arrivals(elapsed_since_sample)
        traffic_sample_at_us = channel.now_us

        training_samples: list[dict[str, object]] = []
        for decision in decisions:
            if job.policy == "pretrain_arm" and decision.reward is not None:
                previous_context = previous_decision_context.get(
                    decision.node_id
                )
                if previous_context is None or decision.previous_arm is None:
                    raise RuntimeError(
                        "pretraining reward is missing prior decision provenance"
                    )
                sequence = local_sequences.get(decision.node_id, 0)
                training_samples.append(
                    {
                        "context": list(previous_context),
                        "arm": decision.previous_arm,
                        "local_reward": decision.reward,
                        "local_sequence": sequence,
                        "node_id": f"{run_id}:{decision.node_id}",
                        "pretraining_seed": job.seed,
                    }
                )
                local_sequences[decision.node_id] = sequence + 1
            previous_decision_context[decision.node_id] = list(
                decision.context
            )

        senders: list[dict[str, object]] = []
        for node_id in result.node_ids:
            node = nodes_by_id[node_id]
            snapshot = snapshots[node_id]
            delay_us: int | None = None
            if len(node.access_delays_us) > snapshot["delay_count"]:
                delay_us = node.access_delays_us[-1]
            observed_interruptions = snapshot["interruptions"]
            if controller is not None and node.policy_kind is PolicyKind.ADAPTIVE:
                latest_attempt = controller.last_attempt(node_id)
                if latest_attempt is None:
                    raise RuntimeError("adaptive sender has no local attempt")
                observed_interruptions = latest_attempt.interruptions
            senders.append(
                {
                    "node_id": node.node_id,
                    "technology": node.technology.value,
                    "selected_backoff_before": snapshot["selected"],
                    "next_selected_backoff": node.selected,
                    "interruptions_before": observed_interruptions,
                    "retries_after": node.db_state.retries,
                    "db_initialized": node.db_initialized,
                    "deterministic_countdown": node.deterministic_countdown,
                    "delay_us": delay_us,
                    "effective_data_us": (
                        result.effective_data_us
                        if result.kind == "success"
                        else 0
                    ),
                }
            )

        yield {
            "schema_version": 1,
            "record_type": "contention_round",
            "run_id": run_id,
            "scenario_id": job.scenario.id,
            "policy": job.policy,
            "seed": job.seed,
            "config_hash": config_hash,
            "round_id": result.round_id,
            "tx_start_us": result.now_us,
            "round_end_us": channel.now_us,
            "kind": result.kind,
            "node_ids": list(result.node_ids),
            "technologies": list(result.technologies),
            "collision_size": result.collision_size,
            "reservation_us": result.reservation_us,
            "effective_data_us": result.effective_data_us,
            "background_busy_us": background_busy_us,
            "active_node_ids": active_node_ids,
            "backlogged_node_ids": backlogged_node_ids,
            "senders": senders,
            "decisions": [_decision_value(item) for item in decisions],
            "training_samples": training_samples,
        }


def _git_revision() -> str:
    repository_root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "failed to resolve the repository Git revision"
        ) from error
    revision = completed.stdout.strip()
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise RuntimeError("Git revision is not a full lowercase commit SHA")
    return revision


def _dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for distribution in (
        "dblbt-fcn",
        "matplotlib",
        "numpy",
        "pandas",
        "pydantic",
        "PyYAML",
        "scipy",
        "typer",
    ):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"failed to resolve dependency version: {distribution}"
            ) from error
    return versions


def run_job(
    job: JobSpec,
    output_dir: str | Path,
    *,
    initial_agent: LinUCB | None = None,
    oracle_arm: int | None = None,
    model_path: str | Path | None = None,
    oracle_artifact_path: str | Path | None = None,
    execution: ExecutionProvenance | None = None,
) -> RunManifest:
    """Persist one job atomically, returning an existing valid run unchanged."""
    if not isinstance(job, JobSpec):
        raise TypeError("job must be a JobSpec")
    oracle_artifact_sha256: str | None = None
    oracle_model_sha256: str | None = None
    source_matrix_sha256: str | None = None
    if job.policy == "fixed_oracle":
        if model_path is None or oracle_artifact_path is None:
            raise ValueError(
                "fixed_oracle requires actual model and Oracle artifact paths"
            )
        from .workflows import load_oracle_arm

        artifact = load_oracle_arm(
            oracle_artifact_path, model_path=model_path
        )
        if oracle_arm is None:
            oracle_arm = artifact.arm
        elif oracle_arm != artifact.arm:
            raise ValueError("oracle_arm does not match Oracle artifact")
        oracle_artifact_sha256 = file_sha256(oracle_artifact_path)
        oracle_model_sha256 = file_sha256(model_path)
        source_matrix_sha256 = artifact.source_matrix_hash
    elif oracle_arm is not None or oracle_artifact_path is not None:
        raise ValueError("Oracle inputs require the fixed_oracle policy")
    expected_execution = execution_provenance(
        job,
        initial_agent=initial_agent,
        model_path=model_path,
        oracle_arm=oracle_arm,
        oracle_artifact_sha256=oracle_artifact_sha256,
        oracle_model_sha256=oracle_model_sha256,
        source_matrix_sha256=source_matrix_sha256,
    )
    if execution is not None and execution != expected_execution:
        raise ValueError("execution provenance does not match actual inputs")
    root = Path(output_dir).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(job, root)
    with job_artifact_lock(job, root) as root_fd:
        ensure_job_config_sidecar(job, root, root_fd)
        if job_is_complete(job, root, expected_execution):
            return load_completed_job_manifest(job, root, expected_execution)

    clear_invalid_job_artifacts(job, root, expected_execution)
    writer_entered = False
    try:
        with job_artifact_lock(job, root) as root_fd:
            writer_entered = True
            ensure_job_config_sidecar(job, root, root_fd)
            if job_is_complete(job, root, expected_execution):
                return load_completed_job_manifest(job, root, expected_execution)

            started_at = datetime.now(UTC)
            revision = _git_revision()
            dependencies = _dependency_versions()
            host = socket.gethostname()
            if not host.strip():
                raise RuntimeError("host name is empty")
            metadata = write_jsonl_gz(
                paths.raw,
                simulate_job_records(
                    job,
                    initial_agent=initial_agent,
                    oracle_arm=oracle_arm,
                ),
            )
            ended_at = datetime.now(UTC)
            manifest = RunManifest(
                run_id=job.run_id,
                scenario_id=job.scenario.id,
                policy=job.policy,
                seed=job.seed,
                config_hash=job.config_hash,
                git_revision=revision,
                dependency_versions=dependencies,
                host=host,
                started_at_utc=started_at,
                ended_at_utc=ended_at,
                elapsed_seconds=(ended_at - started_at).total_seconds(),
                record_path=str(paths.raw),
                record_hash=metadata.sha256,
                row_count=metadata.row_count,
                exit_code=0,
                status="complete",
                execution_provenance=expected_execution,
            )
            write_manifest(paths.manifest, manifest)
            return manifest
    except BaseException as error:
        if writer_entered:
            try:
                clear_invalid_job_artifacts(job, root, expected_execution)
            except BaseException as cleanup_error:
                error.add_note(
                    f"failed to clear incomplete job artifacts: {cleanup_error}"
                )
        raise
