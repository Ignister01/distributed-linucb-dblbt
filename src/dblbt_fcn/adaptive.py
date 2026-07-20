"""Per-node adaptive DB-LBT lifecycle coordination."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from numbers import Integral, Real

import numpy as np

from .channel import Channel, Node, RoundResult
from .config import adaptive_arms
from .linucb import LinUCB
from .metrics import nearest_rank_p95
from .nonideal import InterruptionPerturbation
from .observation import AttemptRecord, LocalWindow
from .reward import RewardComponents, local_reward_from_interval
from .types import PolicyKind, RecoveryProfile


_CONTEXT_DIM = 11
_DECISION_INTERVAL = 32
_FIXED_PROFILE = RecoveryProfile(kappa=7, beta=3, m=6, b_init=15)


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


class ContextFreeUCB:
    """Deterministic context-free UCB1 selector for one local node."""

    def __init__(
        self, num_arms: Integral, exploration: Real = 1.0
    ) -> None:
        arms = _non_negative_int("num_arms", num_arms)
        if arms == 0:
            raise ValueError("num_arms must be positive")
        exploration_value = _finite_real("exploration", exploration)
        if exploration_value < 0:
            raise ValueError("exploration must be non-negative")
        self.num_arms = arms
        self.exploration = exploration_value
        self.counts = np.zeros(arms, dtype=np.int64)
        self.reward_sums = np.zeros(arms, dtype=np.float64)

    def select(self) -> int:
        """Select the lowest untried arm, then the highest UCB1 score."""
        untried = np.flatnonzero(self.counts == 0)
        if untried.size:
            return int(untried[0])
        total = int(self.counts.sum())
        log_total = math.log(total)
        best_arm = 0
        best_score = -math.inf
        for arm in range(self.num_arms):
            count = int(self.counts[arm])
            mean = float(self.reward_sums[arm]) / count
            score = mean + self.exploration * math.sqrt(
                2 * log_total / count
            )
            if score > best_score:
                best_arm = arm
                best_score = score
        return best_arm

    def update(self, arm: object, reward: object) -> None:
        """Record one finite reward for exactly one selected arm."""
        arm_index = _non_negative_int("arm", arm)
        if arm_index >= self.num_arms:
            raise ValueError("arm must identify a configured arm")
        reward_value = _finite_real("reward", reward)
        count = int(self.counts[arm_index])
        if count >= np.iinfo(np.int64).max:
            raise RuntimeError("context-free UCB count overflow")
        reward_sum = float(self.reward_sums[arm_index]) + reward_value
        if not math.isfinite(reward_sum):
            raise RuntimeError("context-free UCB reward sum overflow")
        self.counts[arm_index] = count + 1
        self.reward_sums[arm_index] = reward_sum

    def clone(self) -> ContextFreeUCB:
        """Return an independent copy of this selector state."""
        cloned = type(self)(self.num_arms, self.exploration)
        cloned.counts = self.counts.copy()
        cloned.reward_sums = self.reward_sums.copy()
        return cloned


class FixedArmSelector:
    """Immutable selector that always returns one declared adaptive arm."""

    def __init__(self, num_arms: Integral, arm: Integral) -> None:
        arms = _non_negative_int("num_arms", num_arms)
        if arms == 0:
            raise ValueError("num_arms must be positive")
        selected = _non_negative_int("arm", arm)
        if selected >= arms:
            raise ValueError("arm must identify a configured arm")
        self.num_arms = arms
        self.arm = selected

    def select(self) -> int:
        """Return the configured arm without mutable learning state."""
        return self.arm

    def clone(self) -> FixedArmSelector:
        """Return an independent selector with the same fixed arm."""
        return type(self)(self.num_arms, self.arm)


AdaptiveSelector = LinUCB | ContextFreeUCB | FixedArmSelector


@dataclass(frozen=True, slots=True)
class LocalStepInput:
    """Local queue measurements associated with one controller step."""

    queue_occupancy_ratio: Real = 1.0
    arrivals: Integral = 0

    def __post_init__(self) -> None:
        occupancy = _finite_real(
            "queue_occupancy_ratio", self.queue_occupancy_ratio
        )
        if not 0 <= occupancy <= 1:
            raise ValueError(
                "queue_occupancy_ratio must be between zero and one"
            )
        arrivals = _non_negative_int("arrivals", self.arrivals)
        object.__setattr__(self, "queue_occupancy_ratio", occupancy)
        object.__setattr__(self, "arrivals", arrivals)


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """Immutable provenance for one adaptive decision boundary."""

    round_id: int
    node_id: str
    previous_arm: int | None
    new_arm: int
    context: tuple[float, ...]
    profile: RecoveryProfile
    reward: float | None = None
    reward_components: RewardComponents | None = None

    def __post_init__(self) -> None:
        if type(self.round_id) is not int or self.round_id <= 0:
            raise ValueError("round_id must be a positive integer")
        if type(self.node_id) is not str or not self.node_id.strip():
            raise ValueError("node_id must be a non-empty string")
        arms = adaptive_arms()
        for name, arm in (
            ("previous_arm", self.previous_arm),
            ("new_arm", self.new_arm),
        ):
            if arm is None and name == "previous_arm":
                continue
            if type(arm) is not int or not 0 <= arm < len(arms):
                raise ValueError(f"{name} must identify an adaptive arm")
        context = tuple(float(value) for value in self.context)
        if len(context) != _CONTEXT_DIM or not all(
            math.isfinite(value) for value in context
        ):
            raise ValueError("context must contain eleven finite values")
        if self.profile != arms[self.new_arm]:
            raise ValueError("profile must match new_arm")
        if (self.reward is None) != (self.reward_components is None):
            raise ValueError(
                "reward and reward_components must both be present or absent"
            )
        if self.reward is not None:
            reward = _finite_real("reward", self.reward)
            if reward != self.reward_components.reward:
                raise ValueError("reward must match reward_components")
            object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "context", context)


@dataclass(slots=True)
class _IntervalAccumulator:
    start_us: int
    attempts: int = 0
    collisions: int = 0
    effective_data_us: float = 0.0
    successful_busy_us: float = 0.0
    other_busy_us: float = 0.0
    delays_us: list[float] = field(default_factory=list)

    def reset(self, start_us: int) -> None:
        self.start_us = start_us
        self.attempts = 0
        self.collisions = 0
        self.effective_data_us = 0.0
        self.successful_busy_us = 0.0
        self.other_busy_us = 0.0
        self.delays_us.clear()


@dataclass(frozen=True, slots=True)
class AdaptiveIntervalState:
    """Detached immutable snapshot of one local reward interval."""

    start_us: int
    attempts: int
    collisions: int
    effective_data_us: float
    successful_busy_us: float
    other_busy_us: float
    delays_us: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AdaptiveNodeState:
    """Detached snapshot of one adaptive node's controller state."""

    window: LocalWindow
    agent: AdaptiveSelector
    last_attempt_end_us: int
    interval: AdaptiveIntervalState
    current_arm: int | None = None
    previous_arm: int | None = None
    decision_context: np.ndarray | None = None
    local_busy_us: float = 0.0
    inactive_since_us: int | None = None
    pending_arrivals: int = 0
    latest_queue_occupancy: float = 1.0


@dataclass(slots=True)
class _AdaptiveNodeState:
    window: LocalWindow
    agent: AdaptiveSelector
    last_attempt_end_us: int
    interval: _IntervalAccumulator
    current_arm: int | None = None
    previous_arm: int | None = None
    decision_context: np.ndarray | None = None
    local_busy_us: float = 0.0
    inactive_since_us: int | None = None
    pending_arrivals: int = 0
    latest_queue_occupancy: float = 1.0


@dataclass(frozen=True, slots=True)
class _AttemptSnapshot:
    interruptions: int
    selected: int
    retries: int
    delay_count: int


class AdaptiveController:
    """Coordinate local observations and LinUCB decisions around a channel."""

    def __init__(
        self,
        channel: Channel,
        agent: AdaptiveSelector,
        *,
        feature_mask: Sequence[bool] | Sequence[int] | None = None,
        online_updates: bool = True,
        collision_weight: Real = 0.25,
        interruption_perturbations: Mapping[
            str, InterruptionPerturbation
        ] | None = None,
    ) -> None:
        if not isinstance(channel, Channel):
            raise ValueError("channel must be a Channel")
        if not isinstance(agent, (LinUCB, ContextFreeUCB, FixedArmSelector)):
            raise ValueError(
                "agent must be a LinUCB, ContextFreeUCB, or FixedArmSelector"
            )
        arms = adaptive_arms()
        if agent.num_arms != len(arms):
            raise ValueError("agent num_arms must equal 24")
        if isinstance(agent, LinUCB) and agent.context_dim != _CONTEXT_DIM:
            raise ValueError("agent context_dim must equal 11")
        if type(online_updates) is not bool:
            raise ValueError("online_updates must be a boolean")
        weight = _finite_real("collision_weight", collision_weight)
        if weight < 0:
            raise ValueError("collision_weight must be non-negative")

        self.channel = channel
        self.online_updates = online_updates
        self.collision_weight = weight
        self._feature_mask = self._normalize_feature_mask(feature_mask)
        self._nodes = {
            node.node_id: node
            for node in channel.nodes
            if node.policy_kind is PolicyKind.ADAPTIVE
        }
        if interruption_perturbations is None:
            perturbations: dict[str, InterruptionPerturbation] = {}
        elif not isinstance(interruption_perturbations, Mapping):
            raise ValueError("interruption_perturbations must be a mapping")
        else:
            perturbations = {}
            for node_id, perturbation in interruption_perturbations.items():
                if type(node_id) is not str or node_id not in self._nodes:
                    raise ValueError(
                        "interruption perturbation must identify an adaptive node"
                    )
                if not isinstance(perturbation, InterruptionPerturbation):
                    raise ValueError(
                        "interruption perturbations must contain "
                        "InterruptionPerturbation values"
                    )
                perturbations[node_id] = perturbation
        self._interruption_perturbations = perturbations
        self._states: dict[str, _AdaptiveNodeState] = {}
        for node_id, node in self._nodes.items():
            self.channel.set_recovery_profile(node_id, _FIXED_PROFILE)
            self._states[node_id] = _AdaptiveNodeState(
                window=LocalWindow(max_attempts=64),
                agent=agent.clone(),
                last_attempt_end_us=channel.now_us,
                interval=_IntervalAccumulator(start_us=channel.now_us),
                inactive_since_us=(
                    None
                    if node.active and node.backlogged
                    else channel.now_us
                ),
            )
        self._decisions: list[DecisionRecord] = []
        self._last_attempts: dict[str, AttemptRecord] = {}

    @property
    def decisions(self) -> tuple[DecisionRecord, ...]:
        """Return immutable decision provenance in boundary order."""
        return tuple(self._decisions)

    def decisions_since(self, index: int) -> tuple[DecisionRecord, ...]:
        """Return only decisions at or after one consumed history index."""
        if type(index) is not int or not 0 <= index <= len(self._decisions):
            raise ValueError("decision index must identify a history boundary")
        return tuple(self._decisions[index:])

    def state(self, node_id: str) -> AdaptiveNodeState:
        """Return a detached snapshot for one adaptive node."""
        state = self._state(node_id)
        window = LocalWindow(max_attempts=64)
        for attempt in state.window.attempts:
            window.record(attempt)
        decision_context = (
            None
            if state.decision_context is None
            else state.decision_context.copy()
        )
        if decision_context is not None:
            decision_context.setflags(write=False)
        return AdaptiveNodeState(
            window=window,
            agent=state.agent.clone(),
            last_attempt_end_us=state.last_attempt_end_us,
            interval=AdaptiveIntervalState(
                start_us=state.interval.start_us,
                attempts=state.interval.attempts,
                collisions=state.interval.collisions,
                effective_data_us=state.interval.effective_data_us,
                successful_busy_us=state.interval.successful_busy_us,
                other_busy_us=state.interval.other_busy_us,
                delays_us=tuple(state.interval.delays_us),
            ),
            current_arm=state.current_arm,
            previous_arm=state.previous_arm,
            decision_context=decision_context,
            local_busy_us=state.local_busy_us,
            inactive_since_us=state.inactive_since_us,
            pending_arrivals=state.pending_arrivals,
            latest_queue_occupancy=state.latest_queue_occupancy,
        )

    def last_attempt(self, node_id: str) -> AttemptRecord | None:
        """Return the latest immutable local attempt without cloning state."""
        self._state(node_id)
        return self._last_attempts.get(node_id)

    def _state(self, node_id: str) -> _AdaptiveNodeState:
        if type(node_id) is not str or node_id not in self._states:
            raise ValueError(f"unknown adaptive node_id: {node_id}")
        return self._states[node_id]

    def set_backlogged(self, node_id: str, backlogged: bool) -> None:
        """Change an adaptive queue's eligibility through the Channel API."""
        self._synchronize_all()
        self._state(node_id)
        self.channel.set_backlogged(node_id, backlogged)
        self._synchronize_all()

    def set_active(self, node_id: str, active: bool) -> None:
        """Change adaptive topology participation without measuring the gap."""
        self._synchronize_all()
        self._state(node_id)
        self.channel.set_active(node_id, active)
        self._synchronize_all()

    def apply_background_busy(self, duration_us: int) -> int:
        """Apply external busy time to every eligible adaptive observer."""
        self._synchronize_all()
        eligible = tuple(
            node.node_id
            for node in self._nodes.values()
            if node.active and node.backlogged
        )
        now_us = self.channel.apply_background_busy(duration_us)
        for node_id in eligible:
            state = self._states[node_id]
            state.local_busy_us += duration_us
            state.interval.other_busy_us += duration_us
        return now_us

    def step(
        self,
        local_inputs: Mapping[str, LocalStepInput] | None = None,
    ) -> RoundResult:
        """Advance one channel round and record only actual own attempts."""
        self._synchronize_all()
        inputs = self._local_inputs(local_inputs)
        staged_inputs = self._stage_local_inputs(inputs)
        eligible = tuple(
            node
            for node in self.channel.nodes
            if node.active and node.backlogged
        )
        snapshots = {
            node.node_id: _AttemptSnapshot(
                interruptions=node.db_state.interruptions,
                selected=node.selected,
                retries=node.db_state.retries,
                delay_count=len(node.access_delays_us),
            )
            for node in eligible
            if node.node_id in self._states
        }

        result = self.channel.step()
        self._commit_local_inputs(staged_inputs)
        sender_ids = set(result.node_ids)
        for node_id in result.node_ids:
            if node_id in self._states:
                self._record_attempt(
                    node_id, snapshots[node_id], result
                )

        for node in eligible:
            if node.node_id in self._states and node.node_id not in sender_ids:
                state = self._states[node.node_id]
                state.local_busy_us += self.channel.tx_us
                state.interval.other_busy_us += self.channel.tx_us

        if self.channel.contention_round % _DECISION_INTERVAL == 0:
            self._decision_boundary()
        return result

    def _record_attempt(
        self,
        node_id: str,
        snapshot: _AttemptSnapshot,
        result: RoundResult,
    ) -> None:
        state = self._states[node_id]
        node = self._nodes[node_id]
        attempt_end_us = self.channel.now_us
        elapsed_us = attempt_end_us - state.last_attempt_end_us
        if elapsed_us <= 0:
            raise RuntimeError("adaptive attempt elapsed time must be positive")
        busy_us = min(state.local_busy_us, float(elapsed_us))
        delay_us: float | None = None
        if result.kind == "success":
            added_delays = len(node.access_delays_us) - snapshot.delay_count
            if added_delays == 1:
                delay_us = float(node.access_delays_us[-1])
            elif added_delays != 0:
                raise RuntimeError("channel added an invalid access-delay count")
        effective_data_us = (
            float(result.effective_data_us)
            if result.kind == "success"
            else 0.0
        )
        observed_interruptions = snapshot.interruptions
        perturbation = self._interruption_perturbations.get(node_id)
        if perturbation is not None:
            observed_interruptions = perturbation.apply(
                observed_interruptions
            )
        attempt = AttemptRecord(
            outcome=result.kind,
            elapsed_us=elapsed_us,
            busy_us=busy_us,
            interruptions=observed_interruptions,
            access_delay_us=delay_us,
            queue_occupancy_ratio=state.latest_queue_occupancy,
            arrivals=state.pending_arrivals,
            retries=node.db_state.retries,
            effective_data_us=effective_data_us,
        )
        state.window.record(attempt)
        self._last_attempts[node_id] = attempt
        state.pending_arrivals = 0
        state.last_attempt_end_us = attempt_end_us
        state.local_busy_us = 0.0
        state.interval.attempts += 1
        state.interval.effective_data_us += effective_data_us
        if result.kind == "collision":
            state.interval.collisions += 1
        else:
            state.interval.successful_busy_us += self.channel.tx_us
            if delay_us is not None:
                state.interval.delays_us.append(delay_us)

    def _transition_eligibility(
        self,
        node_id: str,
        state: _AdaptiveNodeState,
        was_eligible: bool,
    ) -> None:
        node = self._nodes[node_id]
        is_eligible = node.active and node.backlogged
        if was_eligible == is_eligible:
            return
        if not is_eligible:
            state.inactive_since_us = self.channel.now_us
            state.local_busy_us = 0.0
            return

        inactive_since_us = state.inactive_since_us
        if inactive_since_us is None:
            raise RuntimeError("adaptive inactive interval is missing")
        inactive_duration_us = self.channel.now_us - inactive_since_us
        if inactive_duration_us < 0:
            raise RuntimeError("adaptive inactive interval cannot be negative")
        state.interval.start_us += inactive_duration_us
        state.last_attempt_end_us = self.channel.now_us
        state.local_busy_us = 0.0
        state.inactive_since_us = None

    def _synchronize_all(self) -> None:
        arms = adaptive_arms()
        for node_id, node in self._nodes.items():
            state = self._states[node_id]
            expected_profile = (
                _FIXED_PROFILE
                if state.current_arm is None
                else arms[state.current_arm]
            )
            if self.channel.recovery_profile(node_id) != expected_profile:
                raise RuntimeError(
                    f"adaptive recovery profile mismatch for {node_id}"
                )
            was_eligible = state.inactive_since_us is None
            is_eligible = node.active and node.backlogged
            if was_eligible != is_eligible:
                self._transition_eligibility(
                    node_id, state, was_eligible
                )

    def _decision_boundary(self) -> None:
        self._synchronize_all()
        for node_id, node in self._nodes.items():
            state = self._states[node_id]
            if not node.active or not state.window.ready:
                continue
            raw_context = state.window.context()
            context = self._masked_context(raw_context)
            previous_arm = state.current_arm
            components = self._interval_reward(state, raw_context)
            reward = components.reward if components is not None else None
            candidate_agent = state.agent.clone()
            if (
                components is not None
                and self.online_updates
                and previous_arm is not None
                and state.decision_context is not None
            ):
                self._update_agent(
                    candidate_agent,
                    previous_arm,
                    state.decision_context,
                    components.reward,
                )

            new_arm = self._select_agent(candidate_agent, context)
            profile = adaptive_arms()[new_arm]
            next_decision_context = context.copy()
            record = DecisionRecord(
                round_id=self.channel.contention_round,
                node_id=node_id,
                previous_arm=previous_arm,
                new_arm=new_arm,
                context=tuple(float(value) for value in context),
                profile=profile,
                reward=reward,
                reward_components=components,
            )
            self.channel.set_recovery_profile(node_id, profile)
            state.agent = candidate_agent
            state.previous_arm = previous_arm
            state.current_arm = new_arm
            state.decision_context = next_decision_context
            state.interval.reset(self.channel.now_us)
            if state.inactive_since_us is not None:
                state.inactive_since_us = self.channel.now_us
            self._decisions.append(record)

    def _interval_reward(
        self, state: _AdaptiveNodeState, raw_context: np.ndarray
    ) -> RewardComponents | None:
        if state.current_arm is None or state.interval.attempts == 0:
            return None
        interval_end_us = (
            self.channel.now_us
            if state.inactive_since_us is None
            else state.inactive_since_us
        )
        elapsed_us = interval_end_us - state.interval.start_us
        if elapsed_us <= 0:
            raise RuntimeError("adaptive decision interval must be positive")
        share_denominator = (
            state.interval.successful_busy_us + state.interval.other_busy_us
        )
        local_share = (
            state.interval.successful_busy_us / share_denominator
            if share_denominator > 0
            else 0.0
        )
        delay_p95_us = (
            float(nearest_rank_p95(state.interval.delays_us))
            if state.interval.delays_us
            else 0.0
        )
        estimated_contenders = float(raw_context[10] * 31 + 1)
        return local_reward_from_interval(
            estimated_contenders=estimated_contenders,
            effective_data_us=state.interval.effective_data_us,
            elapsed_us=elapsed_us,
            delay_p95_us=delay_p95_us,
            local_share=local_share,
            collision_probability=(
                state.interval.collisions / state.interval.attempts
            ),
            collision_weight=self.collision_weight,
        )

    @staticmethod
    def _select_agent(
        agent: AdaptiveSelector, context: np.ndarray
    ) -> int:
        if isinstance(agent, (ContextFreeUCB, FixedArmSelector)):
            return agent.select()
        return agent.select(context)

    @staticmethod
    def _update_agent(
        agent: AdaptiveSelector,
        arm: int,
        context: np.ndarray,
        reward: float,
    ) -> None:
        if isinstance(agent, ContextFreeUCB):
            agent.update(arm, reward)
        elif isinstance(agent, FixedArmSelector):
            raise RuntimeError("fixed-arm selectors cannot be updated")
        else:
            agent.update(arm, context, reward)

    def _local_inputs(
        self, values: Mapping[str, LocalStepInput] | None
    ) -> dict[str, LocalStepInput]:
        if values is None:
            return {}
        if not isinstance(values, Mapping):
            raise ValueError("local_inputs must be a mapping")
        normalized: dict[str, LocalStepInput] = {}
        for node_id, measurement in values.items():
            if type(node_id) is not str or node_id not in self._states:
                raise ValueError(
                    f"local input node_id must identify an adaptive node: {node_id}"
                )
            if not isinstance(measurement, LocalStepInput):
                raise ValueError("local input must be a LocalStepInput")
            normalized[node_id] = measurement
        return normalized

    def _stage_local_inputs(
        self, values: Mapping[str, LocalStepInput]
    ) -> dict[str, tuple[int, float]]:
        updates: dict[str, tuple[int, float]] = {}
        for node_id, measurement in values.items():
            state = self._states[node_id]
            arrivals = state.pending_arrivals + measurement.arrivals
            try:
                finite_arrivals = float(arrivals)
            except (OverflowError, ValueError) as error:
                raise ValueError(
                    f"arrivals overflow for adaptive node {node_id}"
                ) from error
            if not math.isfinite(finite_arrivals):
                raise ValueError(
                    f"arrivals overflow for adaptive node {node_id}"
                )
            updates[node_id] = (
                arrivals,
                measurement.queue_occupancy_ratio,
            )
        return updates

    def _commit_local_inputs(
        self, updates: Mapping[str, tuple[int, float]]
    ) -> None:
        for node_id, (arrivals, occupancy) in updates.items():
            state = self._states[node_id]
            state.pending_arrivals = arrivals
            state.latest_queue_occupancy = occupancy

    def _masked_context(self, context: np.ndarray) -> np.ndarray:
        masked = np.array(context, dtype=np.float64, copy=True)
        if self._feature_mask:
            masked[list(self._feature_mask)] = 0.0
        return masked

    @staticmethod
    def _normalize_feature_mask(
        value: Sequence[bool] | Sequence[int] | None,
    ) -> tuple[int, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError("feature_mask must be booleans or indices")
        items = tuple(value)
        if not items:
            return ()
        if all(type(item) is bool for item in items):
            if len(items) != _CONTEXT_DIM:
                raise ValueError("feature_mask boolean form must have length 11")
            return tuple(index for index, masked in enumerate(items) if masked)
        if any(
            isinstance(item, bool) or not isinstance(item, Integral)
            for item in items
        ):
            raise ValueError("feature_mask must be booleans or indices")
        indices = tuple(int(item) for item in items)
        if len(indices) != len(set(indices)) or any(
            index < 0 or index >= _CONTEXT_DIM for index in indices
        ):
            raise ValueError("feature_mask indices must be unique values from 0 to 10")
        return tuple(sorted(indices))
