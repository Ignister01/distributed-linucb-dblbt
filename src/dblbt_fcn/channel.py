"""Deterministic contention-round channel engine."""

from collections.abc import Iterable, Iterator, MutableSequence, Sequence
from dataclasses import dataclass, field
import random

from .policies import DbState, PrimaryDbLbt, RandomLbt, TmcDbLbt
from .types import PolicyKind, RecoveryProfile, Technology


ContentionPolicy = RandomLbt | PrimaryDbLbt | TmcDbLbt


def _require_non_negative_int(name: str, value: object) -> None:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive_int(name: str, value: object) -> None:
    _require_non_negative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")


class _AccessDelayHistory(MutableSequence[int]):
    """Controlled mutable sequence of exact nonnegative access delays."""

    __slots__ = ("__items",)

    @staticmethod
    def _value(value: object) -> int:
        _require_non_negative_int("access_delays_us", value)
        return value

    @classmethod
    def _values(cls, values: Iterable[object]) -> list[int]:
        return [cls._value(value) for value in values]

    def __init__(self, values: Iterable[object] = ()) -> None:
        self.__items = self._values(values)

    def __len__(self) -> int:
        return len(self.__items)

    def __getitem__(self, index: int | slice) -> int | list[int]:
        return self.__items[index]

    def __setitem__(
        self, index: int | slice, value: int | Iterable[int]
    ) -> None:
        if isinstance(index, slice):
            normalized: int | list[int] = self._values(
                value  # type: ignore[arg-type]
            )
        else:
            normalized = self._value(value)
        self.__items[index] = normalized  # type: ignore[index, assignment]

    def __delitem__(self, index: int | slice) -> None:
        del self.__items[index]

    def insert(self, index: int, value: int) -> None:
        self.__items.insert(index, self._value(value))

    def extend(self, values: Iterable[int]) -> None:
        normalized = self._values(values)
        self.__items.extend(normalized)

    def __iadd__(self, values: Iterable[int]) -> "_AccessDelayHistory":
        self.extend(values)
        return self

    def __iter__(self) -> Iterator[int]:
        return iter(self.__items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _AccessDelayHistory):
            return self.__items == other.__items
        if isinstance(other, Sequence):
            return self.__items == list(other)
        return False

    def __repr__(self) -> str:
        return repr(self.__items)


@dataclass
class Node:
    """Mutable local state for one channel contender."""

    node_id: str
    technology: Technology
    policy_kind: PolicyKind
    selected: int
    remaining: int
    active: bool = True
    backlogged: bool = True
    db_state: DbState = field(default_factory=DbState)
    db_initialized: bool = False
    deterministic_countdown: bool = False
    last_success_end_us: int | None = None
    access_delays_us: MutableSequence[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._validate_state()
        self.access_delays_us = _AccessDelayHistory(self.access_delays_us)

    def _validate_state(self) -> None:
        if type(self.node_id) is not str or not self.node_id.strip():
            raise ValueError("node_id must be a non-empty string")
        if not isinstance(self.technology, Technology):
            raise ValueError("technology must be a Technology")
        if not isinstance(self.policy_kind, PolicyKind):
            raise ValueError("policy_kind must be a PolicyKind")
        _require_non_negative_int("selected", self.selected)
        _require_non_negative_int("remaining", self.remaining)
        if self.remaining > self.selected:
            raise ValueError("remaining must not exceed selected")
        if type(self.active) is not bool:
            raise ValueError("active must be a boolean")
        if type(self.backlogged) is not bool:
            raise ValueError("backlogged must be a boolean")
        if not isinstance(self.db_state, DbState):
            raise ValueError("db_state must be a DbState")
        if type(self.db_initialized) is not bool:
            raise ValueError("db_initialized must be a boolean")
        if type(self.deterministic_countdown) is not bool:
            raise ValueError("deterministic_countdown must be a boolean")
        _require_non_negative_int(
            "interruptions", self.db_state.interruptions
        )
        _require_non_negative_int("retries", self.db_state.retries)
        if (
            self.policy_kind in (PolicyKind.TMC_DB, PolicyKind.ADAPTIVE)
            and not self.db_initialized
        ):
            if self.deterministic_countdown:
                raise ValueError(
                    "deterministic_countdown must be false before "
                    "DB-LBT initialization"
                )
            if self.db_state.retries != 0:
                raise ValueError(
                    "retries must be zero before DB-LBT initialization"
                )
            if self.db_state.interruptions != 0:
                raise ValueError(
                    "interruptions must be zero before DB-LBT initialization"
                )
        if self.last_success_end_us is not None:
            _require_non_negative_int(
                "last_success_end_us", self.last_success_end_us
            )
        if not isinstance(
            self.access_delays_us, (list, _AccessDelayHistory)
        ):
            raise ValueError("access_delays_us must be a list")
        if not isinstance(self.access_delays_us, _AccessDelayHistory):
            for delay in self.access_delays_us:
                _require_non_negative_int("access_delays_us", delay)


@dataclass(frozen=True)
class RoundResult:
    """Immutable observable result of one contention round."""

    round_id: int
    now_us: int
    kind: str
    node_ids: Sequence[str]
    technologies: Sequence[str]
    collision_size: int
    reservation_us: int
    effective_data_us: int

    def __post_init__(self) -> None:
        _require_non_negative_int("round_id", self.round_id)
        _require_non_negative_int("now_us", self.now_us)
        _require_non_negative_int("collision_size", self.collision_size)
        _require_non_negative_int("reservation_us", self.reservation_us)
        _require_non_negative_int("effective_data_us", self.effective_data_us)
        if self.kind not in ("success", "collision"):
            raise ValueError("kind must be success or collision")
        node_ids = self._freeze_strings("node_ids", self.node_ids)
        technologies = self._freeze_strings(
            "technologies", self.technologies
        )
        if len(node_ids) != len(technologies):
            raise ValueError(
                "technologies must have one entry for every node_id"
            )
        if any(not node_id.strip() for node_id in node_ids):
            raise ValueError("node_ids must contain only non-empty strings")
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node_ids must be unique")
        valid_technologies = {technology.value for technology in Technology}
        if any(
            technology not in valid_technologies
            for technology in technologies
        ):
            raise ValueError(
                "technologies must contain only Technology values"
            )
        if self.kind == "success":
            if len(node_ids) != 1:
                raise ValueError("success must have exactly one sender")
            if self.collision_size != 0:
                raise ValueError("collision_size must be zero for success")
        else:
            if len(node_ids) < 2:
                raise ValueError("collision must have at least two senders")
            if self.collision_size != len(node_ids):
                raise ValueError(
                    "collision_size must equal the number of senders"
                )
            if self.reservation_us != 0:
                raise ValueError("reservation_us must be zero for collision")
            if self.effective_data_us != 0:
                raise ValueError(
                    "effective_data_us must be zero for collision"
                )
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "technologies", technologies)

    @staticmethod
    def _freeze_strings(
        name: str, values: Sequence[str]
    ) -> tuple[str, ...]:
        if isinstance(values, (str, bytes)):
            raise ValueError(f"{name} must be a sequence of strings")
        try:
            frozen = tuple(values)
        except TypeError as error:
            raise ValueError(
                f"{name} must be a sequence of strings"
            ) from error
        if not all(type(value) is str for value in frozen):
            raise ValueError(f"{name} must contain only strings")
        return frozen


class Channel:
    """Advance contenders through deterministic contention rounds."""

    def __init__(
        self,
        nodes: Sequence[Node],
        seed: int,
        slot_us: int = 1,
        tx_us: int = 2_000,
        wifi_ack_us: int = 0,
        nru_sync_us: int = 250,
    ) -> None:
        self.nodes = list(nodes)
        if not all(isinstance(node, Node) for node in self.nodes):
            raise ValueError("nodes must contain only Node instances")
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node_id values must be unique")
        db_states = [node.db_state for node in self.nodes]
        if len(db_states) != len({id(state) for state in db_states}):
            raise ValueError("each node must own a distinct db_state")
        self._registered_nodes = tuple(self.nodes)
        self._registered_node_ids = tuple(node_ids)
        self._registered_technologies = tuple(
            node.technology for node in self.nodes
        )
        self._registered_policy_kinds = tuple(
            node.policy_kind for node in self.nodes
        )
        self._registered_db_states = tuple(db_states)
        self._registered_access_delay_histories = tuple(
            node.access_delays_us for node in self.nodes
        )
        _require_non_negative_int("seed", seed)
        _require_positive_int("slot_us", slot_us)
        _require_positive_int("tx_us", tx_us)
        _require_non_negative_int("wifi_ack_us", wifi_ack_us)
        _require_positive_int("nru_sync_us", nru_sync_us)
        if wifi_ack_us > tx_us:
            raise ValueError("wifi_ack_us must not exceed tx_us")
        if nru_sync_us > tx_us:
            raise ValueError("nru_sync_us must not exceed tx_us")
        self.seed = seed
        self.slot_us = slot_us
        self.tx_us = tx_us
        self.wifi_ack_us = wifi_ack_us
        self.nru_sync_us = nru_sync_us
        self.now_us = max(
            (
                node.last_success_end_us
                for node in self.nodes
                if node.last_success_end_us is not None
            ),
            default=0,
        )
        self.contention_round = 0
        # Keep the import at construction time so experiment orchestration may
        # import Channel in the future without creating a module cycle.
        from .experiment import derive_stream_seed

        self._rngs = {
            node.node_id: random.Random(
                derive_stream_seed(seed, node.node_id, "backoff")
            )
            for node in self.nodes
        }
        self._policies = {
            node.node_id: self._make_policy(node)
            for node in self.nodes
        }

    def step(self) -> RoundResult:
        """Advance one minimum-countdown contention round."""
        self._validate_registered_nodes()
        eligible = [
            node for node in self.nodes if node.active and node.backlogged
        ]
        if not eligible:
            raise ValueError("channel has no eligible active backlogged nodes")

        minimum = min(node.remaining for node in eligible)
        round_start_us = self.now_us + minimum * self.slot_us
        for node in self.nodes:
            last_success_end_us = node.last_success_end_us
            if (
                last_success_end_us is not None
                and last_success_end_us > round_start_us
            ):
                raise ValueError(
                    "last_success_end_us cannot exceed success start time"
                )
        self.now_us += minimum * self.slot_us
        for node in eligible:
            node.remaining -= minimum
        senders = [node for node in eligible if node.remaining == 0]
        for node in eligible:
            if node.remaining > 0 and self._records_interruption(node):
                node.db_state.interruptions += 1
        collision = len(senders) > 1
        reservation_us = 0
        effective_data_us = 0
        if not collision:
            successful = senders[0]
            reservation_us, effective_data_us = self._success_airtime(
                successful
            )
            if successful.last_success_end_us is not None:
                delay_us = self.now_us - successful.last_success_end_us
                successful.access_delays_us.append(delay_us)
            successful.last_success_end_us = self.now_us + self.tx_us

        result = RoundResult(
            round_id=self.contention_round,
            now_us=self.now_us,
            kind="collision" if collision else "success",
            node_ids=tuple(node.node_id for node in senders),
            technologies=tuple(node.technology.value for node in senders),
            collision_size=len(senders) if collision else 0,
            reservation_us=reservation_us,
            effective_data_us=effective_data_us,
        )
        for node in senders:
            self._rearm_after(node, collision=collision)
        self.now_us += self.tx_us
        self.contention_round += 1
        return result

    def apply_background_busy(self, duration_us: int) -> int:
        """Apply one explicit external busy period between rounds."""
        self._validate_registered_nodes()
        for node in self.nodes:
            if (
                node.last_success_end_us is not None
                and node.last_success_end_us > self.now_us
            ):
                raise ValueError(
                    "last_success_end_us cannot exceed current channel time"
                )
        _require_positive_int("duration_us", duration_us)
        eligible = [
            node for node in self.nodes if node.active and node.backlogged
        ]
        if not eligible:
            raise ValueError("channel has no eligible active backlogged nodes")

        for node in eligible:
            if self._records_interruption(node):
                node.db_state.interruptions += 1
        self.now_us += duration_us
        return self.now_us

    def run(self, rounds: int) -> list[RoundResult]:
        """Run a fixed number of contention rounds."""
        _require_non_negative_int("rounds", rounds)
        return [self.step() for _ in range(rounds)]

    def recovery_profile(self, node_id: str) -> RecoveryProfile:
        """Return the future recovery profile for one adaptive node."""
        self._validate_registered_nodes()
        node = self._node(node_id)
        policy = self._adaptive_policy(node)
        return policy.profile

    def set_recovery_profile(
        self, node_id: str, profile: RecoveryProfile
    ) -> None:
        """Replace only the future recovery rule for one adaptive node."""
        self._validate_registered_nodes()
        node = self._node(node_id)
        self._adaptive_policy(node)
        if not isinstance(profile, RecoveryProfile):
            raise ValueError("profile must be a RecoveryProfile")
        self._policies[node.node_id] = TmcDbLbt(profile)

    def set_backlogged(self, node_id: str, backlogged: bool) -> None:
        """Deactivate a queue or rearm it under its current policy."""
        self._validate_registered_nodes()
        node = self._node(node_id)
        if type(backlogged) is not bool:
            raise ValueError("backlogged must be a boolean")
        if node.backlogged is backlogged:
            return
        if not backlogged:
            node.backlogged = False
            return

        policy = self._policies[node.node_id]
        rng = self._rngs[node.node_id]
        if isinstance(policy, RandomLbt):
            selected = policy.draw(rng)
            deterministic = False
        elif isinstance(policy, PrimaryDbLbt):
            deterministic = node.db_state.retries % policy.m < policy.beta
            selected = policy.next_backoff(node.db_state, rng)
        elif not node.db_initialized:
            selected = policy.initial_backoff(rng)
            deterministic = False
        else:
            deterministic = (
                node.db_state.retries % policy.profile.kappa
                < policy.profile.beta
            )
            selected = policy.next_backoff(node.db_state, rng)

        node.selected = selected
        node.remaining = selected
        node.deterministic_countdown = deterministic
        node.backlogged = True

    def set_active(self, node_id: str, active: bool) -> None:
        """Change one registered node's topology participation state."""
        self._validate_registered_nodes()
        node = self._node(node_id)
        if type(active) is not bool:
            raise ValueError("active must be a boolean")
        node.active = active

    def _validate_registered_nodes(self) -> None:
        if (
            len(self.nodes) != len(self._registered_nodes)
            or any(
                current is not registered
                for current, registered in zip(
                    self.nodes, self._registered_nodes, strict=True
                )
            )
        ):
            raise ValueError("nodes cannot be replaced after registration")

        for node, node_id, technology, policy_kind, db_state, history in zip(
            self.nodes,
            self._registered_node_ids,
            self._registered_technologies,
            self._registered_policy_kinds,
            self._registered_db_states,
            self._registered_access_delay_histories,
            strict=True,
        ):
            if node.node_id != node_id:
                raise ValueError("node_id cannot change after registration")
            if node.technology is not technology:
                raise ValueError(
                    "technology cannot change after registration"
                )
            if node.policy_kind is not policy_kind:
                raise ValueError(
                    "policy_kind cannot change after registration"
                )
            if node.db_state is not db_state:
                raise ValueError("db_state cannot change after registration")
            if node.access_delays_us is not history:
                raise ValueError(
                    "access_delays_us cannot change after registration"
                )
            node._validate_state()

    def _node(self, node_id: str) -> Node:
        if type(node_id) is not str or not node_id.strip():
            raise ValueError("node_id must identify a registered node")
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise ValueError(f"unknown node_id: {node_id}")

    def _adaptive_policy(self, node: Node) -> TmcDbLbt:
        policy = self._policies[node.node_id]
        if node.policy_kind is not PolicyKind.ADAPTIVE or not isinstance(
            policy, TmcDbLbt
        ):
            raise ValueError("recovery profiles are available only to adaptive nodes")
        return policy

    @staticmethod
    def _make_policy(node: Node) -> ContentionPolicy:
        policy_kind = node.policy_kind
        if policy_kind is PolicyKind.RANDOM:
            cw_max = (
                1023
                if node.technology is Technology.LEGACY_STA
                else 63
            )
            return RandomLbt(cw_min=15, cw_max=cw_max)
        if policy_kind is PolicyKind.PRIMARY_DB:
            return PrimaryDbLbt(alpha=11, m=4, beta=3)
        if policy_kind in (PolicyKind.TMC_DB, PolicyKind.ADAPTIVE):
            return TmcDbLbt(
                RecoveryProfile(kappa=7, beta=3, m=6, b_init=15)
            )
        raise ValueError(f"unsupported policy kind: {policy_kind}")

    def _rearm_after(self, node: Node, collision: bool) -> None:
        policy = self._policies[node.node_id]
        rng = self._rngs[node.node_id]

        if isinstance(policy, RandomLbt):
            policy.collision() if collision else policy.success()
            selected = policy.draw(rng)
            node.deterministic_countdown = False
        elif isinstance(policy, PrimaryDbLbt):
            if collision:
                node.db_state.collision()
            else:
                node.db_state.success()
                node.db_initialized = True
            node.deterministic_countdown = (
                node.db_state.retries % policy.m < policy.beta
            )
            selected = policy.next_backoff(node.db_state, rng)
        else:
            if collision and not node.db_initialized:
                selected = policy.initial_backoff(rng)
                node.deterministic_countdown = False
            else:
                if collision:
                    node.db_state.collision()
                else:
                    node.db_state.success()
                    node.db_initialized = True
                node.deterministic_countdown = (
                    node.db_state.retries % policy.profile.kappa
                    < policy.profile.beta
                )
                selected = policy.next_backoff(node.db_state, rng)

        node.selected = selected
        node.remaining = selected

    @staticmethod
    def _records_interruption(node: Node) -> bool:
        if node.policy_kind in (PolicyKind.TMC_DB, PolicyKind.ADAPTIVE):
            return node.deterministic_countdown
        return True

    def _success_airtime(self, node: Node) -> tuple[int, int]:
        if node.technology is Technology.NRU:
            reservation = (
                self.nru_sync_us - self.now_us % self.nru_sync_us
            ) % self.nru_sync_us
            return reservation, self.tx_us - reservation
        return 0, self.tx_us - self.wifi_ack_us
