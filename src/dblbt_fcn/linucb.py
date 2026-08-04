"""Independent per-arm LinUCB state, updates, and persistence."""

import math
import os
from collections.abc import Sequence
from numbers import Integral, Real
from os import PathLike
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np


_SCHEMA_VERSION = 1
_REQUIRED_FIELDS = frozenset(
    {
        "A",
        "b",
        "num_arms",
        "context_dim",
        "ridge",
        "exploration",
        "schema_version",
        "action_grid_hash",
    }
)


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    numeric = int(value)
    if numeric <= 0:
        raise ValueError(f"{name} must be positive")
    return numeric


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


def _action_grid_hash(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("action_grid_hash must be a SHA-256 hex string")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("action_grid_hash must be a SHA-256 hex string")
    return normalized


def _path(value: str | PathLike[str]) -> Path:
    try:
        return Path(value)
    except TypeError as error:
        raise ValueError("path must be a filesystem path") from error


def _scalar(name: str, value: np.ndarray) -> object:
    if value.shape != ():
        raise ValueError(f"{name} must be a scalar")
    return value.item()


def _state_arrays(
    A: object,
    b: object,
    num_arms: int,
    context_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw_A = np.asarray(A)
    raw_b = np.asarray(b)
    if raw_A.ndim != 3 or raw_A.shape[1:] != (
        raw_A.shape[1],
        raw_A.shape[1],
    ):
        raise ValueError("A has an invalid shape")
    if raw_b.ndim != 2 or raw_b.shape != (
        raw_A.shape[0],
        raw_A.shape[1],
    ):
        raise ValueError("b has an invalid shape")
    if raw_A.shape[0] != num_arms:
        raise ValueError("num_arms metadata conflicts with state arrays")
    if raw_A.shape[1] != context_dim:
        raise ValueError("context_dim metadata conflicts with state arrays")
    if raw_A.dtype.kind not in "iuf" or raw_b.dtype.kind not in "iuf":
        raise ValueError("A and b must contain real numeric values")

    A_array = np.array(raw_A, dtype=np.float64, copy=True)
    b_array = np.array(raw_b, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(A_array)):
        raise ValueError("A must contain only finite values")
    if not np.all(np.isfinite(b_array)):
        raise ValueError("b must contain only finite values")

    for arm in range(num_arms):
        if not np.array_equal(A_array[arm], A_array[arm].T):
            raise ValueError(f"A for arm {arm} must be symmetric")
        try:
            np.linalg.cholesky(A_array[arm])
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f"A for arm {arm} must be positive definite"
            ) from error
    return A_array, b_array


class LinUCB:
    """Maintain an independent ridge-regression system for every arm."""

    schema_version = _SCHEMA_VERSION

    def __init__(
        self,
        num_arms: Integral,
        context_dim: Integral,
        ridge: Real = 1.0,
        exploration: Real = 0.5,
        *,
        action_grid_hash: str | None = None,
    ) -> None:
        self.num_arms = _positive_int("num_arms", num_arms)
        self.context_dim = _positive_int("context_dim", context_dim)
        self.ridge = _finite_real("ridge", ridge)
        if self.ridge <= 0:
            raise ValueError("ridge must be positive")
        self.exploration = _finite_real("exploration", exploration)
        if self.exploration < 0:
            raise ValueError("exploration must be non-negative")
        self.action_grid_hash = (
            None
            if action_grid_hash is None
            else _action_grid_hash(action_grid_hash)
        )

        identity = np.eye(self.context_dim, dtype=np.float64)
        self.A = np.repeat(
            (self.ridge * identity)[np.newaxis, :, :],
            self.num_arms,
            axis=0,
        )
        self.b = np.zeros(
            (self.num_arms, self.context_dim), dtype=np.float64
        )

    def select(
        self,
        context: object,
        *,
        candidate_arms: object = None,
    ) -> int:
        """Return the lowest-index arm attaining the largest LinUCB score."""
        x = self._context(context)
        if candidate_arms is None:
            candidates = tuple(range(self.num_arms))
        else:
            if isinstance(candidate_arms, (str, bytes)) or not isinstance(
                candidate_arms, Sequence
            ):
                raise ValueError("candidate_arms must be a nonempty sequence")
            candidates = tuple(candidate_arms)
            if (
                not candidates
                or any(type(arm) is not int for arm in candidates)
                or any(not 0 <= arm < self.num_arms for arm in candidates)
                or len(set(candidates)) != len(candidates)
            ):
                raise ValueError(
                    "candidate_arms must contain unique configured arm integers"
                )
            candidates = tuple(sorted(candidates))
        best_arm = candidates[0]
        best_score = -math.inf
        for arm in candidates:
            try:
                estimate = np.linalg.solve(self.A[arm], self.b[arm])
                uncertainty_vector = np.linalg.solve(self.A[arm], x)
            except np.linalg.LinAlgError as error:
                raise RuntimeError(
                    f"failed to solve LinUCB system for arm {arm}"
                ) from error
            if not np.all(np.isfinite(estimate)) or not np.all(
                np.isfinite(uncertainty_vector)
            ):
                raise RuntimeError(
                    f"non-finite LinUCB score for arm {arm}"
                )
            with np.errstate(over="ignore", invalid="ignore"):
                exploitation = float(estimate @ x)
                variance = float(x @ uncertainty_vector)
                score = exploitation + self.exploration * math.sqrt(
                    max(variance, 0.0)
                )
            if not math.isfinite(score):
                raise RuntimeError(
                    f"non-finite LinUCB score for arm {arm}"
                )
            if score > best_score:
                best_arm = arm
                best_score = score
        return best_arm

    def update(self, arm: object, context: object, reward: object) -> None:
        """Update only the selected arm with one finite observation."""
        arm_index = self._arm(arm)
        x = self._context(context)
        reward_value = _finite_real("reward", reward)
        with np.errstate(over="ignore", invalid="ignore"):
            updated_A = self.A[arm_index] + np.outer(x, x)
            updated_b = self.b[arm_index] + reward_value * x
        if not np.all(np.isfinite(updated_A)) or not np.all(
            np.isfinite(updated_b)
        ):
            raise RuntimeError(
                f"update produced non-finite state for arm {arm_index}"
            )
        self.A[arm_index] = updated_A
        self.b[arm_index] = updated_b

    def clone(self) -> "LinUCB":
        """Return an equal agent whose mutable arrays share no memory."""
        cloned = type(self)(
            self.num_arms,
            self.context_dim,
            ridge=self.ridge,
            exploration=self.exploration,
            action_grid_hash=self.action_grid_hash,
        )
        cloned.A = self.A.copy()
        cloned.b = self.b.copy()
        return cloned

    def save(
        self,
        path: str | PathLike[str],
        *,
        action_grid_hash: str | None = None,
    ) -> None:
        """Save validated state to a non-pickled NPZ archive."""
        target = _path(path)
        hash_value = _action_grid_hash(
            self.action_grid_hash
            if action_grid_hash is None
            else action_grid_hash
        )
        A, b = _state_arrays(
            self.A, self.b, self.num_arms, self.context_dim
        )
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as destination:
                temporary = Path(destination.name)
                np.savez(
                    destination,
                    A=A,
                    b=b,
                    num_arms=np.array(self.num_arms, dtype=np.int64),
                    context_dim=np.array(
                        self.context_dim, dtype=np.int64
                    ),
                    ridge=np.array(self.ridge, dtype=np.float64),
                    exploration=np.array(
                        self.exploration, dtype=np.float64
                    ),
                    schema_version=np.array(
                        self.schema_version, dtype=np.int64
                    ),
                    action_grid_hash=np.array(hash_value),
                )
                destination.flush()
                os.fsync(destination.fileno())

            saved = type(self).load(
                temporary, expected_action_grid_hash=hash_value
            )
            if (
                saved.num_arms != self.num_arms
                or saved.context_dim != self.context_dim
                or saved.ridge != self.ridge
                or saved.exploration != self.exploration
                or not np.array_equal(saved.A, A)
                or not np.array_equal(saved.b, b)
            ):
                raise ValueError(
                    "temporary LinUCB NPZ does not match the requested state"
                )
            os.replace(temporary, target)
        except Exception as error:
            raise OSError(
                f"failed to save LinUCB NPZ to {target}: {error}"
            ) from error
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def load(
        cls,
        path: str | PathLike[str],
        *,
        expected_action_grid_hash: str | None = None,
    ) -> "LinUCB":
        """Load and validate a LinUCB NPZ archive without enabling pickle."""
        target = _path(path)
        if not target.exists():
            raise FileNotFoundError(f"LinUCB NPZ file not found: {target}")
        if not target.is_file():
            raise OSError(f"LinUCB NPZ path is not a file: {target}")

        try:
            archive = np.load(target, allow_pickle=False)
        except (OSError, ValueError, EOFError, BadZipFile) as error:
            raise ValueError(
                "failed to load LinUCB NPZ from "
                f"{target}: invalid or corrupt NPZ archive"
            ) from error

        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError(f"{target} is not an NPZ archive")
        with archive:
            missing = sorted(_REQUIRED_FIELDS.difference(archive.files))
            if missing:
                raise ValueError(
                    "LinUCB NPZ is missing required fields: "
                    + ", ".join(missing)
                )
            try:
                values = {
                    name: archive[name].copy() for name in _REQUIRED_FIELDS
                }
            except (OSError, ValueError, EOFError, BadZipFile) as error:
                raise ValueError(
                    "failed to load LinUCB NPZ from "
                    f"{target}: invalid or corrupt NPZ archive"
                ) from error

        schema_version = _scalar(
            "schema_version", values["schema_version"]
        )
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, Integral)
            or int(schema_version) != _SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported schema_version: {schema_version!r}"
            )

        num_arms = _positive_int(
            "num_arms", _scalar("num_arms", values["num_arms"])
        )
        context_dim = _positive_int(
            "context_dim", _scalar("context_dim", values["context_dim"])
        )
        ridge = _finite_real(
            "ridge", _scalar("ridge", values["ridge"])
        )
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        exploration = _finite_real(
            "exploration", _scalar("exploration", values["exploration"])
        )
        if exploration < 0:
            raise ValueError("exploration must be non-negative")
        hash_value = _action_grid_hash(
            _scalar("action_grid_hash", values["action_grid_hash"])
        )
        if expected_action_grid_hash is not None:
            expected = _action_grid_hash(expected_action_grid_hash)
            if hash_value != expected:
                raise ValueError(
                    "action_grid_hash does not match the expected grid"
                )

        A, b = _state_arrays(
            values["A"], values["b"], num_arms, context_dim
        )
        agent = cls(
            num_arms,
            context_dim,
            ridge=ridge,
            exploration=exploration,
            action_grid_hash=hash_value,
        )
        agent.A = A
        agent.b = b
        return agent

    def _context(self, context: object) -> np.ndarray:
        try:
            raw = np.asarray(context)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"context must have exact shape ({self.context_dim},)"
            ) from error
        if raw.shape != (self.context_dim,):
            raise ValueError(
                f"context must have exact shape ({self.context_dim},)"
            )
        if raw.dtype.kind not in "iuf":
            raise ValueError("context must contain real numeric values")
        values = np.array(raw, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(values)):
            raise ValueError("context must contain only finite values")
        return values

    def _arm(self, arm: object) -> int:
        if isinstance(arm, bool) or not isinstance(arm, Integral):
            raise ValueError("arm must be an integer")
        arm_index = int(arm)
        if not 0 <= arm_index < self.num_arms:
            raise ValueError(
                f"arm must be between 0 and {self.num_arms - 1}"
            )
        return arm_index
