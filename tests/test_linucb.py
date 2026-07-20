"""Independent LinUCB agents and their persisted state contract."""

from decimal import Decimal
import math
from pathlib import Path
import zipfile

import numpy as np
import pytest

from dblbt_fcn.linucb import LinUCB


GRID_HASH = "a" * 64
OTHER_GRID_HASH = "b" * 64


def test_initializes_independent_ridge_systems() -> None:
    agent = LinUCB(2, 2, ridge=2.5, exploration=0.0)

    assert agent.A.shape == (2, 2, 2)
    assert agent.b.shape == (2, 2)
    assert agent.A.dtype == np.float64
    assert agent.b.dtype == np.float64
    np.testing.assert_array_equal(
        agent.A, np.stack([2.5 * np.eye(2), 2.5 * np.eye(2)])
    )
    np.testing.assert_array_equal(agent.b, np.zeros((2, 2)))
    assert not np.shares_memory(agent.A[0], agent.A[1])


def test_update_makes_rewarded_arm_the_selected_arm() -> None:
    agent = LinUCB(2, 2, ridge=1.0, exploration=0.0)
    context = [1, 0]

    agent.update(1, context, 1)

    assert agent.select(context) == 1


def test_default_exploration_uses_the_preregistered_value() -> None:
    agent = LinUCB(2, 2)
    agent.A[0] = [[100.0, 0.0], [0.0, 1.0]]
    agent.b[0] = [100.0, 0.0]
    agent.A[1] = [[1.0, 0.0], [0.0, 1.0]]
    agent.b[1] = [0.75, 0.0]

    # At exploration 0.5, the scores for x=[1, 0] are 1.05 and 1.25.
    assert agent.select([1, 0]) == 1
    assert agent.exploration == 0.5


def test_select_uses_the_hand_calculated_confidence_score() -> None:
    agent = LinUCB(2, 2, ridge=1.0, exploration=2.0)
    agent.A[0] = [[4.0, 0.0], [0.0, 1.0]]
    agent.b[0] = [4.0, 0.0]
    agent.A[1] = [[1.0, 0.0], [0.0, 4.0]]
    agent.b[1] = [0.0, 4.0]

    # For x=[2, 1], scores are 2 + 2*sqrt(2) and 1 + 2*sqrt(4.25).
    assert agent.select([2, 1]) == 1


def test_equal_scores_choose_the_smallest_arm() -> None:
    assert LinUCB(3, 2).select([1, 0]) == 0


def test_update_changes_only_the_requested_arm() -> None:
    agent = LinUCB(3, 2)
    original_A = agent.A.copy()
    original_b = agent.b.copy()

    agent.update(1, [2, 3], 0.5)

    np.testing.assert_array_equal(agent.A[0], original_A[0])
    np.testing.assert_array_equal(agent.A[2], original_A[2])
    np.testing.assert_array_equal(agent.b[0], original_b[0])
    np.testing.assert_array_equal(agent.b[2], original_b[2])
    np.testing.assert_array_equal(
        agent.A[1], original_A[1] + np.array([[4.0, 6.0], [6.0, 9.0]])
    )
    np.testing.assert_array_equal(agent.b[1], np.array([1.0, 1.5]))


def test_clone_is_equal_but_has_independent_state() -> None:
    agent = LinUCB(
        2,
        2,
        ridge=1.5,
        exploration=0.25,
        action_grid_hash=GRID_HASH,
    )
    agent.update(1, [1, 2], 0.75)

    cloned = agent.clone()

    assert cloned is not agent
    assert cloned.num_arms == agent.num_arms
    assert cloned.context_dim == agent.context_dim
    assert cloned.ridge == agent.ridge
    assert cloned.exploration == agent.exploration
    assert cloned.action_grid_hash == agent.action_grid_hash
    np.testing.assert_array_equal(cloned.A, agent.A)
    np.testing.assert_array_equal(cloned.b, agent.b)
    assert not np.shares_memory(cloned.A, agent.A)
    assert not np.shares_memory(cloned.b, agent.b)

    cloned.update(0, [1, 0], 1)
    assert not np.array_equal(cloned.A, agent.A)
    assert not np.array_equal(cloned.b, agent.b)


def test_preregistered_agent_clone_preserves_independent_byte_state() -> None:
    agent = LinUCB(24, 11, ridge=1.0, exploration=0.5)
    agent.update(23, np.arange(1, 12), 0.75)

    cloned = agent.clone()

    assert agent.A.shape == cloned.A.shape == (24, 11, 11)
    assert agent.b.shape == cloned.b.shape == (24, 11)
    np.testing.assert_array_equal(cloned.A, agent.A)
    np.testing.assert_array_equal(cloned.b, agent.b)
    assert not np.shares_memory(cloned.A, agent.A)
    assert not np.shares_memory(cloned.b, agent.b)

    original_A_bytes = agent.A.tobytes()
    original_b_bytes = agent.b.tobytes()
    cloned.update(0, np.arange(11), 1.0)

    assert agent.A.tobytes() == original_A_bytes
    assert agent.b.tobytes() == original_b_bytes
    assert cloned.A.tobytes() != original_A_bytes
    assert cloned.b.tobytes() != original_b_bytes


@pytest.mark.parametrize("field", ["num_arms", "context_dim"])
@pytest.mark.parametrize("value", [0, -1, True, 1.0, np.float64(2)])
def test_constructor_requires_positive_exact_integer_dimensions(
    field: str, value: object
) -> None:
    arguments: dict[str, object] = {"num_arms": 2, "context_dim": 2}
    arguments[field] = value

    with pytest.raises(ValueError, match=field):
        LinUCB(**arguments)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ridge", 0),
        ("ridge", -1),
        ("ridge", True),
        ("ridge", math.nan),
        ("ridge", math.inf),
        ("ridge", 1 + 0j),
        ("ridge", Decimal("1")),
        ("exploration", -1),
        ("exploration", True),
        ("exploration", math.nan),
        ("exploration", math.inf),
        ("exploration", 1 + 0j),
        ("exploration", Decimal("1")),
    ],
)
def test_constructor_rejects_invalid_numeric_parameters(
    field: str, value: object
) -> None:
    arguments: dict[str, object] = {
        "num_arms": 2,
        "context_dim": 2,
        "ridge": 1.0,
        "exploration": 0.0,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=field):
        LinUCB(**arguments)


@pytest.mark.parametrize(
    "context",
    [
        [1],
        [1, 2, 3],
        [[1, 2]],
        1,
        [1, math.nan],
        [1, math.inf],
        [1, 2 + 0j],
        np.array([1, object()], dtype=object),
        [1, "2"],
        [1, Decimal("2")],
    ],
)
def test_select_rejects_invalid_context_without_reshaping(
    context: object,
) -> None:
    with pytest.raises(ValueError, match="context"):
        LinUCB(2, 2).select(context)


def test_context_is_calculated_as_float64() -> None:
    agent = LinUCB(2, 2, exploration=0.0)

    assert agent.select(np.array([1, 0], dtype=np.int16)) == 0
    agent.update(1, np.array([1, 0], dtype=np.float32), np.float32(1))
    assert agent.A.dtype == np.float64
    assert agent.b.dtype == np.float64


@pytest.mark.parametrize("arm", [-1, 2, True, 1.0, np.float64(1)])
def test_update_rejects_invalid_arm(arm: object) -> None:
    with pytest.raises(ValueError, match="arm"):
        LinUCB(2, 2).update(arm, [1, 0], 1)


@pytest.mark.parametrize(
    "reward",
    [True, math.nan, math.inf, -math.inf, 1 + 0j, "1", Decimal("1")],
)
def test_update_rejects_invalid_reward(reward: object) -> None:
    with pytest.raises(ValueError, match="reward"):
        LinUCB(2, 2).update(0, [1, 0], reward)


def test_select_does_not_use_matrix_inverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("matrix inversion is forbidden")

    monkeypatch.setattr(np.linalg, "inv", forbidden)
    monkeypatch.setattr(np.linalg, "pinv", forbidden)

    agent = LinUCB(2, 2, exploration=0.5)
    agent.update(1, [1, 0], 1)
    assert agent.select([1, 0]) == 1


def test_select_reports_solve_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> np.ndarray:
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(np.linalg, "solve", fail)

    with pytest.raises(RuntimeError, match="solve.*arm 0"):
        LinUCB(2, 2).select([1, 0])


def test_select_reports_non_finite_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def non_finite(*args: object, **kwargs: object) -> np.ndarray:
        return np.array([math.inf, 0.0])

    monkeypatch.setattr(np.linalg, "solve", non_finite)

    with pytest.raises(RuntimeError, match="non-finite.*arm 0"):
        LinUCB(2, 2).select([1, 0])


def test_save_requires_a_valid_action_grid_hash(tmp_path: Path) -> None:
    path = tmp_path / "agent.npz"
    agent = LinUCB(2, 2)

    with pytest.raises(ValueError, match="action_grid_hash"):
        agent.save(path)
    with pytest.raises(ValueError, match="action_grid_hash"):
        agent.save(path, action_grid_hash="")
    with pytest.raises(ValueError, match="action_grid_hash"):
        agent.save(path, action_grid_hash="not-a-sha256")

    agent.save(path, action_grid_hash=GRID_HASH.upper())
    loaded = LinUCB.load(path, expected_action_grid_hash=GRID_HASH)
    assert loaded.action_grid_hash == GRID_HASH


def test_npz_round_trip_preserves_exact_independent_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent.npz"
    agent = LinUCB(
        2,
        2,
        ridge=1.5,
        exploration=0.25,
        action_grid_hash=GRID_HASH,
    )
    agent.update(1, [1, 2], 0.75)

    agent.save(path)
    loaded = LinUCB.load(path, expected_action_grid_hash=GRID_HASH)

    assert loaded.num_arms == 2
    assert loaded.context_dim == 2
    assert loaded.ridge == 1.5
    assert loaded.exploration == 0.25
    assert loaded.action_grid_hash == GRID_HASH
    np.testing.assert_array_equal(loaded.A, agent.A)
    np.testing.assert_array_equal(loaded.b, agent.b)
    assert not np.shares_memory(loaded.A, agent.A)
    assert not np.shares_memory(loaded.b, agent.b)

    loaded.update(0, [1, 0], 1)
    assert not np.array_equal(loaded.A, agent.A)
    assert not np.array_equal(loaded.b, agent.b)


def test_failed_save_preserves_existing_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "agent.npz"
    original = LinUCB(2, 2, exploration=0.25, action_grid_hash=GRID_HASH)
    original.update(1, [1, 2], 0.75)
    original.save(path)
    original_bytes = path.read_bytes()

    replacement = LinUCB(
        2, 2, exploration=0.75, action_grid_hash=GRID_HASH
    )
    replacement.update(0, [2, 1], 1.0)

    def interrupt_save(destination: object, **values: object) -> None:
        destination.write(b"partial")  # type: ignore[attr-defined]
        raise OSError("simulated write failure")

    monkeypatch.setattr(np, "savez", interrupt_save)

    with pytest.raises(
        OSError, match="failed to save.*simulated write failure"
    ):
        replacement.save(path)

    assert path.read_bytes() == original_bytes
    loaded = LinUCB.load(path, expected_action_grid_hash=GRID_HASH)
    np.testing.assert_array_equal(loaded.A, original.A)
    np.testing.assert_array_equal(loaded.b, original.b)
    assert loaded.exploration == original.exploration
    assert list(tmp_path.iterdir()) == [path]


def test_failed_first_save_leaves_no_final_or_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "agent.npz"

    def interrupt_save(destination: object, **values: object) -> None:
        destination.write(b"partial")  # type: ignore[attr-defined]
        raise OSError("simulated write failure")

    monkeypatch.setattr(np, "savez", interrupt_save)

    with pytest.raises(
        OSError, match="failed to save.*simulated write failure"
    ):
        LinUCB(2, 2, action_grid_hash=GRID_HASH).save(path)

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_load_disables_pickle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "agent.npz"
    LinUCB(1, 1, action_grid_hash=GRID_HASH).save(path)
    original_load = np.load
    observed: dict[str, object] = {}

    def recording_load(*args: object, **kwargs: object):
        observed.update(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", recording_load)

    LinUCB.load(path)
    assert observed["allow_pickle"] is False


def test_load_rejects_schema_and_expected_hash_mismatches(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent.npz"
    agent = LinUCB(2, 2, action_grid_hash=GRID_HASH)
    agent.save(path)

    with pytest.raises(ValueError, match="action_grid_hash"):
        LinUCB.load(path, expected_action_grid_hash=OTHER_GRID_HASH)

    values = _read_npz(path)
    values["schema_version"] = np.array(999, dtype=np.int64)
    np.savez(path, **values)
    with pytest.raises(ValueError, match="schema_version"):
        LinUCB.load(path)


def test_load_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "missing.npz"
    _save_valid_npz(path)
    values = _read_npz(path)
    del values["b"]
    np.savez(path, **values)

    with pytest.raises(ValueError, match="missing.*b"):
        LinUCB.load(path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"A": np.ones((2, 2))}, "A.*shape"),
        ({"b": np.ones((2, 2))}, "b.*shape"),
        ({"A": np.array([[[1.0, 0.0], [0.0, math.nan]]])}, "A.*finite"),
        ({"b": np.array([[math.inf, 0.0]])}, "b.*finite"),
        ({"num_arms": np.array(2)}, "num_arms.*conflict"),
        ({"context_dim": np.array(3)}, "context_dim.*conflict"),
        ({"A": np.array([[[1.0, 1.0], [0.0, 1.0]]])}, "symmetric"),
        ({"A": np.array([[[1.0, 0.0], [0.0, 0.0]]])}, "positive definite"),
        ({"action_grid_hash": np.array("bad")}, "action_grid_hash"),
    ],
)
def test_load_rejects_corrupt_state(
    tmp_path: Path, change: dict[str, np.ndarray], message: str
) -> None:
    path = tmp_path / "corrupt.npz"
    _save_valid_npz(path)
    values = _read_npz(path)
    values.update(change)
    np.savez(path, **values)

    with pytest.raises(ValueError, match=message):
        LinUCB.load(path)


def test_load_reports_bad_npz_and_path_errors(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.npz"
    malformed = tmp_path / "malformed.npz"
    malformed.write_bytes(b"not an npz file")

    with pytest.raises(FileNotFoundError, match="does-not-exist"):
        LinUCB.load(missing)
    with pytest.raises(ValueError, match="NPZ"):
        LinUCB.load(malformed)
    with pytest.raises(OSError, match="directory"):
        LinUCB(1, 1, action_grid_hash=GRID_HASH).save(tmp_path)


def test_load_rejects_npy_content_disguised_as_npz(tmp_path: Path) -> None:
    path = tmp_path / "disguised.npz"
    with path.open("wb") as destination:
        np.save(destination, np.array([1.0]))

    with pytest.raises(ValueError, match="NPZ"):
        LinUCB.load(path)


def test_load_reports_member_crc_corruption_as_invalid_npz(
    tmp_path: Path,
) -> None:
    path = tmp_path / "crc-corrupt.npz"
    _save_valid_npz(path)

    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo("A.npy")
    with path.open("r+b") as handle:
        handle.seek(member.header_offset)
        local_header = handle.read(30)
        filename_size = int.from_bytes(local_header[26:28], "little")
        extra_size = int.from_bytes(local_header[28:30], "little")
        data_offset = member.header_offset + 30 + filename_size + extra_size
        handle.seek(data_offset + member.file_size - 1)
        original = handle.read(1)
        handle.seek(-1, 1)
        handle.write(bytes([original[0] ^ 0xFF]))

    with pytest.raises(ValueError, match="invalid or corrupt NPZ"):
        LinUCB.load(path)


def _save_valid_npz(path: Path) -> None:
    LinUCB(1, 2, action_grid_hash=GRID_HASH).save(path)


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}
