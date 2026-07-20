"""Paper DB-LBT and Random LBT state-transition invariants."""

import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dblbt_fcn.policies import DbState, PrimaryDbLbt, RandomLbt, TmcDbLbt
from dblbt_fcn.types import RecoveryProfile


def test_db_state_starts_clear() -> None:
    state = DbState()

    assert state.interruptions == 0
    assert state.retries == 0


def test_tmc_deterministic_choice_resets_interruptions() -> None:
    policy = TmcDbLbt(RecoveryProfile(kappa=7, beta=3, m=6, b_init=15))
    state = DbState(interruptions=5, retries=1)

    assert policy.next_backoff(state, random.Random(1)) == 16
    assert state.interruptions == 0


def test_tmc_random_choice_preserves_interruptions() -> None:
    policy = TmcDbLbt(RecoveryProfile(kappa=7, beta=3, m=6, b_init=15))
    state = DbState(interruptions=5, retries=3)

    value = policy.next_backoff(state, random.Random(1))

    assert 0 <= value <= 6
    assert state.interruptions == 5


def test_tmc_initial_backoff_includes_both_bounds() -> None:
    policy = TmcDbLbt(RecoveryProfile(kappa=7, beta=3, m=6, b_init=15))

    assert policy.initial_backoff(random.Random(31)) == 0
    assert policy.initial_backoff(random.Random(12)) == 15


def test_primary_policy_uses_modulo_m() -> None:
    policy = PrimaryDbLbt(alpha=11, m=4, beta=3)
    state = DbState(interruptions=2, retries=4)

    assert policy.next_backoff(state, random.Random(1)) == 13
    assert state.interruptions == 0


def test_primary_random_choice_preserves_interruptions() -> None:
    policy = PrimaryDbLbt(alpha=11, m=4, beta=3)
    state = DbState(interruptions=2, retries=3)

    value = policy.next_backoff(state, random.Random(1))

    assert 0 <= value <= 3
    assert state.interruptions == 2


def test_primary_deterministic_choice_does_not_consume_rng() -> None:
    policy = PrimaryDbLbt(alpha=11, m=4, beta=3)
    state = DbState(interruptions=2, retries=4)
    rng = random.Random(1)
    initial_rng_state = rng.getstate()

    policy.next_backoff(state, rng)

    assert rng.getstate() == initial_rng_state


def test_tmc_deterministic_choice_does_not_consume_rng() -> None:
    policy = TmcDbLbt(RecoveryProfile(kappa=7, beta=3, m=6, b_init=15))
    state = DbState(interruptions=5, retries=1)
    rng = random.Random(1)
    initial_rng_state = rng.getstate()

    policy.next_backoff(state, rng)

    assert rng.getstate() == initial_rng_state


def test_random_lbt_draw_includes_current_cw() -> None:
    policy = RandomLbt(cw_min=1, cw_max=1)

    assert policy.draw(random.Random(2)) == 0
    assert policy.draw(random.Random(0)) == 1


def test_random_lbt_doubles_cw_and_resets_after_success() -> None:
    policy = RandomLbt(cw_min=15, cw_max=63)

    policy.collision()
    assert policy.cw == 31
    policy.collision()
    assert policy.cw == 63
    policy.success()
    assert policy.cw == 15


@given(
    retries=st.integers(min_value=0, max_value=100),
    interruptions=st.integers(min_value=0, max_value=64),
)
def test_db_state_success_resets_retries(
    retries: int, interruptions: int
) -> None:
    state = DbState(interruptions=interruptions, retries=retries)

    state.success()

    assert state.retries == 0
    assert state.interruptions == interruptions


@given(
    retries=st.integers(min_value=0, max_value=100),
    interruptions=st.integers(min_value=0, max_value=64),
)
def test_db_state_collision_increments_retries(
    retries: int, interruptions: int
) -> None:
    state = DbState(interruptions=interruptions, retries=retries)

    state.collision()

    assert state.retries == retries + 1
    assert state.interruptions == interruptions


@given(
    retries=st.integers(min_value=0, max_value=100),
    interruptions=st.integers(min_value=0, max_value=64),
)
def test_primary_backoff_matches_deterministic_or_random_rule(
    retries: int, interruptions: int
) -> None:
    policy = PrimaryDbLbt(alpha=11, m=4, beta=3)
    state = DbState(interruptions=interruptions, retries=retries)

    value = policy.next_backoff(state, random.Random(retries))

    if retries % 4 < 3:
        assert value == 11 + interruptions
        assert value >= 11
        assert state.interruptions == 0
    else:
        assert 0 <= value <= 3
        assert state.interruptions == interruptions


@given(
    retries=st.integers(min_value=0, max_value=100),
    interruptions=st.integers(min_value=0, max_value=64),
)
def test_tmc_backoff_matches_deterministic_or_random_rule(
    retries: int, interruptions: int
) -> None:
    profile = RecoveryProfile(kappa=7, beta=3, m=6, b_init=15)
    policy = TmcDbLbt(profile)
    state = DbState(interruptions=interruptions, retries=retries)

    value = policy.next_backoff(state, random.Random(retries))

    if retries % 7 < 3:
        assert value == 11 + interruptions
        assert value >= 11
        assert state.interruptions == 0
    else:
        assert 0 <= value <= 6
        assert state.interruptions == interruptions


@given(seed=st.integers(), b_init=st.integers(min_value=1, max_value=255))
def test_tmc_initial_backoff_stays_in_configured_range(
    seed: int, b_init: int
) -> None:
    profile = RecoveryProfile(kappa=7, beta=3, m=6, b_init=b_init)

    value = TmcDbLbt(profile).initial_backoff(random.Random(seed))

    assert 0 <= value <= b_init


@given(
    seed=st.integers(),
    cw_min=st.integers(min_value=0, max_value=255),
    growths=st.integers(min_value=0, max_value=16),
)
def test_random_lbt_draw_and_window_transitions(
    seed: int, cw_min: int, growths: int
) -> None:
    cw_max = 2 * cw_min + 63
    policy = RandomLbt(cw_min=cw_min, cw_max=cw_max)

    for _ in range(growths):
        previous = policy.cw
        policy.collision()
        assert policy.cw == min(2 * previous + 1, cw_max)

    assert 0 <= policy.draw(random.Random(seed)) <= policy.cw
    policy.success()
    assert policy.cw == cw_min


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: DbState(interruptions=-1), "interruptions"),
        (lambda: DbState(retries=-1), "retries"),
        (lambda: PrimaryDbLbt(alpha=10, m=4, beta=3), "alpha"),
        (lambda: PrimaryDbLbt(alpha=11, m=0, beta=0), "m"),
        (lambda: PrimaryDbLbt(alpha=11, m=4, beta=0), "beta"),
        (lambda: PrimaryDbLbt(alpha=11, m=4, beta=4), "beta"),
        (lambda: RandomLbt(cw_min=-1, cw_max=63), "cw_min"),
        (lambda: RandomLbt(cw_min=16, cw_max=15), "cw_max"),
    ],
)
def test_policy_constructors_reject_invalid_bounds(
    factory: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (lambda: DbState(interruptions=True), "interruptions"),
        (lambda: DbState(interruptions=1.0), "interruptions"),
        (lambda: DbState(retries=True), "retries"),
        (lambda: DbState(retries=1.0), "retries"),
        (lambda: PrimaryDbLbt(alpha=True, m=4, beta=3), "alpha"),
        (lambda: PrimaryDbLbt(alpha=11.0, m=4, beta=3), "alpha"),
        (lambda: PrimaryDbLbt(alpha=11, m=True, beta=1), "m"),
        (lambda: PrimaryDbLbt(alpha=11, m=4.0, beta=3), "m"),
        (lambda: PrimaryDbLbt(alpha=11, m=4, beta=True), "beta"),
        (lambda: PrimaryDbLbt(alpha=11, m=4, beta=3.0), "beta"),
        (lambda: RandomLbt(cw_min=True, cw_max=63), "cw_min"),
        (lambda: RandomLbt(cw_min=15.0, cw_max=63), "cw_min"),
        (lambda: RandomLbt(cw_min=0, cw_max=True), "cw_max"),
        (lambda: RandomLbt(cw_min=15, cw_max=63.0), "cw_max"),
    ],
)
def test_policy_constructors_require_exact_integers(
    factory: object, field: str
) -> None:
    with pytest.raises(
        ValueError, match=rf"^{field} must be an integer$"
    ):
        factory()  # type: ignore[operator]
