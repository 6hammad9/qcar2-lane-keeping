"""Lane probing (IDEAM LP) wired into the overtake state machine.

The behaviour under test: a pass forbidden by curvature must not also forbid
creeping forward in the lane we already occupy. Before this, the car stopped
at the detection distance and sat in WAIT_FOR_CLEAR indefinitely, because
most of this 38.8 m route is above the curvature threshold.
"""

from qcar_science_night_pkg.overtake_state_machine import OvertakeStateMachine
from qcar_science_night_pkg.overtake_types import ObstacleStatus


def status(
    *,
    obstacle=False,
    emergency=False,
    left=True,
    right=True,
    front_min=None,
    front_count=0,
    left_count=0,
    right_count=0,
    front_narrow_min=-1.0,
    front_narrow_count=0,
):
    return ObstacleStatus(
        obstacle_ahead=obstacle,
        emergency=emergency,
        left_clear=left,
        right_clear=right,
        front_min=(
            (0.8 if obstacle else 2.0) if front_min is None else front_min
        ),
        left_min=2.0,
        right_min=2.0,
        front_count=front_count,
        left_count=left_count,
        right_count=right_count,
        front_narrow_min=front_narrow_min,
        front_narrow_count=front_narrow_count,
    )


def machine(**kwargs):
    defaults = dict(
        obstacle_confirm_required=1,
        left_clear_confirm_required=1,
        no_obstacle_confirm_required=2,
        right_clear_confirm_required=1,
        min_overtake_progress=0.8,
        min_return_progress=0.3,
        probe_min_gap_m=0.5,
    )
    defaults.update(kwargs)
    return OvertakeStateMachine(**defaults)


def test_curve_forbidding_the_pass_probes_instead_of_freezing():
    sm = machine()

    # overtake_allowed=False is the curvature gate: this is the case that
    # left the car stationary with a clear lane beside it.
    decision = sm.update(
        status(obstacle=True, front_count=1), False, progress=1.0
    )

    assert decision.state == sm.PROBE
    assert decision.motion_enabled is True
    assert decision.offset == 0.0


def test_probing_stops_once_the_gap_closes():
    sm = machine()
    sm.update(status(obstacle=True, front_count=1), False, progress=1.0)

    decision = sm.update(
        status(obstacle=True, front_min=0.45, front_count=1),
        False,
        progress=1.3,
    )

    assert decision.state == sm.WAIT
    assert decision.motion_enabled is False


def test_probing_resumes_driving_when_the_obstacle_clears():
    sm = machine()
    sm.update(status(obstacle=True, front_count=1), False, progress=1.0)

    decision = sm.update(status(), False, progress=1.3)

    assert decision.state == sm.DRIVE
    assert decision.motion_enabled is True


def test_probing_commits_to_a_pass_when_the_route_straightens():
    sm = machine()
    sm.update(status(obstacle=True, front_count=1), False, progress=1.0)

    decision = sm.update(
        status(obstacle=True, front_count=1), True, progress=1.3
    )

    assert decision.state == sm.OVERTAKE
    assert decision.offset == sm.overtake_offset


def test_a_blocked_passing_lane_still_probes_rather_than_stopping():
    sm = machine()

    # Both lanes blocked was previously an immediate stop. Probing forward in
    # our own lane is still available and still bounded by the gap.
    decision = sm.update(
        status(obstacle=True, left=False, right=False, front_count=1),
        True,
        progress=1.0,
    )

    assert decision.state == sm.PROBE
    assert decision.motion_enabled is True


def test_probe_disabled_restores_the_stop_dead_behaviour():
    sm = machine(probe_enabled=False)

    decision = sm.update(
        status(obstacle=True, front_count=1), False, progress=1.0
    )

    assert decision.state == sm.WAIT
    assert decision.motion_enabled is False


def test_an_interrupted_pass_never_probes_forward():
    sm = machine()
    sm.update(status(obstacle=True, front_count=1), True, progress=5.0)

    # Side obstacle interrupts the pass: the car is half in the passing lane.
    # Creeping forward there is not probing, it is driving into the gap.
    decision = sm.update(status(left=False, left_count=2), True, progress=5.0)
    assert decision.state == sm.WAIT
    assert decision.motion_enabled is False

    decision = sm.update(status(left=False, left_count=2), False, progress=5.0)
    assert decision.state == sm.WAIT
    assert decision.motion_enabled is False


def test_emergency_during_a_probe_still_stops():
    sm = machine()
    sm.update(status(obstacle=True, front_count=1), False, progress=1.0)

    decision = sm.update(
        status(obstacle=True, emergency=True, front_count=1),
        False,
        progress=1.2,
    )

    assert decision.state == sm.ESTOP
    assert decision.motion_enabled is False


def test_the_curve_corridor_measures_the_gap_not_the_wide_box():
    sm = machine()

    # The wide front box reads 0.80 m at the outside road edge; the corridor
    # the car will actually sweep has the obstacle at 0.45 m. The corridor is
    # the one that must end the probe.
    decision = sm.update(
        status(
            obstacle=True,
            front_min=0.80,
            front_count=1,
            front_narrow_min=0.45,
            front_narrow_count=3,
        ),
        False,
        progress=1.0,
    )

    assert decision.state == sm.WAIT
    assert decision.motion_enabled is False


def test_an_empty_corridor_does_not_read_as_a_zero_gap():
    sm = machine()

    # -1.0 is the analyzer's "nothing detected". Treating it as a distance
    # would stop the car every time the corridor happened to be empty.
    decision = sm.update(
        status(
            obstacle=True,
            front_min=0.80,
            front_count=1,
            front_narrow_min=-1.0,
            front_narrow_count=0,
        ),
        False,
        progress=1.0,
    )

    assert decision.state == sm.PROBE
