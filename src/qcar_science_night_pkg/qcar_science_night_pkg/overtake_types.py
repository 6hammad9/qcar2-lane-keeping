from dataclasses import dataclass


@dataclass(frozen=True)
class ObstacleStatus:
    obstacle_ahead: bool
    emergency: bool
    left_clear: bool
    right_clear: bool
    front_min: float
    left_min: float
    right_min: float
    front_count: int
    left_count: int
    right_count: int

    # Narrow, curvature-following corridor: the volume the car body will
    # actually sweep, as opposed to the wide straight-ahead lane rectangle.
    # In a curve the wide box points at the outside road edge and cannot be
    # trusted, but this one still can. See overtake_safety.
    front_narrow_min: float = -1.0
    front_narrow_count: int = 0

    # The emergency box's own measurements. Previously the emergency
    # distance gate was applied to front_min, which is a different box.
    emergency_min: float = -1.0
    emergency_count: int = 0

    # The lane being vacated, measured ALONGSIDE and BEHIND the car rather
    # than ahead of it. right_clear answers "is the lane I am merging into
    # clear in front of me"; every forward box necessarily reports clear once
    # the vehicle being passed is level with the bumper. This answers "am I
    # actually past it yet". Defaults to clear so a status built without the
    # measurement behaves as it did before; the distance backstop in
    # OvertakeStateMachine.min_overtake_progress covers that case.
    flank_clear: bool = True
    flank_min: float = -1.0
    flank_count: int = 0


@dataclass(frozen=True)
class OvertakeDecision:
    state: str
    offset: float
    motion_enabled: bool
