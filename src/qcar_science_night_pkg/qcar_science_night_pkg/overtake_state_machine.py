from qcar_science_night_pkg.overtake_types import OvertakeDecision


class OvertakeStateMachine:
    DRIVE = "DRIVE"
    WAIT = "WAIT_FOR_CLEAR"
    OVERTAKE = "OVERTAKE_LEFT"
    RETURN = "RETURN_RIGHT"
    ESTOP = "EMERGENCY_STOP"

    def __init__(
        self,
        overtake_offset=0.65,
        obstacle_confirm_required=3,
        left_clear_confirm_required=2,
        no_obstacle_confirm_required=15,
        right_clear_confirm_required=10,
        return_confirm_required=30,
        min_overtake_steps=30,
        min_overtake_progress=0.80,
        min_return_progress=0.35,
    ):
        self.state = self.DRIVE
        self.overtake_offset = float(overtake_offset)

        self.obstacle_confirm_required = obstacle_confirm_required
        self.left_clear_confirm_required = left_clear_confirm_required
        self.no_obstacle_confirm_required = no_obstacle_confirm_required
        self.right_clear_confirm_required = right_clear_confirm_required
        self.return_confirm_required = return_confirm_required
        # ``min_overtake_steps`` is retained as an API compatibility shim for
        # older launch files.  Scan callbacks are not vehicle motion and must
        # never decide that a pass has completed, so completion is now based
        # exclusively on measured path progress.
        self.min_overtake_steps = min_overtake_steps
        self.min_overtake_progress = float(min_overtake_progress)
        self.min_return_progress = float(min_return_progress)

        self.obstacle_counter = 0
        self.left_clear_counter = 0
        self.right_clear_counter = 0
        self.no_obstacle_counter = 0
        self.return_counter = 0
        self.overtake_counter = 0
        self.overtake_start_progress = None
        self.return_start_progress = None
        self.wait_resume_overtake = False

    def reset_counters(self):
        self.obstacle_counter = 0
        self.left_clear_counter = 0
        self.right_clear_counter = 0
        self.no_obstacle_counter = 0
        self.return_counter = 0
        self.overtake_counter = 0

    @staticmethod
    def _progress_since(start, current):
        """Distance travelled since a state transition.

        ``current`` is supplied by the node as a monotonically increasing,
        unwrapped path distance.  Missing progress is deliberately treated as
        zero: without evidence that the car moved it is safer to remain in the
        passing lane than to return based only on a stream of LiDAR scans.
        """
        if start is None or current is None:
            return 0.0
        return max(0.0, float(current) - float(start))

    def update_counters(self, status):
        if status.obstacle_ahead:
            self.obstacle_counter += 1
            self.no_obstacle_counter = 0
        else:
            self.no_obstacle_counter += 1
            self.obstacle_counter = 0

        if status.left_clear:
            self.left_clear_counter += 1
        else:
            self.left_clear_counter = 0

        if status.right_clear:
            self.right_clear_counter += 1
        else:
            self.right_clear_counter = 0

    def start_overtake(self, progress=None):
        self.state = self.OVERTAKE
        self.overtake_counter = 0
        self.return_counter = 0
        self.overtake_start_progress = progress
        self.return_start_progress = None
        self.wait_resume_overtake = False
        return OvertakeDecision(self.state, self.overtake_offset, True)

    def update(
        self,
        status,
        overtake_allowed,
        yaw_stable=True,
        progress=None,
        pass_confirmed=True,
    ):
        self.update_counters(status)

        confirmed_obstacle = (
            self.obstacle_counter >= self.obstacle_confirm_required
        )
        confirmed_left_clear = (
            self.left_clear_counter >= self.left_clear_confirm_required
        )
        confirmed_no_obstacle = (
            self.no_obstacle_counter >= self.no_obstacle_confirm_required
        )
        confirmed_right_clear = (
            self.right_clear_counter >= self.right_clear_confirm_required
        )

        if status.emergency:
            self.state = self.ESTOP
            self.return_counter = 0
            self.overtake_counter = 0
            self.overtake_start_progress = None
            self.return_start_progress = None
            self.wait_resume_overtake = False
            return OvertakeDecision(self.state, 999.0, False)

        if self.state == self.DRIVE:
            if not confirmed_obstacle:
                return OvertakeDecision(self.state, 0.0, True)

            both_blocked = (
                not status.left_clear
                and not status.right_clear
            )

            if both_blocked:
                self.state = self.WAIT
                self.wait_resume_overtake = False
                return OvertakeDecision(self.state, 999.0, False)

            can_avoid = (
                confirmed_obstacle
                and overtake_allowed
                and status.left_clear
                and confirmed_left_clear
            )

            if can_avoid:
                return self.start_overtake(progress)

            self.state = self.WAIT
            self.wait_resume_overtake = False
            return OvertakeDecision(self.state, 999.0, False)

        if self.state == self.WAIT:
            # If a side obstacle interrupted an active pass, do not interpret
            # loss of the front return as successful completion. Resume the
            # same pass only when its lane is clear; path progress remains
            # anchored at the original start.
            if self.wait_resume_overtake:
                if status.left_clear and overtake_allowed:
                    self.state = self.OVERTAKE
                    self.wait_resume_overtake = False
                    return OvertakeDecision(
                        self.state, self.overtake_offset, True
                    )
                return OvertakeDecision(self.state, 999.0, False)

            if not confirmed_obstacle:
                self.state = self.DRIVE
                self.reset_counters()
                return OvertakeDecision(self.state, 0.0, True)

            both_blocked = (
                not status.left_clear
                and not status.right_clear
            )

            if both_blocked:
                return OvertakeDecision(self.state, 999.0, False)

            can_avoid = (
                confirmed_obstacle
                and overtake_allowed
                and status.left_clear
                and confirmed_left_clear
            )

            if can_avoid:
                return self.start_overtake(progress)

            return OvertakeDecision(self.state, 999.0, False)

        if self.state == self.OVERTAKE:
            # Kept only for diagnostics/backwards compatibility; this count
            # no longer participates in any transition.
            self.overtake_counter += 1

            # Stop only if the left/overtake lane is blocked.
            # Right-side obstacle should only delay return, not stop.
            if not status.left_clear and status.left_count > 0:
                self.state = self.WAIT
                self.wait_resume_overtake = True
                return OvertakeDecision(self.state, 999.0, False)

            right_side_confirmed_empty = (
                confirmed_right_clear
                and status.right_clear
            )

            if (
                confirmed_no_obstacle
                and right_side_confirmed_empty
                # A return is a lane change too.  Starting it on a curve can
                # drive the reference across the inside road edge even when
                # the original lane is clear at the current scan.  The MPC
                # publishes this permission only for a sufficiently long,
                # low-curvature preview of the recorded route.
                and overtake_allowed
                and yaw_stable
                and pass_confirmed
                and self._progress_since(
                    self.overtake_start_progress, progress
                ) >= self.min_overtake_progress
            ):
                self.state = self.RETURN
                self.return_counter = 0
                self.return_start_progress = progress
                return OvertakeDecision(self.state, 0.0, True)

            return OvertakeDecision(self.state, self.overtake_offset, True)

        if self.state == self.RETURN:
            # If the original lane becomes blocked during a committed return,
            # stop at the current lateral position.  Commanding the full
            # passing-lane offset here repeatedly reversed and restarted the
            # MPC's return S-curve, producing an unsafe left/right oscillation.
            # With motion disabled, measured progress and the swept-corridor
            # estimate both freeze.  A clear scan resumes the same return.
            if not status.right_clear and status.right_count >= 2:
                return OvertakeDecision(
                    self.state,
                    999.0,
                    False,
                )

            self.return_counter += 1

            if (
                yaw_stable
                and self._progress_since(
                    self.return_start_progress, progress
                ) >= self.min_return_progress
            ):
                self.state = self.DRIVE
                self.return_counter = 0
                self.overtake_counter = 0
                self.reset_counters()
                self.overtake_start_progress = None
                self.return_start_progress = None
                self.wait_resume_overtake = False

            return OvertakeDecision(self.state, 0.0, True)

        if self.state == self.ESTOP:
            if status.emergency:
                return OvertakeDecision(self.state, 999.0, False)

            self.state = self.WAIT
            self.reset_counters()
            self.overtake_start_progress = None
            self.return_start_progress = None
            self.wait_resume_overtake = False
            return OvertakeDecision(self.state, 999.0, False)

        self.state = self.DRIVE
        self.reset_counters()
        self.overtake_start_progress = None
        self.return_start_progress = None
        self.wait_resume_overtake = False
        return OvertakeDecision(self.state, 0.0, True)
