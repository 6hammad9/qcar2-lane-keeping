from qcar_science_night_pkg.overtake_types import OvertakeDecision


class OvertakeStateMachine:
    DRIVE = "DRIVE"
    WAIT = "WAIT_FOR_CLEAR"
    PROBE = "LANE_PROBE"
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
        flank_clear_confirm_required=4,
        return_confirm_required=30,
        min_overtake_steps=30,
        min_overtake_progress=0.80,
        min_return_progress=0.35,
        probe_enabled=True,
        probe_min_gap_m=0.50,
        flank_override_progress=1.20,
    ):
        self.state = self.DRIVE
        self.overtake_offset = float(overtake_offset)

        self.obstacle_confirm_required = obstacle_confirm_required
        self.left_clear_confirm_required = left_clear_confirm_required
        self.no_obstacle_confirm_required = no_obstacle_confirm_required
        self.right_clear_confirm_required = right_clear_confirm_required
        self.flank_clear_confirm_required = flank_clear_confirm_required
        self.return_confirm_required = return_confirm_required
        # ``min_overtake_steps`` is retained as an API compatibility shim for
        # older launch files.  Scan callbacks are not vehicle motion and must
        # never decide that a pass has completed, so completion is now based
        # exclusively on measured path progress.
        self.min_overtake_steps = min_overtake_steps
        self.min_overtake_progress = float(min_overtake_progress)
        self.min_return_progress = float(min_return_progress)

        # Lane probing, after IDEAM's LP state (Shu, Zhou & Zhang, T-ITS
        # 2025, Sec. V-A2). Where a lane change is forbidden -- a curve, or
        # both lanes blocked -- the car previously stopped dead and waited.
        # It should instead keep closing on the obstacle under a gap bound,
        # so it is already in position when a pass becomes legal. On this
        # route the curvature gate is false over most of the lap, so
        # "stop dead" meant stranding the car with a clear lane beside it.
        self.probe_enabled = bool(probe_enabled)
        self.probe_min_gap_m = float(probe_min_gap_m)

        # Distance past a completed pass after which the flank corridor stops
        # being allowed to veto the merge.  The corridor cannot tell a lead
        # from scenery, and this course has walls inside the band it samples,
        # so on the vehicle it latched at flank_count=55 and never released.
        # A lead that was ahead at the commit point cannot still be alongside
        # after the car has driven min_overtake_progress plus this much
        # further, so beyond that the corridor is reporting on the world, not
        # on traffic, and its veto is unsound.  The front box, the emergency
        # corridor and the MPC barrier all remain in force.
        self.flank_override_progress = float(flank_override_progress)

        self.obstacle_counter = 0
        self.left_clear_counter = 0
        self.right_clear_counter = 0
        self.flank_clear_counter = 0
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
        self.flank_clear_counter = 0
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

        if status.flank_clear:
            self.flank_clear_counter += 1
        else:
            self.flank_clear_counter = 0

    def _flank_veto_expired(self, progress):
        """True once the flank corridor can no longer be seeing the lead.

        Distance is measured from the commit point, so this is the length of
        the whole encounter plus a margin.  Past it the only things the
        corridor can be reporting are static, and a static return beside the
        car is a wall, not a reason to stay in the passing lane forever.
        """
        return self._progress_since(
            self.overtake_start_progress, progress
        ) >= self.min_overtake_progress + self.flank_override_progress

    def start_overtake(self, progress=None):
        self.state = self.OVERTAKE
        self.overtake_counter = 0
        self.return_counter = 0
        self.overtake_start_progress = progress
        self.return_start_progress = None
        self.wait_resume_overtake = False
        return OvertakeDecision(self.state, self.overtake_offset, True)

    def probe_gap_allows(self, status):
        """True while there is still room to creep forward.

        The gap comes from the curvature-following corridor when it has one,
        not from the wide front box. In a curve the wide box is looking at
        the outside road edge, so its range is not the distance to anything
        the car will actually reach; the corridor is the volume it will
        sweep. Falling back to front_min keeps a status without corridor
        data behaving sensibly.

        front_narrow_min of -1.0 is the analyzer's "nothing detected", NOT a
        zero-metre gap. Treating the sentinel as a distance would end the
        probe every time the corridor happened to be empty, which is exactly
        when it is safe to continue.
        """
        gap = None
        if getattr(status, "front_narrow_count", 0) > 0:
            narrow = float(getattr(status, "front_narrow_min", -1.0))
            if narrow > 0.0:
                gap = narrow
        if gap is None:
            front = float(status.front_min)
            gap = front if front > 0.0 else None

        if gap is None:
            return True
        return gap > self.probe_min_gap_m

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
        confirmed_flank_clear = (
            self.flank_clear_counter >= self.flank_clear_confirm_required
        )

        if status.emergency:
            self.state = self.ESTOP
            self.return_counter = 0
            self.overtake_counter = 0
            self.overtake_start_progress = None
            self.return_start_progress = None
            self.wait_resume_overtake = False
            return OvertakeDecision(self.state, 999.0, False)

        if self.state in (self.DRIVE, self.PROBE):
            if not confirmed_obstacle:
                self.state = self.DRIVE
                return OvertakeDecision(self.state, 0.0, True)

            can_avoid = (
                confirmed_obstacle
                and overtake_allowed
                and status.left_clear
                and confirmed_left_clear
            )

            if can_avoid:
                return self.start_overtake(progress)

            # A pass is not available: curve, blocked passing lane, or both.
            # Creep forward instead of stopping, while a gap remains.
            if self.probe_enabled and self.probe_gap_allows(status):
                self.state = self.PROBE
                self.wait_resume_overtake = False
                return OvertakeDecision(self.state, 0.0, True)

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
                # The lane being vacated must be empty BESIDE and BEHIND the
                # car, not merely ahead of it.  Every forward box reports
                # clear the instant the lead draws level with the bumper --
                # which is precisely when merging back is a collision.  This
                # replaces a gate on overtake_allowed, which asked the wrong
                # question: that is the MPC's curvature permission for
                # STARTING a pass, false over most of this route, so it both
                # stranded the car in the passing lane for whole curves and
                # said nothing at all about whether the lead had been cleared.
                and (
                    (confirmed_flank_clear and status.flank_clear)
                    or self._flank_veto_expired(progress)
                )
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
            #
            # The flank is checked here for the same reason it gates entry:
            # a lead that is still level with the car's tail is invisible to
            # right_clear.  It stops mattering by itself as the offset
            # collapses, because the corridor is then inside the car's own
            # body and the analyzer stops reporting on it.
            flank_blocks_merge = (
                not status.flank_clear
                and not self._flank_veto_expired(progress)
            )
            if (
                not status.right_clear and status.right_count >= 2
            ) or flank_blocks_merge:
                # Hold the passing lane and KEEP DRIVING rather than braking.
                # Whatever blocks a merge is alongside or behind the car, and
                # no amount of braking clears it -- only forward travel does.
                # Stopping here therefore cannot resolve into anything: it was
                # measured stuck at flank_count=55 with motion disabled and no
                # exit, because the flank corridor had settled on the static
                # right-hand wall, which never moves. It sat in the passing lane
                # indefinitely and never returned to DRIVE.
                #
                # Commanding the full offset rather than 999.0 also preserves
                # what the previous stop-in-place was protecting: the lateral
                # command stays where the car already is, so the MPC's return
                # S-curve is never reversed mid-flight, which is the left/right
                # oscillation that motivated stopping in the first place.
                # Re-anchoring progress keeps min_return_progress measured from
                # the point the merge actually begins.
                self.return_start_progress = progress
                return OvertakeDecision(
                    self.state,
                    self.overtake_offset,
                    True,
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
