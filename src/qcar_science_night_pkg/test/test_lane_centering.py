from qcar_science_night_pkg.lane_utils import lane_reference_offset


def test_image_right_requires_vehicle_right_correction():
    # MPC reference offsets are left-positive. If the lane center appears to
    # the image right, the vehicle is left of center and must move right.
    assert lane_reference_offset(0.4, 0.25) == -0.1


def test_image_left_requires_vehicle_left_correction():
    assert lane_reference_offset(-0.4, 0.25) == 0.1


def test_centered_image_has_no_correction():
    assert lane_reference_offset(0.0, 0.25) == 0.0
