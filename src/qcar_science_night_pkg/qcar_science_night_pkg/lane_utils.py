"""Pure helpers shared by the camera lane detector and its tests."""


def lane_reference_offset(normalized_image_error, gain):
    """Convert image error to the MPC's left-positive path-offset frame.

    An image feature to the right means the vehicle is left of the desired
    lane center, so the path correction must be to vehicle-right (negative).
    """
    return -float(gain) * float(normalized_image_error)
