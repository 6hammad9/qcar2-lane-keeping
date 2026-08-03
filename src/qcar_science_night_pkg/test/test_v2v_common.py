import pytest

from qcar_science_night_pkg import v2v_common
from qcar_science_night_pkg.v2v_common import (
    PacketError,
    pack_state,
    parse_packet,
)


def test_predicted_yaw_uses_angle_bound_not_position_bound():
    packet = pack_state(
        vehicle_id="rosbot3",
        seq=1,
        stamp=1.0,
        localized=True,
        x=0.0,
        y=0.0,
        yaw=0.0,
        v=0.1,
        moving=True,
        horizon_dt=0.08,
        predicted=[(0.0, 0.0, 100.0)],
    )

    with pytest.raises(PacketError, match="predicted yaw"):
        parse_packet(packet, expected_id="rosbot3")


# ---- Schema 2: obstacle report and the command channel ----

def test_obstacle_report_survives_a_roundtrip():
    packet = v2v_common.pack_state(
        "rosbot3", 1, 1000.0, True, 1.0, 2.0, 0.5, 0.1, True, 0.08,
        [(1.0, 2.0, 0.5)],
        blocked=True, blocked_distance=0.62, detour_intent=True,
    )
    parsed = v2v_common.parse_packet(packet, expected_id="rosbot3")
    assert parsed["blocked"] is True
    assert parsed["blocked_distance"] == 0.62
    assert parsed["detour_intent"] is True


def test_absent_obstacle_report_decodes_as_unknown_not_clear():
    # An un-upgraded broadcaster cannot tell us about its lane. Decoding that
    # silence as "clear" would let QCar act on a fact it was never given.
    packet = v2v_common.pack_state(
        "rosbot3", 1, 1000.0, True, 1.0, 2.0, 0.5, 0.1, True, 0.08,
        [(1.0, 2.0, 0.5)],
    )
    parsed = v2v_common.parse_packet(packet, expected_id="rosbot3")
    assert parsed["blocked"] is None
    assert parsed["detour_intent"] is False


def test_schema_1_sender_is_still_accepted():
    import json
    packet = v2v_common.pack_state(
        "rosbot3", 1, 1000.0, True, 1.0, 2.0, 0.5, 0.1, True, 0.08,
        [(1.0, 2.0, 0.5)],
    )
    legacy = json.loads(packet.decode())
    legacy["s"] = 1
    parsed = v2v_common.parse_packet(
        json.dumps(legacy).encode(), expected_id="rosbot3"
    )
    assert parsed["blocked"] is None
    assert parsed["x"] == 1.0


def test_command_roundtrip_and_addressing():
    packet = v2v_common.pack_command("qcar2", "rosbot3", 3, 1000.0, True, 1.0)
    parsed = v2v_common.parse_command(
        packet, expected_target="rosbot3", expected_issuer="qcar2"
    )
    assert parsed["hold"] is True
    assert parsed["ttl_sec"] == 1.0
    assert parsed["seq"] == 3

    # Addressed elsewhere: not ours to obey.
    for kwargs in (
        {"expected_target": "rosbot9"},
        {"expected_issuer": "someone_else"},
    ):
        try:
            v2v_common.parse_command(packet, **kwargs)
        except v2v_common.PacketError:
            pass
        else:
            raise AssertionError(f"should have rejected {kwargs}")


def test_corrupt_command_never_decodes_as_proceed():
    # "Proceed" releases a robot into a lane QCar may still occupy, so an
    # unrecognised word has to be an error, never a default.
    import json
    packet = v2v_common.pack_command("qcar2", "rosbot3", 1, 1000.0, True, 1.0)
    for mutation in ({"c": "X"}, {"c": None}, {"c": ""}, {"ttl": 0.0},
                     {"ttl": 99.0}, {"ttl": "1.0"}, {"q": "abc"}):
        broken = json.loads(packet.decode())
        broken.update(mutation)
        try:
            v2v_common.parse_command(broken and json.dumps(broken).encode())
        except v2v_common.PacketError:
            continue
        raise AssertionError(f"should have rejected {mutation}")


def test_command_ttl_is_bounded_at_pack_time():
    for bad_ttl in (0.0, -1.0, v2v_common.MAX_COMMAND_TTL_SEC + 0.1):
        try:
            v2v_common.pack_command("qcar2", "rosbot3", 1, 1000.0, True, bad_ttl)
        except ValueError:
            continue
        raise AssertionError(f"should have rejected ttl {bad_ttl}")
