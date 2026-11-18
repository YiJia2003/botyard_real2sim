"""Golden wire examples for the offline H7 A0 payload codec."""

import math
import struct
import subprocess
import sys

import pytest

from botyard_real2sim.protocol import DofCommand, decode_a0_payload, encode_a0_payload

pytestmark = pytest.mark.unit


def test_golden_payload_is_mask_ordered_signed_little_endian():
    commands = {
        "MFJ3": DofCommand(math.radians(-12.34), math.radians(-50.0), -2.5),
        "THJ1": DofCommand(math.radians(45.0), math.radians(120.0), 20.0),
    }
    golden = bytes.fromhex("2100009411b004d0072efb0cfe06ff")
    assert encode_a0_payload(commands) == golden
    decoded = decode_a0_payload(golden)
    assert list(decoded) == ["THJ1", "MFJ3"]
    for name, cmd in commands.items():
        assert decoded[name].angle_rad == pytest.approx(cmd.angle_rad)
        assert decoded[name].speed_rad_s == pytest.approx(cmd.speed_rad_s)
        assert decoded[name].tension_command_n == cmd.tension_command_n
    assert encode_a0_payload(decoded) == golden


def test_sparse_mask_does_not_autopair_and_empty_mask_is_valid():
    payload = encode_a0_payload({"LFJ3": DofCommand(0.0, 0.0, 0.0)})
    assert payload == bytes.fromhex("000200000000000000")
    assert list(decode_a0_payload(payload)) == ["LFJ3"]
    assert encode_a0_payload({}) == b"\0\0\0"
    assert decode_a0_payload(b"\0\0\0") == {}


def test_int16_endpoints_and_360_degree_overflow():
    payload = b"\x01\0\0" + struct.pack("<hhh", -32768, 32767, -32768)
    assert encode_a0_payload(decode_a0_payload(payload)) == payload
    for cmd in (
        DofCommand(math.radians(360), 0, 0),
        DofCommand(0, math.radians(4000), 0),
        DofCommand(0, 0, 400),
    ):
        with pytest.raises(ValueError, match="int16"):
            encode_a0_payload({"THJ1": cmd})


@pytest.mark.parametrize("payload", [b"", b"\0\0", b"\0\0\0\0", b"\x01\0\0", b"\0\x04\0", b"\0\0\x80"])
def test_rejects_bad_lengths_and_unsupported_bits(payload):
    with pytest.raises(ValueError):
        decode_a0_payload(payload)


def test_nonfinite_unknown_names_and_quantization():
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            DofCommand(bad, 0, 0)
    with pytest.raises(ValueError, match="unsupported"):
        encode_a0_payload({"FFJ1": DofCommand(0, 0, 0)})
    result = decode_a0_payload(
        encode_a0_payload({"THJ2": DofCommand(math.radians(12.344), math.radians(20.04), 1.234)})
    )
    assert result["THJ2"].angle_rad == pytest.approx(math.radians(12.34))
    assert result["THJ2"].speed_rad_s == pytest.approx(math.radians(20.0))
    assert result["THJ2"].tension_command_n == pytest.approx(1.23)


def test_codec_import_does_not_load_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from botyard_real2sim import protocol; assert 'torch' not in sys.modules",
        ],
        check=True,
    )
