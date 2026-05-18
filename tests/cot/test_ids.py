from evalhub.cot import decode, encode
from evalhub.cot.ids import GENERATION_ID_SEP, GenerationId

import pytest


def test_round_trip_simple():
    gid = encode("AIME2025/12", 7)
    assert gid == f"AIME2025/12{GENERATION_ID_SEP}7"
    decoded = decode(gid)
    assert decoded == GenerationId("AIME2025/12", 7)


def test_round_trip_with_separator_inside_task_id():
    """An original task id may itself contain the separator substring; the
    decoder must split on the rightmost occurrence so the round-trip survives."""
    pathological = f"weird{GENERATION_ID_SEP}name/5"
    gid = encode(pathological, 3)
    decoded = decode(gid)
    assert decoded.original_task_id == pathological
    assert decoded.generation_idx == 3


def test_decode_rejects_missing_separator():
    with pytest.raises(ValueError):
        decode("no-separator-here")


def test_decode_rejects_non_integer_suffix():
    with pytest.raises(ValueError):
        decode(f"task{GENERATION_ID_SEP}abc")
