"""Correctness tests for error_study.common, boxed/number helpers, the
wrong-number menu, and the type-aware answer corruption.

Run from the repository root:  PYTHONPATH=. python -m pytest error_study/tests -q
"""

from __future__ import annotations

import random
import re

import pytest

from error_study import common


# boxed helpers
def test_find_last_boxed_basic():
    s = "answer is \\boxed{277} done"
    start, end, inner = common.find_last_boxed_span(s)
    assert inner == "277"
    assert s[start:end] == "\\boxed{277}"


def test_find_last_boxed_takes_last_and_nested():
    s = "\\boxed{1} then \\boxed{\\frac{52}{5}}"
    start, end, inner = common.find_last_boxed_span(s)
    assert inner == "\\frac{52}{5}"
    assert s[start:end] == "\\boxed{\\frac{52}{5}}"


def test_find_last_boxed_none():
    assert common.find_last_boxed_span("no box here") is None


def test_set_last_boxed_inner():
    s = "x \\boxed{5} y \\boxed{9}"
    out = common.set_last_boxed_inner(s, "42")
    assert out == "x \\boxed{5} y \\boxed{42}"


# number extraction / boundary safety
def test_question_numbers():
    assert common.question_numbers("Find N with 12 boxes and 40 items.") == {12, 40}


def test_has_embedded_occurrence():
    assert common.has_embedded_occurrence("the number 125 here", "12") is True  # 12 inside 125
    assert common.has_embedded_occurrence("x312y", "12") is True  # 312
    assert common.has_embedded_occurrence("a 12 b", "12") is False  # standalone


def _first_int_span(t):
    m = next(re.finditer(r"\d+", t))
    return m.start(), m.end()


@pytest.mark.parametrize(
    "text,is_label",
    [
        ("the 2nd quadrant", True),  # ordinal
        ("21st term", True),
        ("3rd case", True),
        ("s_1 + s_2", True),  # subscript
        ("z_{2} value", True),  # latex subscript
        ("= 4 + 2i", False),  # real computational number
        ("1 - 0.7 = 0.3", False),  # decimal digit is computational
    ],
)
def test_is_label_number(text, is_label):
    st, en = _first_int_span(text)
    assert common.is_label_number(text, st, en) is is_label


def test_boundary_safe_replace_int():
    out, n, first = common.boundary_safe_replace_int("12 and 125 and x12", "12", "99")
    assert out == "99 and 125 and x99"  # 125's 12 untouched, standalone ones replaced
    assert n == 2
    assert first == 0  # offset of the first real replacement


# answer-type classification
@pytest.mark.parametrize(
    "gt,expected",
    [
        ("277", "int"),
        ("-4", "int"),
        ("0", "int"),
        ("1,1", "decimal"),
        ("1.5", "decimal"),
        ("\\frac{52}{5}", "frac"),
        ("-\\frac{49}{8}", "frac"),
        ("3\\sqrt{10}", "sqrt"),
        ("40\\%", "percent"),
        ("9a-4", "expr"),
        ("A", "mc"),
    ],
)
def test_classify_answer_type(gt, expected):
    assert common.classify_answer_type(gt) == expected


# menu generators (properties)
def test_pm12_candidates():
    got = set(common._pm12_candidates(50, random.Random(0)))
    assert got == {48, 49, 51, 52}


def test_digit_swap_two_digit_is_reverse():
    assert 74 in common._digit_swap_candidates(47, random.Random(0))


def test_digit_swap_single_digit_infeasible():
    assert common._digit_swap_candidates(5, random.Random(0)) == []


def test_digit_swap_no_leading_zero():
    # 100 -> any swap creates a leading zero -> no candidate
    assert common._digit_swap_candidates(100, random.Random(0)) == []


def test_digit_swap_three_digit():
    got = set(common._digit_swap_candidates(277, random.Random(0)))
    assert got == {727, 772}


def test_single_digit_changes_one_digit():
    got = common._single_digit_candidates(392, random.Random(0))
    assert 382 in got  # middle 9 -> 8
    assert all(len(str(abs(v))) == 3 for v in got)  # length preserved
    assert 392 not in got


# corrupt_integer + fallback
def test_corrupt_integer_primary_used():
    v, cat, fb = common.corrupt_integer(50, "pm12", lambda c: True, random.Random(1))
    assert cat == "pm12" and fb is False and v in {48, 49, 51, 52}


def test_corrupt_integer_fallback_when_primary_infeasible():
    # digit_swap on a 1-digit number is infeasible -> must fall back + flag it
    r = common.corrupt_integer(5, "digit_swap", lambda c: True, random.Random(1))
    assert r is not None
    v, cat, fb = r
    assert cat != "digit_swap" and fb is True


def test_corrupt_integer_none_when_all_forbidden():
    # forbid everything -> None
    assert common.corrupt_integer(50, "pm12", lambda c: False, random.Random(1)) is None


# corrupt_answer_value: the shared wrong number (type1 & type3)
def test_corrupt_answer_int_aime_in_range():
    w = common.corrupt_answer_value("277", "int", True, set(), "pm12", random.Random(2))
    assert w is not None
    assert w["wrong_str"] != "277"
    assert 0 <= int(w["wrong_str"]) <= 999


def test_corrupt_answer_respects_question_numbers():
    # if pm12 would land on a question number it must be avoided
    for seed in range(20):
        w = common.corrupt_answer_value("50", "int", True, {48, 49, 51, 52}, "pm12", random.Random(seed))
        assert w is not None
        assert int(w["wrong_str"]) not in {48, 49, 50, 51, 52}


def test_corrupt_answer_never_equals_real_answer():
    for seed in range(50):
        w = common.corrupt_answer_value("100", "int", True, set(), "single_digit", random.Random(seed))
        assert w is None or w["wrong_str"] != "100"


def test_corrupt_answer_negative_integer_is_corrupted_not_skipped():
    # REGRESSION: negative integer answers (-1,-2,-4 exist in the data) must get a
    # valid wrong number, not be force-skipped. A negative answer's wrong value may
    # be negative (fits the type) and must differ from the answer.
    for gt in ("-4", "-2", "-1"):
        got = [
            common.corrupt_answer_value(gt, "int", False, set(), cat, random.Random(s))
            for cat in common.config.MENU_ORDER
            for s in range(5)
        ]
        assert any(w is not None for w in got), f"{gt} was never corrupted (force-skipped)"
        for w in got:
            if w is not None:
                assert w["wrong_str"] != gt


def test_corrupt_answer_fraction_keeps_shape():
    w = common.corrupt_answer_value("\\frac{52}{5}", "frac", False, set(), "single_digit", random.Random(3))
    assert w is not None
    assert w["wrong_str"].startswith("\\frac{") and w["wrong_str"] != "\\frac{52}{5}"


def test_corrupt_answer_decimal_comma():
    w = common.corrupt_answer_value("1,1", "decimal", False, set(), "pm12", random.Random(4))
    assert w is not None and "," in w["wrong_str"] and w["wrong_str"] != "1,1"


def test_corrupt_answer_percent():
    w = common.corrupt_answer_value("40\\%", "percent", False, set(), "pm12", random.Random(5))
    assert w is not None and w["wrong_str"].endswith("\\%") and w["wrong_str"] != "40\\%"


def test_corrupt_answer_pure_symbolic_skipped():
    # no numeric token -> None (caller skips types 1 & 3)
    assert common.corrupt_answer_value("A", "mc", False, set(), "pm12", random.Random(6)) is None
    assert common.corrupt_answer_value("x+y", "expr", False, set(), "pm12", random.Random(6)) is None


def test_corrupt_answer_deterministic():
    a = common.corrupt_answer_value("277", "int", True, set(), "pm12", random.Random(42))
    b = common.corrupt_answer_value("277", "int", True, set(), "pm12", random.Random(42))
    assert a == b
