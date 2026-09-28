"""Correctness tests for the four error-injection primitives + corrupt_one.

Includes regression tests for the bugs found in the adversarial review:
 - type3 change_location must point at a REAL replacement, not a pre-existing
   occurrence of the wrong number;
 - type4 must keep the boxed even when the 75% cut lands inside it;
 - type2 must not partially capture a number straddling the 40%/70% boundary.
"""

from __future__ import annotations

import random
import re

from error_study import common
from error_study.corrupt import corrupt_one


def _w(wrong, old, new, cat="pm12"):
    return {"wrong_str": wrong, "old_token": old, "new_token": new, "category": cat, "fallback": False}


# type1: boxed only
def test_type1_changes_only_boxed():
    content = "we get 277 in the middle and finally \\boxed{277}"
    r = common.apply_type1(content, "277", _w("278", "277", "278"))
    out = r["corrupted_solution"]
    assert common.find_last_boxed_span(out)[2] == "278"  # boxed changed
    assert out.count("277") == 1  # body 277 kept, only boxed changed
    assert out.count("278") == 1


def test_type1_boxed_with_spaces_matches_gt():
    content = "x \\boxed{ 277 }"
    r = common.apply_type1(content, "277", _w("300", "277", "300"))
    assert common.find_last_boxed_span(r["corrupted_solution"])[2] == "300"


# type3: all occurrences, shared wrong number
def test_type3_replaces_all_boundary_safe():
    content = "answer 42 ... also 42 here ... \\boxed{42} and 7"
    r = common.apply_type3(content, "42", "int", _w("43", "42", "43"), set())
    out = r["corrupted_solution"]
    assert out.count("43") == 3  # all 3 answer occurrences replaced
    assert r["num_occurrences_replaced"] == 3


def test_type3_skips_when_answer_embedded_in_larger_number():
    # 42 appears inside 425 -> whole solution skipped per spec
    r = common.apply_type3("42 here and 425 there \\boxed{42}", "42", "int", _w("43", "42", "43"), set())
    assert r["skip_reason"] == "answer_inside_larger_number"


def test_type3_change_location_is_real_not_preexisting():
    # REGRESSION: the wrong number 38 pre-exists (40+38) BEFORE the real answer 39.
    content = "step 40 + 38 = 78 ... the answer is 39. \\boxed{39}"
    r = common.apply_type3(content, "39", "int", _w("38", "39", "38"), set())
    loc = r["change_location_char"]
    # recorded location must be a site that originally held the answer "39",
    # not the pre-existing "38".
    assert content[loc : loc + 2] == "39"
    assert loc > content.find("38")


def test_type3_skip_embedded():
    r = common.apply_type3("value 12 and 125 \\boxed{12}", "12", "int", _w("13", "12", "13"), set())
    assert r["skip_reason"] == "answer_inside_larger_number"


def test_type3_skip_answer_equals_question_number():
    r = common.apply_type3("x \\boxed{40}", "40", "int", _w("41", "40", "41"), {40})
    assert r["skip_reason"] == "answer_equals_question_number"


# type2: intermediate slip
def test_type2_picks_region_number_not_answer_or_question():
    content = "A" * 40 + " 88 " + "B" * 40 + " \\boxed{5}"
    r = common.apply_type2(content, "5", "int", set(), "pm12", random.Random(0))
    assert r["old_number"] == "88"
    assert r["new_number"] != "88" and r["new_number"] not in ("0", "1")
    lo = int(0.40 * len(content))
    assert r["change_location_char"] >= lo


def test_type2_boundary_straddle_excluded():
    # REGRESSION: 12345 straddles lo (starts at 44, lo=46); it must NOT be captured
    # as a partial "345". No other number in region -> skip, not a mangled token.
    content = "A" * 44 + "12345" + "B" * 56 + " \\boxed{7}"
    lo = int(0.40 * len(content))
    assert 44 < lo < 49  # 12345 (chars 44-49) straddles lo
    r = common.apply_type2(content, "7", "int", set(), "pm12", random.Random(0))
    assert r["skip_reason"] == "no_intermediate_number_in_region"


def test_type2_skips_ordinal_label_numbers():
    # region's only number is the ordinal "2" in "2nd" -> excluded -> skip (not "7nd")
    content = "A" * 40 + " this is the 2nd case here " + "B" * 40 + " \\boxed{5}"
    lo = int(0.40 * len(content))
    assert content.index("2nd") >= lo  # the ordinal sits inside the region
    r = common.apply_type2(content, "5", "int", set(), "pm12", random.Random(0))
    assert r["skip_reason"] == "no_intermediate_number_in_region"


def test_type2_recorded_old_number_is_whole_token():
    content = "start " + "z " * 60 + "the value is 137 here " + "q " * 60 + "\\boxed{9}"
    r = common.apply_type2(content, "9", "int", set(), "single_digit", random.Random(1))
    if "corrupted_solution" in r:  # applied
        old = r["old_number"]
        assert re.search(r"(?<!\d)" + re.escape(old) + r"(?!\d)", content)


# type4: truncate keep boxed
def test_type4_normal_keeps_boxed_and_shortens():
    content = "reasoning " * 50 + "\\boxed{5}" + " trailing verification " * 20
    r = common.apply_type4(content)
    out = r["corrupted_solution"]
    assert common.find_last_boxed_span(out) is not None and "\\boxed{5}" in out
    assert len(out) < len(content)


def test_type4_boxed_inside_cut_preserved():
    # REGRESSION: the box straddles the 75% cut -> must be preserved, not sliced.
    content = "x" * 100 + "\\boxed{47}" + "." * 30  # L=140, cut=105, box spans 100-110
    r = common.apply_type4(content)
    out = r["corrupted_solution"]
    assert "\\boxed{47}" in out
    assert common.find_last_boxed_span(out)[2] == "47"


# corrupt_one integration, the shared-wrong-number invariant
def _base(sol, gt, bench="aime2026", q="Find N with 12 and 40."):
    return {
        "row_id": "M|B|T/1|g0",
        "model": "M",
        "state": "think",
        "benchmark": bench,
        "group": "M|" + bench,
        "task_id": "T/1",
        "question_text": q,
        "ground_truth": gt,
        "answer_type": common.classify_answer_type(gt),
        "source_variant": 65536,
        "source_gen_ordinal": 0,
        "solution_text": sol,
        "solution_len_tokens": 700,
        "solution_len_chars": len(sol),
    }


def test_corrupt_one_five_rows_and_shared_number():
    sol = "we compute 90 then 123 and the answer is 277. \\boxed{277}"
    rows = corrupt_one(_base(sol, "277"), "pm12", "pm12")
    by = {r["error_type"]: r for r in rows}
    assert set(by) == {"clean", "boxed_only", "intermediate_error", "consistent_error", "truncated"}
    assert by["clean"]["corrupted_solution"] == sol
    if by["boxed_only"]["corruption_applied"] and by["consistent_error"]["corruption_applied"]:
        assert by["boxed_only"]["new_number"] == by["consistent_error"]["new_number"]


def test_corrupt_one_negative_answer_not_forcemskipped():
    # REGRESSION: negative integer answer must be corrupt-able (non-AIME).
    sol = "after algebra the result is -4 so \\boxed{-4}"
    rows = corrupt_one(_base(sol, "-4", bench="pt_exams_math", q="no numbers"), "pm12", "pm12")
    by = {r["error_type"]: r for r in rows}
    assert by["boxed_only"]["corruption_applied"] is True
    assert by["boxed_only"]["new_number"] != "-4"
