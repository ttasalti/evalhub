import pytest

from evalhub.benchmarks.math.tubitak_math2026 import (
    TUBITAK_MATH2026,
    TubitakMath2026Dataset,
)
from evalhub.benchmarks.math.verifier import grade_answer
from evalhub.benchmarks.registry import DATASET_MAP


def test_registry_entry():
    assert TUBITAK_MATH2026 in DATASET_MAP
    assert DATASET_MAP[TUBITAK_MATH2026] is TubitakMath2026Dataset


def test_load_tasks_count():
    ds = TubitakMath2026Dataset()
    ds.load_tasks()
    assert len(ds.tasks) == 32
    assert len(ds.groundtruth) == 32


def test_task_id_format_matches_cot_judge_derivation():
    # cot/judge.py:97 splits on "/", lowercases, replaces "-" with "_". The
    # benchmark_name it derives must equal the DATASET_MAP key.
    ds = TubitakMath2026Dataset()
    ds.load_tasks()
    sample_id = next(iter(ds.tasks))
    derived = sample_id.split("/", 1)[0].lower().replace("-", "_")
    assert derived == TUBITAK_MATH2026


def test_strip_outer_dollars_idempotent():
    strip = TubitakMath2026Dataset._strip_outer_dollars
    assert strip("$2$") == "2"
    assert strip("2") == "2"
    assert strip("$\\frac{1}{2}$") == "\\frac{1}{2}"
    assert strip("\\frac{1}{2}") == "\\frac{1}{2}"
    assert strip("  $105^\\circ$  ") == "105^\\circ"


@pytest.mark.parametrize(
    "ground_truth,given_answer",
    [
        ("\\frac{52}{5}", "\\frac{52}{5}"),
        ("\\frac{52}{5}", "52/5"),
        ("105^\\circ", "105"),
        ("3\\sqrt{10}", "3\\sqrt{10}"),
        ("\\sqrt{7}+1", "\\sqrt{7}+1"),
        ("-\\frac{49}{8}", "-49/8"),
        ("8^5", "32768"),
        ("1600", "1600"),
    ],
)
def test_grading_handles_olympiad_answer_forms(ground_truth, given_answer):
    assert grade_answer(given_answer, ground_truth)
