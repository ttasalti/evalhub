import orjson

from evalhub.cot import aggregate_judge_votes
from evalhub.cot.ids import encode


def _write_jsonl(path, records):
    with open(path, "wb") as f:
        for r in records:
            f.write(orjson.dumps(r) + b"\n")


def _read_jsonl(path):
    return [orjson.loads(line) for line in open(path, "rb") if line.strip()]


def test_majority_strict_greater_than(tmp_path):
    """A 2-1 yes majority approves; a 1-1 tie does NOT approve."""
    gid_yes = encode("T", 0)
    gid_tie = encode("T", 1)
    gid_invalid_majority = encode("T", 2)

    judge_solutions = tmp_path / "judge.jsonl"
    out = tmp_path / "majority.jsonl"
    _write_jsonl(
        judge_solutions,
        [
            {"task_id": gid_yes, "solution": "yes"},
            {"task_id": gid_yes, "solution": "yes"},
            {"task_id": gid_yes, "solution": "no"},
            {"task_id": gid_tie, "solution": "yes"},
            {"task_id": gid_tie, "solution": "no"},
            {"task_id": gid_invalid_majority, "solution": "invalid_format"},
            {"task_id": gid_invalid_majority, "solution": "invalid_format"},
            {"task_id": gid_invalid_majority, "solution": "yes"},
        ],
    )

    aggregate_judge_votes(judge_solutions, out)
    rows = {r["task_id"]: r for r in _read_jsonl(out)}

    assert rows[gid_yes]["majority_correct"] is True
    assert rows[gid_tie]["majority_correct"] is False
    assert rows[gid_invalid_majority]["majority_correct"] is True  # 1 yes vs 0 no
    assert rows[gid_invalid_majority]["invalid_count"] == 2


def test_handles_list_solution_field(tmp_path):
    """Some upstream code paths emit a list under 'solution'; flatten it."""
    gid = encode("T", 0)
    judge_solutions = tmp_path / "judge.jsonl"
    out = tmp_path / "out.jsonl"
    _write_jsonl(
        judge_solutions,
        [{"task_id": gid, "solution": ["yes", "yes", "no"]}],
    )
    aggregate_judge_votes(judge_solutions, out)
    rows = _read_jsonl(out)
    assert len(rows) == 1
    assert rows[0]["yes_count"] == 2 and rows[0]["no_count"] == 1
    assert rows[0]["majority_correct"] is True


def test_normalises_case_and_whitespace(tmp_path):
    gid = encode("T", 0)
    js = tmp_path / "judge.jsonl"
    out = tmp_path / "out.jsonl"
    _write_jsonl(
        js,
        [
            {"task_id": gid, "solution": " YES "},
            {"task_id": gid, "solution": "No"},
            {"task_id": gid, "solution": "yes"},
        ],
    )
    aggregate_judge_votes(js, out)
    rows = _read_jsonl(out)
    assert rows[0]["yes_count"] == 2 and rows[0]["no_count"] == 1
    assert rows[0]["majority_correct"] is True
