import orjson

from evalhub.cot import extract_correct_generations
from evalhub.cot.ids import encode


def _write_jsonl(path, records):
    with open(path, "wb") as f:
        for r in records:
            f.write(orjson.dumps(r) + b"\n")


def _read_jsonl(path):
    out = []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(orjson.loads(line))
    return out


def test_extract_keeps_only_correct_generations_in_order(tmp_path):
    base_results = tmp_path / "bench_results.jsonl"
    base_raw = tmp_path / "bench_raw.jsonl"
    output = tmp_path / "judge_input.jsonl"

    _write_jsonl(
        base_results,
        [
            {
                "task_id": "T1",
                "ground_truth": "42",
                "correct": [True, False, True],
                "solutions": ["42", "wrong", "42"],
            },
            {
                "task_id": "T2",
                "ground_truth": "7",
                "correct": [False, False],
                "solutions": ["nope", "nope"],
            },
        ],
    )
    _write_jsonl(
        base_raw,
        [
            {"task_id": "T1", "response": {"choices": [{"message": {"content": "first"}}]}},
            {"task_id": "T1", "response": {"choices": [{"message": {"content": "second"}}]}},
            {"task_id": "T1", "response": {"choices": [{"message": {"content": "third"}}]}},
            {"task_id": "T2", "response": {"choices": [{"message": {"content": "x"}}]}},
            {"task_id": "T2", "response": {"choices": [{"message": {"content": "y"}}]}},
        ],
    )

    written = extract_correct_generations(base_results, base_raw, output)
    assert written == 2

    records = _read_jsonl(output)
    assert [r["task_id"] for r in records] == [encode("T1", 0), encode("T1", 2)]
    assert records[0]["generation_idx"] == 0 and records[1]["generation_idx"] == 2
    assert records[0]["original_task_id"] == "T1"
    assert records[0]["generated_answer"] == "42"
    assert records[1]["raw_response"]["choices"][0]["message"]["content"] == "third"


def test_extract_tolerates_interleaved_task_ids(tmp_path):
    base_results = tmp_path / "bench_results.jsonl"
    base_raw = tmp_path / "bench_raw.jsonl"
    output = tmp_path / "judge_input.jsonl"

    _write_jsonl(
        base_results,
        [
            {"task_id": "A", "ground_truth": "1", "correct": [True, False], "solutions": ["1", "x"]},
            {"task_id": "B", "ground_truth": "2", "correct": [False, True], "solutions": ["x", "2"]},
        ],
    )
    # Raw is *not* per-task contiguous: A0, B0, A1, B1.
    _write_jsonl(
        base_raw,
        [
            {"task_id": "A", "response": {"choices": [{"message": {"content": "a-0"}}]}},
            {"task_id": "B", "response": {"choices": [{"message": {"content": "b-0"}}]}},
            {"task_id": "A", "response": {"choices": [{"message": {"content": "a-1"}}]}},
            {"task_id": "B", "response": {"choices": [{"message": {"content": "b-1"}}]}},
        ],
    )

    extract_correct_generations(base_results, base_raw, output)
    records = _read_jsonl(output)
    ids = sorted(r["task_id"] for r in records)
    assert ids == sorted([encode("A", 0), encode("B", 1)])


def test_extract_returns_zero_when_no_correct(tmp_path):
    base_results = tmp_path / "r.jsonl"
    base_raw = tmp_path / "raw.jsonl"
    output = tmp_path / "out.jsonl"

    _write_jsonl(
        base_results,
        [{"task_id": "T", "ground_truth": "x", "correct": [False, False], "solutions": ["a", "b"]}],
    )
    _write_jsonl(
        base_raw,
        [
            {"task_id": "T", "response": {}},
            {"task_id": "T", "response": {}},
        ],
    )
    assert extract_correct_generations(base_results, base_raw, output) == 0
    assert output.read_text() == ""
