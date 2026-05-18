"""Encoding and decoding of CoT generation identifiers.

A generation identifier deterministically pairs an original ``task_id`` with the
ordinal index of a single sample drawn for that task. It is the only join key
used between the base evaluation, the judge generation, and the final CoT
metric computation, so all three stages must agree on its exact form.
"""

from __future__ import annotations

from dataclasses import dataclass

GENERATION_ID_SEP = "_gen_"


@dataclass(frozen=True)
class GenerationId:
    original_task_id: str
    generation_idx: int

    def encode(self) -> str:
        return f"{self.original_task_id}{GENERATION_ID_SEP}{self.generation_idx}"

    @classmethod
    def decode(cls, generation_id: str) -> "GenerationId":
        marker_pos = generation_id.rfind(GENERATION_ID_SEP)
        if marker_pos == -1:
            raise ValueError(f"Not a valid generation_id: {generation_id!r}")
        suffix = generation_id[marker_pos + len(GENERATION_ID_SEP):]
        try:
            idx = int(suffix)
        except ValueError as exc:
            raise ValueError(f"Generation index is not an integer in {generation_id!r}") from exc
        return cls(original_task_id=generation_id[:marker_pos], generation_idx=idx)


def encode(original_task_id: str, generation_idx: int) -> str:
    return GenerationId(original_task_id, generation_idx).encode()


def decode(generation_id: str) -> GenerationId:
    return GenerationId.decode(generation_id)
