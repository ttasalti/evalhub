"""Parameterised LLM-as-a-Judge dataset for CoT-Pass@K.

Reads a JSONL of correct base generations (produced by ``evalhub cot extract``),
resolves each generation back to its original question by looking up the source
benchmark in :data:`DATASET_MAP`, and emits one judge task per generation that
asks the judge model to return ``\\boxed{yes}`` or ``\\boxed{no}``.

A new language variant is registered by adding an entry to
:data:`COT_JUDGE_VARIANTS`; no new subclasses are required.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

import orjson

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.math.verifier.rllm import extract_boxed_answer
from evalhub.benchmarks.registry import DATASET_MAP, register_dataset
from evalhub.benchmarks.cot.prompts import get_judge_prompt
from evalhub.utils.logger import logger

JUDGE_YES = "yes"
JUDGE_NO = "no"
MISSING_QUESTION_SENTINEL = "MISSING_QUESTION_IN_DATA"


class CoTJudgeDataset(MathDataset):
    """LLM-as-a-Judge dataset that evaluates the *reasoning* of past generations."""

    language: ClassVar[str] = "en"

    def __init__(
        self,
        name: str | None = None,
        meta_data: dict[str, Any] | None = None,
        **kwargs,
    ):
        meta_data = dict(meta_data) if meta_data else {}
        meta_data.setdefault("file_path", "")
        file_path = meta_data.get("file_path")
        if file_path and os.path.exists(file_path):
            meta_data["file_mtime"] = os.path.getmtime(file_path)
        super().__init__(name=name, meta_data=meta_data, **kwargs)

    def load_tasks(self) -> None:
        file_path = self.meta_data.get("file_path")
        if not file_path or not os.path.exists(file_path):
            logger.error(
                f"CoT judge input file not found: {file_path!r}. "
                f"Pass it via --override-args '{{\"file_path\": \"...\"}}'."
            )
            return

        benchmark_cache: dict[str, MathDataset | None] = {}

        with open(file_path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = orjson.loads(line)
                question = self._lookup_original_question(item, benchmark_cache)
                solution_text = self._extract_solution_text(item)

                prompt = get_judge_prompt(self.language).format(
                    question=question,
                    solution=solution_text,
                )

                task_id = item["task_id"]
                self.add_task(
                    Task(
                        task_id=task_id,
                        prompt=prompt,
                        metadata={
                            "original_task_id": item.get("original_task_id"),
                            "generation_idx": item.get("generation_idx"),
                            "language": self.language,
                        },
                    )
                )
                self.add_groundtruth(GroundTruth(task_id=task_id, answer=JUDGE_YES))

    def _lookup_original_question(
        self,
        item: dict[str, Any],
        cache: dict[str, MathDataset | None],
    ) -> str:
        original_task_id = item.get("original_task_id")
        if not original_task_id:
            return MISSING_QUESTION_SENTINEL

        benchmark_name = original_task_id.split("/", 1)[0].lower()
        if benchmark_name not in cache:
            cache[benchmark_name] = self._load_source_benchmark(benchmark_name)

        source = cache[benchmark_name]
        if source is None:
            return MISSING_QUESTION_SENTINEL

        original_task = source.tasks.get(original_task_id)
        if original_task is None:
            for k, v in source.tasks.items():
                if str(k).lower() == str(original_task_id).lower():
                    original_task = v
                    break
        if original_task is None:
            logger.warning(
                f"Could not resolve original task {original_task_id!r} in {benchmark_name!r}"
            )
            return MISSING_QUESTION_SENTINEL
        return original_task.prompt

    @staticmethod
    def _load_source_benchmark(benchmark_name: str) -> MathDataset | None:
        dataset_cls = DATASET_MAP.get(benchmark_name)
        if dataset_cls is None:
            logger.error(f"Source benchmark {benchmark_name!r} not in DATASET_MAP")
            return None
        try:
            instance = dataset_cls()
            instance.load_tasks()
            return instance
        except Exception as exc:  # noqa: BLE001 — surface any loader failure
            logger.error(f"Failed to load source benchmark {benchmark_name!r}: {exc}")
            return None

    @staticmethod
    def _extract_solution_text(item: dict[str, Any]) -> str:
        raw = item.get("raw_response")
        if isinstance(raw, dict):
            choices = raw.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                content = message.get("content")
                if content:
                    return content
        return str(item.get("generated_answer", ""))

    def format_prompt(self, item: dict[str, Any]) -> str:
        return get_judge_prompt(self.language).format(
            question=item.get("question", MISSING_QUESTION_SENTINEL),
            solution=item.get("solution", ""),
        )

    def extract_solution(self, task_id: str, response: str) -> str:
        answer = extract_boxed_answer(response or "")
        if not answer:
            return "invalid_format"
        return answer.strip().lower()

    def check_correct(
        self,
        extracted_answer: str | None,
        ground_truth: str,
        task_id: str | None = None,
    ) -> bool:
        return extracted_answer == JUDGE_YES


COT_JUDGE_VARIANTS: dict[str, str] = {
    "cot_judge": "en",
    "cot_judge_tr": "tr",
    "cot_judge_pt": "pt",
}


def _make_variant(name: str, language: str) -> type[CoTJudgeDataset]:
    cls_name = f"CoTJudgeDataset_{language.upper()}"
    cls = type(
        cls_name,
        (CoTJudgeDataset,),
        {"language": language, "name": name},
    )
    return register_dataset((name, f"local/{name}", True))(cls)


COT_JUDGE_CLASSES: dict[str, type[CoTJudgeDataset]] = {
    name: _make_variant(name, language) for name, language in COT_JUDGE_VARIANTS.items()
}

CoTJudgeDatasetEN = COT_JUDGE_CLASSES["cot_judge"]
CoTJudgeDatasetTR = COT_JUDGE_CLASSES["cot_judge_tr"]
CoTJudgeDatasetPT = COT_JUDGE_CLASSES["cot_judge_pt"]
