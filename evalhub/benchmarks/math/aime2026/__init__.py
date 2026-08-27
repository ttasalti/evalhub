from typing import Any

from datasets import load_dataset

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.registry import register_dataset

AIME2026 = "aime2026"
AIME2026_HUB = "MathArena/aime_2026"

@register_dataset((AIME2026, AIME2026_HUB, True))
class AIME2026Dataset(MathDataset):
    """Dataset class for AIME2026 problems."""

    def __init__(self, name: str = AIME2026, **kwargs):
        super().__init__(name, **kwargs)

    def load_tasks(self):
        r"""Load tasks from AIME2026 dataset."""
        dataset = load_dataset(AIME2026_HUB, split="train") 
        for _, item in enumerate(dataset):
            task = Task(
                task_id=f"AIME2026/{item['problem_idx']}",
                prompt=self.format_prompt(item),
            )
            groundtruth = GroundTruth(
                task_id=f"AIME2026/{item['problem_idx']}",
                answer=str(item["answer"]),
            )
            self.add_task(task)
            self.add_groundtruth(groundtruth)

    def format_prompt(self, item: dict[str, Any]) -> str:
        r"""Format the prompt for AIME2026 task."""
        question = item["problem"].strip()
        instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

        question += " " + instruction_following
        return question