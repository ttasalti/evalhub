import os
from typing import Any
import pandas as pd

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.registry import register_dataset

AIME2026_TR = "aime2026_tr"

@register_dataset((AIME2026_TR, None, True))
class AIME2026TRDataset(MathDataset):
    """Dataset class for Turkish AIME2026 problems."""

    def __init__(self, name: str = AIME2026_TR, **kwargs):
        super().__init__(name, **kwargs)

    def load_tasks(self):
        r"""Load tasks directly from the local Parquet file."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "aime2026_tr.parquet")
        
        # Yerel Parquet dosyasını oku
        df = pd.read_parquet(data_path)
        
        for _, row in df.iterrows():
            task_id = f"{AIME2026_TR}/{row['problem_idx']}"
            task = Task(
                task_id=task_id,
                prompt=self.format_prompt(row.to_dict()),
            )
            groundtruth = GroundTruth(
                task_id=task_id,
                answer=str(row["answer"]),
            )
            self.add_task(task)
            self.add_groundtruth(groundtruth)

    def format_prompt(self, item: dict[str, Any]) -> str:
        r"""Format the prompt for the Turkish AIME2026 task."""
        question = item["problem"].strip()
        instruction_following = "Adım adım düşün ve nihai cevabı \\boxed{} içerisinde ver."

        question += " " + instruction_following
        return question