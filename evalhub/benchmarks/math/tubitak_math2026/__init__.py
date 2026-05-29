import os
from typing import Any

import pandas as pd

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.registry import register_dataset

TUBITAK_MATH2026 = "tubitak_math2026"


@register_dataset((TUBITAK_MATH2026, None, True))
class TubitakMath2026Dataset(MathDataset):
    """TÜBİTAK Matematik Olimpiyatı 2026 (Turkish math olympiad, 32 problems, mixed integer + LaTeX answers)."""

    def __init__(self, name: str = TUBITAK_MATH2026, **kwargs):
        super().__init__(name, **kwargs)

    def load_tasks(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "tubitak_math2026.csv")
        df = pd.read_csv(data_path, encoding="utf-8")

        for _, row in df.iterrows():
            task_id = f"{TUBITAK_MATH2026}/{row['Question_Number']}"
            answer_clean = self._strip_outer_dollars(str(row["Answer"]))
            self.add_task(Task(task_id=task_id, prompt=self.format_prompt(row.to_dict())))
            self.add_groundtruth(GroundTruth(task_id=task_id, answer=answer_clean))

    @staticmethod
    def _strip_outer_dollars(s: str) -> str:
        # Idempotent: "$2$" → "2", "2" → "2", "$\frac{1}{2}$" → "\frac{1}{2}".
        s = s.strip()
        if len(s) >= 2 and s.startswith("$") and s.endswith("$"):
            return s[1:-1].strip()
        return s

    def format_prompt(self, item: dict[str, Any]) -> str:
        question = item["Question_Text"].strip()
        instruction_following = "Adım adım düşün ve nihai cevabı \\boxed{} içerisinde ver."
        return question + " " + instruction_following
