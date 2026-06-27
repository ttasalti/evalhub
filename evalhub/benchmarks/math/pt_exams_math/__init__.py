import os
import re
from typing import Any

import pandas as pd
import sympy
from sympy.parsing.sympy_parser import parse_expr

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.registry import register_dataset

PT_EXAMS_MATH = "pt_exams_math"


@register_dataset((PT_EXAMS_MATH, None, True))
class PTExamsMathDataset(MathDataset):
    """Portuguese national math-exam questions (open-ended, mixed integer + LaTeX answers)."""

    def __init__(self, name: str = PT_EXAMS_MATH, **kwargs):
        super().__init__(name, **kwargs)

    def load_tasks(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "pt_exams_math.csv")
        df = pd.read_csv(data_path, encoding="utf-8")

        for idx, row in df.iterrows():
            task_id = f"{PT_EXAMS_MATH}/{idx}"
            answer_clean = self._normalize_answer(str(row["ground_truth"]))
            self.add_task(Task(task_id=task_id, prompt=self.format_prompt(row.to_dict())))
            self.add_groundtruth(GroundTruth(task_id=task_id, answer=answer_clean))

    @classmethod
    def _normalize_answer(cls, s: str) -> str:
        s = cls._strip_outer_dollars(s)
        s = re.sub(r"\s*°\s*$", "", s).strip()
        s = re.sub(r"\s*graus\s*$", "", s, flags=re.IGNORECASE).strip()
        return s

    @staticmethod
    def _strip_outer_dollars(s: str) -> str:
        # Idempotent: "$2$" → "2", "2" → "2", "$\frac{1}{2}$" → "\frac{1}{2}".
        s = s.strip()
        if len(s) >= 2 and s.startswith("$") and s.endswith("$"):
            return s[1:-1].strip()
        return s

    def patch(self, extracted_answer: str, ground_truth: str, task_id: str = None) -> bool:
        gt_value = self._to_float(ground_truth)
        extracted_value = self._to_float(extracted_answer)
        if gt_value is None or extracted_value is None:
            return False

        if ground_truth.rstrip().endswith("%"):
            # "30%" should also match "0.3" or "3/10".
            return any(abs(extracted_value - candidate) < 1e-4 for candidate in (gt_value, gt_value / 100))

        if gt_value == 0:
            return False
        # Non-terminating fractions (e.g. "\frac{1}{7}") rounded to a decimal
        # by the model ("0.143") have no exact match in grade_answer(); accept
        # within a tolerance that absorbs 3-4 significant-digit rounding.
        return abs(extracted_value - gt_value) < max(1e-2 * abs(gt_value), 5e-4)

    @staticmethod
    def _to_float(s: str) -> float | None:
        s = s.strip().strip("$").strip()
        s = s.replace("\\%", "").replace("%", "")
        s = re.sub(r"(\d)\s+(\d)", r"\1\2", s)
        s = s.replace(",", ".")
        s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
        s = s.replace("\\left", "").replace("\\right", "").replace("\\circ", "").replace("°", "")
        s = re.sub(r"\s*graus\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"(\d)\\pi\b", r"\1*pi", s)
        s = s.replace("\\pi", "pi")
        s = re.sub(r"(\d)\\sqrt\{([^{}]+)\}", r"\1*sqrt(\2)", s)
        s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
        s = s.strip()
        try:
            return float(sympy.N(parse_expr(s, evaluate=True)))
        except Exception:
            return None

    def format_prompt(self, item: dict[str, Any]) -> str:
        question = item["question"].strip()
        instruction_following = "Vamos pensar passo a passo e apresentar a resposta final dentro de \\boxed{}."
        question += " " + instruction_following
        return question
