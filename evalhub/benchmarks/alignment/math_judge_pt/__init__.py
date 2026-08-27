import json
import os
from typing import Any

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.registry import register_dataset, DATASET_MAP
from evalhub.benchmarks.math.verifier.rllm import extract_boxed_answer
from evalhub.utils.logger import logger

MATH_JUDGE_PT= "math_judge_pt"

JUDGE_PROMPT_TEMPLATE = """You are an expert in mathematics and logical reasoning. Your task is to evaluate the correctness of a solution to a given math problem **(written in European Portuguese)**, with a **strong emphasis on the reasoning process**, not just the final answer.
Below is the **Problem** and the **Solution (Provided by another AI model)**:
—
**Problem**:
{question}
**Solution (Provided by another AI model)**:
{solution}
—
Please perform the following tasks:
1. **Analyze the solution step-by-step**, paying close attention to: - Computational accuracy - Logical consistency - Conceptual understanding - Whether the reasoning is valid and complete
2. **Identify any issues or errors in the reasoning**, even if the final answer is correct. Classify them into the following categories (if applicable): - **Calculation Error**: Mistakes in arithmetic, algebraic manipulation, or numerical computation. - **Logical Error**: Invalid reasoning, flawed logic, or incorrect inference. - **Conceptual Error**: Misunderstanding or misuse of mathematical concepts or definitions. - **Omission / Incompleteness**: Missing steps, incomplete justification, or not addressing all parts of the question. - **Other**: Any other type of error that does not fit into the above categories.
3. **Provide a final judgment** on whether the solution is logically sound and free of errors in reasoning.
4. **Language Consideration**: Ignore whether the solution is provided in European Portuguese, English, or a combination of both (language switching). Focus exclusively on mathematical and logical correctness, disregarding the language used in the evaluation.
Please format your response as follows:
—
**Issues Identified:**
- [Issue 1]: [Classification] - [Brief explanation] - [Issue 2]: [Classification] - [Brief explanation] - ...
Let’s think step by step and output your final judgment within \\boxed{{}}
\\boxed{{yes}} or \\boxed{{no}}"""

MATH_JUDGE_PT_META_DATA = {
    "file_path": "" 
}

@register_dataset((MATH_JUDGE_PT, "local/math_judge_pt", True))
class MathJudgePTDataset(MathDataset):
    """Dataset class for LLM-as-a-Judge on generated math solutions for European Portuguese Benchmarks."""

    def __init__(self, name: str = MATH_JUDGE_PT, meta_data: dict[str, Any] = None, **kwargs):
            if meta_data is None:
                meta_data = MATH_JUDGE_PT_META_DATA.copy()
                
            file_path = meta_data.get("file_path")
            if file_path and os.path.exists(file_path):
                meta_data["file_mtime"] = os.path.getmtime(file_path)
                
            super().__init__(name, meta_data=meta_data, **kwargs)

    def load_tasks(self) -> None:
        """Load tasks from the local JSONL file and dynamically fetch original questions."""
        file_path = self.meta_data.get("file_path")
        if not file_path or not os.path.exists(file_path):
            logger.error(f"File not found or not provided: {file_path}. Please provide a valid file_path via --override-args.")
            return

        loaded_benchmarks = {}

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                
                orig_id = item.get("original_task_id") 
                
                if orig_id:
                    benchmark_name = orig_id.split('/')[0].lower()
                    
                    if benchmark_name not in loaded_benchmarks:
                        try:
                            dataset_cls = DATASET_MAP.get(benchmark_name)
                            
                            if dataset_cls is None:
                                raise ValueError(f"'{benchmark_name}' was not found in DATASET_MAP.")
                                
                            benchmark_ds = dataset_cls()
                            benchmark_ds.load_tasks()
                            loaded_benchmarks[benchmark_name] = benchmark_ds
                        except Exception as e:
                            logger.error(f"Failed to load benchmark {benchmark_name}: {e}")
                            loaded_benchmarks[benchmark_name] = None
                    
                    benchmark_ds = loaded_benchmarks.get(benchmark_name)
                    if benchmark_ds:
                        original_task = benchmark_ds.tasks.get(orig_id)
                        
                        if not original_task:
                            for k, v in benchmark_ds.tasks.items():
                                if str(k).lower() == str(orig_id).lower():
                                    original_task = v
                                    break
                        
                        if original_task:
                            item["question"] = original_task.prompt
                        else:
                            item["question"] = "MISSING_QUESTION_IN_DATA"
                            logger.warning(f"Could not find original task for {orig_id} in {benchmark_name}")
                    else:
                        item["question"] = "MISSING_QUESTION_IN_DATA"
                else:
                    item["question"] = "MISSING_QUESTION_IN_DATA"

                task = Task(
                    task_id=item["task_id"],
                    prompt=self.format_prompt(item),
                    metadata={
                        "original_task_id": item.get("original_task_id"),
                        "generation_idx": item.get("generation_idx")
                    }
                )
                
                groundtruth = GroundTruth(
                    task_id=item["task_id"],
                    answer="yes" # İleride şablonu Türkçe yaparsan burayı "evet" olarak güncellemelisin
                )
                
                self.add_task(task)
                self.add_groundtruth(groundtruth)

    def format_prompt(self, item: dict[str, Any]) -> str:
        """Format the judge prompt with the question and the generated solution."""
        question = item.get("question", "MISSING_QUESTION_IN_DATA")
        
        try:
            solution = item["raw_response"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            solution = item.get("generated_answer", "")

        return JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            solution=solution
        )

    def extract_solution(self, task_id: str, response: str) -> str:
        """Extract 'yes' or 'no' from the judge's response."""
        answer = extract_boxed_answer(response)
        if answer:
            return answer.lower().strip()
        return "invalid_format"
        
    def check_correct(self, extracted_answer: str | None, ground_truth: str, task_id: str = None) -> bool:
        """Check if the judge evaluated it as 'yes'."""
        if extracted_answer is None:
            return False
        return extracted_answer == "yes" # İleride şablonu Türkçe yaparsan burayı "evet" olarak güncellemelisin