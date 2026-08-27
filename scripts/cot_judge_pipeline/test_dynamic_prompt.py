import logging
import argparse
from pathlib import Path
from evalhub.benchmarks.alignment.math_judge import MathJudgeDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test dynamic prompt loading for MathJudge.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the JSONL test file")
    return parser.parse_args()

def test_dynamic_pipeline(test_file: Path) -> None:
    if not test_file.exists():
        logger.error("Test file not found: %s", test_file)
        return

    try:
        dataset = MathJudgeDataset(meta_data={"file_path": str(test_file)})
        dataset.load_tasks()
        
        if not dataset.tasks:
            logger.warning("No tasks loaded from the dataset.")
            return

        first_task = list(dataset.tasks.values())[0] if isinstance(dataset.tasks, dict) else dataset.tasks[0]
        prompt_text = first_task.prompt
        
        if "MISSING_QUESTION_IN_DATA" in prompt_text:
            logger.error("Failed to fetch the original question for task: %s", first_task.task_id)
        
        dummy_response = "Here is my step-by-step reasoning... Therefore, the final answer is \\boxed{no}."
        extracted = dataset.extract_solution("dummy_id", dummy_response)
        
        if extracted not in ["yes", "no"]:
            logger.error("Regex extraction failed. Expected 'yes' or 'no', got: '%s'", extracted)
        else:
            logger.info("Dynamic pipeline test completed successfully.")
            
    except Exception as e:
        logger.error("Pipeline testing failed: %s", e, exc_info=True)

if __name__ == "__main__":
    args = parse_args()
    test_dynamic_pipeline(Path(args.test_file))
