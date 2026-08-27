# CoT & LLM-as-a-Judge Pipeline

This pipeline is responsible for filtering correct generations and evaluating them using a stronger LLM as a judge to verify Chain-of-Thought reasoning.

* **`01_extract_corrects.py` / `01_run_extraction.sh`**: Extracts the raw text of correct generations from the base model's outputs.
* **`02_submit_judges.sh`**: Submits the judge evaluation tasks to the Slurm cluster.
* **`03_run_judge_worker.sh`**: The core worker that prompts the judge model to evaluate the extracted solutions.
* **`04_aggregate_votes.py`**: Computes the majority vote from multiple judge responses (e.g., majority "yes" or "no").
* **`05_apply_metrics.py`**: Merges the judge's majority vote with the original results, updating the true correct counts and Pass@K metrics.
* **`04_05_run_postprocess.sh`**: Automates the aggregation and metric application steps (runs `04_aggregate_votes.py` and `05_apply_metrics.py` sequentially).
* **`test_dynamic_prompt.py`**: A testing utility to verify that the dynamic prompt formatting and dataset loading work correctly for the judge.
