# Pass@K Evaluation Pipeline

This pipeline handles the standard evaluation of LLMs on various benchmarks to compute the Pass@K metric.

* **`01_submit_evals.sh`**: Submits parallel Slurm jobs for each model defined in the configuration.
* **`02_run_eval_worker.sh`**: The core worker script that starts a local vLLM/SGLang server, generates responses, and evaluates them.
* **`03_analyze_stats.py`**: A Python script to extract generation statistics (correct/wrong counts).
* **`04_run_analyze.sh`**: A Slurm wrapper to run the statistical analysis across multiple models and benchmarks.
