# Utilities & Deployment

This directory contains shared utility scripts, standalone evaluation recipes, and deployment tools.

* **`deployment/`**: Scripts for serving models (`start_vllm_nodes.sh`) and setting up load balancers (`start_sglang_router.sh`).
* **`downsample_data.py` / `run_downsample.sh`**: Tools to filter and reduce the number of samples in a dataset for quicker testing.
* **Standalone Recipes**: Custom, single-file evaluation scripts (e.g., `eval_code.sh`, `r1_recipe.sh`) that do not strictly fit into the main pipelines.
