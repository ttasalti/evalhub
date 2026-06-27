"""Command-line interface for EvalHub."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from evalhub.benchmarks import DATASET_HUB, DATASET_MAP, EVALUATE_DATASETS, THIRD_PARTY_DATASETS
from evalhub.benchmarks.base import Dataset
from evalhub.cot.aggregate import aggregate_judge_votes
from evalhub.cot.extract import extract_correct_generations
from evalhub.cot.metrics import apply_cot_metrics
from evalhub.cot.pipeline import finalize_cot_pipeline
from evalhub.gen import generate
from evalhub.inference.schemas import GenerationConfig
from evalhub.report._cli import (
    DEFAULT_CSV,
    DEFAULT_PLOT_DIR,
    cmd_aggregate,
    cmd_plot,
    cmd_upsert,
)
from evalhub.utils.typer import options
from evalhub.view import view_results

console = Console()

# Create the main Typer app
app = typer.Typer(
    name="evalhub",
    help="EvalHub - All-in-one benchmarking platform for evaluating LLMs.",
    add_completion=True,
    rich_markup_mode="rich",
)

cot_app = typer.Typer(
    name="cot",
    help="CoT-Pass@K post-processing pipeline (extract / aggregate / metrics).",
    add_completion=False,
    rich_markup_mode="rich",
)
app.add_typer(cot_app, name="cot")

report_app = typer.Typer(
    name="report",
    help="Aggregate evaluation summaries into the master wide CSV (full rebuild or incremental upsert).",
    add_completion=False,
    rich_markup_mode="rich",
)
app.add_typer(report_app, name="report")


@app.command()
@options(GenerationConfig)
def gen(
    config: GenerationConfig,
    *,
    override_args: Annotated[str | None, typer.Option(help="Override dataset arguments in json string format")] = None,
):
    r"""Run generation on a model with specified dataset."""
    console.print(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for task in config.tasks:
        generate(config=config, task=task, override_args=override_args)


@app.command()
def eval(
    tasks: Annotated[str, typer.Option(help="Tasks to evaluate on, separated by commas")],
    solutions: Annotated[str, typer.Option(help="Solutions to evaluate on, separated by commas")],
    output_dir: Annotated[str, typer.Option(help="Output directory")],
    override_args: Annotated[str | None, typer.Option(help="Override dataset arguments in json string format")] = None,
):
    r"""Evaluate the model on the tasks."""
    tasks = [task.strip().lower() for task in tasks.split(",")]
    solutions = [solution.strip() for solution in solutions.split(",")]
    assert len(tasks) == len(solutions), "Number of tasks and solutions must be the same"
    for task, solution in zip(tasks, solutions, strict=False):
        assert task in EVALUATE_DATASETS, f"Dataset {task} is not supported for evaluation"
        dataset: Dataset = DATASET_MAP[task](name=task.lower(), override_args=override_args)
        dataset.evaluate(solution, output_dir)


@app.command()
def view(
    results: Annotated[str, typer.Option(help="Results file path")],
    max_display: Annotated[int, typer.Option(help="Maximum number of samples to display")] = -1,
    false_only: Annotated[bool, typer.Option(help="Only display false samples")] = True,
):
    r"""View and analyze evaluation results with rich formatting.

    Automatically detects the result format:
    - JSONL files: Math evaluation results (GSM8K, etc.)
    - JSON files: LiveCodeBench results
    """
    view_results(
        results_path=Path(results),
        max_display=max_display,
        false_only=false_only,
    )


@app.command(name="tasks")
def list_tasks():
    r"""List all supported tasks and evaluable tasks."""
    # Create a table for all tasks
    task_table = Table(title="EvalHub Supported Tasks")

    task_table.add_column("Task", style="cyan")
    task_table.add_column("Evaluable", style="green")
    task_table.add_column("Huggingface", style="blue")

    # Sort tasks alphabetically for better readability
    sorted_tasks = sorted(DATASET_MAP.keys())

    for task in sorted_tasks:
        evaluable = "✅" if task in EVALUATE_DATASETS else "❌(Third-party)"
        hf_name = DATASET_HUB[task]
        task_table.add_row(task, evaluable, hf_name)

    console.print(task_table)

    # Print usage examples
    console.print("\n[bold yellow]Generation Examples:[/bold yellow]")
    console.print('evalhub gen --model "Qwen2.5-7B-Instruct" --tasks humaneval,mbpp --output-dir ./results')

    console.print("\n[bold yellow]Evaluation Examples:[/bold yellow]")
    for task in list(DATASET_MAP.keys())[:6]:
        if task == "bigcodebench":
            continue
        if task in THIRD_PARTY_DATASETS:
            console.print(f"evalplus.evaluate --dataset {task} --samples ./results/{task}.jsonl")
        else:
            assert task in EVALUATE_DATASETS, f"Dataset {task} is not supported for evaluation"
            console.print(f"evalhub eval --tasks {task} --solutions ./results/{task}.jsonl --output-dir ./results")


@cot_app.command("extract")
def cot_extract(
    base_results: Annotated[Path, typer.Option(help="Path to the base eval *_results.jsonl")],
    base_raw: Annotated[Path, typer.Option(help="Path to the base gen *_raw.jsonl")],
    output: Annotated[Path, typer.Option(help="Output JSONL: one record per correct generation")],
    max_tasks: Annotated[int | None, typer.Option(help="Optional cap on number of tasks to process")] = None,
):
    r"""Extract correct base generations into a judge-input JSONL."""
    written = extract_correct_generations(
        base_results_path=base_results,
        base_raw_path=base_raw,
        output_path=output,
        max_tasks=max_tasks,
    )
    console.print(f"[green]Wrote {written} judge-input records -> {output}[/green]")


@cot_app.command("aggregate")
def cot_aggregate(
    judge_solutions: Annotated[Path, typer.Option(help="Path to the judge gen *_solutions/.jsonl")],
    output: Annotated[Path, typer.Option(help="Output JSONL: one majority verdict per generation")],
):
    r"""Majority-vote yes/no judge outputs into per-generation verdicts."""
    written = aggregate_judge_votes(judge_solutions_path=judge_solutions, output_path=output)
    console.print(f"[green]Wrote {written} majority verdicts -> {output}[/green]")


@cot_app.command("metrics")
def cot_metrics(
    base_results: Annotated[Path, typer.Option(help="Path to the base eval *_results.jsonl")],
    majority: Annotated[Path, typer.Option(help="Path to the majority verdicts JSONL")],
    output: Annotated[Path, typer.Option(help="Output JSONL: results with CoT-adjusted correct[] and pass_at_k")],
    summary: Annotated[Path, typer.Option(help="Output JSON: aggregate pass_at_k and cons_at_k under the CoT veto")],
    stats: Annotated[Path | None, typer.Option(help="Optional output JSON: per-class generation counts")] = None,
):
    r"""Apply the CoT veto and recompute Pass@K / Cons@K."""
    result = apply_cot_metrics(
        base_results_path=base_results,
        majority_path=majority,
        output_results_path=output,
        summary_path=summary,
        stats_path=stats,
    )
    console.print(f"[green]CoT metrics -> {summary}[/green]")
    console.print(result)


@cot_app.command("finalize")
def cot_finalize(
    base_results: Annotated[Path, typer.Option(help="Path to the base eval *_results.jsonl")],
    base_raw: Annotated[Path, typer.Option(help="Path to the base gen *_raw.jsonl")],
    judge_solutions: Annotated[Path, typer.Option(help="Path to the judge gen *_solutions/.jsonl")],
    output_dir: Annotated[Path, typer.Option(help="Directory to write intermediate and final files")],
    benchmark: Annotated[str, typer.Option(help="Benchmark short name (used as filename stem)")],
):
    r"""Run extract -> aggregate -> metrics in one shot from local files."""
    result = finalize_cot_pipeline(
        base_results_path=base_results,
        base_raw_path=base_raw,
        judge_solutions_path=judge_solutions,
        output_dir=output_dir,
        benchmark=benchmark,
    )
    console.print(f"[green]CoT pipeline complete -> {result.summary_path}[/green]")
    console.print(result.summary)


@report_app.command("aggregate")
def report_aggregate(
    results_root: Annotated[
        Path, typer.Option(help="Root directory produced by evalhub eval / cot finalize")
    ] = Path("results"),
    output: Annotated[Path, typer.Option(help="Destination CSV for the wide aggregated table")] = DEFAULT_CSV,
):
    r"""Walk ``results_root`` and write the master **wide** CSV (full rebuild).

    Every ``*_summary.json`` (No-Judge) and ``*_cot_summary.json`` (judged)
    contributes one row carrying every metric at every K and τ. The
    ``judge_model`` column is empty for No-Judge rows and set for cot rows.
    Defaults to ``results/report.csv`` (the CSV lives inside the results tree).
    """
    out = cmd_aggregate(results_root, output)
    console.print(f"[green]Aggregated wide CSV -> {out}[/green]")


@report_app.command("upsert")
def report_upsert(
    summary: Annotated[Path, typer.Option(help="A single *_summary.json / *_cot_summary.json to add")],
    csv: Annotated[Path, typer.Option(help="Master CSV to append to / update")] = DEFAULT_CSV,
    results_root: Annotated[
        Path | None,
        typer.Option(help="Optional results root (only sets the row's source_root)"),
    ] = None,
):
    r"""Add or replace **one** row in the master CSV for the given summary.

    Idempotent: re-running for the same (model, state, benchmark, judge) replaces
    that row; a new key appends; an unseen K grows the schema. Drop this into a
    pipeline to grow the CSV one evaluation at a time.
    """
    out = cmd_upsert(summary, csv, results_root)
    console.print(f"[green]Upserted 1 row -> {out}[/green]")


@report_app.command("plot")
def report_plot(
    csv: Annotated[Path, typer.Option(help="Wide master CSV produced by `report aggregate`")] = DEFAULT_CSV,
    output_dir: Annotated[Path, typer.Option(help="Directory for the plot suite")] = DEFAULT_PLOT_DIR,
):
    r"""Render the Pass@K vs CoT-Pass@K visualisation suite from the wide CSV.

    Line matrices (No-Judge vs each judge over K), benchmark/size comparisons,
    veto-effect curves, summary tables and multilingual Δ heatmaps — written
    under ``results/report_plots`` by default.
    """
    written = cmd_plot(csv, output_dir)
    total = sum(len(paths) for paths in written.values())
    console.print(f"[green]Rendered {total} file(s) under {output_dir}[/green]")


def main():
    r"""Run the CLI entry point for EvalHub."""
    app()


if __name__ == "__main__":
    main()
