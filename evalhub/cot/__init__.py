"""CoT-Pass@K post-processing pipeline.

Three pure-Python stages run after a standard ``evalhub gen`` + ``evalhub eval``
pass plus a judge ``gen`` + ``eval`` pass:

  * ``extract``   — filter correct generations into a judge-input file.
  * ``aggregate`` — majority-vote the judge verdicts per generation.
  * ``metrics``   — apply the CoT veto and recompute Pass@K / Cons@K.

The ``finalize`` helper composes all three when the judge has already run.
"""

from evalhub.cot.aggregate import aggregate_judge_votes
from evalhub.cot.extract import extract_correct_generations
from evalhub.cot.ids import GenerationId, decode, encode
from evalhub.cot.metrics import COT_FALSE_LABEL, apply_cot_metrics
from evalhub.cot.pipeline import FinalizeResult, finalize_cot_pipeline

__all__ = [
    "GenerationId",
    "encode",
    "decode",
    "extract_correct_generations",
    "aggregate_judge_votes",
    "apply_cot_metrics",
    "COT_FALSE_LABEL",
    "finalize_cot_pipeline",
    "FinalizeResult",
]
