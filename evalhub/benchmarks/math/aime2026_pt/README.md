# `aime2026_pt`

European Portuguese translation of the 30 problems of AIME 2026 (AIME I and II, February
2026). The English problems are by the MAA and were taken from
[MathArena/aime_2026](https://huggingface.co/datasets/MathArena/aime_2026)
(CC BY-NC-SA 4.0); the translation was machine-made and then audited by native
speakers, who corrected wording and notation so that each problem states the
same problem as the original. `problem_idx` follows the MathArena numbering.

- Data: `aime2026_pt.parquet` with `problem_idx`, `problem`, `answer` (0 to 999).
- Licence: CC BY-NC-SA 4.0, as a derivative of the MathArena set; credit the MAA and MathArena.
- On Hugging Face: [tariktuna/aime2026-tr-pt](https://huggingface.co/datasets/tariktuna/aime2026-tr-pt), configuration `pt`.
- CoT-Pass@K: use the matching judge prompt, `JUDGE_TASK=cot_judge_pt`.
