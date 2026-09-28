# `pt_exams_math`

The 166 Mathematics A questions of PHEB, the Portuguese national
secondary-school exams from 2006 to 2023, converted by hand from multiple
choice to open-answer form. PHEB gives four options and the index of the
correct one; here the options are dropped from the prompt and the reference
value is given directly in `ground_truth`, so a model has to produce the answer
rather than pick it. The original `choices` and `answer` index are kept for
traceability. All material is European Portuguese; two graduate students,
native speakers, checked the converted questions and answers.

- Source: [PHEB](https://github.com/AMALIA-LLM/pheb) (Tavares et al., LREC 2026),
  on Hugging Face as [amalia-llm/pt_exams](https://huggingface.co/datasets/amalia-llm/pt_exams),
  configuration `mathematics_a`.
- This subset on Hugging Face: [tariktuna/pt-exams-math-open](https://huggingface.co/datasets/tariktuna/pt-exams-math-open).
- Terms: PHEB declares no licence; the subset is redistributed for research
  with attribution, as agreed with the PHEB authors. Please cite PHEB when you
  use it (see `THIRD_PARTY_NOTICES.md`).

`pt_exams_math.csv` columns: `question`, `ground_truth` (open-answer reference
as in the exam key: `2`, `1,1`, `4320`, `\frac{1}{2}`, `50%`), `choices`,
`answer` (index into `choices`), `is_completion`, `phase`, `question_group`,
`question_number`, `subject`, `year`. The loader adds the Portuguese solver
instruction and grades through the shared verifier with a tolerance for
decimal and percentage answers; a comma between digits is read as a decimal
mark. Use the Portuguese judge prompt (`JUDGE_TASK=cot_judge_pt`) for
CoT-Pass@K.
