"""Judge prompt templates per language.

Each template has two named placeholders: ``{question}`` and ``{solution}``.
Adding support for a new language is a single dictionary entry.
"""

from __future__ import annotations

JUDGE_PROMPT_EN = """You are an expert in mathematics and logical reasoning. Your task is to evaluate the correctness of a solution to a given math problem, with a **strong emphasis on the reasoning process**, not just the final answer.
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
Please format your response as follows:
—
**Issues Identified:**
- [Issue 1]: [Classification] - [Brief explanation] - [Issue 2]: [Classification] - [Brief explanation] - ...
Let's think step by step and output your final judgment within \\boxed{{}}
\\boxed{{yes}} or \\boxed{{no}}"""

JUDGE_PROMPT_TR = """You are an expert in mathematics and logical reasoning. Your task is to evaluate the correctness of a solution to a given math problem **(written in Turkish)**, with a **strong emphasis on the reasoning process**, not just the final answer.
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
4. **Language Consideration**: Ignore whether the solution is provided in Turkish, English, or a combination of both (language switching). Focus exclusively on mathematical and logical correctness, disregarding the language used in the evaluation.
Please format your response as follows:
—
**Issues Identified:**
- [Issue 1]: [Classification] - [Brief explanation] - [Issue 2]: [Classification] - [Brief explanation] - ...
Let's think step by step and output your final judgment within \\boxed{{}}
\\boxed{{yes}} or \\boxed{{no}}"""

JUDGE_PROMPT_PT = """You are an expert in mathematics and logical reasoning. Your task is to evaluate the correctness of a solution to a given math problem **(written in Portuguese)**, with a **strong emphasis on the reasoning process**, not just the final answer.
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
4. **Language Consideration**: Ignore whether the solution is provided in Portuguese, English, or a combination of both (language switching). Focus exclusively on mathematical and logical correctness, disregarding the language used in the evaluation.
Please format your response as follows:
—
**Issues Identified:**
- [Issue 1]: [Classification] - [Brief explanation] - [Issue 2]: [Classification] - [Brief explanation] - ...
Let's think step by step and output your final judgment within \\boxed{{}}
\\boxed{{yes}} or \\boxed{{no}}"""

JUDGE_PROMPTS: dict[str, str] = {
    "en": JUDGE_PROMPT_EN,
    "tr": JUDGE_PROMPT_TR,
    "pt": JUDGE_PROMPT_PT,
}


def get_judge_prompt(language: str) -> str:
    try:
        return JUDGE_PROMPTS[language]
    except KeyError as exc:
        raise ValueError(
            f"Unknown CoT judge language {language!r}. Known: {sorted(JUDGE_PROMPTS)}"
        ) from exc
