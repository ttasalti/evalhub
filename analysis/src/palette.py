"""Colour palette shared by the figure scripts.

Three roles, three distinct colour families: solver generation, judge and
benchmark. The keys are the short names used in the data tables.
"""

# Solver generation (earlier: Qwen2.5 solvers; current: Qwen3.5 and Gemma-4 solvers).
GEN = {"earlier": "#08519c", "current": "#a63603"}
GEN_LIGHT = {"earlier": "#c6dbef", "current": "#fdd0a2"}

JUDGE = {"Qwen3.6": "#ee854a", "V4-Flash": "#4878cf", "R1-distill": "#6acc65", "Gemma4": "#8172b3"}

# Benchmarks: Okabe-Ito colours (colour-blind safe), palette "B" below.
BENCH = {
    "aime2026": "#0072b2",
    "aime2026_pt": "#009e73",
    "aime2026_tr": "#e69f00",
    "tubitak_math2026": "#cc79a7",
    "pt_exams_math": "#56b4e9",
}
BENCH_ORDER = ["aime2026", "aime2026_pt", "aime2026_tr", "tubitak_math2026", "pt_exams_math"]
BENCH_LABEL = {
    "aime2026": "AIME (EN)",
    "aime2026_pt": "AIME (PT)",
    "aime2026_tr": "AIME (TR)",
    "tubitak_math2026": "TÜBİTAK (TR)",
    "pt_exams_math": "PT exams",
}

# Alternative benchmark palettes. BENCH above is palette "B".
PALETTES = {
    # A: seaborn "deep" with a brown for the Portuguese exams.
    "A": {
        "aime2026": "#4c72b0",
        "aime2026_pt": "#55a868",
        "aime2026_tr": "#dd8452",
        "tubitak_math2026": "#8172b3",
        "pt_exams_math": "#937860",
    },
    # C: purple, red, blue, green and mustard.
    "C": {
        "aime2026": "#8c4fa8",
        "aime2026_pt": "#c8552c",
        "aime2026_tr": "#2f6a9e",
        "tubitak_math2026": "#3a7d44",
        "pt_exams_math": "#b8860b",
    },
    # B: Okabe-Ito.
    "B": {
        "aime2026": "#0072b2",
        "aime2026_pt": "#009e73",
        "aime2026_tr": "#e69f00",
        "tubitak_math2026": "#cc79a7",
        "pt_exams_math": "#56b4e9",
    },
}
