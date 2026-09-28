# Third-party notices

This repository is a fork of [ysy-phoenix/evalhub](https://github.com/ysy-phoenix/evalhub)
(MIT License, Copyright (c) 2025 Shengyu Ye). The additions of this fork are
released under the same MIT License; see `LICENSE`.

The code and data below come from other projects and keep their own terms.

## Code

| Component | Location in this repository | Origin | Licence |
|---|---|---|---|
| LiveCodeBench evaluation code | `evalhub/benchmarks/code/livecodebench/` | [LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) | MIT, Copyright (c) 2024 LiveCodeBench |
| BigCodeBench sanitizer | `evalhub/benchmarks/code/bigcodebench/sanitize.py` | [bigcode-project/bigcodebench](https://github.com/bigcode-project/bigcodebench) | Apache License 2.0 |
| Math answer verifier (DAPO) | `evalhub/benchmarks/math/verifier/dapo.py` | [volcengine/verl](https://github.com/volcengine/verl), after EleutherAI and Hugging Face code | Apache License 2.0, Copyright 2024 Bytedance Ltd., Copyright 2022 EleutherAI and the HuggingFace Inc. team |
| G-Pass@k and mG-Pass@k | `evalhub/utils/metrics.py` | implements the definition of Liu et al. (2024), "Are Your LLMs Capable of Stable Reasoning?"; reference implementation [open-compass/GPassK](https://github.com/open-compass/GPassK) | formula implementation; the reference repository states no licence |
| MaskLID | `analysis/src/vendor/masklid.py` | [cisnlp/MaskLID](https://github.com/cisnlp/MaskLID) | MIT, Copyright (c) 2024 Deep NLP @ CIS - LMU (`analysis/src/vendor/MASKLID_LICENSE`) |
| GlotLID model (downloaded at run time, not stored here) | `analysis/src/code_switching.py` | [cis-lmu/glotlid](https://huggingface.co/cis-lmu/glotlid) | Apache License 2.0 with notices, see the model card |

The upstream project also acknowledges EvalPlus, deepscaler,
math-evaluation-harness and verl as sources of design and code.

## Data

| Dataset | Location | Origin | Terms |
|---|---|---|---|
| AIME 2026 (English) | downloaded at run time | [MathArena/aime_2026](https://huggingface.co/datasets/MathArena/aime_2026); problems by the Mathematical Association of America (MAA) | CC BY-NC-SA 4.0 (MathArena); problems copyright MAA |
| AIME 2026 Turkish and Portuguese translations | `evalhub/benchmarks/math/aime2026_tr/`, `aime2026_pt/` | translations of the MathArena set made for this project | CC BY-NC-SA 4.0, credit MAA and MathArena |
| TÜBİTAK Mathematics Olympiad 2026, first stage | `evalhub/benchmarks/math/tubitak_math2026/` | [TÜBİTAK past exams](https://bilimolimpiyatlari.tubitak.gov.tr/tr/gecmis-sinav-sorulari) | copyright TÜBİTAK, all rights reserved; reproduced with attribution for non-commercial research; removed on request |
| Portuguese national exam mathematics questions | `evalhub/benchmarks/math/pt_exams_math/` | PHEB (Tavares et al., LREC 2026), [AMALIA-LLM/pheb](https://github.com/AMALIA-LLM/pheb) | derived from PHEB: the mathematics questions, filtered by hand and converted to open answer; the PHEB repository declares no licence, so the material is used with attribution for research |

The other benchmark loaders under `evalhub/benchmarks/` download their data
from the sources named in each loader and are unchanged from upstream.
