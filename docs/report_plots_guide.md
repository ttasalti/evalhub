# Reading the plot suite — what each folder/PNG means and how to interpret it

This is the interpretation manual for everything under `results/report_plots/`
(produced by `evalhub report plot`). It is **separate from the highlights PDF**:
the highlights tell you *what we found*; this guide teaches you *how to read every
figure yourself* and what conclusions are (and are **not**) supported.

> **Prefer to flip through the plots themselves?** `evalhub report atlas` builds
> `results/report_plots_atlas.pdf` — a curated visual index that embeds
> representative plots from each family with these same explanations, plus a
> complete file index. This document is the full prose reference behind it.

The one question the whole suite exists to answer:

> **When does a model's final answer outrun its reasoning?** i.e. how far does the
> CoT judge "veto" pull a metric down from its No-Judge value — sliced by model,
> mode, language, benchmark and judge, never averaged away.

Two papers frame it: **2504.13837** (Pass@K curves over K, base vs RL → here
**No-Judge vs Judge**) and **2506.14245** (**CoT-Pass@K**: a generation counts only
if answer *and* reasoning are correct → our judged rows).

---

## 0. Conventions that apply to EVERY figure (read this first)

These are global; the per-folder sections below assume you know them.

- **Solid black line / "No-Judge"** = the plain metric (`pass@k`, `g-pass@k`,
  `mg-pass@k`) — answer-only correctness, no reasoning check.
- **Dashed coloured line / "cot"** = the same metric *after* a CoT judge vetoes
  answer-correct-but-reasoning-wrong generations (`cot-pass@k`, …). It is **always
  ≤** the solid line; **the vertical gap between them is the veto effect.** That gap
  is the entire point — read gaps, not absolute heights.
- **Judge is always `think`.** There are exactly three judges, distinguished by
  colour: `gemma-4-26B-A4B-it`, `Qwen3.6-35B-A3B`, `Ministral-3-14B-Reasoning-2512`.
  A "non-think judge" never appears (it would be a migration mislabel).
- **X axis = K on a log₂ scale** (1, 2, 4, …, 64). Moving right = "allow more
  samples / more lenient." Pass@k rises with k; g-pass/mg-pass are stricter.
- **⚠️ Y axis = 0 → that cell's own maximum, NOT a fixed 0–100 %.** This is
  deliberate (so tiny gaps stay visible), but it has one hard consequence:
  **you cannot compare bar/line *heights* across two different cells.** A line that
  looks "high" in one panel may be 5 % and in the next panel 90 %. Always read the
  y tick labels. *Within* a cell, comparing the solid vs dashed line is exactly
  what the axis is tuned for. (The summary **tables** and **heatmaps** give absolute
  numbers — use those when you need cross-cell comparison.)
- **The metric "lenses"** (each is its own PNG):
  - `pass` — **Pass@K**: ≥1 of k generations answer-correct. Most lenient.
  - `gpass_t{0.25,0.5,0.75,1.0}` — **G-Pass@K at threshold τ**: at least a fraction
    τ of the k generations correct. τ=0.25 is lenient, **τ=1.0 = all k must be
    correct** (very strict, near-zero for hard items). Rising τ = rising stringency.
  - `mgpass` — **mG-Pass@K**: G-Pass integrated over τ∈(0.5,1] → a single
    "consistency" score (reward stable correctness, not lucky single hits).
- **States / modes:** `base` (pretrained), `non-think` (instruct, no reasoning),
  `think` (instruct, reasoning on). `base` rows are *different checkpoints* from the
  instruct rows of the same size — only `mode_compare` pairs them deliberately.
- **Languages / benchmarks:** `aime2026`=EN, `aime2026_pt`=PT, `aime2026_tr`=TR
  (the same AIME items translated), `tubitak_math2026`=TR-OL (a *different*, harder
  Turkish olympiad set — not a translation of AIME).

**Mental model for any curve panel:** the solid line is "can it get the answer?",
the dashed line is "can it get the answer *for the right reason*?", and the shaded
gap between them is "how much of its success is unjustified." Everything else is
just which slice you're holding fixed.

---

## 1. `judge_effect/` — THE core figure (Pass@K vs CoT-Pass@K)

**Files:** `{metric}__{state}.png` → 6 metrics × 3 states = 18 PNGs
(e.g. `pass__think.png`, `gpass_t0.5__non-think.png`, `mgpass__base.png`).

**Layout:** one big matrix per file. **Rows = model**, **columns = benchmark
(EN/PT/TR/TR-OL)**. Each cell: x=k, **solid black = No-Judge**, **one dashed line
per judge = cot**. Greyed cells = that (model, benchmark) wasn't run.

**How to read a single cell:**
- Big vertical gap between solid and dashed → the model is frequently *right for the
  wrong reason*; its answer-only score overstates real competence.
- Gap ≈ 0 (dashed hugs solid) → faithful reasoning; the score is trustworthy.
- The three dashed lines fanning apart → the judges disagree about this model
  (treat the cot number as judge-dependent, not absolute — see `comparisons/`).

**What to infer / how to use it:** This is your first stop. Open `pass__think.png`
to see, for reasoning-mode models, where the answer outruns the reasoning. Scan
across a row to see **language sensitivity** of one model; scan down a column to see
**which models** are most/least faithful on one benchmark. Switch the `{state}` file
to ask the same question for pretrained vs non-think vs think.

**Caveat:** y is 0→cell-max, so do **not** compare gap *heights* between two cells;
compare "gap relative to the solid line" (a 5-pt gap on a 10-pt line is huge; on a
90-pt line it's minor). For absolute magnitudes use `tables/` or `comparisons/`.

---

## 2. `bench_compare/` — language transfer (is the model's skill language-bound?)

**Files:** `{metric}__{state}__nojudge.png` and `{metric}__{state}__cot.png`
(6 metrics × 3 states × 2 variants = 36 PNGs).

**Layout:**
- `__nojudge`: a grid of **per-model panels**; inside each panel, **4 coloured
  curves = the 4 languages** (EN blue, PT green, TR red, TR-OL purple), No-Judge only.
- `__cot`: **rows = model**, **columns = [No-Judge | each judge]**; each cell again
  holds the 4 language curves. Reading left→right across the columns shows how the
  veto reshapes the language picture.

**How to read / infer:**
- In `__nojudge`, curves bunched together → the model transfers across languages
  (same math skill EN/PT/TR). Curves spread → language-bound performance. Note TR-OL
  is a *harder different* set, so it legitimately sits apart — don't read its gap as
  pure "translation loss."
- In `__cot`, compare the No-Judge column to each judge column: if one language's
  curve *drops more* than others when you move into a judge column, that language has
  more unfaithful CoT (the veto bites it harder). This is the multilingual veto story
  in curve form (the heatmap version is in `comparisons/`).

**Use it to answer:** "Does this model actually reason in PT/TR, or just pattern-match
the answer?" — a language whose curve collapses under the judge is being answered
without sound reasoning.

---

## 3. `size_compare/` — scaling (does competence/faithfulness grow with parameters?)

**Files:** `{metric}__{state}__nojudge.png` and
`{metric}__{state}__cot__{judge}.png` (per-judge for the cot overlay).

**Layout:** **rows = model family** (Qwen / gemma / Ministral), **columns =
benchmark**. Inside each cell, **one curve per model, coloured by size** (viridis:
dark/purple = small, yellow = large). The `__cot__{judge}` variant adds, for each
model, a **dashed** line of the same colour = that judge's cot — so within one cell
you see both "bigger = better?" and "bigger = more faithful?".

**How to read / infer:**
- `__nojudge`: curves ordered by colour from low (small) to high (large) → clean
  size scaling. Crossing/overlapping curves → scaling has broken down (or a small
  model is punching above its size).
- `__cot__{judge}`: look at the **solid–dashed gap per colour**. If the gap shrinks
  as colour goes from dark→yellow, **faithfulness improves with scale** (big models
  earn their answers; small models guess). This is one of the strongest effects in
  the data.

**Use it to answer:** "Is the veto effect a small-model artifact?" Compare the gap of
the smallest vs largest curve in a cell.

---

## 4. `veto_curve/` — Δ(k): how much the judge removes, as a function of K

**Files:** `{metric}__{state}.png` (18). **Layout:** rows = model, cols =
benchmark; each cell plots **y = No-Judge − cot (the veto Δ itself)** vs k, one
dashed line per judge, with a grey zero line.

**Why it exists:** in `judge_effect` both lines saturate near the top at high k, so
the gap gets visually squeezed. Here the gap *is* the y value, so you can see it even
when pass@k is ~90 %. **A rising curve means the veto grows with more sampling** —
extra samples buy more *answer*-correct hits than *reasoning*-correct ones.

**How to read / infer:**
- Upward slope → "lucky guesses" accumulate faster than justified solutions as k
  grows (the headline global trend: mean Δ rises 2.8→7.0 from k=1 to k=64).
- Flat near zero → faithful at all sampling budgets.
- A hump (rise then fall) → the model "catches up" with justified solutions at high k.
- Lines far apart → judge-dependent; pick the judge deliberately.

**Use it to answer:** "If I sample more, am I just inflating an unjustified score?"

---

## 5. `mode_compare/` — pretrained vs instruct-non-think vs instruct-think

**Files:** `{metric}__{family}.png` (6 metrics × 3 families = 18). This is the only
family that **pairs the base checkpoint with its instruct sibling of the same size**.

**Layout:** **rows = model size**, **columns = benchmark**. Each cell overlays three
**solid** No-Judge curves — **grey = pretrained, blue = non-think, red = think** —
plus a **dashed** cot line per mode (the representative/most-covered judge).

**How to read / infer:**
- Compare the three solid lines: does instruct-tuning help (blue/red above grey)?
  Does reasoning help (red above blue)? — **this can invert**: for Qwen on AIME-EN,
  `think` (red) sits *below* `non-think` (blue) at high k (reasoning *hurts* raw
  accuracy there; see the anomaly in the highlights).
- Compare each solid line to its own dashed line (the veto for that mode). A key
  pattern: **think's solid–dashed gap is the smallest** → reasoning mode produces the
  most faithful CoT even when its raw accuracy isn't the highest.

**Use it to answer:** "Does turning on reasoning make the model *more correct* or
just *more justified*?" — those are two different axes and this plot separates them.

---

## 6. `per_model/` — one model's full fingerprint

**Files:** `{short_model}__{state}.png` (one per model×mode actually run, ~23).

**Layout:** **rows = the 6 metrics** (pass, g-pass τ×4, mg-pass), **columns =
benchmark**. Each cell is a `judge_effect`-style cell (solid No-Judge + dashed cot
per judge) for that one model+mode.

**How to read / infer:** this is the deep-dive once a model catches your eye in
`judge_effect`. Reading **down a column** (fixed language) shows how the veto behaves
as the metric gets stricter: typically the gap is largest for `pass` and shrinks for
`mg-pass` (the veto mostly removes lucky single hits, not consistent correctness).
Reading **across a row** (fixed metric) shows that model's language profile.

**Use it to answer:** "Give me the complete story for *this* model" — for a report
appendix or to sanity-check one checkpoint.

---

## 7. `tables/` — absolute numbers (use these for cross-cell comparison)

Unlike the curve families, tables give you **comparable absolute values** (×100).
Three kinds, at **k=1 and k=64**:

- **`{benchmark}__k{k}__nojudge.png`** — rows = `(model·mode)`, cols = the 6 metrics,
  values = No-Judge ×100, **blue = higher**. The plain leaderboard per benchmark.
- **`{benchmark}__k{k}__cotdelta__{judge}.png`** — same grid but values =
  **(No-Judge − cot) ×100 = the veto**, **diverging colour (red = bigger veto)**.
  This is the numeric companion to the heatmaps; scan for red rows/cells.
- **`headline__{judge}__k{k}.png`** — rows = `(model·mode)`, cols = benchmark, each
  cell = **`pass / cot-pass`** (both numbers ×100), coloured by the veto. The
  one-glance "answer vs justified-answer" scoreboard.

**How to read / infer:** because these are absolute, you *can* compare across rows
and columns here. Use `nojudge` to rank capability, `cotdelta` to rank
*un*faithfulness, and `headline` to see both at once. Switch `{judge}` to see how
much the conclusion depends on who graded.

---

## 8. `comparisons/` — the multilingual veto heatmaps + the raw CSV

**Files:** `veto__{metric}__k{k}__{judge}.png` (heatmaps) and
**`pass_vs_cot_k{1,64}.csv`** (the data behind everything).

**Heatmap layout:** **rows = `(model·mode)`**, **columns = language**, cell value =
**No-Judge − cot (×100)**, **diverging RdBu_r centred at 0** (deep red = large veto),
annotated with the number. One heatmap per (metric, k, judge).

**How to read / infer:** this is the cleanest view of the *main* question — the
veto across languages. Scan a **column** to find which language is hardest-vetoed
overall (TR-OL columns run reddest); scan a **row** to see a model's per-language
faithfulness. Because it's centred at 0 with a shared scale, colour intensity *is*
comparable across cells here (unlike the curve families).

**`pass_vs_cot_k{k}.csv`** is long-format: one row per
`(model, mode, language, judge, metric)` with `nojudge`, `cot`, `delta`. Load it in
pandas/Excel to compute any ranking yourself, e.g. group by language or judge — it's
the audit trail for the highlights PDF numbers.

---

## 9. "Which folder answers which question?" — quick recipe

| Your question | Open |
|---|---|
| Where does answer outrun reasoning, overall? | `judge_effect/pass__{state}.png` |
| Does the model really reason in PT/TR? | `bench_compare/pass__{state}__cot.png` |
| Is the veto just a small-model thing? | `size_compare/pass__{state}__cot__{judge}.png` |
| Does sampling more inflate an unjustified score? | `veto_curve/pass__{state}.png` |
| Does reasoning make it *correct* or just *justified*? | `mode_compare/pass__{family}.png` |
| Full story for one checkpoint? | `per_model/{model}__{state}.png` |
| I need exact, comparable numbers / a leaderboard | `tables/*` |
| The multilingual veto, at a glance, comparable | `comparisons/veto__*__{judge}.png` |
| I want to compute my own stat | `comparisons/pass_vs_cot_k{k}.csv` |

## 10. Three interpretation traps to avoid

1. **Don't compare heights across curve cells** — y is 0→cell-max. Compare gaps
   *within* a cell, or switch to `tables/` / `comparisons/` for absolute values.
2. **Don't read a single judge as ground truth** — judges disagree (max spread on
   one cell ≈ 47 pts). If a conclusion flips between the `{judge}` files, it's a
   judge artifact, not a model fact. The heatmaps/tables let you check all three.
3. **Keep "correct" and "justified" separate** — a high solid line with a big gap is
   a model that *answers* well but *reasons* poorly; that is a worse outcome than a
   slightly lower solid line that the dashed line hugs. The whole suite is built to
   stop you from rewarding the first case.
