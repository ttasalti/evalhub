"""Curated **plot-atlas PDF** — a guided tour of ``results/report_plots/``.

The full suite is ~245 PNGs; this assembles a single self-contained catalogue:
a cover page (suite map + reading conventions), then one section per plot family
with a deep interpretation caption, one or two *representative* plots embedded at
full size, and a compact index of every file in that family so nothing is hidden.

It pairs with the two text companions: the highlights PDF (``report highlights``,
the findings) and ``docs/report_plots_guide.md`` (the full prose manual). This is
the *visual* index. No new dependencies — it embeds the existing PNGs.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from evalhub.utils.logger import logger  # noqa: E402

A4 = (8.27, 11.69)
ACCENT = "#c0392b"
INK = "#222222"

# Render order; only families present on disk are emitted.
FAMILY_ORDER = [
    "judge_effect", "bench_compare", "size_compare", "veto_curve",
    "mode_compare", "per_model", "tables", "comparisons",
]

FAMILY_TITLES = {
    "judge_effect": "judge_effect — Pass@K vs CoT-Pass@K (core)",
    "bench_compare": "bench_compare — language transfer",
    "size_compare": "size_compare — scaling",
    "veto_curve": "veto_curve — Δ(k), how the veto grows",
    "mode_compare": "mode_compare — pretrained vs instruct modes",
    "per_model": "per_model — single-checkpoint fingerprints",
    "tables": "tables — absolute numbers & headline scoreboards",
    "comparisons": "comparisons — multilingual veto heatmaps (+ CSV)",
}

# Deep interpretation per family (condensed from docs/report_plots_guide.md).
ATLAS_CAPTIONS = {
    "judge_effect": (
        "Rows = model, columns = benchmark. Solid black = No-Judge (pass@k); each dashed colour = a "
        "judge's cot-pass@k. The vertical GAP is the veto = unjustified success. Big gap → right answer, "
        "wrong reasoning; dashed hugging solid → faithful. One file per (metric, mode); switch the file "
        "to change the lens (pass / g-pass τ / mg-pass) or the mode (base / non-think / think)."
    ),
    "bench_compare": (
        "Four language curves (EN/PT/TR/TR-OL) per cell. The __nojudge variant shows raw language "
        "transfer (curves bunched = transfers; spread = language-bound). The __cot variant adds a column "
        "per judge: a language whose curve drops more across judge columns has more unfaithful CoT there. "
        "TR-OL is a harder different set, so its offset is expected, not pure translation loss."
    ),
    "size_compare": (
        "Rows = family, cols = benchmark; one curve per model coloured by size (dark=small → yellow=large). "
        "The __cot__<judge> variant overlays each model's cot dashed: if the solid–dashed gap shrinks as "
        "colour warms, faithfulness improves with scale. Use it to test whether the veto is a small-model "
        "artifact — it largely is."
    ),
    "veto_curve": (
        "y = No-Judge − cot (the veto itself) vs K. judge_effect squeezes the gap near saturation; here the "
        "gap IS the y-value, so it stays visible at high pass@k. Rising line → extra sampling buys more "
        "answer-correct than reasoning-correct hits (lucky guesses accumulate)."
    ),
    "mode_compare": (
        "The only family that pairs a base checkpoint with its instruct sibling of the same size. Rows = "
        "size, cols = benchmark; grey/blue/red solids = pretrained / non-think / think (No-Judge), dashed = "
        "cot. Compare solids for 'does reasoning help accuracy?' (can invert!) and each solid–dash gap for "
        "'does reasoning help faithfulness?' (think's gap is smallest)."
    ),
    "per_model": (
        "One model+mode per file. Rows = the six metrics (pass, g-pass τ×4, mg-pass), cols = benchmark; "
        "each cell is a No-Judge-vs-cot cell. Read down a column to see the veto shrink as the metric gets "
        "stricter (it removes lucky single hits, not consistent skill); across a row for the language profile."
    ),
    "tables": (
        "Absolute, cross-comparable numbers (×100). __nojudge = the leaderboard per benchmark; "
        "__cotdelta__<judge> = the veto (No-Judge − cot), red = bigger; headline__<judge> = 'pass / cot-pass' "
        "per cell. Unlike the curve families you CAN compare across rows/columns here."
    ),
    "comparisons": (
        "Multilingual veto heatmaps: rows = (model·mode), cols = language, value = No-Judge − cot, diverging "
        "RdBu centred at 0 (deep red = large veto) — colour intensity is comparable across cells here. The "
        "pass_vs_cot_k{1,64}.csv files are the long-format audit trail behind every number in the suite."
    ),
}

# Representative file picks per family (matched by exact name then prefix);
# missing picks fall back to the first sorted files.
_PREFERRED = {
    "judge_effect": ["pass__think.png", "gpass_t0.5__think.png"],
    "bench_compare": ["pass__think__cot.png", "pass__think__nojudge.png"],
    "size_compare": ["pass__think__nojudge.png", "pass__think__cot__G4-26B.png"],
    "veto_curve": ["pass__think.png", "mgpass__think.png"],
    "mode_compare": ["pass__Qwen.png", "pass__gemma.png"],
    "per_model": ["Q-9B__think.png", "G4-26B__think.png"],
    "tables": ["headline__G4-26B__k64.png", "aime2026__k64__cotdelta__G4-26B.png"],
    "comparisons": ["veto__pass__k64__G4-26B.png", "veto__mgpass__k64__G4-26B.png"],
}


def _png_files(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.glob("*.png"))


def _pick(files: list[str], preferred: list[str], n: int = 2) -> list[str]:
    """Pick up to ``n`` representative files: preferred (exact/prefix) then sorted fill."""
    chosen: list[str] = []
    for want in preferred:
        if want in files and want not in chosen:
            chosen.append(want)
        else:
            stem = want.rsplit(".png", 1)[0]
            for f in files:
                if f.startswith(stem) and f not in chosen:
                    chosen.append(f)
                    break
        if len(chosen) >= n:
            break
    for f in files:  # fill if preferred didn't yield enough
        if len(chosen) >= n:
            break
        if f not in chosen:
            chosen.append(f)
    return chosen[:n]


def _wrap(text: str, width: int = 104) -> str:
    return "\n".join(textwrap.fill(p, width=width) for p in text.split("\n"))


def _save(pdf: PdfPages, fig) -> None:
    pdf.savefig(fig)
    plt.close(fig)


def _cover_page(pdf: PdfPages, plot_dir: Path, present: list[str]) -> None:
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.90, "Plot Atlas", ha="center", fontsize=26, fontweight="bold", color=INK)
    fig.text(0.5, 0.862, "A guided tour of results/report_plots/", ha="center", fontsize=13,
             color=ACCENT, style="italic")

    fig.text(0.1, 0.80, "SUITE MAP", fontsize=10, color=ACCENT, fontweight="bold")
    lines = []
    for fam in present:
        n = len(_png_files(plot_dir / fam))
        lines.append(f"{fam:<16s} {n:>3d} plot(s)   —  {FAMILY_TITLES[fam].split('—',1)[-1].strip()}")
    fig.text(0.1, 0.775, "\n".join(lines), ha="left", va="top", fontsize=9.5, family="monospace",
             color=INK, linespacing=1.6)

    fig.text(0.1, 0.46, "HOW TO READ EVERY FIGURE", fontsize=10, color=ACCENT, fontweight="bold")
    conv = [
        "Solid black = No-Judge (answer-only: pass / g-pass / mg-pass). Dashed colour = cot (a judge "
        "also vetoes reasoning-wrong generations). The gap between them is the veto = unjustified success.",
        "X axis = K on a log2 scale (1→64). The judge is ALWAYS think; there are no non-think judge series.",
        "Y axis = 0 → that cell's own max (not a fixed 0–100%). So compare GAPS within a cell — never bar/line "
        "heights ACROSS cells. For comparable absolute numbers use the tables/ and comparisons/ families.",
        "Six metric lenses: pass (lenient) · g-pass τ=0.25/0.5/0.75/1.0 (τ=1.0 = all k correct) · mg-pass "
        "(consistency). Modes: base / non-think / think.",
    ]
    fig.text(0.1, 0.435, "\n".join("•  " + _wrap(c, 96).replace("\n", "\n    ") for c in conv),
             ha="left", va="top", fontsize=9.7, color=INK, linespacing=1.5)
    fig.text(0.5, 0.06, "Companions: report_highlights.pdf (findings) · docs/report_plots_guide.md (full manual)",
             ha="center", fontsize=8.5, color="#888")
    _save(pdf, fig)


def _image_page(pdf: PdfPages, family: str, img_path: Path, *, caption: str | None,
                tag: str) -> None:
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.962, FAMILY_TITLES.get(family, family), ha="center", fontsize=13,
             fontweight="bold", color=INK)
    fig.text(0.5, 0.938, tag, ha="center", fontsize=9, color=ACCENT)

    img_top = 0.915
    if caption:
        fig.text(0.07, 0.915, _wrap(caption, 110), ha="left", va="top", fontsize=8.8,
                 color=INK, linespacing=1.4)
        # reserve space for the wrapped caption (rough: lines × line-height)
        n_lines = sum(len(textwrap.fill(p, 110).split("\n")) for p in caption.split("\n"))
        img_top = max(0.60, 0.905 - 0.022 * n_lines)

    try:
        img = mpimg.imread(str(img_path))
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"[atlas] could not read {img_path}: {e}")
        plt.close(fig)
        return
    ax = fig.add_axes((0.05, 0.05, 0.90, img_top - 0.07))
    ax.imshow(img)  # default aspect='equal' letterboxes without distortion
    ax.axis("off")
    fig.text(0.5, 0.022, img_path.name, ha="center", fontsize=7.5, color="#666", family="monospace")
    _save(pdf, fig)


def _index_page(pdf: PdfPages, family: str, files: list[str]) -> None:
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.955, FAMILY_TITLES.get(family, family), ha="center", fontsize=12.5,
             fontweight="bold", color=INK)
    fig.text(0.5, 0.930, f"complete index — {len(files)} file(s) in results/report_plots/{family}/",
             ha="center", fontsize=9, color=ACCENT)
    items = [f"• {f}" for f in files]
    half = (len(items) + 1) // 2
    left, right = items[:half], items[half:]
    fig.text(0.07, 0.90, "\n".join(left), ha="left", va="top", fontsize=7.4,
             family="monospace", color=INK, linespacing=1.5)
    if right:
        fig.text(0.53, 0.90, "\n".join(right), ha="left", va="top", fontsize=7.4,
                 family="monospace", color=INK, linespacing=1.5)
    _save(pdf, fig)


def build_atlas(plot_dir: Path | str, output_pdf: Path | str) -> Path:
    """Render the curated plot-atlas PDF from a ``report_plots`` directory."""
    plot_dir = Path(plot_dir)
    out = Path(output_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)

    present = [f for f in FAMILY_ORDER if (plot_dir / f).is_dir() and _png_files(plot_dir / f)]
    if not present:
        raise FileNotFoundError(
            f"No plot families with PNGs under {plot_dir} (run `evalhub report plot` first)."
        )

    with PdfPages(out) as pdf:
        _cover_page(pdf, plot_dir, present)
        for fam in present:
            files = _png_files(plot_dir / fam)
            picks = _pick(files, _PREFERRED.get(fam, []), n=2)
            for i, name in enumerate(picks):
                _image_page(
                    pdf, fam, plot_dir / fam / name,
                    caption=ATLAS_CAPTIONS.get(fam) if i == 0 else None,
                    tag=f"representative {i + 1}/{len(picks)}",
                )
            _index_page(pdf, fam, files)
        d = pdf.infodict()
        d["Title"] = "EvalHub Plot Atlas"
        d["Subject"] = "Curated visual index of results/report_plots/"
    logger.info(f"[atlas] wrote {out}")
    return out
