"""Smoke tests for the curated plot-atlas PDF :mod:`evalhub.report.atlas`.

Builds a tiny on-disk ``report_plots`` tree (real PNGs, so ``imread`` works),
then checks representative selection, the multipage PDF, and graceful skipping.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from evalhub.report import atlas as A  # noqa: E402,N812


def _png(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(2, 2))
    plt.plot([0, 1], [0, 1])
    fig.savefig(path)
    plt.close(fig)


@pytest.fixture()
def plot_dir(tmp_path: Path) -> Path:
    root = tmp_path / "report_plots"
    layout = {
        "judge_effect": ["pass__think.png", "gpass_t0.5__think.png", "mgpass__base.png"],
        "tables": ["headline__G4-26B__k64.png", "aime2026__k64__nojudge.png"],
    }
    for fam, names in layout.items():
        for n in names:
            _png(root / fam / n)
    return root


def test_pick_prefers_then_prefix_then_fills():
    files = ["pass__think.png", "z.png", "gpass_t0.5__think.png"]
    assert A._pick(files, ["pass__think.png", "gpass_t0.5__think.png"], 2) == [
        "pass__think.png", "gpass_t0.5__think.png",
    ]
    # prefix fallback when the exact name is absent
    assert A._pick(["pass__think__cot.png", "a.png"], ["pass__think.png"], 1) == ["pass__think__cot.png"]
    # fill from sorted when no preferred matches
    assert A._pick(["a.png", "b.png"], ["nope.png"], 2) == ["a.png", "b.png"]


def test_build_atlas_writes_multipage_pdf(plot_dir, tmp_path):
    out = tmp_path / "nested" / "atlas.pdf"
    res = A.build_atlas(plot_dir, out)
    assert res == out
    assert out.exists() and out.stat().st_size > 0
    data = out.read_bytes()
    pages = data.count(b"/Type /Page") - data.count(b"/Type /Pages")
    # cover + 2 families x (<=2 representative images + 1 index)
    assert pages >= 4


def test_build_atlas_raises_on_empty(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        A.build_atlas(empty, tmp_path / "x.pdf")


def test_build_atlas_skips_pngless_families(tmp_path):
    root = tmp_path / "pd"
    (root / "per_model").mkdir(parents=True)  # exists but empty -> skipped
    _png(root / "veto_curve" / "pass__think.png")
    out = tmp_path / "a.pdf"
    A.build_atlas(root, out)
    assert out.exists() and out.stat().st_size > 0
