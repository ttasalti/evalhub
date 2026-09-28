# `analysis/src/`

One script per figure, table or quoted statistic; the mapping from paper
element to script is in [`../README.md`](../README.md). `common.py` resolves
every path, so a script runs from any directory as
`PYTHONPATH=analysis/src python analysis/src/<name>.py`. `palette.py` holds the
shared colours; `vendor/` holds MaskLID (MIT) and the FLORES-200 label list.
Scripts whose name starts with `fig_` write a PNG, `tab_` a LaTeX table; the
rest write text summaries or data-layer CSVs.
