
from __future__ import annotations

from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike

def nicetable(modelname: str, tabvals: ArrayLike, countries: ArrayLike, index_freq: int, nametable: str, out_dir: str | Path = ".") -> Path:
    """
    Port of nicetable (in nicetable2.m) that writes a self-contained LaTeX file.

    Parameters
    ----------
    modelname : str
    tabvals : (n,10)
    countries : (nc, >=13) array-like of country names/labels (strings)
    index_freq : 0 yearly, 1 quarterly, 2 monthly
    nametable : base name for output file
    out_dir : output directory
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if index_freq == 1:
        fname = f"{nametable}_tablesQ.tex"
        freq = "Quarterly"
    elif index_freq == 2:
        fname = f"{nametable}_tablesM.tex"
        freq = "Monthly"
    else:
        fname = f"{nametable}_tablesA.tex"
        freq = "Annual"

    fpath = out_dir / fname

    tab = np.asarray(tabvals, dtype=float)
    ctry = np.asarray(countries)

    def _uname(jj: int) -> str:
        jj1 = jj
        if jj >= ctry.shape[0]:
            jj1 = jj - ctry.shape[0]
        # countries(jj1,1:13) => first 13 chars/cols
        s = "".join([str(x) for x in ctry[jj1].tolist()]) if ctry.ndim > 1 else str(ctry[jj1])
        return s[:13]

    lines = []
    lines.append(r"\documentclass{article}")
    lines.append(r"\title{AR Bench., Rolling Forecasts}")
    lines.append(r"\begin{document}")
    lines.append(r"\begin{tabular}{ l | c | c | c | c | c | c | c | c | c | c }")
    lines.append(rf"\multicolumn{{11}}{{c}}{{\textbf{{{modelname} Model in {freq} Data   }} }} \\ ")
    lines.append(r"\hline \hline")
    lines.append(r"Country & {GC} & {Rossi} & {RMSErw} &  {RMSER} & {pvDM} & {pvCW} & {Fluct} & {pvFluct} & {P} & {IR} \\ \hline \hline")

    for jj in range(tab.shape[0]):
        uname = _uname(jj)
        row = tab[jj, :]
        fmt = lambda v: f"{v:10.2f}" if np.isfinite(v) else "   nan   "
        row_str = " &".join([fmt(v) for v in row[:10]])
        lines.append(f"{uname} &{row_str} \\\\")
    lines.append(r"\hline \hline   \end{tabular}")
    lines.append(r"\end{document}")

    fpath.write_text("\r\n".join(lines), encoding="utf-8")
    return fpath
