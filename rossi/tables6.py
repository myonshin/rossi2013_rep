
"""
tables6.py - Python port (scaffold) of tables6.m

This script is the main driver in the original MATLAB codebase. The MATLAB version:
- Loads macro/FX data from Excel workbooks in the same folder
- Applies optional one-sided seasonal adjustment (MA(12) monthly / MA(3) quarterly)
- Constructs relative differentials and first differences
- Runs a menu of forecast-evaluation / stability tests and writes LaTeX tables & EPS figures

What is implemented here
------------------------
1) Data loading from .xls using pandas.
2) Core data transforms (mmult/differ/filter_nan) faithfully ported.
3) LaTeX writer (nicetable) ported.

What is *not* fully implemented yet
-----------------------------------
The original script calls a fairly large suite of econometric test routines
(CW tests, fluctuation tests, panel routines, BMA routines, etc.). Those
are spread across many .m files. This repository includes Python ports for
the key GMM/TVP building blocks (see rossi/tvp.py and rossi/gmm.py). The
high-level forecast test wrappers (e.g., testsoos, testsforBMA, testsPANEL)
are not fully ported in this initial pass.

If you want this script to reproduce the paper tables end-to-end, the next step
is to port the remaining wrappers that orchestrate the forecast comparisons.

Usage
-----
python -m rossi.tables6 --index-freq 2 --seasadj 1

index_freq: 0 yearly, 1 quarterly, 2 monthly
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import mmult, differ, filter_nan
from .dates import tds_m, tds_q, calds_m_string, calds_q_string, calds_a_string

@dataclass
class Tables6Config:
    index_freq: int = 2  # 0 yearly, 1 quarterly, 2 monthly
    seasadj: int = 1
    sel_countries: list[int] = None
    plotvariables: int = 0
    h: int = 1
    h2_base: int = 4  # in MATLAB: 4 quarters; if monthly multiplies by 12
    data_dir: Path = Path(".")

    def __post_init__(self):
        if self.sel_countries is None:
            self.sel_countries = [2,3,4,7,11,13,14,15,16,18,19,20,22,26,30,31,32,36]

def _read_xls(path: Path, sheet: int) -> np.ndarray:
    # MATLAB xlsread(..., sheetIndex) uses 1-based sheet index. pandas uses 0-based.
    df = pd.read_excel(path, sheet_name=sheet-1, header=None, engine="xlrd")
    return df.to_numpy(dtype=float)

def run(cfg: Tables6Config) -> dict[str, np.ndarray]:
    d = cfg.data_dir

    if cfg.index_freq > 0:
        e     = _read_xls(d / "erates.xls",  cfg.index_freq)
        i     = _read_xls(d / "irates.xls",  cfg.index_freq)
        m     = _read_xls(d / "money.xls",   cfg.index_freq)
        y     = _read_xls(d / "rgdp.xls",    cfg.index_freq)
        cpi   = _read_xls(d / "prices.xls",  cfg.index_freq)
        tbill = _read_xls(d / "tbill.xls",   cfg.index_freq)
        i3    = _read_xls(d / "i3month.xls", cfg.index_freq)
        ca    = _read_xls(d / "bop.xls",     2)
        tb    = _read_xls(d / "bop.xls",     3)
        i5    = _read_xls(d / "ilong.xls",   cfg.index_freq)
        ppi   = _read_xls(d / "ppi.xls",     cfg.index_freq)
    else:
        # Yearly path in tables6.m down-samples quarterly sheets; we implement a direct yearly loader fallback.
        e     = _read_xls(d / "erates.xls",  1)[55::4, :]
        i     = _read_xls(d / "irates.xls",  1)[55::4, :]
        m     = _read_xls(d / "money.xls",   1)[55::4, :]
        y     = _read_xls(d / "rgdp.xls",    1)[55::4, :]
        cpi   = _read_xls(d / "prices.xls",  1)[55::4, :]
        tbill = _read_xls(d / "tbill.xls",   1)[55::4, :]
        i3    = _read_xls(d / "i3month.xls", 1)[55::4, :]
        ca    = _read_xls(d / "bop.xls",     2)[55::4, :]
        tb    = _read_xls(d / "bop.xls",     3)[55::4, :]
        i5    = _read_xls(d / "ilong.xls",   1)[55::4, :]
        ppi   = _read_xls(d / "ppi.xls",     1)[55::4, :]

    # Calendar sequences
    if cfg.index_freq == 2:
        tds = tds_m(1957, 1, e.shape[0])
        calds = calds_m_string(1957, 1, e.shape[0])
    elif cfg.index_freq == 1:
        tds = tds_q(1957, 1, e.shape[0])
        calds = calds_q_string(1957, 1, e.shape[0])
    else:
        tds = np.arange(1957, 1957 + e.shape[0], dtype=float)
        calds = calds_a_string(1957, e.shape[0])

    # Seasonal adjustment: one-sided moving average (backward) via filter then NaN burn-in
    if cfg.seasadj == 1 and cfg.index_freq in (1,2):
        if cfg.index_freq == 1:
            MAweight = np.array([1/3, 1/3, 1/3], dtype=float)
        else:
            MAweight = np.ones(12, dtype=float) / 12.0
        ppi = filter_nan(MAweight, ppi)
        m   = filter_nan(MAweight, m)
        y   = filter_nan(MAweight, y)
        cpi = filter_nan(MAweight, cpi)
        ca  = filter_nan(MAweight, ca)
        tb  = filter_nan(MAweight, tb)

    # Transforms used throughout tables6.m
    # MATLAB: idiff=i - mmult(i(:,end), ones(size(i)))
    idiff = i - mmult(i[:, -1], np.ones_like(i))
    D_idiff = differ(idiff, 1)

    ydiff = np.log(y) - mmult(np.log(y[:, -1]), np.ones_like(y))
    D_ydiff = differ(ydiff, 1)

    mdiff = np.log(m) - mmult(np.log(m[:, -1]), np.ones_like(m))
    D_mdiff = differ(mdiff, 1)

    cpidiff = np.log(cpi) - mmult(np.log(cpi[:, -1]), np.ones_like(cpi))
    D_cpidiff = differ(cpidiff, 1)

    Dcpi = np.log(cpi)
    D_cpi = differ(Dcpi, 1)

    e_log = np.log(e)
    D_e = differ(e_log, 1)

    tbilldiff = np.log(tbill) - mmult(np.log(tbill[:, -1]), np.ones_like(tbill))
    D_tbilldiff = differ(tbilldiff, 1)

    i3diff = np.log(i3) - mmult(np.log(i3[:, -1]), np.ones_like(i3))
    D_i3diff = differ(i3diff, 1)

    # Forecast horizons
    h = cfg.h
    h2 = cfg.h2_base
    if cfg.index_freq == 2:
        h2 = 12 * h2

    return {
        "tds": tds,
        "calds": np.array(calds, dtype=object),
        "D_e": D_e,
        "D_idiff": D_idiff,
        "D_ydiff": D_ydiff,
        "D_mdiff": D_mdiff,
        "D_cpidiff": D_cpidiff,
        "D_cpi": D_cpi,
        "D_tbilldiff": D_tbilldiff,
        "D_i3diff": D_i3diff,
        "h": np.array([h], dtype=int),
        "h2": np.array([h2], dtype=int),
    }

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-freq", type=int, default=2)
    ap.add_argument("--seasadj", type=int, default=1)
    ap.add_argument("--data-dir", type=str, default=".")
    args = ap.parse_args()

    cfg = Tables6Config(index_freq=args.index_freq, seasadj=args.seasadj, data_dir=Path(args.data_dir))
    out = run(cfg)
    print("Loaded and transformed data.")
    print("Keys:", ", ".join(out.keys()))
    print("D_e shape:", out["D_e"].shape)

if __name__ == "__main__":
    main()
