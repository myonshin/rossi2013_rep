# Rossi JEL (Revised) — Python translation (partial)

This is a pragmatic Python port of the core MATLAB utilities and the main driver `tables6.m`.

## What's included
- `rossi/utils.py`: ports of `mmult.m`, `differ.m`, `filterNAN.m`, `cleanNaN.m` (as `clean_nan_rows`)
- `rossi/gmm.py`: ports of `gmmbeta.m`, `gmmvar.m`, `gmmres.m`
- `rossi/tvp.py`: ports of `cdfchic.m`, `pvcalc.m`, `chowgmm.m`, `chowgmmstar.m`, `nyblomstar.m`
- `rossi/latex.py`: port of `nicetable` (in `nicetable2.m`)
- `rossi/dates.py`: ports of `tds_m.m`, `tds_q.m`, `calds_m_string.m`, `calds_q_string.m` plus a small yearly helper
- `rossi/tables6.py`: **scaffold** of the MATLAB main script that loads the Excel data and reproduces
  the key data transformations used in the tables.

## Data
The original `.xls` and `.txt` lookup tables are included under `./data/`.

## Install
This code expects:
- Python 3.10+
- numpy, pandas, scipy
- xlrd (to read `.xls` via pandas)

Example:
```bash
pip install numpy pandas scipy xlrd
```

## Run the `tables6` scaffold
```bash
python -m rossi.tables6 --index-freq 2 --seasadj 1 --data-dir ./data
```

## What's not yet ported
The high-level forecast evaluation wrappers called by `tables6.m` (e.g. `testsoos`, `testsPANEL`,
`testsforBMA`, CW tests, fluctuation tests, etc.) are not fully implemented in this first pass.
The building blocks to port those routines are included in `rossi/gmm.py` and `rossi/tvp.py`.

If you want end-to-end reproduction of the paper tables, we should next port:
- `testsoos*.m` (out-of-sample forecast evaluation and CW tests)
- `CW_test_nan_general.m`
- `Fluctuationh.m`
- `testsPANEL.m`
- `testsforBMA.m` / BMA helpers
