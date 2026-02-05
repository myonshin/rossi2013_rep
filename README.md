# Rossi (2013, JEL) Replication (wip as of Feb 2026)

This is a partial replication of `tables6.m` in "Rossi, Barbara (2013). "Exchange Rate Predictability." Journal of Economic Literature 51(4): 1063-1119", and its dependencies in python.

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

## TBD
The high-level forecast evaluation wrappers called by `tables6.m` are not fully implemented. 
- `testsoos*.m` (out-of-sample forecast evaluation and CW tests)
- `CW_test_nan_general.m`
- `Fluctuationh.m`
- `testsPANEL.m`
- `testsforBMA.m` / BMA helpers

The building blocks to port those routines are included in `rossi/gmm.py` and `rossi/tvp.py`.

Also, calds_a_string.m is missing as it was missing also in the original replication material.
